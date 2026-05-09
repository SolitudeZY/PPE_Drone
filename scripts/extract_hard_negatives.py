"""Extract hard-negative training samples from inference alerts.

Most false positives in this project come from textured non-person objects
(e.g. red/white traffic cones being mistaken for vest/helmet) and from
far/blurry persons. The most effective fix is to add the offending frames to
the training set as background images (YOLO treats an empty .txt label as
"this image is pure negative").

This script reads `alerts.jsonl` produced by `infer_video.py`, extracts the
corresponding frames from the source video, and writes:

    out/images/<stem>_f{frame:06d}_{alert}.png
    out/labels/<stem>_f{frame:06d}_{alert}.txt   (empty file)

Manual workflow afterwards:

    1. Browse `out/images/`, DELETE images that actually contain a real
       violation (true positive). Keep only the false-positive frames.
    2. For frames that contain BOTH a real PPE violation and a false
       positive (e.g. a real worker plus a misclassified cone), instead of
       keeping the empty .txt, label the frame properly with a tool such as
       LabelImg / Roboflow and replace the empty .txt with real annotations.
    3. Move the kept image/label pairs into
       `datasets/wrj_ppe/images/train/` and
       `datasets/wrj_ppe/labels/train/`, then retrain.

Usage:
    python scripts/extract_hard_negatives.py \
        --alerts runs/ppe/infer/alerts.jsonl \
        --video  'test-video/test-video-1倍焦距切换.mp4' \
        --out    runs/ppe/hard_negatives \
        --min-gap 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alerts", required=True,
                   help="path to alerts.jsonl from infer_video.py")
    p.add_argument("--video", required=True, help="source video path")
    p.add_argument("--out", default=str(ROOT / "runs" / "ppe" / "hard_negatives"))
    p.add_argument("--min-gap", type=int, default=30,
                   help="minimum frame gap between exports of the same track "
                        "and alert type (deduplication)")
    p.add_argument("--max-per-alert", type=int, default=0,
                   help="0 = no cap; otherwise stop after exporting this many "
                        "frames per alert tag")
    p.add_argument("--alert-types", nargs="*", default=None,
                   help="only export these alert tags (e.g. no_vest no_helmet)")
    p.add_argument("--ext", default="png", choices=["png", "jpg"])
    return p.parse_args()


def load_alerts(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_frames(
    alerts: List[dict],
    min_gap: int,
    max_per_alert: int,
    alert_types: List[str] | None,
) -> List[dict]:
    """Deduplicate alerts so we don't export 50 nearly-identical frames.

    Key = (track_id, alert). Within the same key, only keep events that are
    at least `min_gap` frames apart. Globally cap per-alert-type count.
    """
    last_seen: Dict[Tuple[int, str], int] = {}
    per_alert_count: Dict[str, int] = {}
    selected = []
    alerts_sorted = sorted(alerts, key=lambda r: r["frame"])
    for rec in alerts_sorted:
        tag = rec["alert"]
        if alert_types and tag not in alert_types:
            continue
        key = (rec["track_id"], tag)
        last = last_seen.get(key, -10**9)
        if rec["frame"] - last < min_gap:
            continue
        if max_per_alert and per_alert_count.get(tag, 0) >= max_per_alert:
            continue
        last_seen[key] = rec["frame"]
        per_alert_count[tag] = per_alert_count.get(tag, 0) + 1
        selected.append(rec)
    return selected


def extract(video: Path, frames_to_save: Dict[int, List[dict]],
            out_img_dir: Path, out_lbl_dir: Path,
            stem: str, ext: str) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")

    target_frames = sorted(frames_to_save.keys())
    saved = 0
    cur_idx = 0
    target_ptr = 0
    while target_ptr < len(target_frames):
        target = target_frames[target_ptr]
        # Seek if jumping forward by a lot, otherwise read sequentially.
        if target - cur_idx > 30:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            cur_idx = target
        while cur_idx < target:
            ok = cap.grab()
            if not ok:
                break
            cur_idx += 1
        ok, frame = cap.read()
        if not ok:
            break
        cur_idx += 1
        for rec in frames_to_save[target]:
            tag = rec["alert"]
            tid = rec["track_id"]
            name = f"{stem}_f{target:06d}_t{tid}_{tag}"
            img_path = out_img_dir / f"{name}.{ext}"
            lbl_path = out_lbl_dir / f"{name}.txt"
            cv2.imwrite(str(img_path), frame)
            lbl_path.write_text("", encoding="utf-8")  # empty = pure negative
            saved += 1
        target_ptr += 1
    cap.release()
    return saved


def main() -> None:
    args = parse_args()
    alerts_path = Path(args.alerts)
    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    alerts = load_alerts(alerts_path)
    print(f"Loaded {len(alerts)} alert records from {alerts_path}")

    selected = select_frames(alerts, args.min_gap, args.max_per_alert,
                             args.alert_types)
    print(f"Selected {len(selected)} frames after dedup (min_gap={args.min_gap}"
          f"{', max_per_alert=' + str(args.max_per_alert) if args.max_per_alert else ''})")

    # Group selected records by frame number for one-pass video read.
    frames_to_save: Dict[int, List[dict]] = {}
    for rec in selected:
        frames_to_save.setdefault(rec["frame"], []).append(rec)

    stem = video_path.stem.replace(" ", "_")
    saved = extract(video_path, frames_to_save, out_img_dir, out_lbl_dir,
                    stem, args.ext)

    print(f"\nSaved {saved} frame(s).")
    print(f"  Images: {out_img_dir}")
    print(f"  Labels: {out_lbl_dir}  (all empty == background negatives)")
    print("\nNext steps:")
    print("  1) Manually delete TRUE-POSITIVE frames from the images dir")
    print("     (their label .txt should be deleted too).")
    print("  2) For frames mixing real violations + false alarms, replace the")
    print("     empty .txt with proper YOLO annotations.")
    print("  3) Move kept pairs into datasets/wrj_ppe/{images,labels}/train/")
    print("     and retrain.")


if __name__ == "__main__":
    main()
