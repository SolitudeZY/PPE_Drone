"""Video PPE inference for the local Hexmon Hugging Face model.

Example:
    python other/HF-Hexmon/infer_video.py \
        --weights /home/fs-ai/ppe_drone/other/HF-Hexmon/model/best.pt \
        --source /home/fs-ai/ppe_drone/test-video/test-video-1倍焦距切换.mp4 \
        --device 0 --imgsz 1280 --no-show
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils_ppe import TrackVoter, associate_ppe, containment


COLOR_OK = (0, 200, 0)
COLOR_BAD = (0, 0, 255)
COLOR_UNK = (180, 180, 180)
COLOR_PPE = (255, 200, 0)
COLOR_HEAD = (200, 0, 200)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(root / "model" / "best.pt"))
    p.add_argument(
        "--source",
        default=str(ROOT / "test-video" / "test-video-1倍焦距切换.mp4"),
        help="video path or 0 for webcam",
    )
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--person-conf", type=float, default=0.25)
    p.add_argument("--helmet-conf", type=float, default=0.45)
    p.add_argument("--vest-conf", type=float, default=0.30)
    p.add_argument("--helmet-negative-min-px", type=int, default=48)
    p.add_argument("--vest-negative-min-px", type=int, default=80)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true", help="enable FP16 inference")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--vote-window", type=int, default=15)
    p.add_argument("--vote-thresh", type=int, default=10)
    p.add_argument("--min-person-px", type=int, default=24)
    p.add_argument("--min-person-width-px", type=int, default=12)
    p.add_argument("--person-max-aspect", type=float, default=4.8)
    p.add_argument("--top-edge-ignore-px", type=int, default=6)
    p.add_argument("--top-edge-person-aspect", type=float, default=2.6)
    p.add_argument("--contain-thr", type=float, default=0.5)
    p.add_argument("--save-dir", default=str(root / "runs" / "infer_video"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--draw-ppe-boxes", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def status_label(h: Optional[bool], v: Optional[bool]) -> str:
    helmet_txt = "helmet" if h is True else ("no-helmet" if h is False else "helmet?")
    vest_txt = "vest" if v is True else ("no-vest" if v is False else "vest?")
    return f"{helmet_txt} | {vest_txt}"


def status_color(h: Optional[bool], v: Optional[bool]) -> Tuple[int, int, int]:
    if h is False or v is False:
        return COLOR_BAD
    if h is True and v is True:
        return COLOR_OK
    return COLOR_UNK


def _resolve_class_id(model_names: Dict[int, str], *aliases: str) -> Optional[int]:
    alias_set = {alias.lower() for alias in aliases}
    for class_id, name in model_names.items():
        if str(name).lower() in alias_set:
            return int(class_id)
    return None


def resolve_model_classes(model: YOLO) -> Dict[str, Optional[int]]:
    names = model.names
    return {
        "person": _resolve_class_id(names, "person", "human", "worker"),
        "helmet": _resolve_class_id(names, "helmet", "hardhat", "hard-hat"),
        "vest": _resolve_class_id(names, "vest", "safety-vest", "safety vest"),
        "head": _resolve_class_id(names, "head", "no-helmet", "no_helmet", "no helmet"),
    }


def has_contained_item(
    person_box: np.ndarray,
    items: List[np.ndarray],
    contain_thr: float,
    y_lo: float = 0.0,
    y_hi: float = 1.0,
) -> bool:
    if not items:
        return False
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    region = np.array([x1, y1 + h * y_lo, x2, y1 + h * y_hi], dtype=np.float32)
    return any(containment(item, region) >= contain_thr for item in items)


def is_likely_false_person(
    box: np.ndarray,
    helmets: List[np.ndarray],
    heads: List[np.ndarray],
    vests: List[np.ndarray],
    args: argparse.Namespace,
) -> bool:
    w = float(box[2] - box[0])
    h = float(box[3] - box[1])
    if w < args.min_person_width_px:
        return True

    aspect = h / max(w, 1.0)
    has_ppe_evidence = (
        has_contained_item(box, helmets, args.contain_thr, 0.0, 0.45)
        or has_contained_item(box, heads, args.contain_thr, 0.0, 0.45)
        or has_contained_item(box, vests, args.contain_thr, 0.15, 0.8)
    )

    if aspect > args.person_max_aspect and not has_ppe_evidence:
        return True

    if box[1] <= args.top_edge_ignore_px and aspect > args.top_edge_person_aspect and not has_ppe_evidence:
        return True

    return False


def apply_missing_ppe_fallback(
    status,
    box: np.ndarray,
    args: argparse.Namespace,
) -> None:
    short_side = float(min(box[2] - box[0], box[3] - box[1]))
    if status.has_helmet is None and short_side >= args.helmet_negative_min_px:
        status.has_helmet = False
    if status.has_vest is None and short_side >= args.vest_negative_min_px:
        status.has_vest = False


def _resolve_ffmpeg() -> Optional[str]:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class FFmpegWriter:
    def __init__(self, path: Path, fps: float, size: Tuple[int, int], ffmpeg_bin: str) -> None:
        w, h = size
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._size = size

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self._size[0] or frame.shape[0] != self._size[1]:
            frame = cv2.resize(frame, self._size)
        assert self.proc.stdin is not None
        self.proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        self.proc.wait()

    def isOpened(self) -> bool:
        return self.proc.poll() is None


def open_writer(path: Path, fps: float, size: Tuple[int, int]):
    ffmpeg_bin = _resolve_ffmpeg()
    if ffmpeg_bin:
        print(f"VideoWriter: ffmpeg libx264 ({ffmpeg_bin})")
        return FFmpegWriter(path, fps, size, ffmpeg_bin)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")
    print("VideoWriter codec: mp4v")
    return writer


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = save_dir / "output.mp4"
    alerts_path = save_dir / "alerts.jsonl"

    model = YOLO(args.weights)
    class_ids = resolve_model_classes(model)
    if class_ids["person"] is None:
        raise RuntimeError(f"Could not find a person/human class in model names: {model.names}")
    if class_ids["helmet"] is None:
        raise RuntimeError(f"Could not find a helmet class in model names: {model.names}")
    if class_ids["vest"] is None:
        raise RuntimeError(f"Could not find a vest class in model names: {model.names}")

    track_classes = [class_ids["person"], class_ids["helmet"], class_ids["vest"]]
    if class_ids["head"] is not None:
        track_classes.append(class_ids["head"])
    print(f"Resolved model classes: {class_ids}")
    print(f"Using device: {args.device}")

    cap = cv2.VideoCapture(args.source if not args.source.isdigit() else int(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = open_writer(out_video_path, fps, (width, height))
    voter = TrackVoter(window=args.vote_window, thresh=args.vote_thresh)
    alerts_f = alerts_path.open("w", encoding="utf-8")

    stream = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        tracker=args.tracker,
        classes=track_classes,
        persist=True,
        stream=True,
        verbose=False,
    )

    t0 = time.time()
    frame_idx = 0
    n_alerts = 0

    try:
        for result in stream:
            frame = result.orig_img.copy()
            boxes = result.boxes

            persons: List[Dict[str, object]] = []
            helmets: List[np.ndarray] = []
            heads: List[np.ndarray] = []
            vests: List[np.ndarray] = []

            if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else np.full(len(cls), -1, dtype=int)
                )
                for i in range(len(cls)):
                    c = int(cls[i])
                    b = xyxy[i]
                    cf = float(conf[i])
                    if c == class_ids["person"]:
                        if cf < args.person_conf:
                            continue
                        persons.append({"box": b, "id": int(ids[i])})
                    elif c == class_ids["helmet"]:
                        if cf >= args.helmet_conf:
                            helmets.append(b)
                            if args.draw_ppe_boxes:
                                cv2.rectangle(frame, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), COLOR_PPE, 1)
                    elif class_ids["head"] is not None and c == class_ids["head"]:
                        heads.append(b)
                        if args.draw_ppe_boxes:
                            cv2.rectangle(frame, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), COLOR_HEAD, 1)
                    elif c == class_ids["vest"]:
                        if cf >= args.vest_conf:
                            vests.append(b)
                            if args.draw_ppe_boxes:
                                cv2.rectangle(frame, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), COLOR_PPE, 1)

            for person in persons:
                box = person["box"]
                tid = int(person["id"])
                assert isinstance(box, np.ndarray)
                short_side = min(box[2] - box[0], box[3] - box[1])

                if short_side < args.min_person_px:
                    cv2.rectangle(
                        frame,
                        tuple(box[:2].astype(int)),
                        tuple(box[2:].astype(int)),
                        COLOR_UNK,
                        1,
                    )
                    continue

                if is_likely_false_person(box, helmets, heads, vests, args):
                    continue

                status = associate_ppe(
                    box,
                    helmets,
                    heads,
                    vests,
                    contain_thr=args.contain_thr,
                    vest_negative_min_px=args.vest_negative_min_px,
                )
                apply_missing_ppe_fallback(status, box, args)

                if tid < 0:
                    h_val, v_val = status.has_helmet, status.has_vest
                    color = status_color(h_val, v_val)
                    label = f"id? {status_label(h_val, v_val)}"
                else:
                    h_val, v_val, new_alerts = voter.update(tid, status)
                    color = status_color(h_val, v_val)
                    label = f"id{tid} {status_label(h_val, v_val)}"
                    for tag in new_alerts:
                        n_alerts += 1
                        rec = {
                            "frame": frame_idx,
                            "time_sec": frame_idx / fps,
                            "track_id": tid,
                            "alert": tag,
                            "bbox_xyxy": [float(v) for v in box.tolist()],
                        }
                        alerts_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            voter.step()

            elapsed = time.time() - t0
            cur_fps = (frame_idx + 1) / max(elapsed, 1e-6)
            hud = f"frame {frame_idx + 1}/{total} fps {cur_fps:.1f} alerts {n_alerts}"
            cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            if not args.no_show:
                cv2.imshow("PPE", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        writer.release()
        alerts_f.close()
        if not args.no_show:
            cv2.destroyAllWindows()

    dur = time.time() - t0
    print(f"Done. {frame_idx} frames in {dur:.1f}s -> {frame_idx / max(dur, 1e-6):.2f} FPS")
    print(f"Output video: {out_video_path}")
    print(f"Alerts:       {alerts_path} (total {n_alerts})")


if __name__ == "__main__":
    main()
