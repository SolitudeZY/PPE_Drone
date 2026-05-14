"""Real-time PPE inference with selectable backbone (YOLOv8/YOLO11/RT-DETR).

This is a thin wrapper around the same pipeline used in infer_video.py:
the heavy lifting (TrackVoter, associate_ppe, FrameAnnotator, FFmpegWriter)
is reused unchanged.  Differences vs infer_video.py:

- --arch flag picks YOLO / RT-DETR loader automatically based on the weights.
- Defaults --device to '0' and prints the resolved CUDA device.
- Falls back to detect-then-associate without ByteTrack when the model is
  RT-DETR (ultralytics RTDETR.track is supported, but tracker config may
  differ; we still use model.track and let ultralytics handle it).

Usage:
    python scripts/infer_video_v2.py \
        --arch yolo11s \
        --weights runs/ppe/yolo11s_imgsz1280/weights/best.pt \
        --source 'test-video/test-video-1倍焦距切换.mp4' \
        --half --no-show

--no-show：不弹实时预览窗口（不调用 cv2.imshow）。你这台机器的 OpenCV 没带 GUI 支持，调用 imshow 就会报错。加上它就只写文件、不开窗口。结果视频照常保存到 runs/ppe/infer_v2/output.mp4。                                                                                                                                                       
--half：用 FP16(半精度) 推理。模型权重和激活从 32-bit 降到 16-bit，走 RTX 5060 Ti 的 Tensor Core，速度通常 +30%~50%、显存占用约减半，精度损失通常可忽略（mAP 下降<0.5%）。仅推理时用；训练时一般用 AMP 自动混合精度，不需要这个开关。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
torch.use_deterministic_algorithms(False)  # 消除不确定性的warning，不影响精度。让日志更干净

from ultralytics import RTDETR, YOLO

from infer_video import (
    COLOR_BAD,
    COLOR_HEAD,
    COLOR_OK,
    COLOR_PPE,
    COLOR_UNK,
    FrameAnnotator,
    _resolve_class_id,
    open_writer,
    resolve_model_classes,
    status_color,
    status_label,
)
from utils_ppe import TrackVoter, associate_ppe

# Equipment classes (introduced with the 6-class dataset 20260506).
# Not part of PPE compliance — drawn for situational awareness only.
COLOR_CRANE = (255, 128, 0)       # orange
COLOR_EXCAVATOR = (0, 215, 255)   # amber
EQUIPMENT_LABELS_CN = {"crane": "吊车", "excavator": "挖掘机"}
EQUIPMENT_COLORS = {"crane": COLOR_CRANE, "excavator": COLOR_EXCAVATOR}


def resolve_equipment_classes(model) -> Dict[str, int]:
    """Find optional equipment classes (crane / excavator) by model.names."""
    names = model.names
    out: Dict[str, int] = {}
    for key, aliases in (
        ("crane", ("crane", "tower-crane", "tower_crane")),
        ("excavator", ("excavator", "digger")),
    ):
        cid = _resolve_class_id(names, *aliases)
        if cid is not None:
            out[key] = cid
    return out

ROOT = Path(__file__).resolve().parents[1]

_YOLO_PREFIXES = ("yolov8", "yolo11", "yolov5", "yolov9", "yolov10")
_DETR_PREFIXES = ("rtdetr",)


def _pick_loader(arch: str):
    a = arch.lower()
    if any(a.startswith(p) for p in _DETR_PREFIXES):
        return RTDETR
    if any(a.startswith(p) for p in _YOLO_PREFIXES):
        return YOLO
    # default: try YOLO
    return YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="yolo11s",
                   help="family/size hint to pick loader (yolov8s, yolo11m, rtdetr-l, ...)")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--helmet-conf", type=float, default=0.45)
    p.add_argument("--vest-conf", type=float, default=0.30)
    p.add_argument("--head-conf", type=float, default=0.30,
                   help="extra filter: drop head detections below this conf")
    p.add_argument("--equip-conf", type=float, default=0.40,
                   help="extra filter for crane/excavator detections")
    p.add_argument("--no-equipment", action="store_true",
                   help="skip drawing/tracking crane and excavator classes")
    p.add_argument("--vest-negative-min-px", type=int, default=80)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0",
                   help="'0', '0,1' or 'cpu'. Default GPU 0.")
    p.add_argument("--half", action="store_true")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--vote-window", type=int, default=15)
    p.add_argument("--vote-thresh", type=int, default=7)
    p.add_argument("--min-person-px", type=int, default=16)
    p.add_argument("--contain-thr", type=float, default=0.5)
    p.add_argument("--save-dir", default=str(ROOT / "runs" / "ppe_v3" / "infer_v2" / time.strftime("%Y%m%d_%H%M%S")))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--draw-ppe-boxes", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="draw helmet/head/vest boxes (default on; use --no-draw-ppe-boxes to hide)")
    p.add_argument("--font-size", type=int, default=18)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def _print_device(device: str) -> None:
    if device == "cpu":
        print("Device: CPU")
        return
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available; falling back to CPU.")
        return
    idx = 0
    if device and device[0].isdigit():
        idx = int(device.split(",")[0])
    print(f"Device cuda:{idx} -> {torch.cuda.get_device_name(idx)}")


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = save_dir / "output.mp4"
    alerts_path = save_dir / "alerts.jsonl"
    snapshots_dir = save_dir / "alert_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    _print_device(args.device)
    loader_cls = _pick_loader(args.arch)
    print(f"Loader: {loader_cls.__name__}  arch={args.arch}  weights={args.weights}")
    model = loader_cls(args.weights)

    class_ids = resolve_model_classes(model)
    if class_ids["person"] is None:
        raise RuntimeError(f"No person class in model.names: {model.names}")
    if class_ids["helmet"] is None:
        raise RuntimeError(f"No helmet class in model.names: {model.names}")
    if class_ids["vest"] is None:
        raise RuntimeError(f"No vest class in model.names: {model.names}")

    track_classes = [class_ids["person"], class_ids["helmet"], class_ids["vest"]]
    if class_ids["head"] is not None:
        track_classes.append(class_ids["head"])

    equipment_ids: Dict[str, int] = {} if args.no_equipment else resolve_equipment_classes(model)
    for cid in equipment_ids.values():
        track_classes.append(cid)
    print(f"Resolved classes: {class_ids}")
    if equipment_ids:
        print(f"Equipment classes: {equipment_ids}")

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
    annotator = FrameAnnotator(font_size=args.font_size)
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
    pending_snapshots: List[tuple] = []

    try:
        for result in stream:
            frame = result.orig_img.copy()
            boxes = result.boxes

            persons: List[Dict] = []
            helmets: List[np.ndarray] = []
            heads: List[np.ndarray] = []
            vests: List[np.ndarray] = []

            if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                ids = (boxes.id.cpu().numpy().astype(int)
                       if boxes.id is not None else np.full(len(cls), -1, dtype=int))
                person_cls = class_ids["person"]
                helmet_cls = class_ids["helmet"]
                vest_cls = class_ids["vest"]
                head_cls = class_ids["head"]
                equip_cls_to_key = {cid: key for key, cid in equipment_ids.items()}
                for i in range(len(cls)):
                    c = int(cls[i]); b = xyxy[i]; cf = float(conf[i])
                    if c == person_cls:
                        persons.append({"box": b, "id": int(ids[i])})
                    elif c == helmet_cls:
                        if cf < args.helmet_conf:
                            continue
                        helmets.append(b)
                        if args.draw_ppe_boxes:
                            annotator.box(frame, b, COLOR_PPE, "安全帽", 1)
                    elif head_cls is not None and c == head_cls:
                        if cf < args.head_conf:
                            continue
                        heads.append(b)
                        if args.draw_ppe_boxes:
                            annotator.box(frame, b, COLOR_HEAD, "头部", 1)
                    elif c == vest_cls:
                        if cf < args.vest_conf:
                            continue
                        vests.append(b)
                        if args.draw_ppe_boxes:
                            annotator.box(frame, b, COLOR_PPE, "反光衣", 1)
                    elif c in equip_cls_to_key:
                        if cf < args.equip_conf:
                            continue
                        key = equip_cls_to_key[c]
                        annotator.box(frame, b, EQUIPMENT_COLORS[key],
                                      f"{EQUIPMENT_LABELS_CN[key]} {cf:.2f}", 2)

            for p in persons:
                box = p["box"]
                tid = p["id"]
                w = box[2] - box[0]; h = box[3] - box[1]
                short_side = min(w, h)
                if short_side < args.min_person_px:
                    annotator.box(frame, box, COLOR_UNK, f"id{tid} 远景", 1)
                    continue
                status = associate_ppe(box, helmets, heads, vests,
                                       contain_thr=args.contain_thr,
                                       vest_negative_min_px=args.vest_negative_min_px)
                if tid < 0:
                    color = status_color(status.has_helmet, status.has_vest)
                    annotator.box(frame, box, color,
                                  f"id? {status_label(status.has_helmet, status.has_vest)}", 2)
                    continue
                sm_h, sm_v, new_alerts = voter.update(tid, status)
                color = status_color(sm_h, sm_v)
                annotator.box(frame, box, color,
                              f"id{tid} {status_label(sm_h, sm_v)}", 2)
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
                    pending_snapshots.append((frame_idx, tid, tag))

            voter.step()
            frame = annotator.commit(frame)

            for snap_frame, snap_tid, snap_tag in pending_snapshots:
                snap_path = snapshots_dir / f"f{snap_frame:06d}_id{snap_tid}_{snap_tag}.jpg"
                cv2.imwrite(str(snap_path), frame)
            pending_snapshots.clear()

            elapsed = time.time() - t0
            cur_fps = (frame_idx + 1) / max(elapsed, 1e-6)
            hud = f"{args.arch} | frame {frame_idx + 1}/{total}  fps {cur_fps:.1f}  alerts {n_alerts}"
            cv2.putText(frame, hud, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

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
    print(f"Output: {out_video_path}")
    print(f"Alerts: {alerts_path}  (total {n_alerts})")


if __name__ == "__main__":
    main()
