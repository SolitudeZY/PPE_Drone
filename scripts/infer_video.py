"""Real-time PPE compliance detection on drone video.

Pipeline:
    YOLOv8 (4-class) + ByteTrack (on person) -> per-frame PPE association via
    geometric containment -> per-track temporal voting (TrackVoter) -> render
    annotated mp4 (H.264) + alerts.jsonl.

Usage:
    python scripts/infer_video.py \
        --weights runs/ppe/.../best.pt \
        --source 'test-video/test-video-1倍焦距切换.mp4' \
        --imgsz 1280 --conf 0.4 \
        --vote-window 15 --vote-thresh 10 --min-person-px 24 \
        --save-dir runs/ppe/infer
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from utils_ppe import PPEStatus, TrackVoter, associate_ppe

ROOT = Path(__file__).resolve().parents[1]

# BGR colors (used by cv2.rectangle); PIL drawing converts to RGB internally.
COLOR_OK = (0, 200, 0)
COLOR_BAD = (0, 0, 255)
COLOR_UNK = (180, 180, 180)
COLOR_PPE = (255, 200, 0)
COLOR_HEAD = (200, 0, 200)

# Candidate fonts that support CJK (first existing one wins).
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # ASCII fallback
]


def load_font(size: int) -> ImageFont.ImageFont|ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="video path or 0 for webcam")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.4,
                   help="global confidence threshold for the YOLO model")
    p.add_argument("--helmet-conf", type=float, default=0.45,
                   help="extra filter: drop helmet detections below this conf")
    p.add_argument("--vest-conf", type=float, default=0.30,
                   help="extra filter: drop vest detections below this conf")
    p.add_argument("--vest-negative-min-px", type=int, default=80,
                   help="only declare 'no vest' if person shorter side >= this px")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true", help="FP16 inference")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--vote-window", type=int, default=15)
    p.add_argument("--vote-thresh", type=int, default=10)
    p.add_argument("--min-person-px", type=int, default=24,
                   help="persons whose shorter side < this are marked unknown")
    p.add_argument("--contain-thr", type=float, default=0.5)
    p.add_argument("--save-dir", default=str(ROOT / "runs" / "ppe" / "infer"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--draw-ppe-boxes", action="store_true",
                   help="also draw helmet/vest/head boxes (off by default for cleaner output)")
    p.add_argument("--font-size", type=int, default=18)
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 = no limit; otherwise stop after N frames")
    return p.parse_args()


def _bgr_to_rgb(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (c[2], c[1], c[0])


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


class FrameAnnotator:
    """Batch text/box drawing on a frame using PIL (CJK-capable).

    Boxes are drawn with cv2 (fast); text labels are drawn with PIL once per
    frame in a single image conversion to keep overhead low.
    """

    def __init__(self, font_size: int = 18) -> None:
        self.font = load_font(font_size)
        self._labels: List[Tuple[int, int, str, Tuple[int, int, int]]] = []

    def reset(self) -> None:
        self._labels.clear()

    def box(self, frame: np.ndarray, xyxy, color: Tuple[int, int, int],
            label: str = "", thickness: int = 2) -> None:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label:
            self._labels.append((x1, y1, label, color))

    def commit(self, frame: np.ndarray) -> np.ndarray:
        if not self._labels:
            return frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for x, y, text, color_bgr in self._labels:
            color_rgb = _bgr_to_rgb(color_bgr)
            try:
                l, t, r, b = draw.textbbox((0, 0), text, font=self.font)
                tw, th = r - l, b - t
            except AttributeError:
                tw, th = draw.textsize(text, font=self.font)
            pad = 2
            ty = max(0, y - th - 2 * pad)
            draw.rectangle([x, ty, x + tw + 2 * pad, ty + th + 2 * pad],
                           fill=color_rgb)
            draw.text((x + pad, ty + pad), text, font=self.font,
                      fill=(255, 255, 255))
        self.reset()
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def status_label(h: Optional[bool], v: Optional[bool]) -> str:
    helmet_txt = "戴帽" if h is True else ("未戴帽" if h is False else "帽?")
    vest_txt = "穿衣" if v is True else ("未穿衣" if v is False else "衣?")
    return f"{helmet_txt} | {vest_txt}"


def status_color(h: Optional[bool], v: Optional[bool]) -> Tuple[int, int, int]:
    if h is False or v is False:
        return COLOR_BAD
    if h is True and v is True:
        return COLOR_OK
    return COLOR_UNK


def _resolve_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg binary. Prefer system, fall back to imageio-ffmpeg."""
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
    """Pipe BGR frames to ffmpeg for libx264 encoding (much smaller than mp4v)."""

    def __init__(self, path: Path, fps: float, size: Tuple[int, int],
                 ffmpeg_bin: str, crf: int = 23, preset: str = "veryfast") -> None:
        w, h = size
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{fps:.6f}",
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", preset,
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
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

    def isOpened(self) -> bool:  # noqa: N802 (cv2-style API)
        return self.proc.poll() is None


def open_writer(path: Path, fps: float, size: Tuple[int, int]):
    """Prefer ffmpeg/libx264 (smallest); fall back to OpenCV mp4v."""
    ffmpeg_bin = _resolve_ffmpeg()
    if ffmpeg_bin:
        print(f"VideoWriter: ffmpeg libx264 ({ffmpeg_bin})")
        return FFmpegWriter(path, fps, size, ffmpeg_bin)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")
    print("VideoWriter codec: mp4v (ffmpeg not found, file will be larger)")
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

    track_classes = [
        class_ids["person"],
        class_ids["helmet"],
        class_ids["vest"],
    ]
    if class_ids["head"] is not None:
        track_classes.append(class_ids["head"])
    print(f"Resolved model classes: {class_ids}")

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
                        heads.append(b)
                        if args.draw_ppe_boxes:
                            label = "未戴帽" if str(model.names[c]).lower() != "head" else "头部"
                            annotator.box(frame, b, COLOR_HEAD, label, 1)
                    elif c == vest_cls:
                        if cf < args.vest_conf:
                            continue
                        vests.append(b)
                        if args.draw_ppe_boxes:
                            annotator.box(frame, b, COLOR_PPE, "反光衣", 1)

            for p in persons:
                box = p["box"]
                tid = p["id"]
                w = box[2] - box[0]; h = box[3] - box[1]
                short_side = min(w, h)

                if short_side < args.min_person_px:
                    annotator.box(frame, box, COLOR_UNK,
                                  f"id{tid} 远景", 1)
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
                label = f"id{tid} {status_label(sm_h, sm_v)}"
                annotator.box(frame, box, color, label, 2)

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

            voter.step()

            frame = annotator.commit(frame)

            elapsed = time.time() - t0
            cur_fps = (frame_idx + 1) / max(elapsed, 1e-6)
            hud = f"frame {frame_idx + 1}/{total}  fps {cur_fps:.1f}  alerts {n_alerts}"
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
    print(f"Output video: {out_video_path}")
    print(f"Alerts:       {alerts_path}  (total {n_alerts})")


if __name__ == "__main__":
    main()
