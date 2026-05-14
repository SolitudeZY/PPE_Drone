"""Train larger detector variants (YOLOv8 s/m/l/x, YOLO11, RT-DETR) on wrj_ppe.

Reuses dataset / augmentation choices from train.py but exposes a wider model
zoo via --arch.  Default device is GPU 0; pass --device cpu to force CPU.

Usage:
    python scripts/train_v2.py --arch yolo11s --imgsz 1280 --epochs 150
    python scripts/train_v2.py --arch rtdetr-l --imgsz 1280 --epochs 100 --batch 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "20260514" / "data_local.yaml"
RUNS_DIR = ROOT / "runs" / "ppe_v3"

# arch -> (weights file, loader class)
_ARCH_TABLE: dict[str, tuple[str, type]] = {
    # YOLOv8
    "yolov8n": ("yolov8n.pt", YOLO),
    "yolov8s": ("yolov8s.pt", YOLO),
    "yolov8m": ("yolov8m.pt", YOLO),
    "yolov8l": ("yolov8l.pt", YOLO),
    "yolov8x": ("yolov8x.pt", YOLO),
    # YOLO11 (latest Ultralytics family)
    "yolo11n": ("yolo11n.pt", YOLO),
    "yolo11s": ("yolo11s.pt", YOLO),
    "yolo11m": ("yolo11m.pt", YOLO),
    "yolo11l": ("yolo11l.pt", YOLO),
    "yolo11x": ("yolo11x.pt", YOLO),
    # RT-DETR (real-time DETR variant)
    "rtdetr-l": ("rtdetr-l.pt", RTDETR),
    "rtdetr-x": ("rtdetr-x.pt", RTDETR),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="yolo11s", choices=sorted(_ARCH_TABLE.keys()),
                   help="backbone family + size; default yolo11s")
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=-1, help="-1 = AutoBatch")
    p.add_argument("--device", default="0",
                   help="GPU id(s) like '0' or '0,1', or 'cpu'. Default 0.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--name", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "False"])
    p.add_argument("--no-strong-aug", action="store_true",
                   help="disable mosaic/copy_paste/mixup (recommended for RT-DETR)")
    # Note: fl-gamma and image-weights have been removed as they are no longer
    # supported directly in kwargs by the installed version of ultralytics.
    return p.parse_args()


def _print_device(device: str) -> None:
    if device == "cpu":
        print("Device: CPU")
        return
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available; ultralytics will fall back to CPU.")
        return
    ids = [int(x) for x in device.split(",") if x.strip().isdigit()]
    if not ids:
        ids = [0]
    for i in ids:
        print(f"Device cuda:{i} -> {torch.cuda.get_device_name(i)}")


def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"{args.data} not found. Run `python scripts/fix_data_yaml.py` first."
        )

    weights, loader_cls = _ARCH_TABLE[args.arch]
    _print_device(args.device)
    print(f"Architecture: {args.arch}  ({loader_cls.__name__})  weights={weights}")

    model = loader_cls(weights)
    name = args.name or f"{args.arch}_imgsz{args.imgsz}"
    cache = False if args.cache == "False" else args.cache

    is_detr = loader_cls is RTDETR
    use_strong_aug = (not args.no_strong_aug) and (not is_detr)

    # Note: Ultralytics currently doesn't expose fl_gamma and image_weights 
    # directly via the kwargs interface in recent versions.
    # To mitigate class imbalance, we rely on the strong augmentation we already have.
    # Removed invalid kwargs to prevent SyntaxError.
    train_kwargs: dict = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cos_lr=True,
        cache=cache,
        project=str(RUNS_DIR),
        name=name,
        resume=args.resume,
        save=True,
        plots=True,
        exist_ok=True,
    )

    if use_strong_aug:
        # YOLO benefits from these for small / far-field objects.
        train_kwargs.update(
            mosaic=1.0,
            close_mosaic=15,
            copy_paste=0.3,
            mixup=0.10,
            scale=0.5,
            degrees=5.0,
            translate=0.1,
            fliplr=0.5,
            hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        )
    else:
        # RT-DETR is sensitive to mosaic/copy_paste; keep mild geometric aug only.
        train_kwargs.update(
            mosaic=0.0,
            copy_paste=0.0,
            mixup=0.0,
            scale=0.5,
            translate=0.1,
            fliplr=0.5,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        )

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
