"""Train YOLOv8 on the wrj_ppe drone dataset.

Key design choices (see plan file):
- Single-stage 4-class detection (person/helmet/vest/head).
- Large input size (1280) to preserve far-field small objects.
- Strong small-object oriented augmentations.

Usage:
    python scripts/train.py --model n --imgsz 1280 --epochs 150 --batch -1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "wrj_ppe" / "data_local.yaml"
RUNS_DIR = ROOT / "runs" / "ppe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["n", "s", "m"], default="n",
                   help="YOLOv8 size: n=nano (default), s=small, m=medium")
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--name", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "False"],
                   help="dataset cache mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"{args.data} not found. Run `python scripts/fix_data_yaml.py` first."
        )

    weights = f"yolov8{args.model}.pt"
    model = YOLO(weights)

    name = args.name or f"yolov8{args.model}_imgsz{args.imgsz}"
    cache = False if args.cache == "False" else args.cache

    model.train(
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
        # ----- small-object friendly augmentations -----
        mosaic=1.0,
        close_mosaic=15,
        copy_paste=0.3,
        mixup=0.10,
        scale=0.5,
        degrees=5.0,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        # save best + last
        save=True,
        plots=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
