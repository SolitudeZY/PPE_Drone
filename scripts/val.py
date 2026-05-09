"""Evaluate a trained YOLOv8 PPE model on val or test split.

Usage:
    python scripts/val.py --weights runs/ppe/.../best.pt --split val
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "20260506" / "data_local.yaml"
RUNS_DIR = ROOT / "runs" / "ppe_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(RUNS_DIR),
        name=f"val_{args.split}",
        plots=True,
        exist_ok=True,
    )
    print("\n=== Per-class mAP50 ===")
    names = metrics.names
    for i, ap50 in enumerate(metrics.box.ap50):
        print(f"  {names[i]:>8s}: mAP50={ap50:.4f}")
    print(f"\nOverall mAP50    = {metrics.box.map50:.4f}")
    print(f"Overall mAP50-95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
