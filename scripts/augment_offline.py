"""Offline augmentation for the 6-class PPE dataset.

Goal: improve OOD generalization on the test video (different camera/lighting
from the training set) by synthesizing 1 augmented copy per train image that
contains at least one `person` annotation.

What it does
------------
- Iterates `datasets/20260506/labels/train/*.txt`.
- Skips images whose label file has no `person` (class id 0) annotation.
- For each kept image, applies an albumentations pipeline tuned for drone /
  jobsite footage (color jitter, CLAHE, blur, JPEG compression, mild affine,
  optional fog/shadow). Bounding boxes are transformed in YOLO format.
- Writes the new image as `<stem>_aug.<ext>` and the new label as
  `<stem>_aug.txt` next to the originals so they're picked up by the same
  `images/train` and `labels/train` glob.
- Drops a sample entirely if the bbox visibility check causes loss of all
  person boxes (we only augment for person recall, so a frame with no person
  left after transform is useless).

Usage
-----
    python scripts/augment_offline.py
    python scripts/augment_offline.py --dataset datasets/20260506 --seed 0
    python scripts/augment_offline.py --dry-run         # report counts only
    python scripts/augment_offline.py --overwrite       # regenerate _aug files
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import albumentations as A

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "20260506"
PERSON_CLASS_ID = 0
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
AUG_SUFFIX = "_aug"


def build_pipeline() -> A.Compose:
    """Augmentations target jobsite/drone OOD: lighting, compression, blur."""
    return A.Compose(
        [
            # ----- color / lighting -----
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=15, p=0.5
            ),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # ----- blur / compression artefacts -----
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.Defocus(radius=(2, 3), p=1.0),
                ],
                p=0.4,
            ),
            A.ImageCompression(quality_lower=55, quality_upper=90, p=0.4),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            # ----- weather / outdoor -----
            A.OneOf(
                [
                    A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=1.0),
                    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2,
                                   shadow_dimension=5, p=1.0),
                ],
                p=0.15,
            ),
            # ----- mild geometric -----
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-5, 5),
                shear=(-3, 3),
                fit_output=False,
                cval=0,
                p=0.7,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.35,
            min_area=4,
        ),
    )


def read_yolo_label(path: Path) -> Tuple[List[List[float]], List[int]]:
    """Return (bboxes_yolo, class_ids). bboxes are [xc, yc, w, h] normalized."""
    bboxes: List[List[float]] = []
    class_ids: List[int] = []
    if not path.exists():
        return bboxes, class_ids
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            xc, yc, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        xc = min(max(xc, 1e-6), 1 - 1e-6)
        yc = min(max(yc, 1e-6), 1 - 1e-6)
        w = min(max(w, 1e-6), 1.0)
        h = min(max(h, 1e-6), 1.0)
        x1 = max(xc - w / 2, 0.0)
        y1 = max(yc - h / 2, 0.0)
        x2 = min(xc + w / 2, 1.0)
        y2 = min(yc + h / 2, 1.0)
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        class_ids.append(cid)
    return bboxes, class_ids


def write_yolo_label(path: Path, bboxes: List[List[float]],
                     class_ids: List[int]) -> None:
    lines = []
    for cid, (xc, yc, w, h) in zip(class_ids, bboxes):
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(DEFAULT_DATASET),
                   help="dataset root (must contain images/train and labels/train)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="only count candidates, don't write files")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate even if *_aug files already exist")
    p.add_argument("--max-attempts", type=int, default=5,
                   help="per-image: retry transform if all person bboxes get dropped")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ds_root = Path(args.dataset)
    img_dir = ds_root / "images" / "train"
    lbl_dir = ds_root / "labels" / "train"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"Expected {img_dir} and {lbl_dir} to exist")

    pipeline = build_pipeline()

    n_total = 0
    n_with_person = 0
    n_skip_no_person = 0
    n_skip_existing = 0
    n_written = 0
    n_failed = 0

    label_files = sorted(lbl_dir.glob("*.txt"))
    for lbl_path in label_files:
        if AUG_SUFFIX in lbl_path.stem:
            continue
        n_total += 1
        bboxes, class_ids = read_yolo_label(lbl_path)
        if PERSON_CLASS_ID not in class_ids:
            n_skip_no_person += 1
            continue
        n_with_person += 1

        img_path = find_image_for_label(img_dir, lbl_path.stem)
        if img_path is None:
            print(f"[warn] image missing for {lbl_path.name}")
            n_failed += 1
            continue

        out_img = img_dir / f"{lbl_path.stem}{AUG_SUFFIX}{img_path.suffix}"
        out_lbl = lbl_dir / f"{lbl_path.stem}{AUG_SUFFIX}.txt"
        if (not args.overwrite) and out_img.exists() and out_lbl.exists():
            n_skip_existing += 1
            continue

        if args.dry_run:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] cv2 failed to read {img_path}")
            n_failed += 1
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        success = False
        for _ in range(args.max_attempts):
            try:
                out = pipeline(image=img_rgb, bboxes=bboxes, class_ids=class_ids)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] transform failed on {img_path.name}: {e}")
                continue
            new_bboxes = out["bboxes"]
            new_cids = out["class_ids"]
            if any(int(c) == PERSON_CLASS_ID for c in new_cids):
                new_img = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_img), new_img)
                write_yolo_label(out_lbl, [list(b) for b in new_bboxes],
                                 [int(c) for c in new_cids])
                n_written += 1
                success = True
                break
        if not success:
            n_failed += 1

    print("\n=== augment_offline summary ===")
    print(f"label files scanned         : {n_total}")
    print(f"  with person (class 0)     : {n_with_person}")
    print(f"  skipped (no person)       : {n_skip_no_person}")
    print(f"  skipped (already _aug)    : {n_skip_existing}")
    print(f"new pairs written           : {n_written}")
    print(f"failed (no person survived) : {n_failed}")
    if args.dry_run:
        print("(dry-run: no files written)")


if __name__ == "__main__":
    main()
