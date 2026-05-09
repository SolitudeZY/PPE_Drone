"""Generate a local-path data yaml from datasets/wrj_ppe/data.yaml.

The shipped data.yaml has a Windows absolute path (`E:/...`) which makes
ultralytics fail to find the images on this machine. We do NOT modify the
original file; instead we emit `data_local.yaml` next to it, pointing `path`
at the dataset directory on this host.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "wrj_ppe"
SRC = DATA_DIR / "data.yaml"
DST = DATA_DIR / "data_local.yaml"


def main() -> None:
    with SRC.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(DATA_DIR)
    cfg.setdefault("train", "images/train")
    cfg.setdefault("val", "images/val")
    cfg.setdefault("test", "images/test")

    with DST.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote {DST}")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
