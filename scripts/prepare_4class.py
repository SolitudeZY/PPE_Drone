"""
从 6 类数据集创建 4 类训练视图（不修改原始数据集）。
- 新建 data_4class.yaml（只含 person/helmet/vest/head）
- 新建 labels_filtered/ 目录，去掉 crane/excavator 的标注行

原始文件（images, labels）完全不动。

用法：
    python scripts/prepare_4class.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

DATA_DIR = Path("/home/fs-ai/ppe_drone/datasets/20260506")

# 只保留前 4 个类
CLASSES_4 = ["person", "helmet", "vest", "head"]
NC_4 = len(CLASSES_4)
CLASSES_TO_KEEP = {0, 1, 2, 3}  # class 4=crane, 5=excavator 将被丢弃


def create_4class_yaml():
    """新建 data_4class.yaml，标签路径指向 labels_filtered"""
    content = f"""
# Auto-generated 4-class config (原始: 6 classes)
# labels 使用了过滤后的目录，原始标签未修改
path: {DATA_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {NC_4}
names: {CLASSES_4}
"""
    dst = DATA_DIR / "data_4class.yaml"
    dst.write_text(content)
    print(f"  [+] 创建 YAML: {dst}")
    return dst


def filter_labels():
    """遍历 labels/ 中所有 .txt，去掉 class 4/5 的行，写入 labels_filtered/"""
    label_root = DATA_DIR / "labels"
    filtered_root = DATA_DIR / "labels_filtered"

    # 清空重建
    if filtered_root.exists():
        shutil.rmtree(filtered_root)

    total_orig = 0
    total_kept = 0
    total_removed = 0
    removed_this_split = 0

    for split in ["train", "val", "test"]:
        src_dir = label_root / split
        dst_dir = filtered_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"  [W] {src_dir} 不存在，跳过")
            continue

        removed_this_split = 0
        for lbl_file in sorted(src_dir.glob("*.txt")):
            lines = lbl_file.read_text().strip().splitlines()
            kept = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                total_orig += 1
                cls_id = int(line.split()[0])
                if cls_id in CLASSES_TO_KEEP:
                    kept.append(line)
                else:
                    total_removed += 1
                    removed_this_split += 1

            total_kept += len(kept)
            if kept:
                (dst_dir / lbl_file.name).write_text("\n".join(kept) + "\n")

        count_src = len(list(src_dir.glob("*.txt")))
        count_dst = len(list(dst_dir.glob("*.txt")))
        print(f"  {split}: {count_src} files -> {count_dst} files, 去掉 {removed_this_split} 条 class 4/5")

    print(f"\n总计: {total_orig} 条标注 -> 保留 {total_kept} 条, 去掉 {total_removed} 条")
    return filtered_root


def main():
    print("=== 创建 4 类训练视图 ===")
    print(f"数据集: {DATA_DIR}")
    print(f"保留: {CLASSES_4}")
    print(f"丢弃: crane (class 4), excavator (class 5)")
    print()

    create_4class_yaml()
    filter_labels()

    print()
    print("[+] 完成！训练时运行:")
    print(f"    python scripts/train_v2.py --data dataset/20260506/data_4class.yaml")
    print()
    print("注：推理时如果模型只输出 4 类，infer 脚本也需对应调整。")


if __name__ == "__main__":
    main()
