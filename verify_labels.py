"""
可视化验证 YOLO 多边形标注是否正确。

随机或指定挑选一张含裂缝的样本，将 YOLO 多边形画到原图上，输出 verify_output.jpg
用法:
    python verify_labels.py                                    # 默认: train, 默认数据集路径
    python verify_labels.py --split test                       # 验证 test 集
    python verify_labels.py --dataset-root /path/to/dataset    # 自定义数据集路径
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="可视化验证 YOLO 标注是否正确")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./crack_segmentation_dataset",
        help="数据集根目录 (默认: ./crack_segmentation_dataset)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="要验证的子集 (默认: train)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="verify_output.jpg",
        help="可视化输出文件路径 (默认: verify_output.jpg)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    img_dir = dataset_root / args.split / "images"
    label_dir = dataset_root / args.split / "labels"

    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"目录不存在，请确保已运行过 convert_masks.py:\n"
            f"  {img_dir}\n  {label_dir}"
        )

    # 找一个 label 不为空的样本
    img_path = None
    label_path = None
    for cand_img in sorted(img_dir.glob("*.jpg")):
        cand_label = label_dir / (cand_img.stem + ".txt")
        if cand_label.exists() and cand_label.stat().st_size > 10:
            img_path = cand_img
            label_path = cand_label
            break

    if img_path is None:
        print("❌ 没找到任何含裂缝标注的样本")
        return

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # 画出 YOLO 多边形
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            coords = coords.astype(int)
            cv2.polylines(img, [coords], True, (0, 255, 0), 2)

    cv2.imwrite(args.output, img)
    print(f"✅ 已保存 {args.output}，原图: {img_path.name}")


if __name__ == "__main__":
    main()
