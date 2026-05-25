"""
将裂缝分割数据集的 mask 图像转换为 YOLO 多边形标注格式。

用法:
    python convert_masks.py                                    # 默认数据集路径 ./crack_segmentation_dataset
    python convert_masks.py --dataset-root /path/to/dataset    # 自定义路径
    python convert_masks.py --min-area 10                      # 自定义噪点过滤阈值

数据集目录结构要求:
    <dataset_root>/
    ├── train/
    │   ├── images/   *.jpg, *.png
    │   └── masks/    对应文件名的 mask 图
    └── test/
        ├── images/
        └── masks/

输出: 在 train/labels/ 和 test/labels/ 下生成同名 .txt 标注文件
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def mask_to_yolo_polygons(mask_path: Path, min_area: int = 20) -> list:
    """把一张 mask 图转成 YOLO 多边形格式 (归一化坐标列表)。"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # 二值化（白色=裂缝）
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape

    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        # 过滤太小的噪点
        if cv2.contourArea(cnt) < min_area:
            continue
        # 简化轮廓（减少点数，加速训练）
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        # 归一化到 [0, 1]
        coords = approx.reshape(-1, 2).astype(float)
        coords[:, 0] /= w
        coords[:, 1] /= h
        polygons.append(coords.flatten().tolist())

    return polygons


def process_split(dataset_root: Path, split_name: str, min_area: int) -> None:
    """处理一个数据集子集 (train 或 test)。"""
    img_dir = dataset_root / split_name / "images"
    mask_dir = dataset_root / split_name / "masks"
    label_dir = dataset_root / split_name / "labels"

    if not img_dir.exists():
        print(f"⚠️  {img_dir} 不存在，跳过 {split_name}")
        return

    label_dir.mkdir(exist_ok=True)
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"\n处理 {split_name}: 共 {len(img_files)} 张图")

    crack_count = 0
    noncrack_count = 0

    for img_path in tqdm(img_files):
        # 找对应的 mask（先尝试同名，再尝试 .png 后缀）
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = mask_dir / (img_path.stem + ".png")

        label_path = label_dir / (img_path.stem + ".txt")

        # 非裂缝图：写空文件（YOLO 把它当负样本）
        if "noncrack" in img_path.stem.lower() or not mask_path.exists():
            label_path.touch()
            noncrack_count += 1
            continue

        polygons = mask_to_yolo_polygons(mask_path, min_area=min_area)

        with open(label_path, "w") as f:
            for poly in polygons:
                line = "0 " + " ".join(f"{x:.6f}" for x in poly)
                f.write(line + "\n")

        if polygons:
            crack_count += 1
        else:
            noncrack_count += 1

    print(f"  ✅ {split_name}: 含裂缝 {crack_count} 张，无裂缝 {noncrack_count} 张")


def main():
    parser = argparse.ArgumentParser(
        description="将裂缝 mask 数据集转换为 YOLO 多边形标注格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./crack_segmentation_dataset",
        help="数据集根目录 (默认: ./crack_segmentation_dataset)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="噪点过滤阈值，小于此面积的轮廓将被忽略 (默认: 20)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="要处理的数据集子集 (默认: train test)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    print(f"📁 数据集根目录: {dataset_root}")
    print(f"🔧 噪点过滤阈值: {args.min_area}")
    print(f"📦 待处理子集: {args.splits}")

    for split in args.splits:
        process_split(dataset_root, split, args.min_area)

    print("\n🎉 全部转换完成！")


if __name__ == "__main__":
    main()
