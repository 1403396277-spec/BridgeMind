import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 数据集根目录（改成你自己的路径）
DATASET_ROOT = Path("/Users/dangchenghuize/Desktop/项目/crack_segmentation_dataset")


def mask_to_yolo_polygons(mask_path, min_area=20):
    """把一张 mask 图转成 YOLO 多边形格式"""
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
        # 归一化坐标
        coords = approx.reshape(-1, 2).astype(float)
        coords[:, 0] /= w
        coords[:, 1] /= h
        polygons.append(coords.flatten().tolist())

    return polygons


def process_split(split_name):
    """处理 train 或 test"""
    img_dir = DATASET_ROOT / split_name / "images"
    mask_dir = DATASET_ROOT / split_name / "masks"
    label_dir = DATASET_ROOT / split_name / "labels"  # YOLO的标签文件夹
    label_dir.mkdir(exist_ok=True)

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"\n处理 {split_name}: 共 {len(img_files)} 张图")

    crack_count = 0
    noncrack_count = 0

    for img_path in tqdm(img_files):
        # 找对应的 mask
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = mask_dir / (img_path.stem + ".png")

        label_path = label_dir / (img_path.stem + ".txt")

        # 非裂缝图：写空文件（YOLO 把它当负样本）
        if "noncrack" in img_path.stem.lower() or not mask_path.exists():
            label_path.touch()
            noncrack_count += 1
            continue

        # 转换 mask
        polygons = mask_to_yolo_polygons(mask_path)

        with open(label_path, "w") as f:
            for poly in polygons:
                line = "0 " + " ".join(f"{x:.6f}" for x in poly)
                f.write(line + "\n")

        if polygons:
            crack_count += 1
        else:
            noncrack_count += 1

    print(f"  ✅ {split_name}: 含裂缝 {crack_count} 张，无裂缝 {noncrack_count} 张")


if __name__ == "__main__":
    process_split("train")
    process_split("test")
    print("\n🎉 全部转换完成！")