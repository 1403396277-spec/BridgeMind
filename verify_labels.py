import cv2
import numpy as np
from pathlib import Path

DATASET_ROOT = Path("/Users/dangchenghuize/Desktop/项目/crack_segmentation_dataset")
SPLIT = "train"

# 随便挑一张含裂缝的图看
img_dir = DATASET_ROOT / SPLIT / "images"
label_dir = DATASET_ROOT / SPLIT / "labels"

# 找一个 label 不为空的样本
for img_path in img_dir.glob("*.jpg"):
    label_path = label_dir / (img_path.stem + ".txt")
    if label_path.exists() and label_path.stat().st_size > 10:
        break

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

cv2.imwrite("verify_output.jpg", img)
print(f"已保存 verify_output.jpg，原图: {img_path.name}")