"""
BridgeMind 端到端 pipeline:
  桥梁照片 → YOLO 检测 → 几何特征 → RAG + LLM 诊断报告
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Any

# 把项目根目录加入 sys.path，方便 import rag 模块
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from rag.diagnose import diagnose, load_index

# ===== 配置 =====
MODEL_PATH = PROJECT_ROOT / "crack_yolov11s_best.pt"
CONFIDENCE_THRESHOLD = 0.25
PIXEL_TO_MM = 0.5  # 假设：1 像素 ≈ 0.5mm（普通巡检照片，1-2m 距离）
# 实际工程需要现场标定，此处用于演示


# ===== 全局加载（只加载一次，节省启动时间）=====
print("🔧 加载 YOLO 模型...")
yolo_model = YOLO(str(MODEL_PATH))
print(f"✅ YOLO 模型已加载: {MODEL_PATH.name}")


def extract_defect_features(box, mask, conf, img_shape) -> Dict[str, Any]:
    """从 YOLO 输出提取缺陷几何与位置特征

    关键改进：用骨架化算法估算裂缝真实宽度（mask面积/骨架长度），
    避免单纯用 bbox 导致的尺寸严重失真。
    """
    h, w = img_shape[:2]

    # bbox 坐标
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    # ===== 几何分析（基于 mask）=====
    if mask is not None:
        # 取出 mask（二值化数组）
        mask_array = mask.data[0].cpu().numpy().astype(np.uint8)
        # resize 到原图尺寸（YOLO mask 默认是 640x640）
        if mask_array.shape != (h, w):
            mask_array = cv2.resize(mask_array, (w, h), interpolation=cv2.INTER_NEAREST)

        crack_pixels = int(mask_array.sum())

        # 用骨架化提取裂缝中线，长度 = 中线像素数
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask_array > 0)
        length_px = int(skeleton.sum())

        # 真实宽度 = 总面积 / 中线长度
        width_px = crack_pixels / length_px if length_px > 0 else 1
    else:
        # 没有 mask 时降级处理
        length_px = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        width_px = min(x2 - x1, y2 - y1) * 0.3  # 经验系数
        crack_pixels = int((x2 - x1) * (y2 - y1) * 0.3)

    # 像素 → 毫米
    length_mm = round(length_px * PIXEL_TO_MM, 1)
    width_mm = round(width_px * PIXEL_TO_MM, 2)

    # 位置描述
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    v_pos = "上部" if cy < 0.33 else ("中部" if cy < 0.67 else "下部")
    h_pos = "左侧" if cx < 0.33 else ("中间" if cx < 0.67 else "右侧")
    location = f"图像{v_pos}{h_pos} (像素坐标 {int((x1 + x2) / 2)}, {int((y1 + y2) / 2)})"

    # 缺陷类型（基于 bbox 宽高比 + 骨架方向）
    bbox_aspect = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1
    if bbox_aspect > 2:
        defect_type = "横向裂缝"
    elif bbox_aspect < 0.5:
        defect_type = "纵向裂缝"
    else:
        defect_type = "斜向/网状裂缝"

    return {
        "defect_type": defect_type,
        "confidence": round(float(conf), 3),
        "location": location,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "component": "桥梁结构构件",
        "structure_type": "混凝土桥梁",
        "_bbox": [int(x1), int(y1), int(x2), int(y2)],
        "_crack_pixels": crack_pixels,
        "_skeleton_length_px": length_px,
    }


def diagnose_bridge_image(
        image_path: str,
        component: str = "桥梁结构构件",
        structure_type: str = "混凝土桥梁",
        max_defects: int = 2,
) -> Dict[str, Any]:
    """
    端到端诊断：输入图片 → 输出标注图 + 报告

    Args:
        image_path: 桥梁照片路径
        component: 构件类型（如"主梁"、"桥面板"）
        structure_type: 结构类型（如"预应力混凝土桥梁"）
        max_defects: 最多详细分析几处缺陷（节省 token）
    """
    print(f"\n{'=' * 70}")
    print(f"📸 分析图像: {image_path}")
    print(f"{'=' * 70}")

    # ===== Step 1: YOLO 检测 =====
    print("\n🔍 YOLO 检测中...")
    results = yolo_model.predict(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]
    boxes = result.boxes
    masks = result.masks

    # ===== Step 2: 处理无缺陷情况 =====
    if boxes is None or len(boxes) == 0:
        print("✅ 未检测到裂缝缺陷")
        annotated = cv2.imread(image_path)
        return {
            "num_defects": 0,
            "annotated_image": annotated,
            "defects": [],
            "summary": "✅ 该图像中未检测到裂缝缺陷，结构外观状况良好。建议按常规巡检周期复查。"
        }

    # ===== Step 3: 按置信度排序，取 Top-N =====
    confs = boxes.conf.cpu().numpy()
    sorted_idx = np.argsort(-confs)[:max_defects]
    print(f"🎯 共检测到 {len(boxes)} 处疑似缺陷，详细诊断置信度最高的 {len(sorted_idx)} 处...")

    # ===== Step 4: 逐个分析 =====
    img = cv2.imread(image_path)
    defects = []

    for i, idx in enumerate(sorted_idx):
        print(f"\n  ▶ 缺陷 #{i + 1}/{len(sorted_idx)}")
        box = boxes[idx]
        mask = masks[idx] if masks is not None else None

        # 提取特征
        defect_info = extract_defect_features(box, mask, float(confs[idx]), img.shape)
        defect_info["component"] = component
        defect_info["structure_type"] = structure_type

        print(f"    类型: {defect_info['defect_type']}")
        print(f"    尺寸: 长 {defect_info['length_mm']}mm × 宽 {defect_info['width_mm']}mm")
        print(f"    置信度: {defect_info['confidence']}")

        # 调 RAG + LLM 生成报告
        diag_result = diagnose(defect_info)
        defects.append({
            "info": defect_info,
            "report": diag_result["report"],
        })

    # ===== Step 5: 生成可视化（YOLO 自带）=====
    annotated = results[0].plot()  # BGR 格式

    return {
        "num_defects": len(boxes),
        "analyzed_defects": len(defects),
        "annotated_image": annotated,
        "defects": defects,
    }


if __name__ == "__main__":
    # 测试：自动找一张测试集图片
    test_dir = PROJECT_ROOT / "crack_segmentation_dataset" / "test" / "images"
    if test_dir.exists():
        # 优先用 CFD（细裂缝），跳过 CRACK500（厚损伤）
        all_imgs = list(test_dir.glob("*.jpg"))
        cfd_imgs = [p for p in all_imgs if p.name.startswith("CFD")]
        test_image = str(cfd_imgs[0] if cfd_imgs else all_imgs[0])
    else:
        print("❌ 没找到测试集，请把图片路径作为参数传入")
        exit(1)

    print("\n" + "=" * 70)
    print("🏗️  BridgeMind 端到端诊断系统 - 测试")
    print("=" * 70)

    result = diagnose_bridge_image(
        image_path=test_image,
        component="桥面板",
        structure_type="钢筋混凝土桥梁",
        max_defects=2,
    )

    # ===== 保存结果到 pipeline_outputs/ =====
    output_dir = PROJECT_ROOT / "pipeline_outputs"
    output_dir.mkdir(exist_ok=True)
    img_stem = Path(test_image).stem

    # 保存标注图
    annotated_path = output_dir / f"{img_stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), result["annotated_image"])

    # 保存诊断报告 Markdown
    report_path = output_dir / f"{img_stem}_full_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# BridgeMind 桥梁缺陷诊断报告\n\n")
        f.write(f"**输入图像**: `{Path(test_image).name}`\n\n")
        f.write(f"**检测概况**: 共发现 {result['num_defects']} 处缺陷，")
        f.write(f"详细分析 {result.get('analyzed_defects', 0)} 处\n\n")
        f.write(f"![标注图]({annotated_path.name})\n\n")
        f.write("---\n\n")

        if not result["defects"]:
            f.write(result["summary"])
        else:
            for i, d in enumerate(result["defects"], 1):
                f.write(f"# 🔹 缺陷 #{i}\n\n")
                f.write("## 检测特征\n\n")
                for k, v in d["info"].items():
                    if not k.startswith("_"):
                        f.write(f"- **{k}**: {v}\n")
                f.write("\n---\n\n")
                f.write(d["report"])
                f.write("\n\n---\n\n")

    print(f"\n💾 标注图已保存: {annotated_path}")
    print(f"💾 完整报告已保存: {report_path}")
    print(f"\n🎉 Pipeline 端到端测试完成！")