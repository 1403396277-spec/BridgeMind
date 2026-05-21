"""
BridgeMind Gradio Web 界面
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import gradio as gr
from pipeline import diagnose_bridge_image

# ===== 配置 =====
EXAMPLES_DIR = PROJECT_ROOT / "crack_segmentation_dataset" / "test" / "images"


def analyze(image, component, structure_type, max_defects):
    """Gradio 主回调函数"""
    if image is None:
        return None, "❌ 请先上传图片", "请上传一张桥梁照片开始分析"

    # Gradio 给的是 PIL/numpy，写到临时文件
    tmp_path = PROJECT_ROOT / "_tmp_input.jpg"
    if isinstance(image, np.ndarray):
        cv2.imwrite(str(tmp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        image.save(tmp_path)

    # 调用 pipeline
    try:
        result = diagnose_bridge_image(
            image_path=str(tmp_path),
            component=component,
            structure_type=structure_type,
            max_defects=int(max_defects),
        )
    except Exception as e:
        return None, f"❌ 处理出错: {e}", ""

    # 处理结果
    if result["num_defects"] == 0:
        annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
        return annotated_rgb, "✅ 未检测到缺陷", result["summary"]

    # 标注图（BGR → RGB）
    annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)

    # 缺陷摘要表格
    summary = f"### 🔍 检测结果\n\n"
    summary += f"- **共发现缺陷**: {result['num_defects']} 处\n"
    summary += f"- **详细分析**: {result['analyzed_defects']} 处（按置信度排序）\n\n"
    summary += "| # | 类型 | 长度(mm) | 宽度(mm) | 置信度 | 位置 |\n"
    summary += "|---|------|----------|----------|--------|------|\n"
    for i, d in enumerate(result["defects"], 1):
        info = d["info"]
        summary += f"| {i} | {info['defect_type']} | {info['length_mm']} | {info['width_mm']} | {info['confidence']} | {info['location']} |\n"
    summary += "\n> ⚠️ 物理尺寸基于像素假设（1 px = 0.5 mm），实际工程使用须现场标定\n"

    # 拼接所有诊断报告
    all_reports = ""
    for i, d in enumerate(result["defects"], 1):
        all_reports += f"\n\n# 🔹 缺陷 #{i}\n\n"
        all_reports += d["report"]
        all_reports += "\n\n---\n"

    return annotated_rgb, summary, all_reports


# ===== 构建 UI =====
def build_ui():
    with gr.Blocks(title="BridgeMind 桥梁缺陷智能诊断", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌉 BridgeMind - 桥梁缺陷智能诊断系统

            **多模态大模型驱动 · YOLO 检测 + RAG 知识库 + DeepSeek 报告生成**

            上传一张桥梁/混凝土结构照片，系统将自动：
            1. 🔍 YOLO11-seg 检测裂缝位置与形态
            2. 📐 提取几何特征（长度、宽度、类型）  
            3. 📚 从桥梁规范知识库检索相关条文（RAG）
            4. 📄 调用 DeepSeek-V3 生成 5 章节专业诊断报告
            """
        )

        with gr.Row():
            # 左栏：输入
            with gr.Column(scale=1):
                input_image = gr.Image(label="📸 上传桥梁照片", type="numpy", height=400)

                with gr.Accordion("⚙️ 高级选项（可选）", open=False):
                    component = gr.Textbox(
                        label="构件类型",
                        value="桥面板",
                        placeholder="如：主梁、桥墩、桥面板、支座..."
                    )
                    structure_type = gr.Textbox(
                        label="结构类型",
                        value="钢筋混凝土桥梁",
                        placeholder="如：钢筋混凝土桥梁、预应力混凝土桥梁..."
                    )
                    max_defects = gr.Slider(
                        label="最多分析几处缺陷",
                        minimum=1, maximum=5, value=2, step=1,
                        info="缺陷越多耗时越长（每个约 15 秒）"
                    )

                analyze_btn = gr.Button("🚀 开始诊断", variant="primary", size="lg")

            # 右栏：输出
            with gr.Column(scale=1):
                output_image = gr.Image(label="🎯 检测结果可视化", height=400)
                output_summary = gr.Markdown(label="检测摘要")

        # 底部：详细报告
        gr.Markdown("---")
        gr.Markdown("## 📄 完整诊断报告")
        output_report = gr.Markdown()

        # 示例图
        if EXAMPLES_DIR.exists():
            example_files = sorted(EXAMPLES_DIR.glob("CFD_*.jpg"))[:6]
            if example_files:
                gr.Examples(
                    examples=[[str(f), "桥面板", "钢筋混凝土桥梁", 2] for f in example_files],
                    inputs=[input_image, component, structure_type, max_defects],
                    label="📂 示例图片（点击试用）",
                )

        # 绑定事件
        analyze_btn.click(
            fn=analyze,
            inputs=[input_image, component, structure_type, max_defects],
            outputs=[output_image, output_summary, output_report],
        )

        gr.Markdown(
            """
            ---

            ### 🛠️ 技术栈
            - **检测模型**: YOLO11s-seg (在 11K+ 多源融合裂缝数据集上微调, mAP@0.5 = 0.69)
            - **知识库检索**: LlamaIndex + BGE-small-zh embedding
            - **大语言模型**: DeepSeek-V3
            - **前端**: Gradio

            📂 [GitHub 仓库](https://github.com/1403396277-spec/BridgeMind)
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # 改成 True 可以生成公网链接
        inbrowser=True,  # 自动打开浏览器
    )