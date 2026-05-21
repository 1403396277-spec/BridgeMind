"""
专业诊断报告生成器：CV 结构化输出 + RAG 检索 + DeepSeek 生成
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

PROJECT_ROOT = Path(__file__).parent.parent
PERSIST_DIR = PROJECT_ROOT / "rag" / "index_storage"

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    is_chat_model=True,
    context_window=64000,
)

# ===== 核心：专业诊断 Prompt 模板 =====
DIAGNOSIS_PROMPT = """你是一位资深桥梁结构工程师，拥有 20 年桥梁检测与维护经验。请基于以下检测信息和参考规范，生成一份专业的桥梁缺陷诊断报告。

## 检测到的缺陷信息
- 缺陷类型: {defect_type}
- AI 检测置信度: {confidence}
- 位置: {location}
- 估算尺寸: 长 {length_mm} mm, 宽 {width_mm} mm
- 构件类型: {component}
- 结构类型: {structure_type}

## 相关规范条文（检索自知识库）
{context}

## 报告要求
请严格按以下五个章节输出 Markdown 格式诊断报告：

### 1. 缺陷描述
用专业术语描述该缺陷的几何特征、形态特点。

### 2. 严重程度评估
评定为以下四级之一：**轻微 / 中等 / 严重 / 危险**，并给出依据。

### 3. 规范依据
引用上述参考资料中相关条文（标注来源[X]），说明该缺陷为何需要关注。**如规范资料中无明确依据，须明确说明"参考资料中未找到直接相关条文"，禁止编造规范名称或条文号**。

### 4. 处理建议
按优先级给出 2-3 个具体处理方案，每个方案说明：
- 适用条件
- 大致工艺
- 预期效果

### 5. 风险评估
- 若不及时处理的潜在后果
- 建议的复检周期（如 3 个月 / 6 个月 / 12 个月）

报告须**专业、严谨、可操作**，避免空泛表述。"""


def load_index():
    if not PERSIST_DIR.exists():
        raise FileNotFoundError("❌ 索引不存在！请先运行 build_index.py")
    storage = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    return load_index_from_storage(storage)


def retrieve_context(index, defect_info: Dict[str, Any], top_k: int = 5) -> str:
    """根据缺陷信息智能构造检索 query"""
    # 智能拼接 query：缺陷类型 + 构件 + 关键词
    query = f"{defect_info.get('defect_type', '')} {defect_info.get('component', '')} 处理 加固 评估 规范"

    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    # 整合上下文（带来源标注，方便 LLM 引用）
    context_parts = []
    for i, node in enumerate(nodes, 1):
        source = node.metadata.get("source", "未知")
        score = node.score or 0
        context_parts.append(
            f"[{i}] 来源: {source} (相似度: {score:.2f})\n{node.text.strip()}"
        )

    return "\n\n---\n\n".join(context_parts)


def diagnose(defect_info: Dict[str, Any]) -> Dict[str, str]:
    """
    主诊断函数

    Args:
        defect_info: YOLO 输出的结构化缺陷信息

    Returns:
        {"report": "报告内容", "context": "检索到的规范原文"}
    """
    index = load_index()

    print("🔍 检索相关规范条文...")
    context = retrieve_context(index, defect_info)

    prompt = DIAGNOSIS_PROMPT.format(
        defect_type=defect_info.get("defect_type", "未知"),
        confidence=defect_info.get("confidence", "N/A"),
        location=defect_info.get("location", "未指定"),
        length_mm=defect_info.get("length_mm", "未测量"),
        width_mm=defect_info.get("width_mm", "未测量"),
        component=defect_info.get("component", "未指定"),
        structure_type=defect_info.get("structure_type", "桥梁结构"),
        context=context,
    )

    print("🤖 DeepSeek 生成诊断报告中（约 10-20 秒）...")
    response = Settings.llm.complete(prompt)

    return {
        "report": str(response),
        "context": context,
    }


if __name__ == "__main__":
    # 模拟 3 种不同严重程度的缺陷
    test_cases = [
        {
            "name": "案例 1: 轻微表面裂缝",
            "data": {
                "defect_type": "横向表面裂缝",
                "confidence": 0.82,
                "location": "桥面板边缘",
                "length_mm": 80,
                "width_mm": 0.15,
                "component": "桥面板",
                "structure_type": "钢筋混凝土桥梁",
            }
        },
        {
            "name": "案例 2: 严重承重裂缝",
            "data": {
                "defect_type": "纵向受力裂缝",
                "confidence": 0.91,
                "location": "主梁跨中下缘",
                "length_mm": 450,
                "width_mm": 0.8,
                "component": "主梁",
                "structure_type": "预应力混凝土桥梁",
            }
        },
        {
            "name": "案例 3: 中等程度斜裂缝",
            "data": {
                "defect_type": "斜向剪切裂缝",
                "confidence": 0.78,
                "location": "支座附近梁腹",
                "length_mm": 180,
                "width_mm": 0.4,
                "component": "梁腹板",
                "structure_type": "T 形混凝土桥梁",
            }
        },
    ]

    print("=" * 70)
    print("🏗️  BridgeMind 桥梁缺陷智能诊断系统")
    print("=" * 70)

    for case in test_cases:
        print(f"\n\n{'#' * 70}")
        print(f"# {case['name']}")
        print(f"{'#' * 70}")

        print("\n📋 输入缺陷信息:")
        for k, v in case['data'].items():
            print(f"  • {k}: {v}")

        result = diagnose(case['data'])

        # 保存报告到 reports/ 文件夹
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        safe_name = case['name'].replace(' ', '_').replace(':', '')
        report_path = reports_dir / f"{safe_name}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# {case['name']}\n\n")
            f.write("## 输入缺陷信息\n\n")
            for k, v in case['data'].items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n---\n\n")
            f.write(result["report"])

        print(f"\n💾 报告已保存: {report_path}")
        print("\n📄 诊断报告:")
        print("-" * 70)
        print(result["report"])
        print("-" * 70)
        print("\n按 Enter 继续下一个案例...")
        input()