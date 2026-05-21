"""
RAG 查询：从已建好的索引里提问，调用 DeepSeek 生成回答
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

PROJECT_ROOT = Path(__file__).parent.parent
PERSIST_DIR = PROJECT_ROOT / "rag" / "index_storage"

# 模型配置（必须和 build_index.py 一致）
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    is_chat_model=True,
    context_window=64000,
)


def load_index():
    if not PERSIST_DIR.exists():
        raise FileNotFoundError("❌ 索引不存在！先运行 build_index.py")
    storage = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    return load_index_from_storage(storage)


def ask(index, question: str, top_k: int = 5):
    engine = index.as_query_engine(similarity_top_k=top_k, response_mode="compact")

    print(f"\n❓ {question}")
    print("🤔 思考中...")
    response = engine.query(question)

    print(f"\n💡 回答:\n{response}\n")
    print(f"📖 参考来源 (Top {top_k}):")
    for i, node in enumerate(response.source_nodes, 1):
        src = node.metadata.get("source", "未知")
        score = node.score or 0
        snippet = node.text[:80].replace("\n", " ")
        print(f"  [{i}] {src} (相似度 {score:.3f}): {snippet}...")
    print("=" * 60)


if __name__ == "__main__":
    print("📚 加载索引...")
    idx = load_index()
    print("✅ 索引就绪\n")

    # 测试问题（针对桥梁规范）
    test_qs = [
        "桥梁的设计使用年限分为哪几类？",
        "公路桥梁分为哪几个等级？",
        "钢筋混凝土桥梁的设计需要考虑哪些主要荷载？",
    ]

    for q in test_qs:
        ask(idx, q)

    # 交互模式
    print("\n💬 进入交互模式（输入 quit 退出）")
    while True:
        q = input("\n你的问题: ").strip()
        if q.lower() in ["quit", "exit", "q", ""]:
            break
        ask(idx, q)