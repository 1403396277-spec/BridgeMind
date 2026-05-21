"""
RAG 索引构建：解析 PDF → 切块 → 向量化 → 存盘
首次运行会下载 BGE 中文 embedding 模型（约 100MB），耐心等待
"""
import os
from pathlib import Path
from dotenv import load_dotenv

import pymupdf  # PDF 解析（对中文友好）
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# ===== 加载 API Key =====
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ 没找到 DEEPSEEK_API_KEY！检查 .env 文件")

# ===== 配置路径 =====
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
PERSIST_DIR = PROJECT_ROOT / "rag" / "index_storage"

# ===== 配置模型 =====
print("🔧 配置 embedding 模型 (BGE-small-zh)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

print("🔧 配置 DeepSeek LLM...")
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    is_chat_model=True,
    context_window=64000,
)

# 文档切块策略
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def parse_pdf(pdf_path: Path) -> str:
    """用 PyMuPDF 提取 PDF 文本"""
    doc = pymupdf.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text


def build_index():
    # 同时支持 PDF 和 TXT
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    txt_files = list(DOCS_DIR.glob("*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        raise FileNotFoundError(f"❌ {DOCS_DIR} 下没有 PDF 或 TXT！")

    print(f"\n📚 找到 {len(all_files)} 个文件:")

    documents = []
    for f in all_files:
        if f.suffix == ".pdf":
            text = parse_pdf(f)
        else:  # .txt
            text = f.read_text(encoding="utf-8")

        documents.append(Document(text=text, metadata={"source": f.name}))
        print(f"  ✅ {f.name}: {len(text):,} 字符")

    print("\n🧠 构建向量索引...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"\n💾 索引已存到: {PERSIST_DIR}")
    return index

if __name__ == "__main__":
    print("🚀 开始构建 RAG 索引\n" + "=" * 50)
    build_index()
    print("\n✅ 完成！下一步运行 rag/query.py")