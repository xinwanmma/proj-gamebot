"""
PDF 摄取模块 —— 把 manual/ 下的 PDF 解析、切片、存入 Chroma 向量库。

流程：
  读取 PDF（每页一篇文章）
  → 切成 500 字左右的 chunk
  → 用 text2vec 转成向量
  → 写入 Chroma（本地持久化）

跑一次就行了，后面启动 app 直接加载已有向量库。
"""
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .llm import get_embeddings


def load_pdf(path: Path) -> list[Document]:
    """读取一个 PDF，每页变成一个 Document。"""
    print(f"  解析: {path.name}")
    docs = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,     # 文件名
                    "page": i,               # 页码
                },
            ))
    print(f"    → {len(docs)} 页")
    return docs


def get_manual_pdfs() -> list[Path]:
    """获取 manual/ 下所有 PDF 文件。"""
    return sorted(config.MANUAL_DIR.rglob("*.pdf"))


def load_all_manuals() -> list[Document]:
    """读取全部 PDF，每页一篇文章，返回一个大列表。"""
    all_docs = []
    for pdf_path in get_manual_pdfs():
        # 跳过剧本图表（图片为主，向量检索效果差）
        if "scenario" in pdf_path.name.lower():
            print(f"  跳过剧本图表: {pdf_path.name}")
            continue
        all_docs.extend(load_pdf(pdf_path))
    print(f"总计 {len(all_docs)} 页")
    return all_docs


def split_into_chunks(docs: list[Document]) -> list[Document]:
    """
    把"每页一篇文章"切成"每段约 500 字的小块"。
    
    为什么切？
    - 一页可能太长（1000+ 字），LLM 的上下文窗口有限
    - 一段太短也不行，可能信息不完整
    - 500 字 + 80 字重叠是比较稳妥的配置
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"切片: {len(docs)} 页 → {len(chunks)} 个 chunk")
    return chunks


def build_vectorstore_from_scratch():
    """
    从头构建向量库：读取 PDF → 切片 → 向量化 → 写入 Chroma。
    
    这样设计：
    - 不把 Chroma 导入放在文件顶部（因为 pipeline.py 的 _get_vectorstore()
      可能调用这个函数，而 Chroma 不是所有场景都需要）
    - 延迟导入，加快模块加载速度
    """
    from langchain_chroma import Chroma

    print("=" * 40)
    print("构建向量库")
    print("=" * 40)

    # 1. 读取所有 PDF
    print("[1/3] 读取 PDF ...")
    raw_docs = load_all_manuals()

    # 2. 切成 chunk
    print("[2/3] 切片 ...")
    chunks = split_into_chunks(raw_docs)

    # 3. 向量化 + 写入 Chroma
    print("[3/3] 向量化写入 Chroma ...")
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(config.VECTORSTORE_DIR),
    )

    print(f"完成！共 {vectorstore._collection.count()} 条向量")
    return vectorstore
