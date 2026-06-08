"""
PDF 摄取模块 —— 把 manual/ 目录下的所有 PDF 解析、切片、向量化、写入 Chroma。

设计要点：
1. 中文 PDF 用 pdfplumber（保留段落结构），失败回退 pypdf
2. 切片采用 RecursiveCharacterTextSplitter（中文优先按句号、换行切）
3. Chroma 本地持久化，启动时直接加载，无需重新向量化
4. 元数据包含 source 文件名、页码、语言（中/英），便于检索过滤
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pdfplumber
from langchain_core.documents import Document
# langchain 1.x 子包名带下划线；旧版 fallback 到 langchain.text_splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from . import config


# ---------- PDF 解析 ----------

def _extract_pdf_with_pdfplumber(path: Path) -> list[Document]:
    """用 pdfplumber 提取 PDF，每页一个 Document。中文友好。"""
    docs: list[Document] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": i,
                        "lang": "zh" if "游戏手册" in path.name or "修订" in path.name else "en",
                    },
                )
            )
    return docs


def load_pdf(path: Path) -> list[Document]:
    """加载单个 PDF，返回按页切分的 Document 列表。"""
    print(f"[Ingest] 解析 PDF: {path.name}")
    try:
        docs = _extract_pdf_with_pdfplumber(path)
        if docs:
            print(f"  -> 共 {len(docs)} 页有效文本")
            return docs
    except Exception as e:
        print(f"  [warn] pdfplumber 失败: {e}，回退到 pypdf")
    # 回退方案
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": i, "lang": "en"},
                )
            )
    print(f"  -> pypdf 提取 {len(docs)} 页")
    return docs


def iter_manual_pdfs(manual_dir: Path | None = None) -> Iterator[Path]:
    """遍历 manual 目录下所有 PDF（包括子目录）。"""
    base = manual_dir or config.MANUAL_DIR
    yield from sorted(base.rglob("*.pdf"))


def load_all_manuals(manual_dir: Path | None = None) -> list[Document]:
    """加载目录下全部 PDF，返回 Document 列表（每页一个）。"""
    all_docs: list[Document] = []
    for pdf_path in iter_manual_pdfs(manual_dir):
        # 跳过 Scenario Chart（剧本图表），其结构化数据暂不处理
        if "scenario" in pdf_path.name.lower() or "剧本" in pdf_path.name:
            print(f"[Ingest] 跳过剧本图表: {pdf_path.name}")
            continue
        all_docs.extend(load_pdf(pdf_path))
    print(f"[Ingest] 总计加载 {len(all_docs)} 页原始文档")
    return all_docs


# ---------- 切片 ----------

def _make_splitter() -> RecursiveCharacterTextSplitter:
    """
    中文友好的递归切片器。
    分隔符优先级：段落 > 双换行 > 单换行 > 中文句号 > 英文句号 > 空格 > 字符
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=[
            "\n\n", "\n",
            "。", "！", "？",
            ". ", "! ", "? ",
            "；", "; ",
            "，", ", ",
            " ", "",
        ],
        keep_separator=True,
    )


def split_documents(docs: list[Document]) -> list[Document]:
    """把按页的 Document 切成更小的 chunk。"""
    splitter = _make_splitter()
    chunks = splitter.split_documents(docs)
    print(f"[Ingest] 切片完成：{len(docs)} 页 -> {len(chunks)} 个 chunk")
    return chunks


# ---------- 写入 Chroma ----------

def build_vectorstore(
    chunks: list[Document],
    persist_dir: Path | None = None,
) -> "Chroma":  # type: ignore  # noqa: F821
    """把切片写入 Chroma 持久化向量库（首次运行）。"""
    # 优先用 langchain-chroma 子包；旧版 fallback 到 langchain_community
    try:
        from langchain_chroma import Chroma
    except ModuleNotFoundError:
        from langchain_community.vectorstores import Chroma  # type: ignore
    from .embeddings import get_embeddings

    persist_dir = persist_dir or config.VECTORSTORE_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Ingest] 正在向量化并写入 Chroma ({persist_dir}) ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(persist_dir),
    )
    print(f"[Ingest] 完成。共 {vectorstore._collection.count()} 条向量。")  # type: ignore
    return vectorstore


def load_vectorstore(persist_dir: Path | None = None):
    """启动时加载已有 Chroma（无需重新向量化）。"""
    try:
        from langchain_chroma import Chroma
    except ModuleNotFoundError:
        from langchain_community.vectorstores import Chroma  # type: ignore
    from .embeddings import get_embeddings

    persist_dir = persist_dir or config.VECTORSTORE_DIR
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"向量库目录不存在: {persist_dir}\n请先运行 `python ingest_once.py` 进行首次向量化。"
        )
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def get_or_build_vectorstore():
    """
    优先加载已有向量库；若不存在则从 manual/ 重新构建。
    """
    if config.VECTORSTORE_DIR.exists() and any(config.VECTORSTORE_DIR.iterdir()):
        print(f"[Ingest] 检测到已有向量库，直接加载: {config.VECTORSTORE_DIR}")
        return load_vectorstore()
    print("[Ingest] 未检测到向量库，开始首次构建 ...")
    raw = load_all_manuals()
    chunks = split_documents(raw)
    return build_vectorstore(chunks)
