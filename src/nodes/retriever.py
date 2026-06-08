"""
检索节点：从 Chroma 向量库检索与用户问题相关的手册片段。

注意：本节点只做"粗排"——
- 若启用 reranker：取较大的 RERANK_CANDIDATE_K（如 20），交给 rerank 节点精排
- 若禁用 reranker：直接取 RETRIEVE_TOP_K（如 5）作为最终结果
"""
from __future__ import annotations

from .. import config
from ..ingest import get_or_build_vectorstore
from ..state import AgentState


# 进程级缓存，避免每次调用都重新加载
_vectorstore = None


def _get_vs():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_or_build_vectorstore()
    return _vectorstore


def retriever_node(state: AgentState) -> dict:
    """
    根据 state.query 检索 Chroma，写回 state.retrieved_docs。

    取多少条取决于是否启用 reranker：
    - 启用：取 RERANK_CANDIDATE_K 条粗排候选
    - 禁用：直接取 RETRIEVE_TOP_K 条作为最终结果
    """
    query = state["query"]
    vs = _get_vs()
    k = config.RERANK_CANDIDATE_K if config.RERANK_ENABLED else config.RETRIEVE_TOP_K
    docs = vs.similarity_search(query, k=k)
    print(f"[Retriever] 粗排检索到 {len(docs)} 个片段（k={k}）")
    return {"retrieved_docs": docs}


def format_docs(docs) -> str:
    """把检索结果格式化为单段文本，供 prompt 注入。"""
    if not docs:
        return "(未检索到相关手册内容)"
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] 来源: {src} 第 {page} 页\n{d.page_content}")
    return "\n\n---\n\n".join(parts)
