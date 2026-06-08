"""
Rerank 节点：对 retriever 粗排的结果做 Cross-Encoder 精排。

设计原则：
- 失败不阻塞：reranker 加载或推理异常时，原样透传 retrieved_docs
- 原地覆盖：直接把重排结果写回 state["retrieved_docs"]，下游 4 个回答节点零改动
- 可开关：config.RERANK_ENABLED=False 时本节点等价于 no-op
"""
from __future__ import annotations

from .. import config
from ..state import AgentState


def rerank_node(state: AgentState) -> dict:
    """对 retrieved_docs 做 Cross-Encoder 精排，结果原地写回。"""
    docs = state.get("retrieved_docs", []) or []
    if not docs:
        return {}

    # 配置开关：禁用时直接跳过
    if not config.RERANK_ENABLED:
        print("[Rerank] 已禁用，跳过")
        return {}

    query = state["query"]

    # 延迟导入 reranker，避免 graph 构建时就触发模型加载
    try:
        from ..reranker import get_reranker
        reranker = get_reranker()
    except Exception as e:
        print(f"[Rerank] 加载失败，原样透传: {e}")
        return {}

    if reranker is None:
        return {}

    try:
        reranked = reranker.rerank(query, docs, top_k=config.RERANK_TOP_K)
        print(f"[Rerank] 重排完成: {len(docs)} -> {len(reranked)} 条")
        return {"retrieved_docs": reranked}
    except Exception as e:
        print(f"[Rerank] 推理失败，原样透传: {e}")
        return {}
