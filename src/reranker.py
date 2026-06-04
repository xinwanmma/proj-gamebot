"""
本地 Cross-Encoder Reranker 封装。

使用 BAAI/bge-reranker-base（约 280MB）对 (query, doc) 对打分，
相比 Bi-Encoder embedding，精度更高但更慢，常用于粗排后的精排。

加载逻辑参考 embeddings.py：
- 强制 HF 离线模式，避免国内网络超时
- 优先从 HF 缓存 snapshot 加载
- 全局单例（lru_cache）
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# 复用 embeddings.py 的离线模式设置
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_core.documents import Document  # noqa: E402

from . import config  # noqa: E402


def _resolve_local_snapshot(model_name: str) -> str | None:
    """在 HF 缓存中找 snapshot 目录（与 embeddings.py 同逻辑，复用以解耦）。"""
    if "/" not in model_name:
        return None
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub_dir = Path(hf_home) / "hub"
    folder_name = "models--" + model_name.replace("/", "--")
    snap_dir = hub_dir / folder_name / "snapshots"
    if not snap_dir.exists():
        return None
    for sub in snap_dir.iterdir():
        if not sub.is_dir():
            continue
        # reranker 一般是 pytorch_model.bin 或 model.safetensors
        if (sub / "pytorch_model.bin").exists() or (sub / "model.safetensors").exists():
            return str(sub)
    return None


class Reranker:
    """Cross-Encoder reranker 封装。底层用 sentence-transformers 的 CrossEncoder。"""

    def __init__(self, model_path: str, device: str = "cpu"):
        # 延迟导入，避免 app 启动时强制依赖 sentence_transformers
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_path, device=device)
        print(f"[Reranker] 加载完成: {model_path} (device={device})")

    def score(self, query: str, docs: list[str]) -> list[float]:
        """对 (query, doc) 对批量打分，返回与 docs 等长的分数列表。"""
        pairs = [(query, d) for d in docs]
        # batch_size 默认即可；显存吃紧可调小
        scores = self._model.predict(pairs).tolist()
        return scores

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        对 docs 按 query 相关性重排。

        Args:
            query: 用户问题
            docs:  粗排结果（list[Document]）
            top_k: 重排后保留多少条；默认用 config.RERANK_TOP_K

        Returns:
            排序后的 list[Document]，metadata 原样保留，新增 rerank_score 字段
        """
        if not docs:
            return []
        if top_k is None:
            top_k = config.RERANK_TOP_K

        texts = [d.page_content or "" for d in docs]
        scores = self.score(query, texts)

        # 把分数写回 metadata，按分数降序
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        out: list[Document] = []
        for d, s in scored[:top_k]:
            meta = dict(d.metadata or {})
            meta["rerank_score"] = float(s)
            out.append(Document(page_content=d.page_content, metadata=meta))
        return out


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    """
    获取全局共享的 Reranker 实例。
    若 config.RERANK_ENABLED=False，返回 None，调用方需自行判断。
    """
    if not config.RERANK_ENABLED:
        print("[Reranker] 已在 config 中禁用，跳过加载。")
        return None

    model_name = config.RERANKER_MODEL
    # 1) 绝对路径
    if (model_name.startswith(("/", "\\")) or ":" in model_name[:2]) and Path(model_name).exists():
        path = model_name
    else:
        # 2) HF 缓存 snapshot
        snapshot = _resolve_local_snapshot(model_name)
        if snapshot:
            path = snapshot
        else:
            # 3) 兜底：交给 sentence-transformers 默认逻辑（可能联网）
            print(f"[Reranker] 未在本地缓存中找到 {model_name}，将尝试在线下载 ...")
            path = model_name

    return Reranker(path, device=config.RERANKER_DEVICE)
