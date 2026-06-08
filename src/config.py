"""
全局配置模块 —— 集中管理所有可配置项。
所有路径、模型名、API key 都从这里读取。
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env（如果存在）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ===== 项目根目录 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """读取环境变量，required=True 时缺失会抛错。"""
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"环境变量 {key} 未设置，请在 .env 中配置。")
    return val  # type: ignore


# ===== 智谱 LLM 配置 =====
ZHIPU_API_KEY: str = _get_env("ZHIPU_API_KEY", required=True)
ZHIPU_BASE_URL: str = _get_env("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZHIPU_MODEL: str = _get_env("ZHIPU_MODEL", "glm-4-flash")

# LLM 调用温度（路由用低温度 0.1，回答用 0.3，战术建议用 0.5）
LLM_TEMPERATURE_ROUTER: float = 0.1
LLM_TEMPERATURE_ANSWER: float = 0.3
LLM_TEMPERATURE_TACTICS: float = 0.5

# ===== Embedding 配置 =====
EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
EMBEDDING_DEVICE: str = _get_env("EMBEDDING_DEVICE", "cpu")  # cpu / cuda

# ===== 路径配置 =====
MANUAL_DIR: Path = PROJECT_ROOT / _get_env("MANUAL_DIR", "manual")
VECTORSTORE_DIR: Path = PROJECT_ROOT / _get_env("VECTORSTORE_DIR", "vectorstore")
CHECKPOINTER_DB: Path = PROJECT_ROOT / _get_env("CHECKPOINTER_DB", "checkpointer.sqlite")
CHUNKS_CACHE_DIR: Path = PROJECT_ROOT / "data" / "chunks"

# ===== 切片配置 =====
CHUNK_SIZE: int = 500        # 每个切片字符数
CHUNK_OVERLAP: int = 80      # 相邻切片重叠字符数

# ===== 检索配置 =====
RETRIEVE_TOP_K: int = 5      # 每次检索返回的文档数（最终喂给 LLM 的数量）

# ===== Reranker 配置（可选，默认开启） =====
# 流程：Chroma 粗排 top-K1 → Cross-Encoder 精排 → 取 top-K2
# 注意：reranker 与 embedding 不同 ——
#   - embedding 是 Bi-Encoder，把 query/doc 各自编码成向量后算余弦
#   - reranker 是 Cross-Encoder，把 (query, doc) 一起送进模型打分，精度更高但更慢
RERANK_ENABLED: bool = _get_env("RERANK_ENABLED", "1") == "1"
RERANKER_MODEL: str = _get_env("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_DEVICE: str = _get_env("RERANKER_DEVICE", EMBEDDING_DEVICE)  # 默认跟 embedding 同设备
RERANK_CANDIDATE_K: int = int(_get_env("RERANK_CANDIDATE_K", "20"))  # 粗排取多少条
RERANK_TOP_K: int = int(_get_env("RERANK_TOP_K", "5"))                # 重排后保留多少条

# ===== Chroma 集合名 =====
CHROMA_COLLECTION: str = "uoc2_manual"
