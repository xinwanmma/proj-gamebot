"""
本地 Embedding 模型封装。
使用 shibing624/text2vec-base-chinese（768 维中文向量），
通过 LangChain 的 HuggingFaceEmbeddings 接口暴露给 Chroma。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# ===== 关键：开启 HF 离线模式，避免每次加载都去 huggingface.co 检查更新 =====
# 国内网络访问 huggingface.co 不稳定，会超时挂起。
# 模型已在本地缓存时，必须切到离线模式才能正常加载。
# 通过 .env 也可控制：HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 国内镜像（如果以后需要联网下载，走镜像更快）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# langchain 1.x 推荐 langchain_huggingface；旧版 fallback 到 langchain_community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

from . import config


def _resolve_local_snapshot(model_name: str) -> str | None:
    """
    尝试在 HF 默认缓存中找到该模型的 snapshot 目录绝对路径。
    找到就直接返回该路径（绕过 HF 缓存元数据检查），否则返回 None。

    HF 缓存结构：
        ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<commit>/
    其中 <org>--<name> 由 model_name 中的 / 替换为 -- 得到。
    """
    if "/" not in model_name:
        return None
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub_dir = Path(hf_home) / "hub"
    # HF 缓存里把 / 替换成 --
    folder_name = "models--" + model_name.replace("/", "--")
    snap_dir = hub_dir / folder_name / "snapshots"
    if not snap_dir.exists():
        return None
    # 取 snapshots 下第一个有 model.safetensors 或 pytorch_model.bin 的子目录
    for sub in snap_dir.iterdir():
        if not sub.is_dir():
            continue
        if (sub / "model.safetensors").exists() or (sub / "pytorch_model.bin").exists():
            return str(sub)
    return None


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取全局共享的 Embedding 实例。

    加载优先级：
    1. EMBEDDING_MODEL 是本地绝对路径 → 直接加载
    2. 在 HF 默认缓存中找到 snapshot 目录 → 直接加载该目录（绕过联网检查）
    3. 项目 ./models 目录 → 用 cache_folder 参数加载
    4. 兜底：交给 sentence-transformers 默认逻辑（可能联网）
    """
    model_name = config.EMBEDDING_MODEL
    local_root = config.PROJECT_ROOT / "models"

    # 情况 1：用户填的是绝对路径
    if (model_name.startswith(("/", "\\")) or ":" in model_name[:2]) and Path(model_name).exists():
        print(f"[Embedding] 加载本地模型（绝对路径）: {model_name}")
        model_or_path = model_name
        cache_folder = None

    # 情况 2：HF 默认缓存里有 snapshot
    else:
        snapshot = _resolve_local_snapshot(model_name)
        if snapshot:
            print(f"[Embedding] 加载本地模型（HF 缓存 snapshot）: {snapshot}")
            model_or_path = snapshot
            cache_folder = None
        # 情况 3：项目 models 目录有缓存
        elif local_root.exists() and any(local_root.iterdir()):
            print(f"[Embedding] 加载本地模型（项目缓存）: {local_root}")
            model_or_path = model_name
            cache_folder = str(local_root)
        # 情况 4：兜底
        else:
            print(f"[Embedding] 加载模型（默认逻辑）: {model_name}")
            model_or_path = model_name
            cache_folder = None

    embeddings = HuggingFaceEmbeddings(
        model_name=model_or_path,
        cache_folder=cache_folder,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[Embedding] 加载完成。")
    return embeddings
