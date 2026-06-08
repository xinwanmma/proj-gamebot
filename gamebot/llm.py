"""
模型加载模块 —— 三种模型的初始化集中在这里。

包含：
1. ChatLLM  —— 智谱 GLM（负责对话/路由/打分）
2. Embedding —— text2vec 中文向量模型（负责文本→向量）
3. Reranker  —— bge Cross-Encoder（负责检索结果精排）

所有模型都是"懒加载"（用到时才初始化），且只初始化一次。
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from . import config


# ==================== 1. ChatLLM ====================

# ChatOpenAI 是 langchain 对 OpenAI 格式 API 的封装。
# 智谱 API 兼容 OpenAI 格式，所以直接复用，改 base_url 就行。
def _make_llm(temperature: float) -> ChatOpenAI:
    """创建一个智谱 LLM 实例。"""
    return ChatOpenAI(
        model=config.ZHIPU_MODEL,
        api_key=config.ZHIPU_API_KEY,
        base_url=config.ZHIPU_BASE_URL,
        temperature=temperature,
        streaming=True,     # 开启流式，否则拿不到 token 级输出
    )

# 缓存 3 个不同温度的实例，避免重复创建
@lru_cache(maxsize=3)
def get_llm(temperature: float) -> ChatOpenAI:
    """获取指定温度的 LLM 实例。"""
    return _make_llm(temperature)

# 快捷方式：各场景用对应温度
def get_router_llm() -> ChatOpenAI:
    """路由用：低温度，结果稳定。"""
    return get_llm(config.LLM_TEMPERATURE_ROUTER)

def get_answer_llm() -> ChatOpenAI:
    """问答用：中等温度。"""
    return get_llm(config.LLM_TEMPERATURE_ANSWER)

def get_tactics_llm() -> ChatOpenAI:
    """战术建议用：稍高温度，更灵活。"""
    return get_llm(config.LLM_TEMPERATURE_TACTICS)


# ==================== 2. Embedding ====================

# 离线模式：模型已下载到本地缓存时，不联网检查更新
# （国内访问 huggingface.co 不稳定，离线模式避免卡住）
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取文本向量化模型。
    
    作用：把一段文字变成 768 维向量，用于 Chroma 向量检索。
    """
    print("[Embedding] 加载向量模型 ...")
    emb = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},   # 归一化，余弦相似度更稳定
    )
    print("[Embedding] 加载完成。")
    return emb


# ==================== 3. Reranker（可选） ====================

class Reranker:
    """
    Cross-Encoder 精排器。
    
    和 Embedding 的区别：
    - Embedding 是 Bi-Encoder：query 和 doc 各自算向量 → 余弦相似度（快但粗）
    - Reranker 是 Cross-Encoder：把 (query, doc) 拼一起送模型打分（慢但准）
    
    工程实践：先用 Embedding 从全库粗排 top-20，
    再用 Reranker 精排到 top-5 喂给 LLM。
    """

    def __init__(self):
        """延迟导入 CrossEncoder，避免启动时加载过大模型。"""
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(
            config.RERANKER_MODEL,
            device=config.RERANKER_DEVICE,
        )
        print(f"[Reranker] 加载完成: {config.RERANKER_MODEL}")

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        对检索结果按 query 相关性重排。
        
        参数：
            query:  用户问题
            docs:   粗排结果（Chroma 返回的文档列表）
            top_k:  保留前几条（默认 config.RERANK_TOP_K）
        
        返回：
            按分数降序排列的文档列表，每篇 metadata 里多了 rerank_score 字段
        """
        if not docs:
            return []
        if top_k is None:
            top_k = config.RERANK_TOP_K

        # 让模型给每篇文档打分
        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs).tolist()

        # 按分数排序，取 top_k
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored[:top_k]:
            meta = dict(doc.metadata or {})
            meta["rerank_score"] = float(score)     # 记住分数，调试有用
            result.append(Document(
                page_content=doc.page_content,
                metadata=meta,
            ))
        return result


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    """获取 Reranker 实例（可关闭）。"""
    if not config.RERANK_ENABLED:
        return None
    return Reranker()
