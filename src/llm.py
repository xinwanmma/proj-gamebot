"""
智谱 GLM 大模型封装。
智谱 API 兼容 OpenAI 协议，复用 langchain-openai 的 ChatOpenAI 即可。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from . import config


def make_llm(temperature: float = config.LLM_TEMPERATURE_ANSWER) -> ChatOpenAI:
    """
    创建一个智谱 GLM 调用实例。

    Args:
        temperature: 采样温度。
            - 路由/工具调用：0.1（确定性高）
            - 事实问答：0.3（保守）
            - 战术建议：0.5（需要一点创造性）

    Returns:
        ChatOpenAI: 配好智谱 base_url 与 api_key 的 LLM 实例
    """
    return ChatOpenAI(
        model=config.ZHIPU_MODEL,
        api_key=config.ZHIPU_API_KEY,
        base_url=config.ZHIPU_BASE_URL,
        temperature=temperature,
        # 必须 streaming=True 才能拿到 token 级 chunk；
        # 智谱 API 在 streaming 模式下偶尔会把 content 解析成 list（OpenAI 多模态格式），
        # 所有节点都通过 content_to_str() 做了兼容转换
        streaming=True,
    )


@lru_cache(maxsize=4)
def _cached_llm(temperature: float) -> ChatOpenAI:
    """内部缓存，避免重复构造。"""
    return make_llm(temperature=temperature)


def get_router_llm() -> ChatOpenAI:
    """路由节点用：低温度，确定性高。"""
    return _cached_llm(config.LLM_TEMPERATURE_ROUTER)


def get_answer_llm() -> ChatOpenAI:
    """规则问答节点用：中等温度。"""
    return _cached_llm(config.LLM_TEMPERATURE_ANSWER)


def get_tactics_llm() -> ChatOpenAI:
    """战术建议节点用：稍高温度，鼓励多样性。"""
    return _cached_llm(config.LLM_TEMPERATURE_TACTICS)


def content_to_str(content) -> str:
    """
    把 LLM 返回的 content 统一转成 str。

    智谱 / OpenAI 在某些情况下会把 content 返回成 list of parts（多模态格式）：
        [{"type": "text", "text": "..."}, ...]
    这里做兼容处理，提取纯文本拼接。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # {"type": "text", "text": "..."} 这种格式
                t = item.get("text") or item.get("content") or ""
                if t:
                    parts.append(str(t))
        return "".join(parts)
    return str(content)
