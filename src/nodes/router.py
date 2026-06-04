"""
路由节点：判断用户问题意图，决定走哪条边。
"""
from __future__ import annotations

from .. import prompts
from ..llm import content_to_str, get_router_llm
from ..state import AgentState


# 合法标签
_VALID_INTENTS = {"rule", "scenario", "tactics", "chitchat"}


def router_node(state: AgentState) -> dict:
    """
    读取 state.query，调用 LLM 输出意图标签，写回 state.intent。
    """
    query = state["query"]

    # 取最近一轮对话作为上下文（防止"那桥呢"这种代词问题）
    recent = state.get("messages", [])[-3:]
    context_hint = ""
    if recent:
        ctx = "\n".join(
            f"{type(m).__name__}: {content_to_str(m.content)[:200]}"
            for m in recent
        )
        if ctx.strip():
            context_hint = f"\n\n[最近对话上下文]\n{ctx}"

    prompt = prompts.ROUTER_PROMPT.format(query=query + context_hint)
    resp = get_router_llm().invoke(prompt)
    intent = content_to_str(resp.content).strip().lower()

    # 容错：取第一个词，校验
    intent = intent.split()[0] if intent else ""
    if intent not in _VALID_INTENTS:
        # 默认走规则问答（最常见场景）
        intent = "rule"

    print(f"[Router] query='{query[:40]}...' -> intent={intent}")
    return {"intent": intent}  # type: ignore
