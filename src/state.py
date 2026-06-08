"""
LangGraph 全局状态定义。

State 在节点之间流转，每个节点读取需要的字段、写回产出字段。
messages 使用 add_messages reducer，自动追加而非覆盖。
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


# 意图标签，router 节点产出
Intent = Literal["rule", "scenario", "tactics", "chitchat"]


class AgentState(TypedDict):
    """LangGraph 主状态。"""
    # 对话历史（自动追加，由 checkpointer 持久化）
    messages: Annotated[list[BaseMessage], add_messages]

    # 当前用户问题（每轮覆盖）
    query: str

    # router 产出的意图标签
    intent: Intent

    # retriever 产出的检索文档
    retrieved_docs: list[Document]

    # 最终给用户的回答
    answer: str


# ---------- 历史消息格式化工具 ----------

def format_history(messages: list[BaseMessage], max_turns: int = 4) -> str:
    """
    把 messages 列表格式化为 prompt 可用的"对话历史"文本块。

    只保留最近 max_turns 轮（一轮 = 一问一答），避免上下文过长。
    返回的字符串已包含 "对话历史：" 标题；若历史为空，返回空字符串。

    Args:
        messages:   state["messages"]，BaseMessage 列表
        max_turns:  最多保留几轮（一问一答算一轮）

    Returns:
        形如：
            对话历史：
            用户: 补给线是什么？
            AI: 补给线是 ...
        或空字符串
    """
    if not messages:
        return ""

    # 只取 HumanMessage 和 AIMessage，按时间顺序
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for m in messages:
        if isinstance(m, HumanMessage):
            pending_user = m.content if isinstance(m.content, str) else str(m.content)
        elif isinstance(m, AIMessage):
            ai_text = m.content if isinstance(m.content, str) else str(m.content)
            if pending_user is not None:
                pairs.append((pending_user, ai_text))
                pending_user = None
            else:
                pairs.append(("", ai_text))

    # 只保留最近 max_turns 轮
    recent = pairs[-max_turns:]
    if not recent:
        return ""

    lines = ["对话历史："]
    for u, a in recent:
        if u:
            lines.append(f"用户: {u}")
        lines.append(f"AI: {a}")
        lines.append("")
    return "\n".join(lines).strip()
