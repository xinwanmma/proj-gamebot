"""闲聊节点。"""
from __future__ import annotations

from .. import prompts
from ..llm import content_to_str, get_answer_llm
from ..state import AgentState, format_history


def chitchat_node(state: AgentState) -> dict:
    query = state["query"]
    history_block = format_history(state.get("messages", [])[:-1], max_turns=5)
    prompt = prompts.CHITCHAT_PROMPT.format(history_block=history_block, query=query)
    resp = get_answer_llm().invoke(prompt)
    answer = content_to_str(resp.content).strip()
    return {"answer": answer}
