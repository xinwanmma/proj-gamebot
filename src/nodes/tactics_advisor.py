"""战术建议节点：给出可执行的开局/操作建议。"""
from __future__ import annotations

from .. import prompts
from ..llm import content_to_str, get_tactics_llm
from ..state import AgentState, format_history
from .retriever import format_docs


def tactics_node(state: AgentState) -> dict:
    query = state["query"]
    context = format_docs(state.get("retrieved_docs", []))
    history_block = format_history(state.get("messages", [])[:-1], max_turns=3)
    prompt = prompts.TACTICS_PROMPT.format(
        context=context,
        history_block=history_block,
        query=query,
    )
    resp = get_tactics_llm().invoke(prompt)
    answer = content_to_str(resp.content).strip()
    print(f"[TacticsAdvisor] 生成 {len(answer)} 字建议")
    return {"answer": answer}
