"""规则问答节点：基于检索结果回答规则问题。"""
from __future__ import annotations

from .. import prompts
from ..llm import content_to_str, get_answer_llm
from ..state import AgentState, format_history
from .retriever import format_docs


def rule_answer_node(state: AgentState) -> dict:
    query = state["query"]
    context = format_docs(state.get("retrieved_docs", []))
    history_block = format_history(state.get("messages", [])[:-1], max_turns=3)
    prompt = prompts.RULE_ANSWER_PROMPT.format(
        context=context,
        history_block=history_block,
        query=query,
    )
    resp = get_answer_llm().invoke(prompt)
    answer = content_to_str(resp.content).strip()
    print(f"[RuleAnswer] 生成 {len(answer)} 字回答")
    return {"answer": answer}
