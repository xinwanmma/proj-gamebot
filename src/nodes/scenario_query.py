"""剧本/战役查询节点。"""
from __future__ import annotations

from .. import prompts
from ..llm import content_to_str, get_answer_llm
from ..state import AgentState, format_history
from .retriever import format_docs


def scenario_query_node(state: AgentState) -> dict:
    query = state["query"]
    context = format_docs(state.get("retrieved_docs", []))
    history_block = format_history(state.get("messages", [])[:-1], max_turns=3)
    prompt = prompts.SCENARIO_PROMPT.format(
        context=context,
        history_block=history_block,
        query=query,
    )
    resp = get_answer_llm().invoke(prompt)
    answer = content_to_str(resp.content).strip()
    print(f"[ScenarioQuery] 生成 {len(answer)} 字回答")
    return {"answer": answer}
