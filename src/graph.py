"""
LangGraph 主流程：把各节点串联成状态机。

流程图：
            ┌─────┐
   入口 ──▶ │router│
            └───┬──┘
   ┌────────────┼────────────┬───────────┐
   ▼            ▼            ▼           ▼
retriever  (跳过检索)    (跳过)      chitchat
   │            │            │
   ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│rule_ans │ │scenario │ │tactics  │
└─────────┘ └─────────┘ └─────────┘
   │            │            │
   └────────────┴────────────┘
                ▼
              END
"""
from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage

from . import config
from .nodes import (
    chitchat_node,
    rerank_node,
    retriever_node,
    rule_answer_node,
    router_node,
    scenario_query_node,
    tactics_node,
)
from .state import AgentState


# ---------- 路由函数 ----------

def _route_after_router(state: AgentState) -> str:
    """根据 intent 决定下一跳。"""
    intent = state.get("intent", "rule")
    if intent == "chitchat":
        return "chitchat"
    # 其余三类都需要先检索
    return "retriever"


def _route_after_retriever(state: AgentState) -> str:
    """检索完后，根据 intent 走不同回答节点。"""
    intent = state.get("intent", "rule")
    if intent == "scenario":
        return "scenario_query"
    if intent == "tactics":
        return "tactics"
    return "rule_answer"


# ---------- 入口节点（封装用户输入） ----------

def _entry_node(state: AgentState) -> dict:
    """把 query 包装成 HumanMessage，写入 messages。"""
    query = state["query"]
    return {"messages": [HumanMessage(content=query)]}


# ---------- 出口节点（封装 AI 回答） ----------

def _exit_node(state: AgentState) -> dict:
    """把 answer 包装成 AIMessage，写入 messages。"""
    answer = state.get("answer", "(无回答)")
    return {"messages": [AIMessage(content=answer)]}


# ---------- 构建图 ----------

def build_graph():
    """
    构建并编译 LangGraph。

    使用同步 **SqliteSaver** 作为 checkpointer：
    - 节点级流式 + token 级流式都通过 sync stream_events 实现
    - LangGraph 1.x 的 SqliteSaver 必须通过 with 或手动 setup() 初始化表结构
    """
    config.CHECKPOINTER_DB.parent.mkdir(parents=True, exist_ok=True)

    import sqlite3
    conn = sqlite3.connect(str(config.CHECKPOINTER_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    # 关键：手动触发初始化（等价于 with SqliteSaver(conn) as saver: 进入时的操作）
    checkpointer.setup()

    g = StateGraph(AgentState)

    # 添加节点
    g.add_node("entry", _entry_node)
    g.add_node("router", router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("rerank", rerank_node)  # 新增：Cross-Encoder 精排
    g.add_node("rule_answer", rule_answer_node)
    g.add_node("scenario_query", scenario_query_node)
    g.add_node("tactics", tactics_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("exit", _exit_node)

    # 边
    g.add_edge(START, "entry")
    g.add_edge("entry", "router")

    g.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "retriever": "retriever",
            "chitchat": "chitchat",
        },
    )

    # retriever → rerank → 业务节点（线性串联，rerank 内部判断是否禁用）
    g.add_edge("retriever", "rerank")

    g.add_conditional_edges(
        "rerank",
        _route_after_retriever,
        {
            "rule_answer": "rule_answer",
            "scenario_query": "scenario_query",
            "tactics": "tactics",
        },
    )

    # 所有回答节点都汇聚到 exit
    for n in ("rule_answer", "scenario_query", "tactics", "chitchat"):
        g.add_edge(n, "exit")

    g.add_edge("exit", END)

    return g.compile(checkpointer=checkpointer)
