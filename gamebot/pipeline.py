"""
流水线模块 —— LangGraph 状态机，串联整个问答流程。

这是整个助手的核心，理解了这个文件就理解了全部逻辑。

流程：
  用户提问 → 意图路由 → 检索手册 → 精排 → 生成回答 → 返回

所有节点、状态、prompt 模板都在这里，不拆成多个文件，
方便你一口气看完整个链路。
"""
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from . import config
from .llm import content_to_str, get_answer_llm, get_embeddings, get_reranker, get_router_llm, get_tactics_llm


# ==================== 1. 状态定义 ====================

# 意图标签：router 节点判断用户问题属于哪一类
Intent = Literal["rule", "scenario", "tactics", "chitchat"]

class BotState(TypedDict):
    """
    整个流水线的"共享数据总线"。
    每个节点从 state 读入数据，写回结果，传给下一个节点。
    """
    messages: Annotated[list, add_messages]  # 对话历史（自动追加，不覆盖）
    query: str                               # 当前用户问题
    intent: Intent                           # 问题意图（由 router 节点产出）
    retrieved_docs: list[Document]           # 检索到的相关手册段落
    answer: str                              # 最终回答


# ==================== 2. Prompt 模板 ====================

# 所有 prompt 集中在这里，方便统一调整。

ROUTER_PROMPT = """你是一个意图分类器，判断用户问题的类型。

游戏：《统一指挥 II》（Unity of Command II），二战策略游戏。

可选标签：
- rule      : 游戏规则、机制、操作。如"补给怎么算""桥怎么修""HQ 是什么"
- scenario  : 剧本/战役查询。如"Barbarossa 第几回合有援军""Stalingrad 胜利条件"
- tactics   : 战术建议。如"这关怎么打""苏军开局怎么布阵""包围圈怎么破"
- chitchat  : 闲聊、问候。只输出标签本身，不要任何解释。

用户问题：{query}
标签："""

# 规则问答和剧本查询用同一个模板（行为接近）
FACTUAL_ANSWER_PROMPT = """你是一位《统一指挥 II》游戏专家。请基于手册内容回答。

要求：
1. 严格依据"参考资料"，不要编造。资料没提就说"手册中未明确说明"
2. 可标注来源（如 "见手册第 X 页"）
3. 如果用户的话有代词（"它""这个""上面说的"），先看对话历史消歧

参考资料：
{context}

{history}

用户问题：{query}
回答："""

TACTICS_PROMPT = """你是一位《统一指挥 II》资深战术教官。

要求：
1. 优先考虑核心机制：补给线、包围、HQ 范围、Step 损失、桥梁、铁路、Reserve 模式
2. 建议要具体到"做什么"，不是泛泛而谈
3. 如果信息不足，反问 1-2 个关键问题

参考资料：
{context}

{history}

用户问题：{query}
回答："""

CHITCHAT_PROMPT = """你是《统一指挥 II》游戏助手。友好地回应用户。

如果用户问与游戏无关的内容，简短回答后引导回游戏话题。
注意看对话历史，用户可能之前说过名字或偏好。

{history}

用户消息：{query}
回答："""


# ==================== 3. 工具函数 ====================

def _format_docs(docs: list[Document]) -> str:
    """把检索到的文档列表格式化成一段文本，放到 prompt 里。"""
    if not docs:
        return "(未检索到相关手册内容)"
    parts = []
    for i, d in enumerate(docs, 1):
        # 每篇文档标注来源（文件名 + 页码）
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] 来源: {src} 第 {page} 页\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def _format_history(messages: list, max_turns: int = 3) -> str:
    """
    把对话历史格式化成文本，放到 prompt 里。
    只保留最近的 max_turns 轮，避免 prompt 太长。
    """
    if not messages:
        return ""

    # 提取 (用户说, AI说) 配对
    pairs = []
    pending = None
    for m in messages:
        if isinstance(m, HumanMessage):
            pending = str(m.content)
        elif isinstance(m, AIMessage):
            ai = str(m.content)
            pairs.append((pending or "", ai))
            pending = None

    # 取最近几轮
    recent = pairs[-max_turns:]
    lines = ["对话历史："]
    for user, ai in recent:
        if user:
            lines.append(f"用户: {user}")
        lines.append(f"AI: {ai}")
    return "\n".join(lines)

def _answer_with_context(query: str, docs: list[Document],
                          messages: list, prompt_template: str,
                          llm) -> str:
    """
    通用回答函数：4 个回答节点都调用这个。
    
    流程：格式化上下文 → 格式化历史 → 拼 prompt → 调 LLM → 返回文本
    """
    context = _format_docs(docs)
    history = _format_history(messages[:-1])   # 去掉当前问题
    prompt = prompt_template.format(
        context=context, history=history, query=query,
    )
    resp = llm.invoke(prompt)
    return content_to_str(resp.content).strip()


# ==================== 4. 向量检索 ====================

# 缓存向量库（只加载一次）
_vectorstore = None

def _get_vectorstore():
    """获取 Chroma 向量库（懒加载）。"""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    # 优先加载已有向量库
    if config.VECTORSTORE_DIR.exists() and any(config.VECTORSTORE_DIR.iterdir()):
        from langchain_chroma import Chroma
        _vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(config.VECTORSTORE_DIR),
        )
    else:
        # 第一次运行：从 PDF 构建
        print("[检索] 未检测到向量库，开始首次构建 ...")
        from .ingest import build_vectorstore_from_scratch
        _vectorstore = build_vectorstore_from_scratch()
    return _vectorstore

def _retrieve(query: str) -> list[Document]:
    """
    从手册中检索与 query 相关的段落。
    
    两步走：
    1. Chroma 向量检索 top-20（粗排，快）
    2. Reranker 精排 top-5（如果启用，准）
    """
    vs = _get_vectorstore()

    # 1. 粗排：用 Embedding 向量相似度检索 top-20
    candidate_k = config.RERANK_CANDIDATE_K if config.RERANK_ENABLED else config.RETRIEVE_TOP_K
    docs = vs.similarity_search(query, k=candidate_k)

    # 2. 精排：用 Cross-Encoder 重排（如果启用）
    reranker = get_reranker()
    if reranker:
        docs = reranker.rerank(query, docs, top_k=config.RERANK_TOP_K)

    print(f"[检索] 共 {len(docs)} 个相关片段")
    return docs


# ==================== 5. LangGraph 节点 ====================

# LangGraph 中的"节点"就是一个函数：
#   输入：当前 state（所有共享数据）
#   输出：dict（要更新到 state 中的字段）

def entry_node(state: BotState) -> dict:
    """入口节点：把用户问题记入对话历史。"""
    return {"messages": [HumanMessage(content=state["query"])]}

def router_node(state: BotState) -> dict:
    """路由节点：判断用户意图。"""
    query = state["query"]
    prompt = ROUTER_PROMPT.format(query=query)
    resp = get_router_llm().invoke(prompt)
    intent = content_to_str(resp.content).strip().lower().split()[0]

    # 容错：如果不是合法标签，默认走规则问答
    valid = {"rule", "scenario", "tactics", "chitchat"}
    if intent not in valid:
        intent = "rule"

    print(f"[路由] '{query[:30]}...' → {intent}")
    return {"intent": intent}

def retriever_node(state: BotState) -> dict:
    """检索节点：搜索相关手册段落。"""
    docs = _retrieve(state["query"])
    return {"retrieved_docs": docs}

def answer_node(state: BotState) -> dict:
    """
    回答节点：根据意图选不同的 prompt 和 LLM 温度。
    
    4 种意图共用这一个节点函数。
    """
    intent = state.get("intent", "rule")
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    messages = state.get("messages", [])

    if intent == "tactics":
        # 战术建议：用高温度 + 专门 prompt
        llm = get_tactics_llm()
        prompt = TACTICS_PROMPT
    elif intent == "chitchat":
        # 闲聊：不需要检索内容
        prompt = CHITCHAT_PROMPT
        docs = []
        llm = get_answer_llm()
    else:
        # rule 和 scenario：事实问答
        llm = get_answer_llm()
        prompt = FACTUAL_ANSWER_PROMPT

    answer = _answer_with_context(query, docs, messages, prompt, llm)
    print(f"[回答] ({intent}) 生成 {len(answer)} 字")
    return {"answer": answer}

def exit_node(state: BotState) -> dict:
    """出口节点：把回答记入对话历史。"""
    return {"messages": [AIMessage(content=state.get("answer", "(无回答)"))]}


# ==================== 6. 构建 LangGraph 流水线 ====================

def build_graph():
    """
    构建并编译 LangGraph 状态机。
    
    流程图（一目了然）：
    
    entry → router ── chitchat ──┐
                     │           │
                     └─ retriever ── answer_node ── exit → END
                                  （4 种意图共用）
    
    关键点：
    - router 判断意图：chitchat 跳过检索，其余先检索再回答
    - retriever 内部自动处理"是否用 reranker"
    - 所有意图共用同一个 answer_node，内部按 intent 切换 prompt
    """
    import sqlite3

    # 准备数据库：LangGraph 用 SQLite 持久化对话历史
    config.CHECKPOINTER_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.CHECKPOINTER_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    # 建图
    builder = StateGraph(BotState)

    builder.add_node("entry", entry_node)
    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("answer", answer_node)
    builder.add_node("exit", exit_node)

    # 连接边
    builder.add_edge(START, "entry")
    builder.add_edge("entry", "router")

    # 条件边：router 之后分叉
    #   chitchat → 直接回答（不检索）
    #   其他    → 先检索再回答
    builder.add_conditional_edges(
        "router",
        lambda s: "chitchat" if s.get("intent") == "chitchat" else "retriever",
        {"retriever": "retriever", "chitchat": "answer"},
    )

    builder.add_edge("retriever", "answer")
    builder.add_edge("answer", "exit")
    builder.add_edge("exit", END)

    return builder.compile(checkpointer=checkpointer)
