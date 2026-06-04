"""
Gradio Web UI 入口 —— 启动游戏助手聊天界面。

用法：
    python app.py

首次启动前请先运行 `python ingest_once.py` 完成向量化。

特性：
- Token 级流式输出（基于 LangGraph stream_mode="messages"）
- 节点级进度提示（router / retriever / rerank / answer 等节点完成时显示）
- 每个会话独立 thread_id（基于 Gradio session state，多用户不串历史）
- 答案末尾附"参考来源"折叠区（来自 retriever + reranker 的 metadata）
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

# 让脚本可直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src import config  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.llm import content_to_str  # noqa: E402


# 全局编译一次 graph（含 checkpointer）
print("[App] 正在构建 LangGraph ...")
_graph = build_graph()
print("[App] Graph 构建完成。")

# 预热 embedding 模型 + 向量库（避免首次提问时卡顿 20+ 秒）
print("[App] 预热向量库 ...")
try:
    from src.embeddings import get_embeddings  # noqa: E402
    from src.ingest import get_or_build_vectorstore  # noqa: E402
    get_embeddings()                              # 触发模型加载
    get_or_build_vectorstore()                    # 触发 Chroma 加载
    print("[App] 预热完成。")
except Exception as e:
    print(f"[App] 预热失败（不阻塞启动）: {e}")

# 预热 reranker（若启用），避免首次提问多等 1 秒
if config.RERANK_ENABLED:
    try:
        from src.reranker import get_reranker  # noqa: E402
        get_reranker()
    except Exception as e:
        print(f"[App] Reranker 预热失败（不阻塞）: {e}")


# 节点名 -> 用户友好的进度提示
_NODE_LABELS = {
    "entry":          "📝 解析输入",
    "router":         "🤔 判断意图",
    "retriever":      "📚 检索手册",
    "rerank":         "🔬 精排片段",
    "rule_answer":    "✍️ 生成回答",
    "scenario_query": "🗺️ 查询剧本",
    "tactics":        "🪖 思考战术",
    "chitchat":       "💬 组织回复",
    "exit":           "✅ 完成",
}

# 真正产出最终答案的节点 —— 这些节点的 LLM token 才流式输出
_ANSWER_NODES = {"rule_answer", "scenario_query", "tactics", "chitchat"}


def _format_refs(docs) -> str:
    """把 retrieved_docs 渲染成"参考来源"折叠区 markdown。"""
    if not docs:
        return ""
    # 去重：(source, page)
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        key = (str(src), str(page))
        if key in seen:
            continue
        seen.add(key)
        score = meta.get("rerank_score")
        score_str = f"（rerank {score:.3f}）" if isinstance(score, float) else ""
        lines.append(f"- {src}  p.{page} {score_str}")
    return "\n".join(lines)


def chat(user_input: str, history, session_id: str | None = None):
    """
    Gradio ChatInterface 回调 —— 同步 generator + token 级流式。

    用 stream_mode=["messages", "updates"] 同时拿两种事件：
    - "updates"：节点完成时 yield {node_name: state_update}，用于显示进度
    - "messages"：LLM 每个 token chunk 都 yield，用于流式答案

    Args:
        user_input:  用户这一轮的输入
        history:     Gradio 注入的对话历史（此处不用，真实历史走 checkpointer）
        session_id:  Gradio State 注入的会话 ID（thread_id），首次为 None 时生成

    Yields:
        当前累积的 markdown 文本（含进度条 / 答案 / 光标 / 参考来源）
    """
    if not user_input or not user_input.strip():
        yield "(请输入问题)"
        return

    # session 隔离：每个 Gradio 会话用独立 thread_id，多用户不串历史
    if not session_id:
        session_id = f"web-{uuid.uuid4().hex[:12]}"
    thread_id = session_id

    progress_lines: list[str] = []
    answer_parts: list[str] = []
    final_refs_md: str = ""  # 最终答案末尾附上的"参考来源"

    def render(progress_only: bool = False) -> str:
        """渲染当前累积的输出。"""
        # 进度块（斜体）
        progress_md = (
            "\n\n".join(f"_{p}_" for p in progress_lines) + "\n\n"
            if progress_lines
            else ""
        )
        if progress_only:
            return progress_md.rstrip()
        # 答案 + 光标
        answer_md = "".join(answer_parts) + ("▌" if answer_parts else "")
        # 参考来源（只在最终 yield 时附加）
        refs_md = final_refs_md
        return progress_md + answer_md + refs_md

    last_state_docs = None

    for mode, payload in _graph.stream(
        {"query": user_input.strip()},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "updates"],
    ):
        # ---- 节点级进度：节点完成时显示 ----
        if mode == "updates":
            if not isinstance(payload, dict):
                continue
            for node_name, state_update in payload.items():
                label = _NODE_LABELS.get(node_name)
                if label and label not in progress_lines:
                    progress_lines.append(label)
                    yield render(progress_only=not answer_parts)

                # 顺手捕获 rerank/retriever 出口的 retrieved_docs，用于末尾引用
                if isinstance(state_update, dict):
                    docs = state_update.get("retrieved_docs")
                    if docs:
                        last_state_docs = docs

        # ---- token 级流式：只取答案节点 ----
        elif mode == "messages":
            try:
                chunk_msg, metadata = payload
            except (TypeError, ValueError):
                continue
            node = (metadata or {}).get("langgraph_node", "")
            if node in _ANSWER_NODES:
                token = content_to_str(getattr(chunk_msg, "content", ""))
                if token:
                    answer_parts.append(token)
                    yield render()

    # 最终：去掉光标，附上参考来源
    if last_state_docs:
        refs = _format_refs(last_state_docs)
        if refs:
            final_refs_md = "\n\n<details><summary>📚 参考来源</summary>\n\n" + refs + "\n\n</details>"

    if answer_parts:
        yield render()
    else:
        # 兜底：streaming 失败时用 invoke 重试
        try:
            result = _graph.invoke(
                {"query": user_input.strip()},
                config={"configurable": {"thread_id": thread_id}},
            )
            answer = result.get("answer", "(未生成回答)")
            docs = result.get("retrieved_docs") or []
            refs = _format_refs(docs) if docs else ""
            refs_md = (
                "\n\n<details><summary>📚 参考来源</summary>\n\n" + refs + "\n\n</details>"
                if refs
                else ""
            )
            yield answer + refs_md
        except Exception as e:
            yield f"(生成回答失败: {e})"


# ---------- UI（Gradio 5/6 风格） ----------

TITLE = "🎮 统一指挥 II（Unity of Command II）游戏助手"
DESC = """
基于 LangGraph + 智谱 GLM-4.5-flash + 本地 text2vec 中文向量 + bge-reranker 精排

**能力**：规则问答 / 剧本查询 / 战术建议 / 多轮对话 ｜ **Token 级流式输出** ｜ **会话隔离**
"""

EXAMPLES = [
    "HQ 是什么？它和补给有什么关系？",
    "桥梁被炸了怎么修？需要什么单位？",
    "Barbarossa DLC 第 3 回合德军会得到哪些援军？",
    "Stalingrad 苏军开局怎么布阵？",
    "包围（Encirclement）是怎么判定的？",
    "Reserve 模式有什么用？什么时候开启？",
]


# 自定义 ChatInterface 以支持 gr.State 注入 session_id
def _chat_wrapper(user_input, history, session_id):
    """把 generator 的 yield 转发给 Gradio。"""
    yield from chat(user_input, history, session_id)


with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}\n{DESC}")

    session_state = gr.State(None)  # 每个浏览器会话独立一份

    chatbot = gr.Chatbot(height=600, type="messages")
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="问点什么吧…例如：补给线怎么算？桥怎么修？",
            lines=2,
            scale=9,
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=user_input)

    def _respond(user_msg, history, sid):
        # 把 chat generator 的每次 yield 累积到 history 的最后一项
        history = history or []
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ""})
        for chunk in chat(user_msg, history, sid):
            history[-1]["content"] = chunk
            yield history, sid, ""

    submit_btn.click(
        _respond,
        inputs=[user_input, chatbot, session_state],
        outputs=[chatbot, session_state, user_input],
    )
    user_input.submit(
        _respond,
        inputs=[user_input, chatbot, session_state],
        outputs=[chatbot, session_state, user_input],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
