"""
Gradio Web UI —— 游戏助手聊天界面。

用法：
    python app.py

浏览器打开 http://127.0.0.1:7860 即可使用。
支持打字机效果（token 级流式输出）。
"""

# 强制离线模式：HuggingFace 模型已本地缓存，不走网络
# 必须在任何 HF/transformers 导入之前设置
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import uuid
from pathlib import Path

# 确保能找到 gamebot 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from gamebot.llm import content_to_str
from gamebot.pipeline import build_graph
from gamebot.config import RERANK_ENABLED


# ===== 启动时预热 =====
# 提前加载模型，避免第一次提问时卡 20 秒

print("[启动] 构建 LangGraph ...")
graph = build_graph()
print("[启动] Graph 就绪。")

print("[启动] 预热向量库和 embedding 模型 ...")
from gamebot.llm import get_embeddings
get_embeddings()

# 尝试加载已有向量库（没有也不阻塞）
from gamebot.pipeline import _get_vectorstore
try:
    _get_vectorstore()
except Exception as e:
    print(f"[启动] 向量库未找到（首次运行请先执行 python ingest_once.py）: {e}")

if RERANK_ENABLED:
    try:
        from gamebot.llm import get_reranker
        get_reranker()
    except Exception:
        pass

print("[启动] 完成。")

# ===== 聊天逻辑 =====

# 各节点的中文标签（用于流式进度提示）
NODE_LABELS = {
    "entry":     "📝 解析输入",
    "router":    "🤔 判断意图",
    "retriever": "📚 检索手册",
    "answer":    "✍️ 生成回答",
    "exit":      "✅ 完成",
}

def chat_stream(user_input: str, history, session_id: str | None = None):
    """
    聊天生成器：流式输出回答，同时返回 session_id 供 Gradio State 持久化。
    
    参数：
        user_input: 用户当前输入
        history:    Gradio 注入的聊天历史
        session_id: Gradio State，区分不同浏览器会话
    
    用 yield 逐步返回 (文本, session_id) 元组，实现打字机效果。
    第一次调用时生成 session_id，之后 Gr.State 记住它，保证多轮对话共用同一 thread。
    """
    if not user_input or not user_input.strip():
        yield "(请输入问题)", session_id or ""
        return

    # 每个浏览器会话用独立 thread_id，多用户不串历史
    if not session_id:
        session_id = f"web-{uuid.uuid4().hex[:12]}"

    progress = []       # 进度提示（如 "🤔 判断意图"）
    answer_parts = []   # 回答的逐字片段
    last_docs = None    # 最后检索到的文档（用于末尾参考来源）

    # LangGraph 支持两种流式模式：
    #   "updates"  → 节点完成时触发，用于显示进度
    #   "messages" → LLM 每生成一个 token 都触发，用于打字机效果
    for mode, payload in graph.stream(
        {"query": user_input.strip()},
        config={"configurable": {"thread_id": session_id}},
        stream_mode=["messages", "updates"],
    ):
        if mode == "updates":
            # 节点完成：显示进度
            for node_name, state_update in (payload or {}).items():
                label = NODE_LABELS.get(node_name)
                if label and label not in progress:
                    progress.append(label)
                    if not answer_parts:
                        yield "\n\n".join(f"_{p}_" for p in progress), session_id

                # 记下检索结果（用于末尾参考来源）
                if isinstance(state_update, dict):
                    docs = state_update.get("retrieved_docs")
                    if docs:
                        last_docs = docs

        elif mode == "messages":
            # LLM token 流
            try:
                msg, meta = payload
                token = content_to_str(msg.content)
                # 只取回答节点的 token
                if token and (meta or {}).get("langgraph_node") == "answer":
                    answer_parts.append(token)
                    progress_md = "\n\n".join(f"_{p}_" for p in progress) + "\n\n" if progress else ""
                    yield progress_md + "".join(answer_parts) + "▌", session_id
            except (TypeError, ValueError):
                pass

    # 最终输出：去掉光标，加上参考来源
    result = "".join(answer_parts)
    if not result:
        # 流式失败时回退到 invoke，用独立的 thread_id 避免污染对话历史
        fb_id = f"{session_id}-fallback"
        result = graph.invoke(
            {"query": user_input.strip()},
            config={"configurable": {"thread_id": fb_id}},
        ).get("answer", "(未生成回答)")

    # 附上参考来源
    if last_docs:
        refs = _format_refs(last_docs)
        if refs:
            result += f"\n\n<details><summary>📚 参考来源</summary>\n\n{refs}\n\n</details>"

    yield result, session_id


def _format_refs(docs) -> str:
    """把检索结果渲染成"参考来源"折叠区。"""
    if not docs:
        return ""
    seen = set()
    lines = []
    for d in docs:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        key = f"{src}:{page}"
        if key in seen:
            continue
        seen.add(key)
        score = d.metadata.get("rerank_score")
        tag = f"（{score:.3f}）" if isinstance(score, float) else ""
        lines.append(f"- {src} p.{page} {tag}")
    return "\n".join(lines)


# ===== Gradio UI =====

TITLE = "🎮 统一指挥 II 游戏助手"
DESC = "基于 LangGraph + 智谱 GLM + 本地向量检索的游戏问答机器人。支持规则查询/战术建议/多轮对话。"

EXAMPLES = [
    "HQ 是什么？它和补给有什么关系？",
    "桥梁被炸了怎么修？需要什么单位？",
    "Barbarossa DLC 第 3 回合德军会得到哪些援军？",
    "Stalingrad 苏军开局怎么布阵？",
    "Reserve 模式有什么用？什么时候开启？",
    "你好，你能做什么？",
]

def respond(user_msg, history, sid):
    """Gradio 回调：把流式输出累积到聊天框。"""
    history = history or []
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ""})
    for chunk, new_sid in chat_stream(user_msg, history, sid):
        history[-1]["content"] = chunk
        yield history, new_sid, ""

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESC}")

    session_state = gr.State(None)          # 每个浏览器会话独立
    chatbot = gr.Chatbot(height=600)

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="问点什么吧…例如：补给线怎么算？桥怎么修？",
            lines=2, scale=9,
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=user_input)

    submit_btn.click(respond, [user_input, chatbot, session_state],
                     [chatbot, session_state, user_input])
    user_input.submit(respond, [user_input, chatbot, session_state],
                      [chatbot, session_state, user_input])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
