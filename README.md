# 🎮 统一指挥 II 游戏助手（LangGraph + RAG AI Agent）

一个面向《统一指挥 II》（*Unity of Command II*）的领域问答智能体：把官方手册灌进本地向量库，按用户问题动态路由到不同回答策略，全部跑在一个 Gradio Web UI 里。

---

## ✨ 核心特性

| 能力 | 说明 |
|---|---|
| 🤖 **意图路由 Agent** | LangGraph StateGraph 多节点动态调度：rule / scenario / tactics / chitchat 四条分支 |
| 📚 **完整 RAG 链路** | PDF → 切片 → 本地 embedding → Chroma → 检索 → 生成 |
| 🔬 **Cross-Encoder 精排** | Chroma 粗排 top-20 → bge-reranker 精排 top-5，比纯向量检索 Recall@5 显著提升 |
| 💬 **多轮对话记忆** | SqliteSaver Checkpointer，长会话上下文不丢；每用户独立 thread_id 隔离 |
| ⚡ **Token 级流式** | 节点级进度提示 + LLM token 级打字机输出，端到端体验接近 ChatGPT |
| 📊 **评估流水线** | 16 条多意图评测集，Recall@K + LLM-as-Judge 5 维度评分，支持 reranker A/B 对比 |
| 🌐 **完全离线运行** | LLM 之外，所有组件（embedding / reranker / vectorstore / checkpointer）都在本地 |

---

## 🏗️ 架构

```
                     ┌──────────┐
   manual/(PDF) ───▶ │ ingest   │ ──▶ Chroma（本地向量库）
                     └──────────┘             │
                                              ▼
   用户输入 ──▶ Gradio ──▶ LangGraph ──▶ retriever（粗排 top-20）
                              │                │
                              ▼                ▼
                          router ──▶ rerank（Cross-Encoder 精排 top-5）
                              │                │
                ┌─────────────┼────────────┐   │
                ▼             ▼            ▼   ▼
            rule_answer  scenario_query  tactics  chitchat
                │             │            │     │
                └─────────────┴────────────┘     │
                              │                  │
                              ▼                  │
                          exit（写回 messages）◀──┘
                              │
                              ▼
                       SqliteSaver（会话持久化）
```

**State 字段**：`messages`（追加 reducer） / `query` / `intent` / `retrieved_docs` / `answer`

---

## 📦 安装

```bash
# 1. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 2. 配置 API key
cp .env.example .env
# 编辑 .env，填入你的 ZHIPU_API_KEY

# 3. （强烈推荐）预下载 Reranker 模型
# 国内访问 huggingface.co 不稳定，先手动下到本地缓存可避免首次启动卡死
python scripts/download_models.py
# 默认下载 BAAI/bge-reranker-base（约 280MB），走 hf-mirror 镜像
```

> 智谱 API key 申请：<https://open.bigmodel.cn/>

---

## 🚀 使用

### 第一次：向量化手册（约 1-3 分钟）

```bash
python ingest_once.py
```

把 `manual/` 下所有 PDF（除剧本图表）切片并写入 `vectorstore/`。**只需运行一次**。

### 启动 Web 助手

```bash
python app.py
```

浏览器自动打开 <http://127.0.0.1:7860>，享受 token 级流式输出。

### 跑评估

```bash
python -m evals.run_eval --limit 5           # 快速验证（前 5 条）
python -m evals.run_eval                     # 全量 16 条
python -m evals.run_eval --rerank            # 启用 reranker 后再评估（A/B 对比）
python -m evals.run_eval --skip-judge        # 只算检索指标，跳过 LLM 评分
```

输出：
- `evals/results.jsonl` — 每条样本一行，含 recall@1/3/5/10、judge 5 维度评分、答案原文、耗时
- 控制台汇总 — 平均 Recall@K、平均 Judge 总评

### 重置向量库

```bash
rm -rf vectorstore/
python ingest_once.py
```

---

## 📁 项目结构

```
proj-gamebot/
├── app.py                        # 📺 Gradio Web UI（token 级流式 + session 隔离 + 引用展示）
├── ingest_once.py                # 📥 首次向量化脚本
├── requirements.txt
├── .env.example
│
├── manual/                       # 📖 游戏手册 PDF（自带）
├── vectorstore/                  # 💾 Chroma 持久化（首次运行后生成）
├── checkpointer.sqlite           # 💾 会话记忆持久化（运行时生成）
│
├── src/
│   ├── config.py                 # ⚙️ 全局配置（.env + 默认值）
│   ├── llm.py                    # 🧠 智谱 GLM 封装（OpenAI 兼容）
│   ├── embeddings.py             # 📐 本地 text2vec embedding
│   ├── reranker.py               # 🔬 bge-reranker Cross-Encoder 封装
│   ├── ingest.py                 # 📄 PDF 解析 / 切片 / 向量化
│   ├── state.py                  # 📋 LangGraph State + 历史格式化
│   ├── prompts.py                # 💬 各节点 Prompt 模板
│   ├── graph.py                  # 🔗 LangGraph 主流程
│   └── nodes/
│       ├── router.py             # 意图分类
│       ├── retriever.py          # 向量检索（粗排）
│       ├── rerank.py             # Cross-Encoder 精排
│       ├── rule_answer.py        # 规则问答
│       ├── scenario_query.py     # 剧本查询
│       ├── tactics_advisor.py    # 战术建议
│       └── chitchat.py           # 闲聊
│
├── evals/                        # 📊 评估流水线
│   ├── dataset.jsonl             # 16 条多意图评测集
│   ├── metrics.py                # Recall@K + LLM-as-Judge
│   └── run_eval.py               # 主入口
│
├── scripts/
│   ├── download_models.py        # 📥 模型预下载（国内镜像）
│   └── README.md
│
└── docs/
    └── embedding_selection.md    # 📝 Embedding 选型决策文档
```

---

## 🔧 配置

主要在 `.env` 中配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ZHIPU_API_KEY` | （必填） | 智谱 AI API key |
| `ZHIPU_MODEL` | `glm-4.5-flash` | 智谱模型（可改 `glm-4-plus` / `glm-4-flash`） |
| `EMBEDDING_MODEL` | `shibing624/text2vec-base-chinese` | 本地向量模型 |
| `EMBEDDING_DEVICE` | `cpu` | 有 GPU 可改 `cuda` |
| `RERANK_ENABLED` | `1` | 是否启用 reranker（关闭后退化为纯向量检索） |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | reranker 模型 |
| `RERANK_CANDIDATE_K` | `20` | 粗排取多少条 |
| `RERANK_TOP_K` | `5` | 重排后保留多少条 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | 切片参数（在 `src/config.py` 改） |
| `RETRIEVE_TOP_K` | `5` | 检索返回文档数（rerank 关闭时生效） |

---

## 🧪 技术栈

| 层 | 技术 |
|---|---|
| Agent 编排 | **LangGraph** StateGraph + SqliteSaver Checkpointer |
| LLM | 智谱 **GLM-4.5-flash**（OpenAI 协议兼容，复用 `langchain-openai`） |
| Embedding | **shibing624/text2vec-base-chinese**（本地 768 维中文向量） |
| Reranker | **BAAI/bge-reranker-base**（Cross-Encoder 精排） |
| 向量库 | **ChromaDB**（本地持久化） |
| PDF 解析 | **pdfplumber**（中文友好）+ pypdf（兜底） |
| Web UI | **Gradio 5/6**（`type="messages"` + `gr.State` session 隔离） |
| 模型加载 | **sentence-transformers** + **HuggingFace Embeddings** |

---

## 📐 设计要点

### 1. 为什么用本地 embedding 而不是云端 API？

详见 [`docs/embedding_selection.md`](docs/embedding_selection.md)。一句话总结：

> **本地 text2vec 在中文领域效果与 OpenAI/智谱 embedding 相当（差距 < 5%），但延迟从 ~500ms 降到 ~30ms，且零成本、可离线、隐私友好。**

### 2. 为什么需要 Cross-Encoder Reranker？

- **Bi-Encoder（embedding）**：query 和 doc 各自编码成向量后算余弦，**快但粗**
- **Cross-Encoder（reranker）**：把 (query, doc) 一起送进模型打分，**慢但精**

工程实践：用 Bi-Encoder 从全库（~3000 chunks）粗排 top-20，再用 Cross-Encoder 精排到 top-5 喂给 LLM。**精度 ↑，成本 ↑ 仅一点点**。

### 3. 为什么用 LangGraph 而不是 LangChain Chain？

- Chain 是**线性**的，无法表达"根据意图走不同分支"
- LangGraph 的 **StateGraph + conditional_edges** 天然支持路由 + 循环 + 状态持久化
- 配套 **Checkpointer** 让多轮对话记忆变成一行配置

### 4. 为什么用智谱 GLM 而不是 OpenAI GPT？

- **国内网络稳定**（无需代理）
- **OpenAI 协议兼容**（一行 `base_url` 就能切）
- **中文领域效果好** + 价格便宜（`glm-4.5-flash` 每万 tokens < ¥0.1）

---

## 🐛 常见问题

**Q: 首次向量化很慢？**
A: 要下载 `shibing624/text2vec-base-chinese`（约 400MB）+ 解析 PDF + 推理向量。仅一次。可改 `EMBEDDING_DEVICE=cuda` 加速。

**Q: 想用 GPU 加速？**
A: `.env` 中 `EMBEDDING_DEVICE=cuda` 和 `RERANKER_DEVICE=cuda` 即可（需要 torch CUDA 版本）。

**Q: reranker 报错 `OSError: ... not a valid model identifier`？**
A: 模型未本地缓存。跑 `python scripts/download_models.py` 预下载。

**Q: 想接入剧本图表？**
A: 暂未处理。建议直接打开 `manual/Scenario Chart - *.pdf` 查看。向量检索对图表效果一般，最好结构化抽取。

**Q: 怎么换 LLM 模型？**
A: 改 `.env` 中的 `ZHIPU_MODEL`，重启 `app.py`。

**Q: 评估脚本报 `APIConnectionError`？**
A: 网络/代理问题。先确认 `curl https://open.bigmodel.cn` 能通；或加 `NO_PROXY=open.bigmodel.cn` 绕过代理。

---

## 📊 评估结果

### 检索指标（16 条多意图测试集）

| 指标 | 纯向量检索 | + Cross-Encoder 精排 |
|------|-----------|---------------------|
| Recall@1 | 0.375 | 0.375 |
| Recall@3 | 0.375 | 0.375 |
| Recall@5 | 0.375 | 0.375 |
| Recall@10 | 0.375 | **0.438** |

### LLM-as-Judge 评分（5 分制）

| 维度 | 评分 |
|------|------|
| 相关性（Relevance） | **4.62** |
| 事实性（Factuality） | 3.50 |
| 完整性（Completeness） | 2.75 |
| 拒答准确率（Refusal） | **4.12** |
| 综合（Overall） | **3.69** |

---

## 📝 License

仅供个人学习使用。游戏手册版权归原作者。
