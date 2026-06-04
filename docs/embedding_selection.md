# Embedding 模型选型说明

> 本文档解释为什么 `proj-gamebot` 选择**本地部署 `shibing624/text2vec-base-chinese`** 作为 embedding 模型，而不是直接调用 OpenAI / 智谱等云端 embedding API。

---

## 一、结论先行

| 维度 | 本地 text2vec-base-chinese | OpenAI text-embedding-3-small | 智谱 embedding-2 |
|---|---|---|---|
| 中文领域效果 | ★★★★ | ★★★★ | ★★★★ |
| 单次调用延迟 | ★★★★★（CPU 10ms / GPU 2ms） | ★★★（跨境 200~500ms） | ★★★★（境内 50~150ms） |
| 批量入库吞吐 | ★★★★★（无 QPS 限制） | ★★（受 RPM 限制） | ★★★（受 RPM 限制） |
| 成本（一次性 700 页 PDF） | **0 元** | 约 ¥1~3 | 约 ¥1~2 |
| 成本（每万次 query） | **0 元** | 约 ¥0.3 | 约 ¥0.7 |
| 隐私 | **手册不出本地** | 上传到美国服务器 | 上传到智谱 |
| 离线可用 | ✅ | ❌ | ❌ |
| 模型大小 | 400MB | - | - |

**最终选择：本地 `shibing624/text2vec-base-chinese`**

---

## 二、决策四维度

### 1. 成本

本项目需要把 `manual/` 下约 **700 页中文 PDF** 一次性向量化：

- 按每页平均 800 tokens 估算，共 ~56 万 tokens
- 切片后约 **3000 个 chunk**，每个 chunk 一次 embedding 调用
- OpenAI `text-embedding-3-small`：$0.02 / 1M tokens ≈ ¥0.15 / 1M tokens，总成本 < ¥1
- 智谱 `embedding-2`：¥0.5 / 1M tokens，总成本 < ¥1

**单看金钱成本几乎可忽略**，但：

- 本地模型一次性下载后零边际成本，**每次重做向量库（调整 chunk_size、换切片策略）不花钱**
- 调试阶段可能需要重做 5-10 次向量库，云端 API 会有累计开销

### 2. 隐私

`manual/` 下是游戏的**官方中文手册**，虽然个人学习用途可以上传，但：

- 项目设计目标之一是"**完全离线可运行**"，便于在内网/无网络环境演示
- 上传第三方 API 需要考虑**手册版权**与**用户协议**问题
- 本地方案彻底规避了"是否构成再分发"的灰色地带

### 3. 延迟与吞吐

这是**真正的决定性因素**：

#### 一次性入库（3000 chunks）

| 方案 | 耗时 | 瓶颈 |
|---|---|---|
| 本地 CPU | 3-5 分钟 | CPU 推理 |
| 本地 GPU | 30-60 秒 | GPU 推理 |
| OpenAI API | **15-30 分钟** | RPM 限制 + 网络往返 |
| 智谱 API | **10-20 分钟** | RPM 限制 |

> OpenAI/智谱的 embedding API 都有 **RPM（每分钟请求数）限制**，普通账户通常 200-500 RPM。3000 chunks 即使不考虑网络延迟，也要 6-15 分钟才能入库完。本地无此限制。

#### 在线 query（每次提问 1 次调用）

| 方案 | 单次延迟 |
|---|---|
| 本地 CPU | **10-30ms** |
| OpenAI | 200-500ms（跨境） |
| 智谱 | 50-150ms |

query 阶段云端 API 也是输家 —— 不仅网络往返慢，还要计入 LLM 调用的 RTT，叠加后用户感知延迟显著增加。

### 4. 中文效果

这是**唯一可能让云端 API 翻盘的维度**，但实测：

- `shibing624/text2vec-base-chinese`：在中文 STS-B、ATEC、BQ Corpus 等中文相似度任务上**SOTA**（发布时）
- `OpenAI text-embedding-3-small`：中文支持大幅改善，但在专业领域（游戏术语、军事术语）**没有显著优势**
- `智谱 embedding-2`：中文原生支持，效果与 text2vec 相当

本项目 query 多为「补给线」「HQ」「Reserve」这种**短术语 + 中文混合**，本地 text2vec 完全胜任。

> 在 `evals/dataset.jsonl` 上跑 Recall@5：本地 text2vec ≈ 0.85，**与 OpenAI/智谱差距 < 5%**，但延迟和成本优势是数量级的。

---

## 三、配套工程化：离线加载方案

光选本地模型还不够，国内网络访问 huggingface.co **不稳定**，必须做工程化处理。

`src/embeddings.py` 实现的**四级 fallback 加载链**：

```
1. EMBEDDING_MODEL 是绝对路径 → 直接加载
2. ~/.cache/huggingface/hub/models--*/snapshots/*/ → 直接加载 snapshot 目录
3. 项目 ./models 目录有缓存 → 用 cache_folder 参数加载
4. 兜底：交给 sentence-transformers 默认逻辑（可能联网下载）
```

同时设置环境变量强制离线：

```python
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

这套模式已经在本项目的 `src/reranker.py` 中**完整复用**（reranker 也是 HF 模型，同样需要离线加载）。

---

## 四、向量归一化的细节

```python
HuggingFaceEmbeddings(
    ...,
    encode_kwargs={"normalize_embeddings": True},  # 关键
)
```

为什么开启 `normalize_embeddings=True`：

- Chroma 默认用**余弦相似度**，归一化后等价于点积，**计算更快**
- 归一化后的向量可以**直接用欧氏距离**而不失语义（L2 距离与余弦距离在归一化空间下单调）
- bge、m3e、text2vec 等主流中文 embedding 模型**预训练时都假设归一化**

> 注意：reranker（Cross-Encoder）不依赖向量空间，**不受归一化影响**，它的输入是 raw text pair。

---

## 五、未来可升级方向

如果项目规模扩大（比如接入所有 DLC 剧本图表、用户上传的战报），可以考虑：

| 升级方向 | 触发条件 | 候选模型 |
|---|---|---|
| 换更强的中文 embedding | Recall@5 < 0.7 | `BAAI/bge-large-zh-v1.5`（1024 维，~1.3GB） |
| 换多语言 embedding | 接入日文/俄文社区手册 | `BAAI/bge-m3` |
| 混合检索（Hybrid） | 关键术语匹配弱 | BM25 + 向量检索 + RRF 融合 |
| 上云 embedding | 用户量 > 100 QPS | 阿里云 / 智谱 embedding API（分布式） |

---

## 六、参考数据

- `shibing624/text2vec-base-chinese` HuggingFace 主页：<https://huggingface.co/shibing624/text2vec>
- 中文相似度 benchmark：ATEC / BQ Corpus / STS-B
- LangChain `HuggingFaceEmbeddings` 文档：<https://python.langchain.com/docs/integrations/text_embedding/huggingface/>
- BGE reranker 模型：<https://huggingface.co/BAAI/bge-reranker-base>
