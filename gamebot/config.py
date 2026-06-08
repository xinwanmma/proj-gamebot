"""
配置模块 —— 所有可调参数集中在这里。
改配置只需改这一个文件，不用翻代码。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（放 API key 等敏感信息）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ========== 项目路径 ==========

# 项目根目录：gamebot/ 的上一层
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 游戏手册 PDF 存放目录
MANUAL_DIR = PROJECT_ROOT / "manual"

# Chroma 向量库持久化目录
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# 对话历史持久化文件（SQLite）
CHECKPOINTER_DB = PROJECT_ROOT / "checkpointer.sqlite"


# ========== LLM 配置 ==========

# 智谱 API key（必填，从 .env 读取）
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
# 智谱 API 地址（兼容 OpenAI 格式）
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
# 模型名，可选 glm-4-flash / glm-4-plus
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")

# 不同场景的 LLM 温度
#   路由（确定意图）：低温度，稳定
#   回答（事实问答）：中温度，严谨
#   战术建议（创意）：高温度，灵活
LLM_TEMPERATURE_ROUTER = 0.1    # 路由用
LLM_TEMPERATURE_ANSWER = 0.3    # 问答用
LLM_TEMPERATURE_TACTICS = 0.5   # 战术建议用


# ========== Embedding 配置 ==========

# 中文向量模型，把文本转换成 768 维向量
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
# 运行设备：cpu 或 cuda（有 GPU 的话）
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


# ========== 切片配置 ==========

# 每个文档片段的字符数
CHUNK_SIZE = 500
# 相邻片段重叠字符数（防止切在句子中间丢失上下文）
CHUNK_OVERLAP = 80


# ========== 检索配置 ==========

# 最终喂给 LLM 的文档数
RETRIEVE_TOP_K = 5

# Chroma 粗排取 top-K 条，交给 reranker 精排
RERANK_CANDIDATE_K = 20
# 精排后保留几条
RERANK_TOP_K = 5

# 是否启用 reanker（Cross-Encoder 精排）
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
# Reranker 模型名
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
# Reranker 运行设备
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", EMBEDDING_DEVICE)

# Chroma 集合名
CHROMA_COLLECTION = "uoc2_manual"
