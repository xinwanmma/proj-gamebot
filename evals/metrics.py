"""
评估指标实现。

提供两类指标：
1. 检索指标（无 LLM）
   - Recall@K: top-K 检索结果中是否包含至少一个相关 chunk
              判定方式有两种：
              a) source + page 严格匹配（如果 dataset 标注了 gt_source）
              b) 关键词覆盖（gt_keywords 全部出现在 chunk 文本中）
2. 端到端回答评分（LLM-as-Judge）
   - 给定 query / expected_points / answer，让 LLM 给 1-5 分
   - 维度：相关性、事实一致性、完整性、是否拒答（应当拒答时拒答 = 加分）
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


# ---------- 检索指标 ----------

def _chunk_hit_by_metadata(chunk: Document, gt_source: str | None, gt_page: int | None) -> bool:
    """source + page 严格匹配。"""
    if not gt_source:
        return False
    meta = chunk.metadata or {}
    if meta.get("source") != gt_source:
        return False
    if gt_page is not None and meta.get("page") != gt_page:
        return False
    return True


def _chunk_hit_by_keywords(chunk: Document, gt_keywords: list[str]) -> bool:
    """关键词覆盖：所有 gt_keywords 都出现在 chunk 文本中（不区分大小写）。"""
    if not gt_keywords:
        return False
    text = (chunk.page_content or "").lower()
    return any(kw.lower() in text for kw in gt_keywords)


def recall_at_k(
    retrieved_docs: list[Document],
    sample: dict[str, Any],
    k: int,
) -> int:
    """
    Recall@K：top-K 内命中 = 1，否则 = 0。

    判定优先级：
    1. 如果 sample 标注了 gt_source（可选带 gt_page），用 metadata 严格匹配
    2. 否则用 gt_keywords 关键词覆盖判定
    3. 都没有：返回 0 并在调用方报警
    """
    top_k = retrieved_docs[:k]
    gt_source = sample.get("gt_source")
    gt_keywords = sample.get("gt_keywords") or []

    if gt_source:
        hit = any(_chunk_hit_by_metadata(c, gt_source, sample.get("gt_page")) for c in top_k)
    elif gt_keywords:
        hit = any(_chunk_hit_by_keywords(c, gt_keywords) for c in top_k)
    else:
        hit = False
    return 1 if hit else 0


# ---------- LLM-as-Judge ----------

JUDGE_PROMPT = """你是一个严格的评估员，请对 AI 助手的回答打分。

【评估对象】
用户问题：{query}
期望回答应包含的要点：
{expected_points}

AI 实际回答：
{answer}

【打分维度】（每项 1-5 分）
1. 相关性：回答是否针对用户问题
2. 事实一致性：是否与"要点"一致，是否出现明显编造
3. 完整性：是否覆盖了所有要点
4. 拒答合理性：当资料不足时，是否坦诚说明而非硬猜（若要点确实未覆盖但回答合理拒答，本项给高分）

【输出格式】（严格遵守）
相关性: <1-5>
事实一致性: <1-5>
完整性: <1-5>
拒答合理性: <1-5>
总评: <1-5>  # 综合主观分
理由: <一句话说明>
"""


def llm_judge(
    query: str,
    answer: str,
    expected_points: list[str],
    llm=None,
) -> dict[str, int | str]:
    """
    用 LLM 给 answer 打分。

    Args:
        query: 用户问题
        answer: AI 实际回答
        expected_points: 期望回答覆盖的要点
        llm: 可选，传入已构造的 ChatOpenAI；默认用 answer_llm（温度 0）

    Returns:
        dict: {relevance, factuality, completeness, refusal, overall, reason}
    """
    if llm is None:
        # 评估用低温度，临时构造一个
        from src.llm import make_llm
        llm = make_llm(temperature=0.0)

    from src.llm import content_to_str

    points_str = "\n".join(f"- {p}" for p in expected_points) or "(无明确要点)"
    prompt = JUDGE_PROMPT.format(
        query=query,
        expected_points=points_str,
        answer=answer or "(空回答)",
    )
    resp = llm.invoke(prompt)
    text = content_to_str(resp.content).strip()

    # 解析 "字段: 值" 这种行
    fields = {
        "相关性": "relevance",
        "事实一致性": "factuality",
        "完整性": "completeness",
        "拒答合理性": "refusal",
        "总评": "overall",
    }
    result: dict[str, int | str] = {}
    for cn, en in fields.items():
        score = _extract_score(text, cn)
        result[en] = score

    # 抽取理由（"理由:" 后的一行）
    reason = ""
    for line in text.splitlines():
        if line.strip().startswith("理由"):
            reason = line.split(":", 1)[-1].strip()
            break
    result["reason"] = reason
    return result


def _extract_score(text: str, field_cn: str) -> int:
    """从 '字段名: 4' 这种行提取整数分数，失败返回 0。"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(field_cn):
            tail = line.split(":", 1)[-1].strip()
            # 取第一个数字
            for tok in tail.split():
                if tok.isdigit():
                    return int(tok)
    return 0
