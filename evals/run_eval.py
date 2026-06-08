"""
评估脚本 —— 测试检索和回答质量。

用法：
    python -m evals.run_eval                    # 全量跑
    python -m evals.run_eval --limit 5           # 只跑前 5 条
    python -m evals.run_eval --skip-judge        # 跳过 LLM 评分（省 token）
    python -m evals.run_eval --rerank            # 启用 reranker 后对比

输出：
    evals/results.jsonl  ← 每条样本一行，含 recall 和 judge 分数
    控制台汇总          ← 平均 Recall@K、平均总评
"""
import argparse
import json
import sys
import time
from pathlib import Path

# 确保能找到 gamebot 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gamebot.llm import get_embeddings, get_reranker, get_llm
from gamebot.pipeline import _get_vectorstore, build_graph

# ===== 评测数据集 =====
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.jsonl"


# ==================== 指标一：Recall@K ====================

def recall_at_k(docs, sample, k):
    """
    检索召回率：top-K 里有没有找到相关文档？
    返回 0（没找到）或 1（找到了）。
    
    判定规则：
    1. 如果样本标注了 gt_source（文件名），精确匹配 metadata
    2. 否则用 gt_keywords，只要一个关键词出现在文本中就算命中
    """
    top_k = docs[:k]
    gt_source = sample.get("gt_source")
    gt_keywords = sample.get("gt_keywords") or []

    for doc in top_k:
        if gt_source:
            # 精确匹配：文件名 + 页码
            meta = doc.metadata or {}
            if meta.get("source") == gt_source:
                gt_page = sample.get("gt_page")
                if gt_page is None or meta.get("page") == gt_page:
                    return 1
        elif gt_keywords:
            # 模糊匹配：任何一个关键词出现在文本中就命中
            text = (doc.page_content or "").lower()
            if any(kw.lower() in text for kw in gt_keywords):
                return 1
    return 0


# ==================== 指标二：LLM-as-Judge ====================

JUDGE_PROMPT = """你是一个严格的评估员。

【用户问题】
{query}

【期望回答要点】
{expected_points}

【AI 实际回答】
{answer}

请从以下 4 个维度打分（每项 1-5 分）：
1. 相关性：回答是否针对问题
2. 事实一致性：是否与要点一致，没有编造
3. 完整性：是否覆盖了所有要点
4. 拒答合理性：资料不足时是否坦诚说明

【输出格式】
相关性: <1-5>
事实一致性: <1-5>
完整性: <1-5>
拒答合理性: <1-5>
总评: <1-5>
理由: <一句话>
"""

def llm_judge(query, answer, expected_points):
    """让 LLM 给回答打分。"""
    points_str = "\n".join(f"- {p}" for p in expected_points) or "(无明确要点)"
    prompt = JUDGE_PROMPT.format(
        query=query,
        expected_points=points_str,
        answer=answer or "(空回答)",
    )
    # 评估用低温度，保证打分稳定
    resp = get_llm(0.0).invoke(prompt)
    text = str(resp.content).strip()

    # 解析分数
    result = {}
    for cn, en in [("相关性","relevance"),("事实一致性","factuality"),
                    ("完整性","completeness"),("拒答合理性","refusal"),("总评","overall")]:
        for line in text.split("\n"):
            if line.strip().startswith(cn):
                for tok in line.split(":")[-1].split():
                    if tok.isdigit():
                        result[en] = int(tok)
                        break
                break
        if en not in result:
            result[en] = 0

    # 抽取理由
    result["reason"] = ""
    for line in text.split("\n"):
        if line.strip().startswith("理由"):
            result["reason"] = line.split(":", 1)[-1].strip()
            break
    return result


# ==================== 主流程 ====================

def load_dataset(limit=None):
    """加载评测集。"""
    samples = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples[:limit] if limit else samples


def run_one(sample, vs, judge_llm=None, use_rerank=False):
    """跑一条样本：检索 → 可选重排 → 可选 LLM 评分。"""
    query = sample["query"]
    t0 = time.time()

    # 1. 检索 top-20
    docs = vs.similarity_search(query, k=20)

    # 2. 可选：reranker 精排
    if use_rerank:
        reranker = get_reranker()
        if reranker:
            docs = reranker.rerank(query, docs, top_k=10)

    # 3. 算 Recall@K
    result = {
        "query": query,
        "intent": sample.get("intent"),
        "recall@1": recall_at_k(docs, sample, 1),
        "recall@3": recall_at_k(docs, sample, 3),
        "recall@5": recall_at_k(docs, sample, 5),
        "recall@10": recall_at_k(docs, sample, 10),
        "answer": "",
        "judge": {},
        "elapsed_s": round(time.time() - t0, 2),
    }

    # 4. 可选：端到端回答 + LLM 评分
    if judge_llm is not None:
        try:
            g = build_graph()
            r = g.invoke(
                {"query": query},
                config={"configurable": {"thread_id": f"eval-{time.time_ns()}"}},
            )
            answer = r.get("answer", "")
            result["answer"] = answer
            result["judge"] = llm_judge(query, answer, sample.get("expected_points", []))
        except Exception as e:
            result["answer"] = f"(生成失败: {e})"

    return result


def print_summary(results):
    """打印汇总。"""
    n = len(results)
    if n == 0:
        print("(无样本)")
        return

    def avg(field):
        vals = [r.get(field, 0) for r in results]
        return round(sum(vals) / n, 3)

    print(f"\n{'='*50}")
    print(f"评估汇总（共 {n} 条）")
    print(f"{'='*50}")
    print(f"Recall@1  = {avg('recall@1')}")
    print(f"Recall@3  = {avg('recall@3')}")
    print(f"Recall@5  = {avg('recall@5')}")
    print(f"Recall@10 = {avg('recall@10')}")

    judged = [r for r in results if r.get("judge")]
    if judged:
        print(f"{'-'*50}")
        print(f"LLM-as-Judge（{len(judged)} 条有效）")
        for cn, en in [("相关性","relevance"),("事实一致性","factuality"),
                        ("完整性","completeness"),("拒答合理性","refusal"),("总评","overall")]:
            vals = [r["judge"].get(en, 0) for r in judged if isinstance(r["judge"].get(en), int)]
            print(f"  {cn}: {round(sum(vals)/len(vals), 3)}" if vals else f"  {cn}: -")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="运行评估")
    parser.add_argument("--limit", type=int, help="只跑前 N 条")
    parser.add_argument("--skip-judge", action="store_true", help="跳过 LLM 评分")
    parser.add_argument("--rerank", action="store_true", help="启用 reranker 后对比")
    args = parser.parse_args()

    # 加载数据集
    samples = load_dataset(args.limit)
    print(f"评测集: {len(samples)} 条样本")

    # 加载向量库
    print("加载向量库 ...")
    vs = _get_vectorstore()

    # 构造评判 LLM
    judge_llm = None if args.skip_judge else get_llm(0.0)

    # 逐条评测
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {sample['query']}")
        r = run_one(sample, vs, judge_llm, use_rerank=args.rerank)
        results.append(r)
        print(f"  Recall@5={r['recall@5']}  耗时={r['elapsed_s']}s")
        if r.get("judge"):
            print(f"  Judge 总评={r['judge'].get('overall')}  {r['judge'].get('reason','')}")

    # 保存结果
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果已保存: {RESULTS_PATH}")

    print_summary(results)

if __name__ == "__main__":
    main()
