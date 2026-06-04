"""
评估脚本：跑全量 dataset.jsonl，输出 Recall@K 表格 + LLM-as-Judge 评分。

用法：
    python -m evals.run_eval                  # 跑全部样本
    python -m evals.run_eval --limit 5        # 只跑前 5 条
    python -m evals.run_eval --skip-judge     # 只算检索指标，跳过 LLM 评分
    python -m evals.run_eval --rerank         # 启用 reranker 后再评估（需先实现 reranker）

输出：
    evals/results.jsonl   —— 每条样本一行，含 recall@1/3/5、judge 分数
    控制台汇总            —— 平均 Recall@K、平均总评
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 让脚本可直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import get_or_build_vectorstore  # noqa: E402
from src.llm import content_to_str, make_llm  # noqa: E402
from src.state import AgentState  # noqa: E402

from evals.metrics import llm_judge, recall_at_k  # noqa: E402


DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.jsonl"


def load_dataset(limit: int | None = None) -> list[dict]:
    """加载评估集。"""
    samples: list[dict] = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if limit:
        samples = samples[:limit]
    return samples


def run_one(sample: dict, vs, judge_llm=None, use_rerank: bool = False) -> dict:
    """对单条样本跑：检索 + （可选）重排 + （可选）LLM 评分。"""
    query = sample["query"]
    t0 = time.time()

    # 1) 检索 —— 取 top-20 粗排，便于 Recall@5/10 评估
    candidate_k = 20
    docs = vs.similarity_search(query, k=candidate_k)

    # 2) （可选）reranker 重排
    if use_rerank:
        try:
            from src.reranker import get_reranker
            docs = get_reranker().rerank(query, docs, top_k=10)
        except Exception as e:
            print(f"  [warn] rerank 失败，回退到纯检索: {e}")

    # 3) Recall@K
    recall1 = recall_at_k(docs, sample, k=1)
    recall3 = recall_at_k(docs, sample, k=3)
    recall5 = recall_at_k(docs, sample, k=5)
    recall10 = recall_at_k(docs, sample, k=10)

    # 4) （可选）端到端回答 —— 走 graph 拿 answer
    answer = ""
    if judge_llm is not None:
        try:
            from src.graph import build_graph
            graph = build_graph()
            # 评估时每条样本用独立 thread，避免历史串扰
            result = graph.invoke(
                {"query": query},
                config={"configurable": {"thread_id": f"eval-{time.time_ns()}"}},
            )
            answer = result.get("answer", "")
        except Exception as e:
            answer = f"(生成失败: {e})"

    # 5) LLM-as-Judge
    judge_scores: dict = {}
    if judge_llm is not None and answer:
        judge_scores = llm_judge(
            query=query,
            answer=answer,
            expected_points=sample.get("expected_points", []),
            llm=judge_llm,
        )

    elapsed = round(time.time() - t0, 2)
    return {
        "query": query,
        "intent": sample.get("intent"),
        "recall@1": recall1,
        "recall@3": recall3,
        "recall@5": recall5,
        "recall@10": recall10,
        "answer": answer,
        "judge": judge_scores,
        "elapsed_s": elapsed,
    }


def print_summary(results: list[dict]) -> None:
    """控制台打印汇总指标。"""
    n = len(results)
    if n == 0:
        print("(无样本)")
        return

    def avg(field: str) -> float:
        vals = [r.get(field, 0) for r in results]
        return round(sum(vals) / n, 3)

    print("\n" + "=" * 60)
    print(f"评估汇总（共 {n} 条样本）")
    print("=" * 60)
    print(f"Recall@1  = {avg('recall@1')}")
    print(f"Recall@3  = {avg('recall@3')}")
    print(f"Recall@5  = {avg('recall@5')}")
    print(f"Recall@10 = {avg('recall@10')}")

    judged = [r for r in results if r.get("judge")]
    if judged:
        nj = len(judged)

        def javg(field: str) -> float:
            vals = [r["judge"].get(field, 0) for r in judged if isinstance(r["judge"].get(field), int)]
            return round(sum(vals) / max(len(vals), 1), 3)

        print("-" * 60)
        print(f"LLM-as-Judge（共 {nj} 条有效评分）")
        print(f"  相关性    : {javg('relevance')}")
        print(f"  事实一致性: {javg('factuality')}")
        print(f"  完整性    : {javg('completeness')}")
        print(f"  拒答合理性: {javg('refusal')}")
        print(f"  总评      : {javg('overall')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="运行 gamebot 评估")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条样本")
    parser.add_argument("--skip-judge", action="store_true", help="跳过 LLM-as-Judge")
    parser.add_argument("--rerank", action="store_true", help="启用 reranker 后再评估")
    args = parser.parse_args()

    print("[Eval] 加载评估集 ...")
    samples = load_dataset(args.limit)
    print(f"[Eval] 共 {len(samples)} 条样本")

    print("[Eval] 加载向量库 ...")
    vs = get_or_build_vectorstore()

    judge_llm = None if args.skip_judge else make_llm(temperature=0.0)

    results: list[dict] = []
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {sample['query']}")
        r = run_one(sample, vs, judge_llm=judge_llm, use_rerank=args.rerank)
        results.append(r)
        print(f"  Recall@5={r['recall@5']}  耗时={r['elapsed_s']}s")
        if r.get("judge"):
            print(f"  Judge 总评={r['judge'].get('overall')}  理由={r['judge'].get('reason')}")

    # 写入 results.jsonl
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[Eval] 详细结果已写入: {RESULTS_PATH}")

    print_summary(results)


if __name__ == "__main__":
    main()
