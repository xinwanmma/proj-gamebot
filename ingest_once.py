"""
首次向量化脚本 —— 运行一次即可，把 manual/ 下所有 PDF 灌入 Chroma。

用法：
    python ingest_once.py

后续启动 app.py 时会自动加载已有向量库，无需重新向量化。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 让脚本可以直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.ingest import (  # noqa: E402
    build_vectorstore,
    iter_manual_pdfs,
    load_all_manuals,
    split_documents,
)


def main() -> int:
    print("=" * 60)
    print("  Unity of Command II 游戏助手 —— 首次向量化")
    print("=" * 60)

    # 检查 manual 目录
    if not config.MANUAL_DIR.exists():
        print(f"[ERROR] 手册目录不存在: {config.MANUAL_DIR}")
        return 1

    pdfs = list(iter_manual_pdfs())
    print(f"\n[1/3] 发现 {len(pdfs)} 个 PDF 文件：")
    for p in pdfs:
        print(f"  - {p.name}")

    # 过滤掉剧本图表（暂不处理）
    manuals = [
        p for p in pdfs
        if "scenario" not in p.name.lower() and "剧本" not in p.name
    ]
    print(f"\n将处理其中 {len(manuals)} 份手册（已跳过剧本图表）")

    if not manuals:
        print("[ERROR] 没有可处理的手册文件")
        return 1

    # 加载 + 切片
    t0 = time.time()
    docs = load_all_manuals()
    chunks = split_documents(docs)
    t1 = time.time()
    print(f"\n[2/3] 解析 + 切片耗时: {t1 - t0:.1f}s")

    # 写入向量库
    vs = build_vectorstore(chunks)
    t2 = time.time()
    print(f"\n[3/3] 向量化耗时: {t2 - t1:.1f}s")
    print(f"\n✓ 完成！向量库位置: {config.VECTORSTORE_DIR}")
    print(f"  总耗时: {t2 - t0:.1f}s")
    print(f"  向量数: {vs._collection.count()}")
    print("\n现在可以运行 `python app.py` 启动 Web 助手了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
