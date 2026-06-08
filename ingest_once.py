"""
首次向量化脚本 —— 把 manual/ 下所有 PDF 灌入 Chroma 向量库。

用法：
    python ingest_once.py

只需跑一次。以后启动 app.py 会自动加载已有向量库。
"""
import time
from pathlib import Path
import sys

# 确保能找到 gamebot 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gamebot.ingest import build_vectorstore_from_scratch

def main():
    t0 = time.time()
    vs = build_vectorstore_from_scratch()
    elapsed = time.time() - t0
    print(f"\n✓ 完成！耗时 {elapsed:.1f}s")
    print(f"  共 {vs._collection.count()} 条向量")
    print("  现在可以运行 python app.py 启动 Web 助手了。")

if __name__ == "__main__":
    main()
