"""
独立下载 HuggingFace 模型到本地缓存（开发环境初始化用）。

默认下载项目用的 reranker 模型 BAAI/bge-reranker-base（约 280MB）。
也可通过 --model 参数下载其他模型（如 embedding 模型）。

为什么要单独脚本：
1. 国内访问 huggingface.co 不稳定，必须走镜像（默认 hf-mirror.com）
2. src/embeddings.py 和 src/reranker.py 都强制了 HF_HUB_OFFLINE=1，
   运行时不会再触发下载，必须**预先**把模型放进 HF 缓存
3. 显式脚本便于排错、断点续传、批量预下载

用法：
    # 1) 默认下载 reranker（推荐首次部署时跑）
    python scripts/download_models.py

    # 2) 下载 embedding 模型
    python scripts/download_models.py --model shibing624/text2vec-base-chinese

    # 3) 镜像不通时切官方源（需自备代理）
    python scripts/download_models.py --endpoint https://huggingface.co

    # 4) 下载到项目目录（不走 HF 默认缓存）
    python scripts/download_models.py --cache-dir ./models
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 默认配置
DEFAULT_MODEL = "BAAI/bge-reranker-base"
DEFAULT_ENDPOINT = "https://hf-mirror.com"  # 国内镜像，比官方源快很多


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="预下载 HuggingFace 模型到本地缓存",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace repo id，例如 BAAI/bge-reranker-base",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="HF 端点（镜像），默认 https://hf-mirror.com",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="本地缓存目录；不填则用 HF 默认（~/.cache/huggingface/hub）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="并发下载线程数",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过下载后的反向验证（_resolve_local_snapshot）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 关键：下载阶段必须关闭 OFFLINE 模式
    # （src/embeddings.py、src/reranker.py 在 import 时会 setdefault OFFLINE=1，
    #  如果本脚本意外 import 到它们，OFFLINE 会污染下载流程）
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    print("=" * 60)
    print("[Download] 模型:    ", args.model)
    print("[Download] 镜像:    ", args.endpoint)
    print("[Download] 缓存目录:", args.cache_dir or "(HF 默认)")
    print("[Download] 并发:    ", args.max_workers)
    print("=" * 60)

    # 延迟导入，让上面的环境变量先生效
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[Error] 未安装 huggingface_hub，请先执行：")
        print("        pip install --upgrade huggingface_hub")
        return 1

    try:
        path = snapshot_download(
            repo_id=args.model,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            # 较新版 huggingface_hub 默认就支持断点续传，无影响
            resume_download=True,
        )
    except Exception as e:
        print(f"[Error] 下载失败: {type(e).__name__}: {e}")
        print("\n排查建议：")
        print("  1. 切换镜像：--endpoint https://huggingface.co（需代理）")
        print("  2. 检查代理：env | grep -i proxy")
        print("  3. 升级客户端：pip install --upgrade huggingface_hub")
        return 2

    print(f"\n[Download] 完成。snapshot 目录:\n  {path}")

    # 反向验证：能否被 src/reranker.py 的 _resolve_local_snapshot 找到
    if not args.no_verify:
        print("\n[Verify] 反向验证 _resolve_local_snapshot ...")
        try:
            # 把项目根加到 sys.path，复用 reranker 的判定逻辑
            project_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(project_root))
            from src.reranker import _resolve_local_snapshot

            verified = _resolve_local_snapshot(args.model)
            if verified:
                print(f"[Verify] OK，加载路径:\n  {verified}")
            else:
                # 下载成功但找不到 snapshot —— 通常是 HF_HOME 不一致
                hf_home = os.environ.get("HF_HOME") or os.path.expanduser(
                    "~/.cache/huggingface"
                )
                print(f"[Verify] FAILED：snapshot 已下到 {path}")
                print(f"         但 _resolve_local_snapshot 在 {hf_home}/hub 下找不到它")
                print("         解决方案：")
                print(f"           - 检查 HF_HOME 环境变量是否被改过")
                print(f"           - 或显式设置 cache_dir 后再跑一次：")
                print(f"             python scripts/download_models.py --cache-dir <与运行时一致>")
                return 3
        except Exception as e:
            print(f"[Verify] 跳过（{type(e).__name__}: {e}）")

    print("\n✅ 全部完成。现在可以运行 app.py / evals，reranker 会从本地秒级加载。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
