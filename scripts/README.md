# scripts/ — 开发环境初始化脚本

## download_models.py

预下载 HuggingFace 模型到本地缓存，**国内网络友好**（默认走 hf-mirror 镜像）。

### 何时跑

- 首次部署本项目
- 切换 / 升级 reranker 或 embedding 模型时
- `app.py` 启动时报错 `OSError: ... not a valid model identifier`（说明本地无缓存）

### 怎么跑

```bash
# 进入项目根目录
cd proj-gamebot

# 1) 默认下载 reranker（BAAI/bge-reranker-base，约 280MB）
python scripts/download_models.py

# 2) 下载 embedding 模型（首次 ingest 时需要，约 400MB）
python scripts/download_models.py --model shibing624/text2vec-base-chinese

# 3) 镜像不通时切官方源（需自备代理）
python scripts/download_models.py --endpoint https://huggingface.co

# 4) 下载到项目目录而不是 HF 默认缓存
python scripts/download_models.py --cache-dir ./models
```

### 验证

下载完成后，脚本会自动调用 `src.reranker._resolve_local_snapshot()` 反向验证能否被项目代码找到。

也可以手动验证：

```python
python -c "from src.reranker import _resolve_local_snapshot; print(_resolve_local_snapshot('BAAI/bge-reranker-base'))"
```

应输出形如 `C:\Users\<you>\.cache\huggingface\hub\models--BAAI--bge-reranker-base\snapshots\<commit>` 的绝对路径。

### 故障排查

| 现象 | 解决 |
|---|---|
| `ConnectionError: https://hf-mirror.com` | 加 `--endpoint https://huggingface.co` 走官方源（需代理） |
| `HFValidationError` | 检查 `--model` 参数是否拼写正确 |
| 下载中断 / 速度慢 | 重跑即可，`resume_download=True` 支持断点续传 |
| 下载成功但 `_resolve_local_snapshot` 找不到 | 检查 `HF_HOME` 环境变量，或加 `--cache-dir <运行时一致>` |
