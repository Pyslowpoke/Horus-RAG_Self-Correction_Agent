# 中文 Embedding 切换与验证

## 已启用的配置

- 模型：`BAAI/bge-small-zh-v1.5`；本机缓存版本 `7999e1d3359715c523056ef9478215996d62a620`。
- CPU 推理，512 维向量，L2 归一化；实际向量范数约 1。
- 查询前缀：`为这个句子生成表示以用于检索相关文章：`。仅查询添加，文档正文不添加。
- 使用模型提供的 CLS pooling 配置，未自行替换成平均池化。
- 新索引：`chroma_db_zh_bge_small_v15`，从原始文档重新切分和编码，共 62 个独立分块。
- 旧 MiniLM 索引 `chroma_db_v2` 及更早的 `chroma_db` 保留。
- 43 条历史纠偏记录重新编码，存于 `gold_memory_db/3465bd147cb9`，旧库保留。
- 纠偏记忆距离门槛改为 0.35（归一化向量的平方 L2 距离），这是保守初值，不是概率，仍需业务纠偏样本校准。

模型的查询指令、归一化和池化方式参考 [官方模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)。模型文件已下载，当前 `local_files_only: true` 可离线加载。

## 小样本对照

使用 8 条有答案问题（包含 4 条改写问法）和 2 条无答案问题。它们围绕当前项目知识，规模很小，不能代表一般中文问答准确率。

纯向量比较固定使用完全相同的 62 个旧文档块，排除重新切分对结果的影响；每个模型重新编码这些正文及查询，以平方 L2 距离排名。此项没有 BM25、重排器或拒答阈值。

| 指标 | 原 MiniLM | 中文 BGE |
|---|---:|---:|
| 纯向量 Top-5 命中标注证据 | 0/8 | 8/8 |
| 纯向量 MRR | 0.000 | 0.771 |
| 纯向量查询 P50（含编码和距离排序） | 10.9 ms | 11.0 ms |
| 完整混合检索命中标注证据 | 5/8 | 5/8 |
| 混合检索无答案正确返回空结果 | 2/2 | 2/2 |
| 混合检索 P50 | 0.908 s | 1.298 s |
| 混合检索 P95 | 1.414 s | 1.717 s |

中文模型改善了这批问题的纯向量召回，但完整混合链路并未同时提升。融合候选截断和重排相关度门槛仍然影响最终输出，需要在更大的业务标注集上继续校准，不能用“8/8”宣称整体 RAG 准确率达到 100%。

本次替换未修改主答案模型、重排模型或重排门槛。初始化不计入上表；没有调用付费 LLM 或百度 API。

## 使用与复现

重启应用以清除旧模型的进程缓存：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

新机器需先把 `embedding.local_files_only` 设为 `false` 下载中文模型，然后运行 `python -m src.data_ingestion` 构建新索引。模型下载完成后可恢复离线模式。

本机对照数据和配置放在忽略版本控制的 `.evaluation` 目录，可复现：

```powershell
.\.venv\Scripts\python.exe scripts/compare_embeddings.py --baseline-config .evaluation/minilm_config.yaml --candidate-config .evaluation/bge_config.yaml --cases .evaluation/chinese_embedding_cases.json --output .evaluation/chinese_embedding_comparison.json
```

通用评测支持配置文件参数：

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result.json
```

## 回退

切换前完整配置保存于 `backups/config_before_chinese_20260915_223852.yaml`。需要回退时恢复该文件为 `config.yaml` 后重启应用；旧索引和旧纠偏记忆未删除。
