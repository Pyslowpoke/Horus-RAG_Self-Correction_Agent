# Horus — RAG Self-Correction Agent

[English](README.md) | **中文**

*Insight, not imagination.*

面向中文知识库的检索增强问答应用，使用 Streamlit 与 LangGraph 编排检索、回答生成、证据核查和有限次数的修正。支持本地知识库、互联网搜索和智能混合三种模式。

## 当前版本

- **中文 Embedding**：`BAAI/bge-small-zh-v1.5`，512 维、向量归一化；仅查询添加中文检索指令。
- **混合检索**：向量与 BM25 各召回最多 20 条，按稳定 `chunk_id` 去重，RRF 融合后最多重排 10 条，返回最多 5 条有效证据。
- **相关性过滤**：默认使用 `BAAI/bge-reranker-base`；未启用或加载失败时使用词项覆盖过滤。允许返回空结果，无证据时直接说明无法回答。
- **有限自纠正**：图统一控制核查与重写，默认最多重写一次；保留当前核查结果及历史记录，解析失败不算通过。
- **流式输出与观测**：先显示回答草稿，核查后显示最终结果；记录首 token、各阶段耗时及模型调用/token 数。
- **增量入库**：按 tokenizer 分块，默认 220 token、重叠 30 token；支持 UTF-8 TXT 和可提取文字的 PDF。重复入库不会重复追加文档。
- **纠偏记忆**：保存用户纠错，监听 `data/correction_inbox/`；更换 Embedding 时使用独立记忆目录并重新编码。
- **可配置开销**：HyDE 和外部查询改写默认关闭；请求预算默认 60 秒，结果缓存有有效期和容量限制。

配置以 [config.yaml](config.yaml) 为准，修改后重启应用。

## 检索与回答流程

```text
问题 → 日期归一化、按需补充最近对话 → 路由、纠偏记忆
  ├─ 本地：向量 + BM25 → 去重/RRF → 重排/过滤
  ├─ 联网：百度千帆搜索 → 证据过滤
  └─ 混合：先本地；无有效证据且已配置搜索服务时联网
        ↓
统一证据编号与上下文 → 生成草稿 → 单次核查
        ↓ 若失败且预算允许
最多一次重写 → 再核查 → 最终结果及核查状态
```

身份类问题直接生成并跳过检索和核查。启用 HyDE 后，仅在原问题没有有效证据时补充候选，仍按原问题重排。纯联网模式不加载本地 Embedding/重排模型，使用词项覆盖过滤。各节点主要串行执行。

## 快速开始

本机验证环境为 **Python 3.12**。模型在本地执行，需要足够内存和模型缓存空间；回答生成与事实核查调用远程 LLM。

### 1. 克隆并安装

```bash
git clone https://github.com/Pyslowpoke/Horus-RAG_Self-Correction_Agent.git
cd Horus-RAG_Self-Correction_Agent
python -m venv .venv
```

激活环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

当前固定 `sentence-transformers==5.2.0`、`transformers==4.51.3`；Streamlit 要求 `>=1.57,<2`。详细依赖见 [requirements.txt](requirements.txt)。

### 2. 配置 API Key

复制 [.env.example](.env.example) 为 `.env`，填入需要使用的 Key：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
BAIDU_API_KEY=your_baidu_api_key
```

- 至少配置 DeepSeek 或 SiliconFlow 中的一路；要使用完整主备策略，应配置两路。
- 默认主生成模型是 `deepseek-chat`，备用及轻量模型是 `Qwen/Qwen2.5-7B-Instruct`；轻量客户端以 DeepSeek 为备用。
- 百度 Key 用于联网搜索和可选查询改写。纯本地模式不需要它；混合模式未配置时仅使用本地检索。
- TXT/PDF 入库和离线检索评测不调用远程 LLM，不需要这些 API Key。

### 3. 首次下载模型

仓库默认使用本地缓存。**新机器必须先下载模型**：将 `config.yaml` 中下面两个选项改为 `false`：

```yaml
embedding:
  local_files_only: false
reranker:
  local_files_only: false
```

只修改现有配置中的对应字段，不要用上面的片段覆盖整个文件。然后运行：

```bash
python -c "from src.config import load_config; from src.components import load_embeddings, load_reranker; c=load_config(); load_embeddings(c); assert load_reranker(c) is not None"
```

下载成功后可把两处恢复为 `true`，避免运行时联网下载模型。模型文件不包含在 Git 仓库中。

### 4. 构建知识库

创建 `knowledge_base/`，放入自己的 `.txt` 或 `.pdf` 文件，再执行：

```bash
python -m src.data_ingestion
```

程序递归读取文档，生成 `chroma_db_zh_bge_small_v15/` 和索引 manifest。扫描版 PDF 需先进行 OCR；当前入库程序不包含 OCR。知识库文件和生成的索引不随仓库发布。

后续文档更新可重复执行该命令，或使用页面侧边栏的“重建知识库”。更换模型、归一化方式或查询前缀时，应选择新索引目录并重新入库；不能把旧向量直接用于新模型。

### 5. 启动

```bash
python -m streamlit run app.py
```

Windows 未激活环境时也可使用：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

首次请求需要加载模型，通常比后续请求慢。部署到新环境时同样需要下载模型、构建索引并配置进程环境变量；当前项目未提供自动云端初始化脚本。

## 关键配置

| 字段 | 默认值 | 含义 |
|---|---|---|
| `retrieval.chunk_size / chunk_overlap` | `220 / 30` | tokenizer token 数 |
| `retrieval.candidate_k / rerank_k / top_k` | `20 / 10 / 5` | 每路召回、融合后重排、最终证据上限 |
| `retrieval.rrf_k` | `60` | RRF 融合常数 |
| `reranker.min_score` | `0.3` | sigmoid 重排分数门槛，不是校准概率 |
| `retrieval.min_lexical_score` | `0.3` | 词项覆盖过滤门槛 |
| `retrieval.hyde_enabled` | `false` | 无有效证据时是否尝试 HyDE |
| `web.query_rewrite_enabled` | `false` | 是否调用外部查询改写 |
| `generation.max_retries` | `1` | 整条图最多重写次数 |
| `generation.max_tokens` | `512` | 单次 LLM 输出上限 |
| `generation.verification_enabled / streaming` | `true / true` | 核查与流式草稿 |
| `llm.max_retries` | `0` | 单链路额外重试；SDK 重试关闭，仍可切换备用链路 |
| `runtime.request_timeout` | `60` 秒 | 包括初始化的请求时间预算 |
| `runtime.query_cache_ttl / query_cache_max_entries` | `300` 秒 / `64` | 会话内结果缓存 |
| `memory.distance_threshold` | `0.35` | 纠偏记忆平方 L2 距离上限，需按样本校准 |

关闭重排可以降低 CPU 开销，但可能降低返回证据的相关性。相关度门槛应通过自己的问题集校准。

## 验证与评测

```bash
python -m unittest discover -s tests -v
python -m compileall -q src app.py scripts experiments tests
python -m pip check
```

当前 36 项测试已在本机通过，覆盖检索、核查循环、索引隔离、查询前缀、超时、流式输出、纠偏记忆门槛和 Streamlit 请求流程。测试使用假网络/LLM，不消耗 API 额度。

离线检索评测需复制并填写 [评测模板](experiments/retrieval_cases.example.json)，用正确的 `chunk_id` 或原文片段 `evidence_contains` 标注证据：

```bash
python -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result.json
# 对照关闭重排
python -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result_no_reranker.json --no-reranker
```

输出 Precision@K、返回文档精度、Recall@K、MRR、无答案拒答率及检索 P50/P95。关键词命中率不再被当作“忠实度”或“幻觉率”。

中文模型切换时的本机小样本结果：

| 指标 | 原 MiniLM | 中文 BGE |
|---|---:|---:|
| 相同文档上的纯向量 Top-5 证据命中 | 0/8 | 8/8 |
| 完整混合链路证据命中 | 5/8 | 5/8 |
| 混合链路无答案正确返回空结果 | 2/2 | 2/2 |

**这些是小样本检索结果，不代表一般业务准确率或答案幻觉率。** 中文模型改善了纯向量召回，完整链路仍受融合及重排过滤影响。真实 LLM/API 的端到端耗时未实测。详见 [中文模型验证](docs/chinese_embedding.md)；其中本机数据、备份和 `.evaluation/` 文件不随仓库发布，复现需自行准备语料及标注。

## 纠偏记忆与迁移

在 `data/correction_inbox/` 新建 UTF-8 TXT，例如：

```text
Q: 原始问题
Wrong: 错误回答
Correct: 正确回答
```

应用启动时导入已有文件，并监听新文件。旧记忆切换到当前 Embedding：

```bash
python scripts/reembed_memory.py --config config.yaml --source gold_memory_db
```

它读取旧记录并重新编码，保留旧库，目标按模型签名隔离。已有知识库更换为中文模型时直接从原文重新入库。`scripts/migrate_index.py` 只用于**模型配置保持一致**的旧索引去重迁移，不可用它更换 Embedding。

## 项目结构

```text
app.py                         Streamlit 页面、缓存和后台任务
config.yaml                    模型、检索、核查及时间预算
src/config.py                  配置、模型签名及索引校验
src/components.py              模型、数据库、BM25 和客户端构建
src/documents.py                稳定分块 ID、去重和分词
src/data_ingestion.py           TXT/PDF 加载、token 分块和增量入库
src/runtime.py                  请求预算、取消、事件及耗时统计
src/query.py                    日期归一化、指代上下文及缓存键
src/agents/                    路由、检索、记忆、生成及核查节点
src/graph/multi_agent_graph.py  当前实际使用的 LangGraph 流程
src/graph/rag_graph.py          兼容旧调用的构造器
src/retrievers/                混合检索、HyDE 和百度搜索
src/generators/                提示词、流式 LLM 客户端与备用链路
src/verifiers/                 结构化证据核查
src/memory/                    对话与持久化纠偏记忆
scripts/                       索引迁移、记忆重编码、Embedding 对照
experiments/                   检索评测脚本和标注模板
tests/                         离线回归测试
docs/                          优化说明与本机验证记录
```

## 已知边界

- 证据核查依赖 LLM，不保证回答绝对正确；界面区分核查通过、未执行、失败及错误。
- 纯联网模式的词项过滤可能漏掉同义表述；搜索结果使用摘要，不包含自动抓取全文。
- 时间预算可阻止后续调用，但无法强制中断已在运行的本地模型计算。
- 更换 Embedding 会改变距离分布；文档及纠偏记忆都需要重新编码和检查门槛。
- 大规模、多用户持续更新场景还需进一步实现原子索引切换；当前入库不是完整的事务发布流程。

更多说明：[优化说明](docs/optimization.md) · [中文模型验证](docs/chinese_embedding.md) · [历史 MiniLM 验证](docs/validation.md)。

## 许可证

[MIT](LICENSE)
