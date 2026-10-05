<div align="center">

![Horus — Insight, not imagination](docs/assets/banner.svg)

# Horus

**面向中文资料的证据问答助手。看见回答，也看见依据。**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square) ![LangGraph](https://img.shields.io/badge/workflow-LangGraph-d8b16c?style=flat-square) [![MIT](https://img.shields.io/badge/license-MIT-d8b16c?style=flat-square)](LICENSE)

[快速开始](#快速开始) · [评测结果](#验证与评测) · [真实验证](docs/LIVE-VALIDATION.md) · [English](README.md)

</div>

---

Horus 将检索、回答生成、证据核查与有限修正连接起来，支持本地知识库、互联网搜索及混合模式。适合需要追溯资料依据的文档问答；本地检索与远程模型生成分别承担自己的工作。

## 回答之外，提供判断依据

| 检索资料 | 追溯回答 | 按需核查 |
| :--- | :--- | :--- |
| 中文向量、BM25 与重排结合，兼顾语义和关键词。 | 引用关联证据；缺少依据时明确说明无法回答。 | 默认快速回答，用户可开启额外核查；失败时保留已有回答与状态。 |

**一条问题，两种处理深度**

```text
资料 → 混合检索 → 带引用的回答                  快速模式（默认）
                     └→ 核查 → 有限修正 → 再核查   按需开启
```

纠偏记忆、近期对话、流式草稿和阶段耗时记录，帮助减少重复输入并定位等待发生在哪一步。

> **透明的验证，明确的边界。** 历史 60 题评测中，混合检索方案为 53/60 完全正确；加入核查并未提高该次结果。核查依赖模型且仍可能误判，引用也不等于事实已被证明。[查看完整评测](evaluations/20260925_horus/REPORT.md) · [当前改动的真实调用记录](docs/LIVE-VALIDATION.md)

[流程](#检索与回答流程) · [安装](#快速开始) · [响应时间](#本地检索的用途与响应时间) · [评测](#验证与评测) · [已知边界](#已知边界)

---

## 适合问什么

- **制度与产品资料：**“这项政策适用于哪些条件？依据在哪份文件？”
- **跨资料整合：**“把相关文档的要求整理到一起，并保留引用。”
- **检查证据缺口：**“资料没有支持这个数字时，明确指出无法确定。”

首次运行需要下载本地检索模型、建立 TXT/PDF 索引并配置生成模型。默认快速模式保留引用；需要额外检查时再勾选证据核查。[按步骤安装 →](#快速开始)

## 当前版本

- **中文 Embedding**：`BAAI/bge-small-zh-v1.5`，512 维、向量归一化；仅查询添加中文检索指令。
- **混合检索**：向量与 BM25 各召回最多 20 条，按稳定 `chunk_id` 去重，RRF 融合后最多重排 10 条，返回最多 5 条有效证据。
- **相关性过滤**：默认使用 `BAAI/bge-reranker-base`；未启用或加载失败时使用词项覆盖过滤。允许返回空结果，无证据时直接说明无法回答。
- **快速回答与可选核查**：界面默认检索后生成带引用的回答，明确标记未核查。勾选证据核查后，图最多重写一次；核查或重写失败时保留已有回答，解析失败不算通过。
- **流式输出与观测**：先显示回答草稿，核查后显示最终结果；记录首 token、各阶段耗时及模型调用/token 数。
- **增量入库**：按 tokenizer 分块，默认 220 token、重叠 30 token；支持 UTF-8 TXT 和可提取文字的 PDF。重复入库不会重复追加文档。
- **纠偏记忆**：保存用户纠错，监听 `data/correction_inbox/`；更换 Embedding 时使用独立记忆目录并重新编码。
- **可配置开销**：HyDE 和外部查询改写默认关闭；请求预算默认 60 秒，结果缓存有有效期和容量限制。

[config.yaml](config.yaml) 定义图的默认配置；界面核查勾选框按请求覆盖 `generation.verification_enabled`，默认不勾选。配置修改后重启应用。

## 检索与回答流程

```text
问题 → 日期归一化、按需补充最近对话 → 路由、纠偏记忆
  ├─ 本地：向量 + BM25 → 去重/RRF → 重排/过滤
  ├─ 联网：百度千帆搜索 → 证据过滤
  └─ 混合：先本地；无有效证据且已配置搜索服务时联网
        ↓
统一证据编号与上下文 → 带引用的草稿 → 快速结果（界面默认）
        ↓ 用户勾选证据核查
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

## 本地检索的用途与响应时间

本地检索适合内部制度、产品资料、可复用写作规范等范围明确的文档集合：为回答提供可追溯片段，避免每次把整套资料发给模型。生成仍调用远程 API；索引质量、资料冲突和核查误判仍影响结果。本地检索本身不等于自动学习偏好，也不能保证文案风格始终一致。

界面默认快速生成带引用的回答；需要额外检查时再勾选证据核查。重写结果只有核查通过才能替换初稿，失败或无法判断时恢复初稿及对应核查记录；生成后的超时保留已保存的回答与证据，并显示状态。

在公开评测索引的两个问题上使用 DeepSeek：快速图请求约 **1.99–5.96 秒**、核查请求约 **5.29–7.58 秒**，不含首次组件加载约 13 秒。它们是小样本图执行耗时，不是完整 Streamlit 端到端耗时或 P95。核查仍出现对正确 TXT/PDF 回答的误判；保护机制保留初稿，没有将它标为核查通过。详见[真实调用验证](docs/LIVE-VALIDATION.md)。

## 验证与评测

### 自动化回归测试

2026-10-05 发布前检查：**39/39 项产品回归测试通过**，包括重写异常、拒绝未经核查通过的重写及回答检查点保护。详见[真实验证记录](docs/LIVE-VALIDATION.md)。下方历史效果评测未重新运行。

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s evaluations/20260925_horus -p test_evaluation.py -v
```

2026-09-25 本机运行：**36/36 项产品回归、6/6 项评估指标测试通过**。覆盖检索、核查循环、索引隔离、查询前缀、超时、流式输出、纠偏记忆门槛及 Streamlit 请求流程。回归测试使用替身网络/LLM，不消耗 API 额度；通过率不代表回答准确率。评估指标测试检查证据匹配、分母和失败样本处理。

### 固定知识库效果评估（2026-09-25）

使用项目公开文档/源码及明确标注的合成资料，构建 **60 道正式题**，覆盖事实查找、跨段落整合、条件/数字约束、无答案、资料冲突和错误前提，每类 10 题。另设 6 道开发题和 30 条独立核查挑战。正式生成只运行一轮；本地检索三轮共 540 次请求。

- **A：**纯向量 Top-5 检索＋生成。
- **B：**向量/BM25 混合召回＋RRF＋重排过滤＋相同生成流程。
- **C：**复用 B 的原始回答及证据，再执行核查和最多一次纠正，避免重新生成带来的随机差异。

| 指标 | A 纯向量 | B 混合检索 | C 混合＋核查纠正 |
|---|---:|---:|---:|
| 回答完全正确（AI 评分，60 题） | 48/60（80.0%） | **53/60（88.3%）** | 52/60（86.7%） |
| 关键信息完整性（50 道有答案题） | 80% | 86% | 84% |
| 无答案题正确拒答 | 10/10 | 10/10 | 10/10 |
| 有答案题误拒答 | 7/50 | 7/50 | 7/50 |
| 工作流失败（含核查错误） | 1/60 | 0/60 | 24/60 |
| 固定证据图耗时 P50 / P95（秒） | 1.72 / 2.28 | 1.13 / 1.69 | 9.79 / 16.13 |
| API 调用数（C 包含 B 的首次生成） | 60 | 50 | 133 |

**如何解读：**B 在本题集上表现更好，但完全正确率差的配对 bootstrap 95% 区间为 −3.33 至 +20.00 个百分点，不能宣称稳定或显著提升。A/B 的标注原文完整证据命中为 32/50 与 38/50；该对照同时改变融合、重排和过滤，不能全部归因于 BM25。控制相同重排/过滤的补充对照见报告。

C 没有错变对案例；17 题尝试重写，其中 1 题因超时丢失原本正确的回答。独立简单核查挑战正确 28/30，不代表真实回答链路具有同等可靠性。当前证据支持继续完善核查模块，不支持“核查提高准确率”或“消除幻觉”的结论。

**测量边界：**回答与引用语义由本次编程助手 AI 根据参考证据评分，评分者也参与数据准备，存在偏差，尚未人工复核。耗时是重放固定证据后的生成/核查图耗时，**不含检索，不是完整 UI 端到端耗时**。实际响应模型为 `deepseek-flash`（请求别名 `deepseek-chat`）及 `Qwen/Qwen2.5-7B-Instruct`；实验禁用备用切换和重试。未评估联网效果或生产流量。

开发与正式实验合计 243 次 API 调用，已知输入 100,131、输出 21,061 Token；10 次超时缺少 usage，不能视为零计费。未确认单价，不报告货币成本。原始失败样本全部保留。

[完整报告与案例](evaluations/20260925_horus/REPORT.md) · [评分汇总 CSV](evaluations/20260925_horus/runs/paid_test_01/answer_summary.csv) · [待人工复核表](evaluations/20260925_horus/human_review_sample.csv) · [方案、数据与复运行命令](evaluations/20260925_horus/README.md)

### 复现与自定义检索评测

本次评估目录包含固定语料、题目、证据、配置、依赖版本及调用日志；本地模型缓存和生成索引不随仓库发布。按上述复运行说明使用新的运行目录，避免覆盖旧记录。真实 API 重跑需要凭证和新的预算确认；本地检索无需 LLM API。

使用自己的知识库时，可复制 [评测模板](experiments/retrieval_cases.example.json)，用有效 `chunk_id` 或原文片段 `evidence_contains` 标注证据：

```bash
python -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result.json
# 关闭重排的对照；输出到另一文件
python -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result_no_reranker.json --no-reranker
```

输出 Precision@K、返回文档精度、Recall@K、MRR、无答案检索返回空结果的比例及检索 P50/P95。检索命中不等于答案正确、忠实或无幻觉。

历史 Embedding 切换的 8 题/2 题小样本记录仍保留在 [中文模型验证](docs/chinese_embedding.md)，不与本次数据合并；历史 `.evaluation/` 本机语料和备份未随仓库发布。

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
evaluations/                   固定数据集、真实 API 评估与审计记录
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


---

## 反馈与参与

使用中遇到问题？[提交 Issue](https://github.com/Pyslowpoke/Horus-RAG_Self-Correction_Agent/issues/new)，附上复现步骤、环境版本、预期与实际结果；请勿附带 API 密钥或私人数据。文档改进、合成示例和可复现的边界案例都很有帮助。

如果项目对你有用，欢迎 Star 收藏；具体的使用反馈同样重要。
