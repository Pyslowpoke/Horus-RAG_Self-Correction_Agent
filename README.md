# Horus - RAG Self-Correction Agent

> [English](README.md) | **中文**

Insight, not imagination.

基于 RAG（检索增强生成）的自纠正深度研究助手，具有事实核查、混合检索和 Multi-Agent 架构。

## 核心特性

- **混合检索**：向量检索 + BM25 关键词检索 + RRF 融合 + Reranker 重排序
- **Reranker 重排序**：bge-reranker-v2-m3 对检索结果精确重排
- **HyDE 按需触发**：短查询或检索结果不足时自动生成假设性文档
- **相关度检查**：自动评估检索结果质量，不相关时触发联网补充
- **事实核查**：逐句溯源，自动发现并修正幻觉
- **自纠正循环**：核查不通过时自动重写，最多 3 次
- **Multi-Agent 架构**：6 个 Agent 协同工作
- **双模型策略**：简单任务用轻量模型，复杂任务用重量模型
- **黄金记忆库**：从用户纠错中持续学习，支持文件监控自动导入
- **互联网搜索**：支持本地知识库 / 互联网搜索 / 智能混合三种模式
- **百度千帆搜索**：国内稳定的搜索引擎
- **用户偏好**：支持昵称、年龄段、职业、学历、内容偏好、输出格式自定义
- **对话记忆**：支持多轮对话上下文
- **查询重写**：百度 API 指代消解 + 日期硬替换
- **上下文感知**：自动注入时间、用户身份、对话历史
- **自我认知**：支持"你是谁"、"你能做什么"等自我认知类问题
- **知识库管理**：侧边栏一键重建向量数据库

## 架构设计

```
用户输入
    ↓
日期硬替换（Python 代码层）
    ↓
百度 Query 改写（API 调用，指代消解）
    ↓
Multi-Agent 流水线
    ↓
Router Agent（意图分析 + 自我认知检测）
    ↓
Memory Agent（历史纠偏记忆检索）
    ↓
┌──────────────────────────────────────┐
│ Retrieval Agent                      │
│ ├── 短查询 → HyDE 检索               │
│ └── 长查询 → Hybrid 检索             │
│     ├── 向量检索 (top_k * 4)          │
│     ├── BM25 检索 (top_k * 4)         │
│     ├── RRF 融合 → top_k * 2 候选     │
│     ├── Reranker 重排序 → top_k       │
│     └── 相关度检查 → 低则联网补充      │
└──────────────────────────────────────┘
    ↓
Merge Context（上下文合并）
    ↓
Generation Agent（生成回答）
    ↓
Fact Check Agent（事实核查 + 自纠正循环）
    ↓
输出
```

## 代码职能

### 核心模块

| 文件 | 职责 |
|------|------|
| `src/data_ingestion.py` | 文档加载、切分、向量化、入库（支持 txt+pdf） |
| `src/agents/interfaces.py` | Multi-Agent 共享状态定义 |
| `src/agents/router_agent.py` | 路由代理：意图分析 + 自我认知检测 |
| `src/agents/retrieval_agent.py` | 检索代理：HyDE 按需触发 + 混合检索 + 相关度检查 |
| `src/agents/web_search_agent.py` | 互联网搜索代理：百度千帆 API |
| `src/agents/generation_agent.py` | 生成代理：根据检索结果生成回答 |
| `src/agents/fact_check_agent.py` | 事实核查代理：核查 + 自纠正循环 |
| `src/agents/memory_agent.py` | 记忆代理：检索黄金记忆库 |

### 检索模块

| 文件 | 职责 |
|------|------|
| `src/retrievers/hybrid_retriever.py` | 混合检索：向量 + BM25 + RRF + Reranker 重排序 |
| `src/retrievers/hyde_retriever.py` | HyDE 检索：假设性文档嵌入 |
| `src/retrievers/web_search_retriever.py` | 互联网搜索：百度千帆 API |

### 生成与核查

| 文件 | 职责 |
|------|------|
| `src/generators/llm_client.py` | 双路容灾 LLM 客户端（DeepSeek + SiliconFlow） |
| `src/generators/prompt_templates.py` | 所有 Prompt 模板 |
| `src/verifiers/fact_checker.py` | 事实核查器：核查→重写循环（不重新生成） |

### 编排与记忆

| 文件 | 职责 |
|------|------|
| `src/graph/multi_agent_graph.py` | Multi-Agent 图构建（LangGraph） |
| `src/memory/conversation_memory.py` | 对话记忆（会话内） |
| `src/memory/gold_memory_bank.py` | 黄金记忆库（持久化 + 文件监控） |
| `app.py` | Streamlit 前端 |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/rag-2.0-self-correction.git
cd rag-2.0-self-correction
```

### 2. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 API Key：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
BAIDU_API_KEY=your_baidu_api_key  # 用于互联网搜索 + 查询改写
```

### 5. 准备知识库

将文档（.txt / .pdf）放入 `knowledge_base/` 目录，然后运行入库脚本：

```bash
python src/data_ingestion.py
```

或者在 Streamlit 侧边栏点击"🔄 重建知识库"按钮。

### 6. 启动前端

```bash
streamlit run app.py
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 向量数据库 | ChromaDB |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | bge-reranker-v2-m3 |
| 关键词检索 | rank-bm25 + jieba |
| LLM | DeepSeek V3 (主) + Qwen2.5-7B (备) |
| Agent 编排 | LangGraph |
| 前端 | Streamlit |
| 互联网搜索 | 百度千帆搜索 API |
| 查询改写 | 百度千帆 Query Rewrite API |

## 评估实验结果

在 10 条测试用例上对比三种方案：

| 方案 | 忠实度 | 幻觉率 |
|------|--------|--------|
| 基线（纯向量检索） | 0.65 | 0.35 |
| **混合检索（向量+BM25+RRF）** | **0.80** | **0.20** |
| 混合检索 + 事实核查 | 0.66 | 0.34 |

**关键发现：**
- 混合检索效果最好（忠实度 0.80），RRF 融合互补效果显著
- 事实核查在当前测试集上略降忠实度，但能有效抑制幻觉
- 基线表现最差（0.65），纯向量检索无法捕捉关键词精确匹配

## 项目结构

```
rag-2.0-self-correction/
├── app.py                              # Streamlit 前端
├── config.yaml                         # 配置文件
├── requirements.txt                    # 依赖列表
├── .env.example                        # 环境变量模板
├── LICENSE                             # MIT 许可证
├── knowledge_base/                     # 本地知识库（txt + pdf）
├── data/correction_inbox/              # 纠偏文件目录
├── src/
│   ├── data_ingestion.py               # 文档加载→切分→入库
│   ├── agents/
│   │   ├── interfaces.py               # Agent 状态定义
│   │   ├── router_agent.py             # 路由代理
│   │   ├── retrieval_agent.py          # 检索代理（HyDE + 相关度检查）
│   │   ├── web_search_agent.py         # 互联网搜索代理
│   │   ├── generation_agent.py         # 生成代理
│   │   ├── fact_check_agent.py         # 事实核查代理
│   │   └── memory_agent.py             # 记忆代理
│   ├── retrievers/
│   │   ├── hybrid_retriever.py         # 混合检索（向量+BM25+RRF+Reranker）
│   │   ├── hyde_retriever.py           # HyDE 检索
│   │   └── web_search_retriever.py     # 互联网搜索（百度千帆）
│   ├── generators/
│   │   ├── llm_client.py               # 双路容灾 LLM 客户端
│   │   └── prompt_templates.py         # Prompt 模板
│   ├── verifiers/
│   │   └── fact_checker.py             # 事实核查器
│   ├── graph/
│   │   ├── state.py                    # LangGraph 状态定义
│   │   ├── nodes.py                    # LangGraph 节点
│   │   ├── rag_graph.py               # LangGraph 图构建
│   │   └── multi_agent_graph.py        # Multi-Agent 图构建
│   └── memory/
│       ├── conversation_memory.py      # 对话记忆
│       └── gold_memory_bank.py         # 黄金记忆库
├── experiments/
│   ├── test_data.py                    # 测试数据
│   └── evaluate_accuracy.py            # 评估脚本
└── data/
    └── correction_inbox/               # 纠偏文件目录
```

## 部署到 Streamlit Community Cloud

1. 将代码推送到 GitHub
2. 访问 [Streamlit Community Cloud](https://share.streamlit.io/)
3. 登录 GitHub 账号
4. 选择仓库和 `app.py` 文件
5. 在 "Advanced settings" 中添加环境变量（DEEPSEEK_API_KEY、SILICONFLOW_API_KEY、BAIDU_API_KEY）
6. 点击 "Deploy"

## 许可证

MIT License
