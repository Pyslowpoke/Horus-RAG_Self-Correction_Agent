# Horus — RAG Self-Correction Agent

> English | [中文](README.zh-CN.md)

*Insight, not imagination.*

A self-correcting deep research assistant powered by RAG (Retrieval-Augmented Generation), featuring fact-checking, hybrid retrieval with Reranker, and a Multi-Agent architecture.

## Key Features

| Feature | Description |
|---------|-------------|
| Hybrid Retrieval | Vector search + BM25 keyword search + RRF fusion |
| Reranker | bge-reranker-v2-m3 for precise re-ranking of candidate documents |
| HyDE (on-demand) | Generates hypothetical documents for short or low-relevance queries |
| Relevance Check | Automatically evaluates retrieval quality; triggers web fallback when results are irrelevant |
| Fact-Checking | Claim-level verification with self-correction loop (up to 3 retries) |
| Multi-Agent | 6 specialized agents orchestrated via LangGraph |
| Dual-Model Strategy | Lightweight model (Qwen2.5-7B) for simple tasks, heavy model (DeepSeek V3) for complex ones |
| Golden Memory Bank | Persistent correction memory with file-monitoring auto-ingestion |
| Web Search | Baidu Qianfan API for reliable Chinese web search |
| Query Rewriting | Baidu Query Rewrite API for coreference resolution + date normalization |
| Context Awareness | Auto-injects time, user profile, and conversation history into prompts |
| Self-Awareness | Answers questions about its own identity and capabilities |
| Knowledge Base Mgmt | One-click rebuild from the sidebar |

## Architecture

```
User Input
    ↓
Date Hard-Replace (Python code layer)
    ↓
Baidu Query Rewrite (API call, coreference resolution)
    ↓
Multi-Agent Pipeline (LangGraph)
    ↓
Router Agent (intent analysis + self-awareness detection)
    ↓
Memory Agent (historical correction memory retrieval)
    ↓
┌───────────────────────────────────────┐
│ Retrieval Agent                       │
│ ├── Short query → HyDE retrieval      │
│ └── Long query  → Hybrid retrieval    │
│     ├── Vector search (top_k × 4)     │
│     ├── BM25 search   (top_k × 4)     │
│     ├── RRF fusion    → top_k × 2     │
│     ├── Reranker re-rank → top_k      │
│     └── Relevance check → web fallback│
└───────────────────────────────────────┘
    ↓
Merge Context
    ↓
Generation Agent (answer generation)
    ↓
Fact-Check Agent (verification + self-correction)
    ↓
Output
```

## Module Overview

### Core

| File | Responsibility |
|------|---------------|
| `src/data_ingestion.py` | Document loading, chunking, vectorization, ingestion (txt + pdf) |
| `src/agents/interfaces.py` | Shared AgentState definition for all agents |
| `src/agents/router_agent.py` | Router: intent analysis + self-awareness detection |
| `src/agents/retrieval_agent.py` | Retrieval: HyDE on-demand + hybrid retrieval + relevance check |
| `src/agents/web_search_agent.py` | Web search agent: Baidu Qianfan API |
| `src/agents/generation_agent.py` | Generation: produces answers from retrieved context |
| `src/agents/fact_check_agent.py` | Fact-check: claim extraction + verification + rewrite loop |
| `src/agents/memory_agent.py` | Memory: searches Golden Memory Bank for past corrections |

### Retrieval

| File | Responsibility |
|------|---------------|
| `src/retrievers/hybrid_retriever.py` | Vector + BM25 + RRF + Reranker re-ranking |
| `src/retrievers/hyde_retriever.py` | HyDE: hypothetical document embedding |
| `src/retrievers/web_search_retriever.py` | Baidu Qianfan web search API |

### Generation & Verification

| File | Responsibility |
|------|---------------|
| `src/generators/llm_client.py` | Dual-path fault-tolerant LLM client (DeepSeek + SiliconFlow) |
| `src/generators/prompt_templates.py` | All prompt templates |
| `src/verifiers/fact_checker.py` | Fact-checker: verify → rewrite loop (no regeneration) |

### Orchestration & Memory

| File | Responsibility |
|------|---------------|
| `src/graph/multi_agent_graph.py` | Multi-Agent graph construction (LangGraph) |
| `src/memory/conversation_memory.py` | In-session conversation memory |
| `src/memory/gold_memory_bank.py` | Persistent golden memory with file watcher |
| `app.py` | Streamlit frontend |

## Quick Start

### 1. Clone

```bash
git clone https://github.com/your-username/rag-2.0-self-correction.git
cd rag-2.0-self-correction
```

### 2. Create virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
BAIDU_API_KEY=your_baidu_api_key  # web search + query rewrite
```

### 5. Build the knowledge base

Place documents (.txt / .pdf) in `knowledge_base/`, then run:

```bash
python src/data_ingestion.py
```

Or click the **"🔄 Rebuild Knowledge Base"** button in the Streamlit sidebar.

### 6. Launch

```bash
streamlit run app.py
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Database | ChromaDB |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Keyword Search | rank-bm25 + jieba |
| LLM | DeepSeek V3 (primary) + Qwen2.5-7B (fallback) |
| Agent Orchestration | LangGraph |
| Frontend | Streamlit |
| Web Search | Baidu Qianfan Search API |
| Query Rewriting | Baidu Qianfan Query Rewrite API |

## Evaluation Results

Tested on 10 cases:

| Method | Faithfulness | Hallucination Rate |
|--------|:------------:|:------------------:|
| Baseline (vector only) | 0.65 | 0.35 |
| **Hybrid Retrieval (vector + BM25 + RRF)** | **0.80** | **0.20** |
| Hybrid + Fact-Checking | 0.66 | 0.34 |

**Key takeaways:**
- Hybrid retrieval performs best (faithfulness 0.80) — RRF fusion provides significant complementarity
- Fact-checking slightly reduces faithfulness but effectively suppresses hallucination
- Pure vector retrieval is the weakest baseline (0.65)

## Project Structure

```
rag-2.0-self-correction/
├── app.py                              # Streamlit frontend
├── config.yaml                         # Configuration
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variable template
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
├── README.md                           # This file (English)
├── README.zh-CN.md                     # Chinese version
├── knowledge_base/                     # Local knowledge base (txt + pdf)
├── src/
│   ├── data_ingestion.py               # Document → chunks → vector DB
│   ├── agents/
│   │   ├── interfaces.py               # AgentState definition
│   │   ├── router_agent.py             # Router agent
│   │   ├── retrieval_agent.py          # Retrieval agent (HyDE + relevance)
│   │   ├── web_search_agent.py         # Web search agent
│   │   ├── generation_agent.py         # Generation agent
│   │   ├── fact_check_agent.py         # Fact-check agent
│   │   └── memory_agent.py             # Memory agent
│   ├── retrievers/
│   │   ├── hybrid_retriever.py         # Hybrid retrieval (vector+BM25+RRF+Reranker)
│   │   ├── hyde_retriever.py           # HyDE retrieval
│   │   └── web_search_retriever.py     # Web search (Baidu Qianfan)
│   ├── generators/
│   │   ├── llm_client.py               # Fault-tolerant LLM client
│   │   └── prompt_templates.py         # Prompt templates
│   ├── verifiers/
│   │   └── fact_checker.py             # Fact-checker with rewrite loop
│   ├── graph/
│   │   ├── multi_agent_graph.py        # Multi-Agent graph (LangGraph)
│   │   └── rag_graph.py                # Legacy linear graph
│   └── memory/
│       ├── conversation_memory.py      # Conversation memory
│       └── gold_memory_bank.py         # Golden memory bank
├── experiments/
│   ├── test_data.py                    # Test data
│   └── evaluate_accuracy.py            # Evaluation script
└── data/
    └── correction_inbox/               # Correction file inbox
```

## Deploy to Streamlit Community Cloud

1. Push code to GitHub
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Select the repository and `app.py`
5. Add environment variables in "Advanced settings" (DEEPSEEK_API_KEY, SILICONFLOW_API_KEY, BAIDU_API_KEY)
6. Click "Deploy"

## License

MIT License
