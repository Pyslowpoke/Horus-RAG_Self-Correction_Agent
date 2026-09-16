# Horus — RAG Self-Correction Agent

**English** | [中文](README.zh-CN.md)

*Insight, not imagination.*

A retrieval-augmented question-answering application for Chinese knowledge bases. Streamlit and LangGraph coordinate retrieval, answer generation, evidence verification, and bounded correction. Supports local knowledge, web search, and hybrid modes.

## Current version

- **Chinese embeddings:** `BAAI/bge-small-zh-v1.5`, 512 dimensions, normalized vectors, and a Chinese instruction applied only to queries.
- **Hybrid retrieval:** vector search and BM25 each retrieve up to 20 candidates. Stable chunk IDs prevent duplicate voting in RRF fusion; up to 10 candidates are reranked and up to 5 accepted passages are returned.
- **Relevance filtering:** `BAAI/bge-reranker-base` by default, with lexical coverage filtering when reranking is disabled or unavailable. Empty results are valid; requests without evidence receive an explicit refusal.
- **Bounded correction:** the graph owns the verification loop, allowing one rewrite by default. Invalid verification JSON does not count as a pass; current results and historical failures are tracked separately.
- **Streaming and metrics:** an initial draft is streamed before the final verification result. Metrics include first-token latency, stage timings, model calls, and token usage.
- **Incremental ingestion:** tokenizer-based chunks of 220 tokens with 30-token overlap. Supports UTF-8 TXT and text-based PDF files; repeated ingestion does not append duplicate passages.
- **Correction memory:** user corrections persist across sessions, with a watched `data/correction_inbox/` directory. Embedding changes use isolated memory directories and require re-encoding.
- **Controlled overhead:** HyDE and external query rewriting are disabled by default. Requests have a 60-second budget; result caching has a TTL and capacity limit.

[config.yaml](config.yaml) is the source of truth. Restart the application after changing configuration.

## Request flow

```text
Question → normalize dates / add relevant recent conversation → route / correction memory
  ├─ Local: vector + BM25 → deduplicate / RRF → rerank / filter
  ├─ Web: Baidu Qianfan search → evidence filtering
  └─ Hybrid: local first; search the web if no evidence survives and search is configured
        ↓
Shared evidence numbering and context → draft → verification
        ↓ if verification fails and budget permits
At most one rewrite → verify again → final answer and verification status
```

Identity questions skip retrieval and verification. Optional HyDE runs only when the original query has no accepted evidence, and its candidates are reranked against the original query. Web-only mode uses lexical filtering without loading local embedding or reranking models. Graph nodes run primarily in sequence.

## Quick start

The locally verified environment uses **Python 3.12**. Embedding and reranking run locally and require memory and model-cache storage. Generation and verification use remote LLM APIs.

### 1. Clone and install

```bash
git clone https://github.com/Pyslowpoke/Horus-RAG_Self-Correction_Agent.git
cd Horus-RAG_Self-Correction_Agent
python -m venv .venv
```

Activate the environment:

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

Current pins include `sentence-transformers==5.2.0` and `transformers==4.51.3`; Streamlit requires `>=1.57,<2`. See [requirements.txt](requirements.txt).

### 2. Configure API keys

Copy [.env.example](.env.example) to `.env` and fill in the services you use:

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
BAIDU_API_KEY=your_baidu_api_key
```

- Configure at least DeepSeek or SiliconFlow; configure both for the full failover setup.
- The primary generation model is `deepseek-chat`; the fallback and lightweight model is `Qwen/Qwen2.5-7B-Instruct`. The lightweight client uses DeepSeek as its fallback.
- Baidu is used for web search and optional external query rewriting. Local mode does not need it; hybrid mode stays local when search is not configured.
- Ingestion and offline retrieval evaluation do not call remote LLMs and do not require these keys.

### 3. Download models on first installation

The repository defaults to local model caches. **A new machine must download the models first.** Set these two existing fields in `config.yaml` to `false`:

```yaml
embedding:
  local_files_only: false
reranker:
  local_files_only: false
```

Change only the corresponding fields; do not replace the full configuration with this snippet. Then run:

```bash
python -c "from src.config import load_config; from src.components import load_embeddings, load_reranker; c=load_config(); load_embeddings(c); assert load_reranker(c) is not None"
```

After successful download, restore both values to `true` to use the local caches. Model files are not included in Git.

### 4. Build the knowledge base

Create `knowledge_base/`, add your own `.txt` or `.pdf` files, and run:

```bash
python -m src.data_ingestion
```

Ingestion recursively reads documents and creates `chroma_db_zh_bge_small_v15/` with an index manifest. Scanned PDFs need OCR beforehand; the ingestion pipeline does not provide OCR. Knowledge documents and generated indexes are not distributed with the repository.

Run the command again after document changes, or use the sidebar rebuild action. Changing the embedding model, normalization, or query prefix requires a new index directory and ingestion from source documents. Old vectors cannot be reused with a different model.

### 5. Start the application

```bash
python -m streamlit run app.py
```

On Windows, without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first request loads models and is usually slower. Deployment also requires model downloads, index creation, and process environment variables; the project does not currently provide an automatic cloud bootstrap script.

## Key settings

| Setting | Default | Meaning |
|---|---|---|
| `retrieval.chunk_size / chunk_overlap` | `220 / 30` | Tokenizer token counts |
| `retrieval.candidate_k / rerank_k / top_k` | `20 / 10 / 5` | Candidates per retriever, reranking limit, final evidence limit |
| `retrieval.rrf_k` | `60` | RRF fusion constant |
| `reranker.min_score` | `0.3` | Sigmoid score threshold, not a calibrated probability |
| `retrieval.min_lexical_score` | `0.3` | Lexical coverage threshold |
| `retrieval.hyde_enabled` | `false` | Try HyDE when no original evidence is accepted |
| `web.query_rewrite_enabled` | `false` | External query rewriting |
| `generation.max_retries` | `1` | Maximum rewrites across the graph |
| `generation.max_tokens` | `512` | Output limit per LLM call |
| `generation.verification_enabled / streaming` | `true / true` | Verification and streamed drafts |
| `llm.max_retries` | `0` | Additional application retries; SDK retries are disabled, failover remains available |
| `runtime.request_timeout` | `60` seconds | Request budget, including initialization |
| `runtime.query_cache_ttl / query_cache_max_entries` | `300` seconds / `64` | Per-session result cache |
| `memory.distance_threshold` | `0.35` | Squared L2 cutoff for correction memory; calibrate on your data |

Disabling reranking can reduce CPU cost but may reduce relevance. Calibrate thresholds using your own labeled questions.

## Tests and evaluation

```bash
python -m unittest discover -s tests -v
python -m compileall -q src app.py scripts experiments tests
python -m pip check
```

All 36 tests passed locally. Coverage includes retrieval, correction loops, index isolation, query prefixes, timeouts, streaming, correction-memory thresholds, and Streamlit request handling. Tests use fake network/LLM implementations and do not consume API credits.

For offline retrieval evaluation, copy and fill in the [case template](experiments/retrieval_cases.example.json). Identify evidence using valid `chunk_id` values or source-text snippets in `evidence_contains`:

```bash
python -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result.json
# Compare without reranking
python -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result_no_reranker.json --no-reranker
```

Reports include Precision@K, returned-document precision, Recall@K, MRR, unanswerable-query rejection, and retrieval P50/P95. Keyword matching is not treated as answer faithfulness or hallucination rate.

Local small-sample comparison when switching embeddings:

| Metric | Previous MiniLM | Chinese BGE |
|---|---:|---:|
| Raw vector Top-5 evidence hits on identical passages | 0/8 | 8/8 |
| Full hybrid pipeline evidence hits | 5/8 | 5/8 |
| Correct empty results for unanswerable hybrid queries | 2/2 | 2/2 |

**These are small-sample retrieval results, not general business accuracy or answer hallucination measurements.** Chinese embeddings improved raw vector recall; fusion and reranking still limit the full pipeline. End-to-end latency with real LLM APIs has not been measured. See the [Chinese embedding validation](docs/chinese_embedding.md). Its local datasets, backups, and `.evaluation/` artifacts are not distributed; reproduction requires your own corpus and labels.

## Correction memory and migration

Create a UTF-8 TXT file in `data/correction_inbox/`:

```text
Q: Original question
Wrong: Incorrect answer
Correct: Correct answer
```

The application imports existing files at startup and watches for new ones. To re-encode old correction memory with the current embedding configuration:

```bash
python scripts/reembed_memory.py --config config.yaml --source gold_memory_db
```

This preserves the old database and writes re-encoded records into a model-signature-specific directory. Rebuild document indexes from source files when changing embeddings. `scripts/migrate_index.py` only deduplicates old indexes with the **same embedding configuration**; it cannot switch embedding models.

## Project layout

```text
app.py                         Streamlit UI, cache, and background requests
config.yaml                    Models, retrieval, verification, and time budgets
src/config.py                  Configuration, model signatures, index validation
src/components.py              Model, database, BM25, and client construction
src/documents.py                Stable chunk IDs, deduplication, tokenization
src/data_ingestion.py           TXT/PDF loading and incremental token-based ingestion
src/runtime.py                 Request budgets, cancellation, events, and timings
src/query.py                    Date normalization, conversation context, cache keys
src/agents/                    Routing, retrieval, memory, generation, verification
src/graph/multi_agent_graph.py  Active LangGraph pipeline
src/graph/rag_graph.py          Compatibility constructors for older callers
src/retrievers/                 Hybrid retrieval, HyDE, and Baidu search
src/generators/                 Prompts, streaming LLM client, and failover
src/verifiers/                  Structured evidence verification
src/memory/                     Conversation and persistent correction memory
scripts/                       Index migration, memory re-encoding, embedding comparison
experiments/                   Retrieval evaluation and label template
tests/                         Offline regression tests
docs/                          Optimization notes and local validation records
```

## Known limitations

- Verification depends on an LLM and does not guarantee correctness. The UI distinguishes passed, skipped, failed, and errored verification.
- Web-only lexical filtering can miss paraphrases. Search uses result snippets rather than automatically fetching full pages.
- The request budget prevents subsequent calls but cannot forcibly interrupt running local model computations.
- Embedding changes affect distance distributions; re-encode both documents and correction memory and review thresholds.
- Large deployments with concurrent updates still need atomic index switching; current ingestion is not a fully transactional publication process.

More details: [optimization notes](docs/optimization.md) · [Chinese model validation](docs/chinese_embedding.md) · [historical MiniLM validation](docs/validation.md).

## License

[MIT](LICENSE)
