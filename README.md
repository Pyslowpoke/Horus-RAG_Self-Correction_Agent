<div align="center">

![Horus — Insight, not imagination](docs/assets/banner.svg)

# Horus

**Evidence-grounded answers for Chinese knowledge bases.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square) ![LangGraph](https://img.shields.io/badge/workflow-LangGraph-d8b16c?style=flat-square) [![MIT](https://img.shields.io/badge/license-MIT-d8b16c?style=flat-square)](LICENSE)

[Quick start](#quick-start) · [Evaluation](#tests-and-evaluation) · [Live validation](docs/LIVE-VALIDATION.md) · [简体中文](README.zh-CN.md)

</div>

---

Horus connects retrieval, answer generation, evidence verification and bounded correction. Use local knowledge, web search or hybrid retrieval for document questions where source traceability matters. Retrieval runs locally; answer generation uses configured remote models.

## More than an answer: evidence you can inspect

| Retrieve | Trace | Verify when needed |
| :--- | :--- | :--- |
| Chinese vectors, BM25 and reranking combine semantic and keyword matching. | Associate citations with passages; explicitly refuse when no evidence is found. | Quick answers by default, with optional checking and preserved answers on later failures. |

**One question, two depths**

```text
Documents → Hybrid retrieval → Cited answer                 Quick mode (default)
                                    └→ Check → Rewrite → Check   Opt-in verification
```

Correction memory, recent conversation, streamed drafts and stage timings help reduce repeated input and show where time is spent.

> **Transparent evidence, explicit limits.** In the historical 60-question evaluation, hybrid retrieval produced 53/60 fully correct answers; adding verification did not improve that run. Model-based checking can still misjudge claims, and citations alone do not prove correctness. [Full evaluation](evaluations/20260925_horus/REPORT.md) · [Current live-call observations](docs/LIVE-VALIDATION.md)

[Flow](#request-flow) · [Install](#quick-start) · [Latency](#local-retrieval-and-response-time) · [Evaluation](#tests-and-evaluation) · [Limitations](#known-limitations)

---

## Know the boundaries before you run

| Concern | Behavior |
| :--- | :--- |
| Local vs. remote | Embeddings and reranking run locally; generation and verification use remote APIs. |
| Unsupported claims | Retrieval may return no evidence; the answer should explicitly refuse or state the gap. |
| Verification failures | Preserve the available answer and evidence with an explicit failed/error status. |
| Reproducibility | Public evaluation corpus, protocols and results are checked into the repository. Historical scores are not claims about every future request. |

## Current version

- **Chinese embeddings:** `BAAI/bge-small-zh-v1.5`, 512 dimensions, normalized vectors, and a Chinese instruction applied only to queries.
- **Hybrid retrieval:** vector search and BM25 each retrieve up to 20 candidates. Stable chunk IDs prevent duplicate voting in RRF fusion; up to 10 candidates are reranked and up to 5 accepted passages are returned.
- **Relevance filtering:** `BAAI/bge-reranker-base` by default, with lexical coverage filtering when reranking is disabled or unavailable. Empty results are valid; requests without evidence receive an explicit refusal.
- **Quick answers and optional verification:** the UI defaults to retrieval and cited generation, explicitly marked unverified. Opt-in verification allows at most one rewrite. Verification or rewrite failures preserve an existing answer; malformed JSON never counts as a pass.
- **Streaming and metrics:** an initial draft is streamed before the final verification result. Metrics include first-token latency, stage timings, model calls, and token usage.
- **Incremental ingestion:** tokenizer-based chunks of 220 tokens with 30-token overlap. Supports UTF-8 TXT and text-based PDF files; repeated ingestion does not append duplicate passages.
- **Correction memory:** user corrections persist across sessions, with a watched `data/correction_inbox/` directory. Embedding changes use isolated memory directories and require re-encoding.
- **Controlled overhead:** HyDE and external query rewriting are disabled by default. Requests have a 60-second budget; result caching has a TTL and capacity limit.

[config.yaml](config.yaml) defines graph defaults. The UI verification checkbox overrides `generation.verification_enabled` for each request and defaults to off. Restart the application after changing configuration.

## Request flow

```text
Question → normalize dates / add relevant recent conversation → route / correction memory
  ├─ Local: vector + BM25 → deduplicate / RRF → rerank / filter
  ├─ Web: Baidu Qianfan search → evidence filtering
  └─ Hybrid: local first; search the web if no evidence survives and search is configured
        ↓
Shared evidence numbering and context → cited draft → quick result (UI default)
        ↓ optional evidence verification
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

## Local retrieval and response time

Local retrieval is useful for bounded document collections such as internal policies, product documentation and reusable writing guidance: it retrieves traceable passages and avoids sending the entire corpus to the model. Generation still uses a remote API; indexing quality, conflicting sources and verification errors remain relevant. It does not by itself provide automatic preference learning or guarantee consistent writing style.

The UI defaults to quick cited answers. Enable evidence verification when the extra checking cost is appropriate. A rewritten answer replaces the initial answer only after verification passes; failed or inconclusive rewrites restore the initial answer and its verification record. Timeouts after generation preserve the checkpointed answer and evidence with an explicit status.

Two questions from the public evaluation index were tested with DeepSeek: quick graph requests took approximately **1.99–5.96 s**, and verification requests **5.29–7.58 s**, excluding approximately 13 s of initial component loading. These are small-sample graph measurements, not Streamlit end-to-end latency or a P95 benchmark. The checker still falsely rejected a supported TXT/PDF answer; the guard preserved its initial answer without labeling it verified. See [live validation](docs/LIVE-VALIDATION.md).

## Tests and evaluation

### Automated regression tests

Pre-push check on 2026-10-05: **39/39 product regression tests passed**, including failed/rejected rewrite handling and answer checkpoints. See [live validation](docs/LIVE-VALIDATION.md). The historical evaluation below was not rerun.

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s evaluations/20260925_horus -p test_evaluation.py -v
```

Local run on 2026-09-25: **36/36 product regression tests and 6/6 evaluation-metric tests passed**. Coverage includes retrieval, correction loops, index isolation, query prefixes, timeouts, streaming, correction memory, and Streamlit request handling. Regression tests use fake network/LLM implementations and consume no API credits; passing tests do not establish answer accuracy. Metric tests check evidence matching, denominators, and failure handling.

### Fixed-corpus effectiveness evaluation (2026-09-25)

The corpus contains public repository documentation/code and explicitly labeled synthetic material. **60 formal questions** cover factual lookup, integration, conditions/numbers, unanswerable questions, conflicting sources, and false premises, with 10 questions per category. There are also 6 development questions and 30 independent checker challenges. Paid generation ran once; three local retrieval repetitions produced 540 requests.

- **A:** vector Top-5 retrieval plus generation.
- **B:** vector/BM25 retrieval, RRF, reranking/filtering, and the same generation procedure.
- **C:** reuse B's exact initial answer and evidence, then verify and allow at most one rewrite, avoiding regeneration noise.

| Metric | A Vector | B Hybrid | C Hybrid + verification/correction |
|---|---:|---:|---:|
| Fully correct answers (AI review, 60 questions) | 48/60 (80.0%) | **53/60 (88.3%)** | 52/60 (86.7%) |
| Key-fact completeness (50 answerable questions) | 80% | 86% | 84% |
| Correct refusals on unanswerable questions | 10/10 | 10/10 | 10/10 |
| False refusals on answerable questions | 7/50 | 7/50 | 7/50 |
| Workflow failures, including checker errors | 1/60 | 0/60 | 24/60 |
| Fixed-evidence graph P50 / P95 (seconds) | 1.72 / 2.28 | 1.13 / 1.69 | 9.79 / 16.13 |
| API calls (C includes B's initial generation) | 60 | 50 | 133 |

**Interpretation:** B performed better on this dataset, but the paired bootstrap 95% interval for its fully-correct-rate difference versus A is −3.33 to +20.00 percentage points; this does not establish a reliable or statistically significant improvement. Complete annotated-quote evidence hits were 32/50 for A and 38/50 for B. This comparison changes fusion, reranking, and filtering together, so gains cannot be attributed solely to BM25. See the report for a supplementary comparison with matched reranking/filtering.

C produced no wrong-to-correct transitions. It attempted 17 rewrites; one timeout lost an initially correct answer. The independent simple checker challenge scored 28/30, which does not establish equivalent reliability in the full answer workflow. These results motivate further checker development, not claims of improved accuracy or eliminated hallucinations.

**Measurement limits:** semantic answer/citation judgments were made by the coding assistant AI against reference evidence. The reviewer also helped prepare the dataset, introducing bias; human review is pending. Timings measure the generation/verification graph with replayed evidence, **excluding retrieval; they are not full UI end-to-end latency**. Actual response models were `deepseek-flash` (requested alias `deepseek-chat`) and `Qwen/Qwen2.5-7B-Instruct`. Failover and retries were disabled for this experiment. Live web performance and production traffic were not evaluated.

Development and formal runs used 243 API calls in total, with 100,131 known input and 21,061 known output tokens. Ten timed-out calls lack usage data and cannot be assumed free. No verified unit prices were available, so no currency cost is reported. Failed samples are retained.

[Full report and cases (Chinese)](evaluations/20260925_horus/REPORT.md) · [Score summary CSV](evaluations/20260925_horus/runs/paid_test_01/answer_summary.csv) · [Pending human review](evaluations/20260925_horus/human_review_sample.csv) · [Protocol, data, and reproduction commands](evaluations/20260925_horus/README.md)

### Reproduction and custom retrieval evaluation

The evaluation directory includes frozen corpus material, questions, evidence, configuration, dependency versions, and API logs. Model caches and generated indexes are excluded. Follow the reproduction guide with a new run directory to preserve prior records. New paid runs require credentials and a new budget authorization; local retrieval requires no LLM API.

For your own knowledge base, copy the [case template](experiments/retrieval_cases.example.json) and identify evidence using valid `chunk_id` values or source-text snippets in `evidence_contains`:

```bash
python -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result.json
# Compare without reranking; use a separate output file
python -m experiments.evaluate_accuracy --config config.yaml --cases your_cases.json --output result_no_reranker.json --no-reranker
```

Reports include Precision@K, returned-document precision, Recall@K, MRR, empty-retrieval rate for unanswerable queries, and retrieval P50/P95. Retrieval hits do not establish answer correctness, faithfulness, or absence of hallucinations.

The historical 8-question/2-question embedding comparison remains in [Chinese embedding validation](docs/chinese_embedding.md), separate from this evaluation. Its local `.evaluation/` corpus and backups are not distributed.

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
evaluations/                   Frozen datasets, real API evaluation, and audit records
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


---

## Feedback & contributions

[Report a reproducible issue](https://github.com/Pyslowpoke/Horus-RAG_Self-Correction_Agent/issues/new) or [propose a change](https://github.com/Pyslowpoke/Horus-RAG_Self-Correction_Agent/compare). Include environment versions, steps and expected/actual behavior. Remove credentials and private data from examples. Documentation fixes, synthetic fixtures and regression cases are welcome.

If this is useful, a Star helps others discover it. Reproducible feedback helps improve it.
