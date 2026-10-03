# RAG System 2026

A production-oriented Retrieval-Augmented Generation (RAG) system over a corpus of machine-learning research papers. Ask a natural-language question and get a grounded, cited answer synthesized only from the retrieved source material — served through a FastAPI layer with semantic caching, cost tracking, automated evaluation, and observability.

## Architecture

```
                    ┌─────────────┐
   PDFs ──ingest──▶ │  Qdrant DB  │  (vector store, cosine, HNSW)
                    └─────────────┘
                          │
   Question ──▶ Semantic cache (Redis)  ──hit──▶  cached answer (no LLM call)
                          │ miss
                          ▼
        ┌───────────────────────────────────┐
        │ 1. Bi-encoder retrieval (top-20)   │  all-MiniLM-L6-v2  — fast, wide net
        ├───────────────────────────────────┤
        │ 2. Cross-encoder re-rank (top-5)   │  ms-marco-MiniLM   — precise
        ├───────────────────────────────────┤
        │ 3. Lost-in-the-Middle reorder      │  best chunks at start/end
        ├───────────────────────────────────┤
        │ 4. Grounded generation (LLM)       │  gpt-4o-mini, structured output
        └───────────────────────────────────┘
                          ▼
         Validated RagResponse (answer, sources, confidence)
                          │
                          ├──▶ metrics → Postgres → Grafana dashboard
                          └──▶ answer cached in Redis
```

### Why two-stage retrieval?

A **bi-encoder** embeds the query and documents separately, so document vectors are precomputed once and retrieval is a fast approximate-nearest-neighbour search — but it only measures surface semantic similarity. A **cross-encoder** scores each `(query, document)` pair *together* through the transformer, which is far more accurate but too slow to run over the whole corpus. The pipeline uses the bi-encoder to cheaply narrow the corpus to 20 candidates, then the cross-encoder to precisely re-rank those down to the best 5.

### Hallucination guard

The generation prompt instructs the model to answer **only** from the retrieved context and to set confidence to 0 when the answer isn't present. Ask something outside the corpus and the system declines rather than inventing an answer.

### Structured output

The LLM returns a validated `RagResponse` (answer, sources, confidence) via OpenAI's native structured outputs — no fragile JSON parsing, and malformed responses are automatically retried.

### Semantic caching

Incoming queries are embedded and matched against recent queries in Redis by cosine similarity. A hit above the threshold returns the cached answer instantly with no retrieval or LLM call — large latency and cost savings on paraphrased repeat queries. Entries carry a TTL so answers don't go stale.

### Evaluation

A RAGAS pipeline scores the system on faithfulness, answer relevancy, context precision, and context recall against a hand-written question set, with results saved to CSV, plotted, and optionally pushed to a Braintrust dashboard for run-over-run comparison.

### Observability

Every query logs per-stage latency (retrieve / rerank / LLM), token counts, and cost to Postgres. A Grafana dashboard visualizes cache-hit rate, latency by stage, cumulative cost, and query volume. LLM calls use exponential-backoff retry with a fallback path.

## Tech Stack

| Component | Choice |
|---|---|
| Vector DB | Qdrant (local persistent mode) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence-Transformers, 384-dim) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunking | `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) |
| LLM | OpenAI `gpt-4o-mini` (native structured outputs) |
| Cache | Redis (semantic, cosine similarity + TTL) |
| Serving | FastAPI + Uvicorn |
| Evaluation | RAGAS (+ optional Braintrust dashboard) |
| Observability | Postgres + Grafana, tenacity retry/fallback |
| Language | Python 3.12, Poetry |

## Setup

```bash
# 1. Install dependencies
poetry install

# 2. Add your keys to .env
#    OPENAI_API_KEY=sk-...
#    BRAINTRUST_API_KEY=sk-...   (optional, for the eval dashboard)

# 3. Start supporting infra (Redis, Postgres, Grafana)
docker-compose up -d

# 4. Add source PDFs to data/papers/, then ingest
poetry run python src/ingest.py
```

## Usage

```bash
# Retrieval only (see which chunks match)
poetry run python src/retrieve.py "What are SHAP values?"

# Full pipeline (CLI): retrieve → re-rank → generate
poetry run python src/generate.py "What are SHAP values and how are they computed?"

# Production pipeline (cache + structured output + cost tracking + metrics)
poetry run python src/generate_v2.py "What is retrieval augmented generation?"

# Serve it (models load once and stay warm)
poetry run uvicorn src.app:app --host 0.0.0.0 --port 8080
#   POST /query   {"query": "..."}
#   GET  /stats   cache-hit rate, avg latency, total cost
#   GET  /health

# Evaluate with RAGAS
poetry run python evals/run_eval.py
```

## Project Structure

```
rag-system-2026/
├── src/
│   ├── ingest.py        # PDF → chunks → embeddings → Qdrant
│   ├── retrieve.py      # query → bi-encoder search → top-k chunks
│   ├── rerank.py        # cross-encoder re-ranking + Lost-in-the-Middle reorder
│   ├── generate.py      # core pipeline: retrieve → rerank → LLM answer
│   ├── generate_v2.py   # production pipeline: + cache, structured output,
│   │                    #   cost tracking, retry/fallback, metrics logging
│   ├── cache.py         # Redis semantic cache
│   ├── metrics_db.py    # per-query metrics → Postgres (for Grafana)
│   └── app.py           # FastAPI serving layer (/query, /stats, /health)
├── evals/
│   ├── eval_questions.json    # hand-written (question, ground_truth) set
│   ├── run_eval.py            # RAGAS evaluation pipeline
│   ├── push_to_braintrust.py  # log results to Braintrust dashboard
│   ├── ragas_results.csv      # per-question scores
│   └── eval_findings.md       # analysis of results
├── tests/
│   └── ingest_test.py   # retrieval smoke test
├── data/
│   ├── papers/          # source PDFs (git-ignored)
│   └── qdrant_db/       # persisted vector store (git-ignored)
├── docker-compose.yml   # Redis + Postgres + Grafana
└── pyproject.toml
```
