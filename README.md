# RAG System 2026

A production-oriented Retrieval-Augmented Generation (RAG) system over a corpus of machine-learning research papers. Ask a natural-language question and get a grounded, cited answer synthesized only from the retrieved source material.

Built as a portfolio project demonstrating a modern, two-stage retrieval pipeline with generation, hallucination guards, and (upcoming) automated evaluation.

## Architecture

```
                    ┌─────────────┐
   PDFs ──ingest──▶ │  Qdrant DB  │  (vector store, cosine, HNSW)
                    └─────────────┘
                          │
   Question ──────────────┤
                          ▼
        ┌───────────────────────────────────┐
        │ 1. Bi-encoder retrieval (top-20)   │  all-MiniLM-L6-v2  — fast, wide net
        ├───────────────────────────────────┤
        │ 2. Cross-encoder re-rank (top-5)   │  ms-marco-MiniLM   — precise
        ├───────────────────────────────────┤
        │ 3. Lost-in-the-Middle reorder      │  best chunks at start/end
        ├───────────────────────────────────┤
        │ 4. Grounded generation (LLM)       │  gpt-4o-mini, context-only prompt
        └───────────────────────────────────┘
                          ▼
              Cited answer + sources
```

### Why two-stage retrieval?

A **bi-encoder** embeds the query and documents separately, so document vectors are precomputed once and retrieval is a fast approximate-nearest-neighbour search — but it only measures surface semantic similarity. A **cross-encoder** scores each `(query, document)` pair *together* through the transformer, which is far more accurate but too slow to run over the whole corpus. The pipeline uses the bi-encoder to cheaply narrow the corpus to 20 candidates, then the cross-encoder to precisely re-rank those down to the best 5.

### Hallucination guard

The generation prompt instructs the model to answer **only** from the retrieved context and to say it doesn't have enough information when the answer isn't present. Ask something outside the corpus (e.g. "How does HNSW work?" when no ingested paper covers it) and the system declines rather than inventing an answer.

## Tech Stack

| Component | Choice |
|---|---|
| Vector DB | Qdrant (local persistent mode) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence-Transformers, 384-dim) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunking | `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) |
| LLM | OpenAI `gpt-4o-mini` |
| Language | Python 3.12, Poetry |

## Setup

```bash
# 1. Install dependencies
poetry install

# 2. Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Add source PDFs to data/papers/, then ingest
poetry run python src/ingest.py

# 4. Ask a question
poetry run python src/generate.py "What is retrieval augmented generation?"
```

## Usage

```bash
# Retrieval only (see which chunks match)
poetry run python src/retrieve.py "What are SHAP values?"

# Full pipeline: retrieve → re-rank → generate
poetry run python src/generate.py "What are SHAP values and how are they computed?"

# Smoke test over sample queries
poetry run python tests/ingest_test.py
```

## Project Structure

```
rag-system-2026/
├── src/
│   ├── ingest.py      # PDF → chunks → embeddings → Qdrant
│   ├── retrieve.py    # query → bi-encoder search → top-k chunks
│   ├── rerank.py      # cross-encoder re-ranking + Lost-in-the-Middle reorder
│   └── generate.py    # full pipeline: retrieve → rerank → LLM answer
├── tests/
│   └── ingest_test.py # retrieval smoke test
├── data/
│   ├── papers/        # source PDFs (git-ignored)
│   └── qdrant_db/     # persisted vector store (git-ignored)
└── pyproject.toml
```

## Roadmap

- [x] Ingestion pipeline (chunking, embeddings, Qdrant)
- [x] Two-stage retrieval (bi-encoder + cross-encoder re-ranking)
- [x] Grounded generation with hallucination guard
- [ ] Automated evaluation with RAGAS (faithfulness, answer relevancy, context precision/recall)
- [ ] Observability (tracing, cost tracking, Grafana dashboard)
- [ ] Semantic caching + structured outputs
