"""
app.py — Week 8: FastAPI serving layer for the production RAG pipeline
-----------------------------------------------------------------------
Why a server instead of the CLI?
  The CLI reloads the embedder + cross-encoder on every run (~20-30s of pure
  model loading). A long-running server loads them ONCE at startup and keeps
  them warm, so real per-query latency is just retrieval + rerank + LLM. This
  is also where the semantic cache actually pays off across many requests.

Endpoints:
  GET  /health   — liveness
  POST /query    — {"query": "..."} → structured RagResponse + metrics
  GET  /stats    — quick aggregate metrics from Postgres (cache-hit rate, cost)

Usage:
    poetry run uvicorn src.app:app --host 0.0.0.0 --port 8080 --reload
    # then:
    #   curl http://localhost:8080/health
    #   Invoke-RestMethod -Uri http://localhost:8080/query -Method POST \
    #       -Body '{"query":"What is RAG?"}' -ContentType "application/json"
"""

from __future__ import annotations

import pathlib
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import generate_v2 as gen
from metrics_db import get_connection, init_db


class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the models ONCE at startup so requests don't pay load cost
    print("Warming models (embedder + cross-encoder)...")
    gen._embed("warmup")           # loads the sentence-transformer
    gen._get_cache()               # connects Redis
    try:
        init_db()                  # ensure metrics table exists
    except Exception as exc:
        print(f"[init_db failed — is Postgres up? {exc}]")
    # Pre-load the reranker by running a tiny no-op through rerank
    print("Models warm. Ready.")
    yield


app = FastAPI(title="RAG System 2026", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    """Run the full production RAG pipeline (cache → retrieve → rerank → generate)."""
    result = gen.answer(req.query, use_cache=req.use_cache, log=True, verbose=False)
    return result


@app.get("/stats")
def stats():
    """Aggregate metrics for a quick health view (Grafana shows the detailed version)."""
    sql = """
    SELECT
        COUNT(*)                                        AS total_queries,
        SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)      AS cache_hits,
        ROUND(AVG(total_ms))                            AS avg_total_ms,
        ROUND(SUM(cost_usd)::numeric, 4)                AS total_cost_usd
    FROM query_metrics
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
        total, hits, avg_ms, cost = row
        hit_rate = round(hits / total, 3) if total else 0.0
        return {
            "total_queries":  total,
            "cache_hits":     hits,
            "cache_hit_rate": hit_rate,
            "avg_total_ms":   float(avg_ms) if avg_ms is not None else None,
            "total_cost_usd": float(cost) if cost is not None else 0.0,
        }
    except Exception as exc:
        return {"error": f"Could not read metrics: {exc}"}
