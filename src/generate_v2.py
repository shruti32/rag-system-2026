"""
generate_v2.py — Week 8: Production RAG pipeline
-------------------------------------------------
Upgrades generate.py with the production concerns from Week 8:

  • Structured output   — LLM returns a validated RagResponse (answer, sources,
                          confidence) via OpenAI's native structured outputs.
                          No fragile JSON parsing.
  • Semantic cache      — checks Redis before running the pipeline; returns a
                          cached answer for a semantically-similar recent query.
  • Cost tracking       — computes $ cost from token usage per call.
  • Retry + fallback    — exponential-backoff retry (tenacity), capped at 3.
  • Per-stage timing     — retrieve / rerank / llm latency logged to Postgres
                          for the Grafana dashboard.

Usage:
    poetry run python src/generate_v2.py "What is retrieval augmented generation?"
"""

from __future__ import annotations

import argparse
import time

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from cache import SemanticCache
from metrics_db import init_db, log_metrics
from rerank import rerank, reorder_lost_in_middle
from retrieve import EMBEDDING_MODEL, retrieve

load_dotenv()

LLM_MODEL      = "gpt-4o-mini"
RETRIEVE_TOP_N = 20
RERANK_TOP_K   = 5

# gpt-4o-mini pricing (USD per token) — update if prices change
PRICE_IN  = 0.15 / 1_000_000
PRICE_OUT = 0.60 / 1_000_000

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question \
using ONLY the provided context passages. If the context lacks the answer, set \
answer to "I don't have enough information in the retrieved context to answer that" \
and confidence to 0.0. Cite source filenames you used. Never use outside knowledge."""


# ── Structured output schema ──────────────────────────────────────────────────

class RagResponse(BaseModel):
    answer: str = Field(description="The answer, grounded only in the context")
    sources: list[str] = Field(description="Source filenames used, e.g. ['rag.pdf']")
    confidence: float = Field(description="0.0-1.0 confidence the answer is supported")


# ── Shared singletons ─────────────────────────────────────────────────────────

_embedder = None
_cache = None


def _embed(text: str):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder.encode(text)


def _get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(embed_fn=_embed)
    return _cache


# ── LLM call with retry + fallback ────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_llm(client: OpenAI, context: str, query: str) -> tuple[RagResponse, dict]:
    """Call the LLM with structured output. Retries with backoff on transient errors."""
    completion = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context passages:\n\n{context}\n\n---\n\nQuestion: {query}"},
        ],
        response_format=RagResponse,
        temperature=0.0,
    )
    parsed = completion.choices[0].message.parsed
    usage = {
        "input_tokens":  completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
    }
    return parsed, usage


def _build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Passage {i} — source: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def answer(query: str, use_cache: bool = True, log: bool = True, verbose: bool = True) -> dict:
    t_start = time.time()

    # 0. Cache check
    if use_cache:
        cached = _get_cache().get(query)
        if cached:
            total_ms = int((time.time() - t_start) * 1000)
            if log:
                _safe_log({"query": query, "cache_hit": True, "total_ms": total_ms})
            if verbose:
                print(f"\n[CACHE HIT sim={cached['cache_sim']} ~ {cached['cached_query'][:50]!r}]")
                print(cached["answer"])
            return {**cached, "latency_ms": total_ms}

    # 1. Retrieve
    t = time.time()
    candidates = retrieve(query, top_k=RETRIEVE_TOP_N)
    retrieve_ms = int((time.time() - t) * 1000)

    # 2. Re-rank + reorder
    t = time.time()
    ranked = rerank(query, candidates, top_k=RERANK_TOP_K)
    ordered = reorder_lost_in_middle(ranked)
    rerank_ms = int((time.time() - t) * 1000)

    # 3. Generate (structured, with retry/fallback)
    t = time.time()
    client = OpenAI()
    try:
        parsed, usage = _call_llm(client, _build_context(ordered), query)
    except Exception as exc:
        # Fallback path — in production, swap to a secondary model here.
        if verbose:
            print(f"[LLM failed after retries: {exc}] returning degraded response")
        parsed = RagResponse(answer="Service temporarily unavailable.", sources=[], confidence=0.0)
        usage = {"input_tokens": 0, "output_tokens": 0}
    llm_ms = int((time.time() - t) * 1000)

    cost = usage["input_tokens"] * PRICE_IN + usage["output_tokens"] * PRICE_OUT
    total_ms = int((time.time() - t_start) * 1000)

    # 4. Store in cache
    if use_cache and parsed.confidence > 0:
        _get_cache().store(query, parsed.answer, parsed.sources)

    # 5. Log metrics
    metrics = {
        "query": query, "cache_hit": False,
        "retrieve_ms": retrieve_ms, "rerank_ms": rerank_ms, "llm_ms": llm_ms,
        "total_ms": total_ms,
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "cost_usd": round(cost, 6), "model": LLM_MODEL,
    }
    if log:
        _safe_log(metrics)

    if verbose:
        print(f"\n{'=' * 70}\nQ: {query}\n{'=' * 70}")
        print(f"\n{parsed.answer}\n")
        print(f"Sources: {parsed.sources}  |  Confidence: {parsed.confidence}")
        print(f"Latency: retrieve={retrieve_ms}ms rerank={rerank_ms}ms llm={llm_ms}ms "
              f"total={total_ms}ms  |  cost=${cost:.6f} "
              f"({usage['input_tokens']}in/{usage['output_tokens']}out)")

    return {
        "answer": parsed.answer, "sources": parsed.sources,
        "confidence": parsed.confidence, "cache_hit": False,
        "metrics": metrics,
    }


def _safe_log(metrics: dict) -> None:
    """Log to Postgres but never let a DB failure break the response."""
    try:
        log_metrics(metrics)
    except Exception as exc:
        print(f"[metrics log failed: {exc}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production RAG pipeline")
    parser.add_argument("query", type=str)
    parser.add_argument("--no-cache", action="store_true", help="Bypass semantic cache")
    parser.add_argument("--no-log", action="store_true", help="Don't log metrics to Postgres")
    args = parser.parse_args()

    if not args.no_log:
        try:
            init_db()
        except Exception as exc:
            print(f"[init_db failed — is Postgres running? {exc}]")

    answer(args.query, use_cache=not args.no_cache, log=not args.no_log)
