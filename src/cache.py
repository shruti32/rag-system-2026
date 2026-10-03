"""
cache.py — Week 8: Redis Semantic Cache
----------------------------------------
Caches LLM answers keyed by query *meaning*, not exact string.

How it works:
  1. Embed the incoming query (same model as retrieval)
  2. Look up recent cached queries in Redis; compare embeddings by cosine sim
  3. If any cached query is similar above THRESHOLD and not expired → cache HIT,
     return its stored answer (no retrieval, no LLM call — big cost/latency win)
  4. Otherwise cache MISS → caller runs the full pipeline and calls store()

Design notes:
  - THRESHOLD is deliberately high (0.95). Too low causes FALSE HITS — different
    questions collapsed to the same cached answer (see Week 8 self-check Q3).
  - TTL expires entries so stale answers don't linger.
  - This is a simple linear-scan cache (fine for a demo / moderate traffic). At
    scale you'd store embeddings in a vector index (e.g. Redis Vector Search).
"""

from __future__ import annotations

import json
import time

import numpy as np
import redis

SIMILARITY_THRESHOLD = 0.90
TTL_SECONDS = 24 * 3600  # 24h freshness
CACHE_KEY_PREFIX = "ragcache:"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SemanticCache:
    def __init__(self, embed_fn, redis_url: str = "redis://localhost:6379/0"):
        """
        embed_fn: function(str) -> list[float] / np.ndarray — same embedder as retrieval
        """
        self.embed_fn = embed_fn
        self.r = redis.from_url(redis_url, decode_responses=True)

    def get(self, query: str) -> dict | None:
        """Return cached result dict if a semantically-similar query exists, else None."""
        q_vec = np.asarray(self.embed_fn(query), dtype=np.float32)

        best_sim, best_payload = 0.0, None
        for key in self.r.scan_iter(match=f"{CACHE_KEY_PREFIX}*"):
            raw = self.r.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            sim = _cosine(q_vec, np.asarray(entry["embedding"], dtype=np.float32))
            if sim > best_sim:
                best_sim, best_payload = sim, entry

        if best_payload and best_sim >= SIMILARITY_THRESHOLD:
            return {
                "answer": best_payload["answer"],
                "sources": best_payload["sources"],
                "cache_hit": True,
                "cache_sim": round(best_sim, 4),
                "cached_query": best_payload["query"],
            }
        return None

    def store(self, query: str, answer: str, sources: list[str]) -> None:
        """Store an answer with its query embedding and a TTL."""
        q_vec = np.asarray(self.embed_fn(query), dtype=np.float32)
        key = f"{CACHE_KEY_PREFIX}{int(time.time() * 1000)}"
        self.r.setex(
            key,
            TTL_SECONDS,
            json.dumps(
                {
                    "query": query,
                    "answer": answer,
                    "sources": sources,
                    "embedding": q_vec.tolist(),
                }
            ),
        )
