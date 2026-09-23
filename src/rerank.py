"""
rerank.py — Week 6 Hands-On: Cross-Encoder Re-Ranking
------------------------------------------------------
Two-stage retrieval:
  Stage 1 (retrieve.py) — bi-encoder retrieves top-N candidates FAST
                          (query and docs embedded separately, ANN search)
  Stage 2 (this file)   — cross-encoder re-ranks those N candidates PRECISELY
                          (query + doc scored TOGETHER through the transformer)

Why two stages?
  A cross-encoder is far more accurate than cosine similarity because it lets
  the query and document attend to each other. But it needs one forward pass
  per (query, doc) pair — too slow to run over the whole corpus. So we only
  run it on the ~20 candidates the fast bi-encoder already narrowed down.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Trained on MS MARCO (real search queries + relevant passages).
  Small and fast — good default re-ranker.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(CROSS_ENCODER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-rank candidate chunks by cross-encoder relevance to the query.

    Parameters
    ----------
    query      : str          — the search query
    candidates : list[dict]    — chunks from retrieve.py (each has a "text" key)
    top_k      : int           — how many to keep after re-ranking

    Returns
    -------
    Top-k candidates sorted by cross-encoder score (descending),
    each with an added "rerank_score" key.
    """
    if not candidates:
        return []

    model = _get_reranker()

    # Build (query, doc) pairs — the cross-encoder scores each pair
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # Attach scores and sort
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def reorder_lost_in_middle(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to mitigate the "Lost in the Middle" problem.

    LLMs attend best to the START and END of context, so we place the
    highest-scoring chunks at the extremes and weaker ones in the middle.

    Input assumed sorted best→worst. Places best at index 0, 2nd-best at the
    end, 3rd-best at index 1, 4th-best second-from-end, and so on — so the
    two strongest chunks bracket the context and the weakest sit in the middle.
    """
    n = len(chunks)
    result: list[dict | None] = [None] * n
    left, right = 0, n - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk
            left += 1
        else:
            result[right] = chunk
            right -= 1
    return result  # type: ignore[return-value]
