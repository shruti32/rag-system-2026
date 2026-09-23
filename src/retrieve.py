"""
retrieve.py — Week 5 Hands-On: Query → Embed → Qdrant Search → Chunks
----------------------------------------------------------------------
Given a natural language query:
  1. Embed the query with the same model used at ingestion
  2. Search Qdrant for top-k most similar chunks (cosine similarity)
  3. Return chunks with metadata and similarity scores

Usage:
    cd rag-system-2026
    poetry run python src/retrieve.py "What is SHAP and how does it work?"
    poetry run python src/retrieve.py "explain RAG evaluation metrics" --top-k 5
"""

from __future__ import annotations

import argparse
import pathlib

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── Config (must match ingest.py) ────────────────────────────────────────────

COLLECTION_NAME = "ml-papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K   = 5

BASE_DIR    = pathlib.Path(__file__).parents[1]
QDRANT_PATH = BASE_DIR / "data" / "qdrant_db"

# Module-level cache so the model isn't reloaded on every call
_model  = None
_client = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Retrieve top-k chunks most relevant to the query.

    Parameters
    ----------
    query  : str   — natural language question
    top_k  : int   — number of chunks to return

    Returns
    -------
    List of dicts with keys: text, source, chunk_index, score
    """
    model  = _get_model()
    client = _get_client()

    # Embed the query (same model = same vector space as ingestion)
    query_vector = model.encode(query).tolist()

    # Search Qdrant (query_points is the current API; .search() was removed)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "text":        r.payload["text"],
            "source":      r.payload["source"],
            "chunk_index": r.payload["chunk_index"],
            "score":       round(r.score, 4),
        }
        for r in response.points
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve chunks from Qdrant")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Number of results to return")
    args = parser.parse_args()

    print(f"Query: {args.query!r}\n")
    chunks = retrieve(args.query, top_k=args.top_k)

    for i, chunk in enumerate(chunks, 1):
        print(f"── Result {i} (score={chunk['score']}) ─────────────────────")
        print(f"Source: {chunk['source']}  |  Chunk #{chunk['chunk_index']}")
        print(chunk["text"][:400])
        print()
