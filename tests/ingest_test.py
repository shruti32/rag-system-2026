"""
ingest_test.py — Week 5 Hands-On: Smoke test for ingest + retrieve pipeline
----------------------------------------------------------------------------
What this tests:
  1. Verifies the Qdrant collection exists and has vectors
  2. Runs 5 sample queries and prints top-3 results for each
  3. Manually evaluate: are the returned chunks relevant to the query?
     (No automated scoring yet — that's Week 7 with RAGAS)

Usage:
    cd rag-system-2026
    poetry run python tests/ingest_test.py

Prerequisites:
    Run src/ingest.py first to populate the collection.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from qdrant_client import QdrantClient
from retrieve import COLLECTION_NAME, QDRANT_PATH, retrieve

# ── Sample queries — adjust to match the papers you ingested ─────────────────

SAMPLE_QUERIES = [
    "What is retrieval augmented generation?",
    "How do you evaluate a RAG system?",
    "What are SHAP values and how are they computed?",
    "What is the difference between bi-encoder and cross-encoder?",
    "How does HNSW work for approximate nearest neighbour search?",
]


def test_collection_exists():
    client = QdrantClient(path=str(QDRANT_PATH))
    collections = [c.name for c in client.get_collections().collections]
    assert COLLECTION_NAME in collections, \
        f"Collection '{COLLECTION_NAME}' not found. Run ingest.py first."
    count = client.count(collection_name=COLLECTION_NAME).count
    assert count > 0, "Collection is empty. Run ingest.py first."
    print(f"✓ Collection '{COLLECTION_NAME}' exists with {count} vectors\n")


def test_retrieval():
    print("Running 5 sample queries...\n")
    for query in SAMPLE_QUERIES:
        print(f"Query: {query!r}")
        results = retrieve(query, top_k=3)
        assert len(results) > 0, f"No results returned for query: {query!r}"
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']}  source={r['source']}")
            print(f"       {r['text'][:200].strip()}...")
        print()


if __name__ == "__main__":
    test_collection_exists()
    test_retrieval()
    print("✓ All smoke tests passed. Manually review results above for relevance.")
