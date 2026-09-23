"""
generate.py — Week 6 Hands-On: Full RAG Pipeline (Retrieve → Re-rank → Generate)
---------------------------------------------------------------------------------
This is the "G" in RAG. Pipeline:
  1. Retrieve top-20 candidates with the bi-encoder (retrieve.py)
  2. Re-rank down to top-5 with the cross-encoder (rerank.py)
  3. Reorder to mitigate "Lost in the Middle" (best chunks at start/end)
  4. Build a grounded prompt and ask the LLM to answer USING ONLY the context
  5. Return (answer, source_chunks, latency)

The system prompt forces the model to answer only from retrieved context and
to say "I don't know" when the context doesn't contain the answer — this is how
you prevent hallucination and get honest "not in the corpus" responses (e.g. the
HNSW question, which isn't in these papers).

Usage:
    cd rag-system-2026
    poetry run python src/generate.py "What is retrieval augmented generation?"
    poetry run python src/generate.py "How does HNSW work?"   # should say "not in context"
"""

from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from rerank import rerank, reorder_lost_in_middle
from retrieve import retrieve

# Load OPENAI_API_KEY from .env
load_dotenv()

LLM_MODEL       = "gpt-4o-mini"
RETRIEVE_TOP_N  = 20    # bi-encoder casts a wide net
RERANK_TOP_K    = 5     # cross-encoder narrows to the best

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question \
using ONLY the provided context passages. Follow these rules strictly:

1. If the context does not contain enough information to answer, say exactly: \
"I don't have enough information in the retrieved context to answer that."
2. Do not use outside knowledge. Only use what is in the context.
3. Cite which source(s) you used by their filename in brackets, e.g. [rag.pdf].
4. Be concise and factual."""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[Passage {i} — source: {c['source']}]\n{c['text']}")
    return "\n\n".join(blocks)


def answer(query: str, verbose: bool = True) -> dict:
    """
    Run the full RAG pipeline for a single query.

    Returns dict: {answer, sources, latency_seconds, chunks}
    """
    t0 = time.time()

    # 1. Retrieve wide with the bi-encoder
    candidates = retrieve(query, top_k=RETRIEVE_TOP_N)

    # 2. Re-rank precisely with the cross-encoder
    ranked = rerank(query, candidates, top_k=RERANK_TOP_K)

    # 3. Reorder for "Lost in the Middle"
    ordered = reorder_lost_in_middle(ranked)

    # 4. Build grounded prompt and call the LLM
    context = build_context(ordered)
    user_prompt = f"Context passages:\n\n{context}\n\n---\n\nQuestion: {query}"

    client = OpenAI()  # reads OPENAI_API_KEY from environment
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,   # deterministic, factual
    )

    llm_answer = response.choices[0].message.content
    latency = round(time.time() - t0, 2)

    sources = sorted({c["source"] for c in ordered})

    result = {
        "answer":          llm_answer,
        "sources":         sources,
        "latency_seconds": latency,
        "chunks":          ordered,
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Q: {query}")
        print(f"{'=' * 70}")
        print(f"\n{llm_answer}\n")
        print(f"Sources: {', '.join(sources)}")
        print(f"Latency: {latency}s  |  Re-ranked {len(candidates)} → {len(ranked)} chunks")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG: retrieve → re-rank → generate")
    parser.add_argument("query", type=str, help="Question to answer")
    args = parser.parse_args()
    answer(args.query)
