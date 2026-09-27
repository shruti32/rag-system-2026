# RAGAS Evaluation Findings

Evaluation of the RAG system over 12 hand-written questions spanning the RAG (Lewis 2020), RAGAS (Es 2023), and SHAP (Lundberg 2017) papers.

## Mean Scores

| Metric | Score | What it measures | Stage |
|---|---|---|---|
| Faithfulness | **0.917** | Are the answer's claims grounded in retrieved context? | Generation |
| Answer Relevancy | **0.877** | Is the answer on-topic for the question? | Generation |
| Context Precision | **0.904** | Are the retrieved chunks relevant? | Retrieval |
| Context Recall | **0.847** | Did retrieval get *all* the needed info? | Retrieval |

## Headline Finding

**Generation is strong; the weakness is retrieval recall.** Every question that
retrieved the correct context answered it faithfully (faithfulness = 1.00 on 11 of
12 questions). The system's failures are not hallucinations — they are cases where
the chunk containing the answer exists in the corpus but did not make the top-5.

## Per-Question Outliers

### Q2 — "What are the two RAG formulations?" — Faithfulness 0.00, Relevancy 0.00
Not a hallucination — the opposite. The system correctly answered *"I don't have
enough information in the retrieved context to answer that"* because the retriever
returned RAGAS-metric chunks instead of the RAG-Sequence/RAG-Token passage (which
exists in `rag.pdf`). The hallucination guard worked as designed. RAGAS scores a
refusal as 0.00 faithfulness because there are no grounded claims to verify — a
**scoring artifact**. The true defect is **retrieval recall** for this question.

### Q3 — "What retriever does RAG use?" — Context Recall 0.00
The answer was faithful (1.00) and roughly correct (DPR), but the retrieved chunk
was the *Results* section, which mentions DPR only in passing. The ground-truth
details (bi-encoder architecture, BERT query/document encoders, maximum inner
product search) live in a different chunk that was not retrieved. A **retrieval
recall** miss, not a generation problem.

### Lower context recall on RAGAS-paper questions (Q6 0.67, Q7 0.50)
Definitions of RAGAS metrics are spread across several chunks; the top-5 captured
part but not all of the reference content. Same root cause: recall.

## Root Cause & Proposed Fixes

The dominant failure mode is **retrieval recall**: the correct chunk exists but
doesn't reach the top-5. Options, in order of expected impact:

1. **Hybrid search (BM25 + dense) with RRF** — keyword matching would catch exact
   terms ("RAG-Sequence", "RAG-Token", "DPR") that dense retrieval alone misses.
2. **Increase the bi-encoder candidate pool** from 20 → 40 before re-ranking, so
   the cross-encoder has a better chance of surfacing the right chunk.
3. **Revisit chunking** — some definitions are split across chunk boundaries;
   larger chunks or semantic chunking would keep them intact.
4. **Query expansion / multi-query** — paraphrase the question to catch chunks
   phrased differently from the query.

## Method Notes

- Answerer: `gpt-4o-mini`. Judge: `gpt-4o-mini`. In production the judge should be
  a different model family to reduce self-enhancement bias.
- 12 hand-written questions; a larger, harder set (multi-hop, adversarial) would
  give a more stringent picture.
- A refusal scoring 0.00 on faithfulness is expected RAGAS behavior — always read
  the response text before treating a low faithfulness score as a hallucination.
