"""
run_eval.py — Week 7 Hands-On: RAGAS Evaluation Pipeline
---------------------------------------------------------
Steps:
  1. Load hand-written (question, ground_truth) pairs from eval_questions.json
  2. Run the full RAG pipeline on each question to collect (answer, contexts)
  3. Score every sample with RAGAS:
       - faithfulness          (generation: are answer claims grounded in context?)
       - answer_relevancy      (generation: is the answer on-topic?)
       - context_precision     (retrieval: are retrieved chunks relevant?)
       - context_recall        (retrieval: did we retrieve all needed info?  — uses ground_truth)
  4. Save per-question scores to evals/ragas_results.csv
  5. Plot score distributions to evals/ragas_scores.png
  6. Print the 5 worst-performing questions (lowest faithfulness) for eval_findings.md

Note on LLM roles (see Week 7 self-check):
  - Answerer:  gpt-4o-mini  (your RAG system, in generate.py)
  - Judge:     gpt-4o-mini  (RAGAS) — for a real project use a DIFFERENT model
               family as judge to reduce self-enhancement bias. Kept same here
               for cost; swap RAGAS_JUDGE_MODEL to e.g. a Claude/Gemini model in prod.

Usage:
    cd rag-system-2026
    poetry run python evals/run_eval.py

Cost: ~$0.10-0.30 for 12 questions (several LLM calls per question).
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from generate import answer  # our RAG pipeline

load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

EVAL_DIR      = pathlib.Path(__file__).parent
QUESTIONS     = EVAL_DIR / "eval_questions.json"
RESULTS_CSV   = EVAL_DIR / "ragas_results.csv"
SCORES_PLOT   = EVAL_DIR / "ragas_scores.png"

RAGAS_JUDGE_MODEL = "gpt-4o-mini"   # swap to a different family in production


# ── 1 & 2. Collect RAG outputs for every question ────────────────────────────

def collect_samples() -> list[dict]:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = data["questions"]

    samples = []
    for i, item in enumerate(questions, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"[{i}/{len(questions)}] Running RAG for: {q[:60]}...")

        result = answer(q, verbose=False)          # runs retrieve → rerank → generate
        contexts = [c["text"] for c in result["chunks"]]

        samples.append({
            "question":     q,
            "answer":       result["answer"],
            "contexts":     contexts,
            "ground_truth": gt,
        })
    return samples


# ── 3. Run RAGAS ──────────────────────────────────────────────────────────────

def run_ragas(samples: list[dict]) -> pd.DataFrame:
    # RAGAS expects a HF Dataset with these column names
    ds = Dataset.from_list([
        {
            "question":     s["question"],
            "answer":       s["answer"],
            "contexts":     s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in samples
    ])

    print("\nRunning RAGAS evaluation (this makes several LLM calls per question)...")
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    df = result.to_pandas()
    return df


# ── 4, 5, 6. Save, plot, report ───────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> None:
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nPer-question results saved → {RESULTS_CSV}")

    print("\n── Mean scores ──")
    for col in metric_cols:
        print(f"  {col:<20} {df[col].mean():.3f}")

    # Distribution plot
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, metric_cols):
        ax.hist(df[col].dropna(), bins=10, range=(0, 1), color="steelblue", edgecolor="white")
        ax.axvline(df[col].mean(), color="tomato", linestyle="--",
                   label=f"mean={df[col].mean():.2f}")
        ax.set_title(col)
        ax.set_xlim(0, 1)
        ax.legend()
    plt.suptitle("RAGAS Score Distributions", y=1.02)
    plt.tight_layout()
    fig.savefig(SCORES_PLOT, dpi=120, bbox_inches="tight")
    print(f"Distribution plot saved → {SCORES_PLOT}")

    # 5 worst by faithfulness
    # ragas 0.2.x renames the question column to "user_input"
    q_col = "question" if "question" in df.columns else "user_input"
    if "faithfulness" in df.columns and q_col in df.columns:
        print("\n── 5 lowest-faithfulness questions (investigate these) ──")
        worst = df.nsmallest(5, "faithfulness")[[q_col, "faithfulness"]]
        for _, row in worst.iterrows():
            print(f"  [{row['faithfulness']:.2f}] {str(row[q_col])[:70]}")


def main():
    samples = collect_samples()
    df = run_ragas(samples)
    summarize(df)
    print("\nDone. Review ragas_results.csv and write your findings in eval_findings.md")


if __name__ == "__main__":
    main()
