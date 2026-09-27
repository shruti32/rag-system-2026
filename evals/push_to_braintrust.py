"""
push_to_braintrust.py — Week 7: Log RAGAS results to a Braintrust dashboard
----------------------------------------------------------------------------
Reads the already-computed evals/ragas_results.csv and logs each question as a
row in a Braintrust experiment, with the four RAGAS metrics as scores. This
gives you a hosted dashboard to compare eval runs over time (regression testing).

Why read the CSV instead of re-running RAGAS?
  The expensive part (LLM calls) already ran. We just ship those results to
  Braintrust. Re-running would cost money for no new information.

Setup:
  1. pip install braintrust
  2. Add BRAINTRUST_API_KEY to .env
  3. poetry run python evals/push_to_braintrust.py

Each run creates a new experiment version, so you can compare "before vs after"
a pipeline change directly in the Braintrust UI.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys

import braintrust
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("BRAINTRUST_API_KEY")
if not _API_KEY:
    sys.exit(
        "BRAINTRUST_API_KEY not found. Add it to .env as:\n"
        "  BRAINTRUST_API_KEY=sk-...\n"
        "(no quotes, no spaces around '=')"
    )

# Explicit login so a bad/missing key fails immediately with a clear message
braintrust.login(api_key=_API_KEY)

EVAL_DIR    = pathlib.Path(__file__).parent
RESULTS_CSV = EVAL_DIR / "ragas_results.csv"

PROJECT_NAME    = "rag-system-2026"
EXPERIMENT_NAME = "ragas-baseline"

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _parse_contexts(raw: str) -> list[str]:
    """retrieved_contexts is stored as a stringified Python list; parse it back."""
    try:
        val = ast.literal_eval(raw)
        return val if isinstance(val, list) else [str(val)]
    except (ValueError, SyntaxError):
        return [str(raw)]


def main():
    df = pd.read_csv(RESULTS_CSV)

    experiment = braintrust.init(
        project=PROJECT_NAME,
        experiment=EXPERIMENT_NAME,
    )

    for _, row in df.iterrows():
        # ragas 0.2.x column names
        question = row.get("user_input", row.get("question", ""))
        answer   = row.get("response",   row.get("answer", ""))
        contexts = _parse_contexts(str(row.get("retrieved_contexts", "[]")))
        ref      = row.get("reference",  row.get("ground_truth", ""))

        scores = {}
        for col in METRIC_COLS:
            if col in row and pd.notna(row[col]):
                scores[col] = float(row[col])

        experiment.log(
            input=question,
            output=answer,
            expected=ref,
            scores=scores,
            metadata={"contexts": contexts, "n_contexts": len(contexts)},
        )

    summary = experiment.summarize()
    print("Logged to Braintrust.")
    print(summary)


if __name__ == "__main__":
    main()
