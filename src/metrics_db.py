"""
metrics_db.py — Week 8: Per-query metrics logging to Postgres
--------------------------------------------------------------
Logs one row per RAG query with per-stage latency, token counts, and cost.
Grafana reads this table to build the observability dashboard.

This is a pragmatic stand-in for full OpenTelemetry tracing: it captures the
same signals (stage latency, cost, tokens) in a queryable table. The production
upgrade is to emit OpenTelemetry spans to Jaeger/Tempo; the schema here maps
cleanly onto spans (each *_ms column is one span's duration).
"""

from __future__ import annotations

import os

import psycopg2

DATABASE_URL = os.getenv(
    "RAG_DATABASE_URL",
    "postgresql://rag:rag123@localhost:5433/rag_metrics",
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_metrics (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query           TEXT NOT NULL,
    cache_hit       BOOLEAN NOT NULL,
    retrieve_ms     INTEGER,
    rerank_ms       INTEGER,
    llm_ms          INTEGER,
    total_ms        INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        NUMERIC(10, 6),
    model           TEXT
);
"""


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()


def log_metrics(m: dict) -> None:
    """Insert one query's metrics. Missing keys default to None."""
    sql = """
    INSERT INTO query_metrics
        (query, cache_hit, retrieve_ms, rerank_ms, llm_ms, total_ms,
         input_tokens, output_tokens, cost_usd, model)
    VALUES (%(query)s, %(cache_hit)s, %(retrieve_ms)s, %(rerank_ms)s, %(llm_ms)s,
            %(total_ms)s, %(input_tokens)s, %(output_tokens)s, %(cost_usd)s, %(model)s)
    """
    defaults = {k: None for k in (
        "retrieve_ms", "rerank_ms", "llm_ms", "total_ms",
        "input_tokens", "output_tokens", "cost_usd", "model",
    )}
    defaults.update(m)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, defaults)
        conn.commit()
