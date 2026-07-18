"""Add missing indexes for dashboard tab-load queries.

The dashboard tabs poll several endpoints whose queries had no supporting
index and degraded as the DB grew (markets: 4.7M rows, document_market_links:
367k, llm_queries: 33k, source_quality_scores: 21k):

1. ix_markets_status_last_fetched_at — /api/markets list + count filter on
   status ordered by last_fetched_at (was a ~2s parallel seq scan per load),
   and the settlement-sources summary (status='active').
2. ix_markets_last_fetched_at — /api/markets with status=all orders the whole
   table by last_fetched_at.
3. ix_signals_created_at — /api/signals orders by created_at desc; signals is
   append-only so the sort input grows without bound.
4. ix_dml_signal_id — the per-signal RAG doc-count subqueries on /api/signals
   and the document-link fetch on signal/position detail filter
   document_market_links by signal_id (was 20 × 367k-row seq scans per page).
   Partial over signal-link rows only, mirroring ix_dml_doc_market_ingestion
   which covers the signal_id IS NULL half.
5. ix_llm_queries_timestamp — /api/llm/queries orders by timestamp; weekly
   and daily spend sums, spend time series, and the system-health LLM error
   count all range-filter on timestamp.
6. ix_source_quality_scores_computed_at — /api/metrics/source-quality finds
   the latest snapshot via max(computed_at) then an equality fetch on it.

Revision ID: 0057
Revises: 0056
Create Date: 2026-07-17

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0057"
down_revision: str | None = "0056"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_markets_status_last_fetched_at",
        "markets",
        ["status", "last_fetched_at"],
    )
    op.create_index(
        "ix_markets_last_fetched_at",
        "markets",
        ["last_fetched_at"],
    )
    op.create_index(
        "ix_signals_created_at",
        "signals",
        ["created_at"],
    )
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_dml_signal_id
        ON document_market_links (signal_id)
        WHERE signal_id IS NOT NULL
    """)
    op.create_index(
        "ix_llm_queries_timestamp",
        "llm_queries",
        ["timestamp"],
    )
    op.create_index(
        "ix_source_quality_scores_computed_at",
        "source_quality_scores",
        ["computed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_source_quality_scores_computed_at", table_name="source_quality_scores")
    op.drop_index("ix_llm_queries_timestamp", table_name="llm_queries")
    op.execute("DROP INDEX IF EXISTS ix_dml_signal_id")
    op.drop_index("ix_signals_created_at", table_name="signals")
    op.drop_index("ix_markets_last_fetched_at", table_name="markets")
    op.drop_index("ix_markets_status_last_fetched_at", table_name="markets")
