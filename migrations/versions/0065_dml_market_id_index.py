"""Index document_market_links on (market_id, document_id).

Every RAG retrieval resolves its candidate pool through a subquery in
``freqpred/rag/retriever.py`` shaped like::

    SELECT DISTINCT document_id FROM document_market_links WHERE market_id = ?

``document_market_links`` has 404k rows and no index that can serve that
predicate. The pkey is on ``id``; ``ix_dml_doc_market_ingestion`` is UNIQUE on
``(document_id, market_id)`` — wrong leading column — and partial on
``signal_id IS NULL`` besides; ``ix_dml_signal_id`` is partial the other way.
So the planner falls back to a sequential scan, discarding ~134k rows per
retrieval. Measured on the worst-case market (1,356 linked documents), the scan
was the dominant cost of a 295ms query.

The composite ``(market_id, document_id)`` rather than ``(market_id)`` alone
lets the subquery run index-only: both columns it touches are in the index, so
the heap is never visited.

This is a pure performance change. The retriever computes ``cosine_distance``
as a projected column over the full candidate set and ranks in Python — there
is no ``ORDER BY ... LIMIT`` for an ANN index to accelerate, and no approximate
search anywhere in the path. The same rows come back in the same order, faster.

Built with ``CREATE INDEX CONCURRENTLY`` so the ingestion pipeline can keep
writing during the build. That cannot run inside a transaction, hence the
``autocommit_block``. The tradeoff is that a failed concurrent build leaves an
INVALID index behind rather than rolling back; ``IF NOT EXISTS`` makes a re-run
safe, but an interrupted build should be checked for and dropped manually::

    SELECT indexrelid::regclass FROM pg_index
    WHERE NOT indisvalid AND indrelid = 'document_market_links'::regclass;

Revision ID: 0065
Revises: 0064
"""
from alembic import op

revision = "0065"
down_revision = "0064"
branch_labels = None
depends_on = None

_INDEX_NAME = "ix_dml_market_id_document_id"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX_NAME} "
            "ON document_market_links (market_id, document_id)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX_NAME}")
