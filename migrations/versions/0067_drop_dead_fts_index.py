"""Drop ix_documents_fts — a 309 MB index nothing has ever scanned.

The index is a GIN over ``to_tsvector('english', title || ' ' || body)``, added
on the belief that it accelerates BM25 keyword scoring. It cannot. The
retriever computes ``ts_rank`` as a **projected column** over every candidate
row — there is no ``@@`` predicate for a GIN index to serve, so Postgres has no
way to use it. ``EXPLAIN`` on a live market shows only
``ix_dml_market_id_document_id`` and ``documents_pkey``, and
``pg_stat_user_indexes`` agrees::

    indexrelname      | idx_scan | idx_tup_read |  size
    ix_documents_fts  |        0 |            0 | 309 MB

Zero scans since the statistics were last reset, against 1,065,225 on the
primary key. It is the largest index on ``documents`` and it does nothing but
slow every insert and re-upsert — and documents are upserted on every re-fetch,
so that write cost is paid constantly.

Note this is *not* a consequence of the 2026-08-11 BM25 change (which switched
scoring from ``coalesce(summary, body)`` to ``body``, incidentally matching this
index's expression). The index was unusable before that change and remains
unusable after it, for the same structural reason.

``ix_documents_embedding`` is also at 0 scans for the same underlying reason —
cosine distance is likewise a projected column with no ``ORDER BY ... LIMIT``
for an ANN index to accelerate — but it is 968 kB rather than 309 MB, so it is
left alone here rather than bundled into an unrelated migration.

Both directions run ``CONCURRENTLY`` so ingestion keeps writing during the
change, which is why they need ``autocommit_block``. The downgrade rebuild is
slow (a full GIN build over ~126k documents) but correct; a failed concurrent
build leaves an INVALID index behind rather than rolling back, so check for one
before retrying::

    SELECT indexrelid::regclass FROM pg_index
    WHERE NOT indisvalid AND indrelid = 'documents'::regclass;

Revision ID: 0067
Revises: 0066
"""
from alembic import op

revision = "0067"
down_revision = "0066"
branch_labels = None
depends_on = None

_INDEX_NAME = "ix_documents_fts"
_EXPRESSION = "to_tsvector('english'::regconfig, ((title || ' '::text) || body))"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX_NAME}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX_NAME} "
            f"ON documents USING gin ({_EXPRESSION})"
        )
