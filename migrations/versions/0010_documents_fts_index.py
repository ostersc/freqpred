"""Add GIN full-text search index on documents + partial unique index on document_market_links.

1. GIN index on to_tsvector('english', title || ' ' || body) — enables ts_rank BM25
   scoring for hybrid retrieval without a full table scan.

2. Partial unique index on document_market_links(document_id, market_id) WHERE
   signal_id IS NULL — enforces one ingestion-time link per (doc, market) pair so the
   scheduler can safely use ON CONFLICT DO NOTHING.

Revision ID: 0010
Revises: 0009
"""

from alembic import op

revision: str = "0010"
down_revision: str = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_documents_fts
        ON documents
        USING GIN (to_tsvector('english', title || ' ' || body))
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_dml_doc_market_ingestion
        ON document_market_links (document_id, market_id)
        WHERE signal_id IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_dml_doc_market_ingestion")
    op.execute("DROP INDEX IF EXISTS ix_documents_fts")
