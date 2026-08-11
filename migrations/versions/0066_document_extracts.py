"""Add document_extracts — question-focused evidence extraction cache (T101).

``build_prompt`` rendered each retrieved document as ``(summary or body)[:500]``,
a raw prefix cut. Because articles open with navigation chrome, 40.3% of
signal-linked evidence reached the LLM as boilerplate; the worst band averaged
22,372 chars and was shown at 6.5%. T101 replaces the cut with an extract taken
against the market actually being analysed.

The table is signal-pipeline-owned rather than a column on ``documents``.
Writing extracts back into ``documents.summary`` would have the signal path
mutating ingestion-owned rows — against §7's pipeline separation — and would
recreate the coupling T101 removes, since a document summarised for one market
then represents itself to every other market's retrieval.

``prompt_version`` is part of the unique key, not merely recorded on the row:
the triple IS the cache key, so changing the extraction prompt re-extracts
instead of serving text written under different instructions.

``ON DELETE CASCADE`` on ``document_id`` because an extract without its
document is meaningless. ``market_id`` is left restricting — markets are never
hard-deleted (they finalize), so a cascade there would only mask a bug.

Revision ID: 0066
Revises: 0065
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID

revision = "0066"
down_revision = "0065"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_extracts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "market_id",
            sa.VARCHAR(255),
            sa.ForeignKey("markets.id"),
            nullable=False,
        ),
        sa.Column("relevance", sa.VARCHAR(20), nullable=False),
        sa.Column("extract", sa.Text(), nullable=False),
        sa.Column("model_used", sa.VARCHAR(100), nullable=False),
        sa.Column("prompt_version", sa.VARCHAR(100), nullable=False),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "document_id",
            "market_id",
            "prompt_version",
            name="uq_document_extracts_doc_market_version",
        ),
    )
    # The read path is a batch lookup of ~10 documents for one market at one
    # prompt version; the unique constraint's index leads on document_id, so
    # this serves the market-scoped scan the pipeline actually issues.
    op.create_index(
        "ix_document_extracts_market_version",
        "document_extracts",
        ["market_id", "prompt_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_extracts_market_version", table_name="document_extracts")
    op.drop_table("document_extracts")
