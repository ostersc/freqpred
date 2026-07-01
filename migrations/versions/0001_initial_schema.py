"""Initial schema — pgvector extension + all tables.

Revision ID: 0001
Revises:
Create Date: 2026-03-15

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # pgvector extension — must exist before the embedding column
    # ------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------
    # markets — created before signals because signals FK -> markets.
    # current_signal_id FK is added after signals table exists.
    # ------------------------------------------------------------------
    op.create_table(
        "markets",
        sa.Column("id", sa.VARCHAR(255), primary_key=True),
        sa.Column("platform", sa.VARCHAR(50), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("category", sa.VARCHAR(100), nullable=False),
        sa.Column("close_time", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("yes_bid", sa.Float, nullable=False),
        sa.Column("yes_ask", sa.Float, nullable=False),
        sa.Column("mid_price", sa.Float, nullable=False),
        sa.Column("volume_24h", sa.Float, nullable=False),
        sa.Column("open_interest", sa.Float, nullable=False),
        sa.Column("last_fetched_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("price_updated_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("metadata_fetched_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        # current_signal_id FK added below after signals table exists
        sa.Column("current_signal_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", sa.JSON, nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------
    op.create_table(
        "signals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "market_id",
            sa.VARCHAR(255),
            sa.ForeignKey("markets.id"),
            nullable=False,
        ),
        sa.Column("estimated_probability", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("edge", sa.Float, nullable=False),
        sa.Column("market_mid_at_signal", sa.Float, nullable=False),
        sa.Column("direction", sa.VARCHAR(10), nullable=False),
        sa.Column("reasoning", sa.Text, nullable=False),
        sa.Column("sources", postgresql.ARRAY(sa.Text), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("social_sentiment_summary", sa.Text, nullable=True),
        sa.Column("retrieval_hash", sa.VARCHAR(64), nullable=False),
        sa.Column("model_used", sa.VARCHAR(100), nullable=False),
        sa.Column("prompt_version", sa.VARCHAR(100), nullable=False),
        sa.Column("trigger", sa.VARCHAR(50), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("raw_context", sa.Text, nullable=False),
    )

    # Add deferred FK: markets.current_signal_id -> signals.id
    op.create_foreign_key(
        "fk_markets_current_signal",
        "markets",
        "signals",
        ["current_signal_id"],
        ["id"],
    )

    # ------------------------------------------------------------------
    # positions
    # ------------------------------------------------------------------
    op.create_table(
        "positions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "market_id",
            sa.VARCHAR(255),
            sa.ForeignKey("markets.id"),
            nullable=False,
        ),
        sa.Column(
            "signal_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("signals.id"),
            nullable=False,
        ),
        sa.Column("strategy_name", sa.VARCHAR(255), nullable=False),
        sa.Column("strategy_version", sa.VARCHAR(50), nullable=False),
        sa.Column("signal_confidence", sa.Float, nullable=False),
        sa.Column("signal_edge", sa.Float, nullable=False),
        sa.Column("signal_estimated_prob", sa.Float, nullable=False),
        sa.Column("direction", sa.VARCHAR(10), nullable=False),
        sa.Column("contracts", sa.Integer, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("entry_time", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("mode", sa.VARCHAR(10), nullable=False),
        sa.Column("status", sa.VARCHAR(20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("exit_price", sa.Float, nullable=True),
        sa.Column("exit_time", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("resolution", sa.Integer, nullable=True),
        sa.Column("pnl", sa.Float, nullable=True),
        sa.Column("pnl_pct", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_url", sa.Text, nullable=False),
        sa.Column("content_hash", sa.VARCHAR(64), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("source_type", sa.VARCHAR(50), nullable=False),
        sa.Column("source_name", sa.VARCHAR(255), nullable=False),
        sa.Column("category", sa.VARCHAR(100), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("published_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("fetched_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.Column("embedding_model", sa.VARCHAR(100), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ------------------------------------------------------------------
    # document_market_links
    # ------------------------------------------------------------------
    op.create_table(
        "document_market_links",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id"),
            nullable=False,
        ),
        sa.Column(
            "market_id",
            sa.VARCHAR(255),
            sa.ForeignKey("markets.id"),
            nullable=False,
        ),
        sa.Column(
            "signal_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("signals.id"),
            nullable=True,
        ),
        sa.Column("relevance_score", sa.Float, nullable=False),
        sa.Column("linked_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ------------------------------------------------------------------
    # llm_queries
    # ------------------------------------------------------------------
    op.create_table(
        "llm_queries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("strategy", sa.VARCHAR(255), nullable=False),
        sa.Column("query_type", sa.VARCHAR(50), nullable=False),
        sa.Column("market_id", sa.VARCHAR(255), nullable=True),
        sa.Column("signal_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("model_used", sa.VARCHAR(100), nullable=False),
        sa.Column("prompt_version", sa.VARCHAR(100), nullable=False),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("response", sa.Text, nullable=False),
        sa.Column("tokens_input", sa.Integer, nullable=False),
        sa.Column("tokens_output", sa.Integer, nullable=False),
        sa.Column("tokens_total", sa.Integer, nullable=False),
        sa.Column("cost_usd", sa.Float, nullable=False),
        sa.Column("confidence_extracted", sa.Float, nullable=True),
        sa.Column("decision_extracted", sa.VARCHAR(10), nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("success", sa.Boolean, nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    # documents.embedding — ivfflat for ANN search
    # lists=100 is a reasonable starting point; tune after data grows
    op.execute(
        "CREATE INDEX ix_documents_embedding ON documents "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # documents.source_url — already unique (unique index implicit), add named index
    op.create_index("ix_documents_source_url", "documents", ["source_url"], unique=True)

    # signals.market_id
    op.create_index("ix_signals_market_id", "signals", ["market_id"])

    # positions.status
    op.create_index("ix_positions_status", "positions", ["status"])


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table("llm_queries")
    op.drop_table("document_market_links")
    op.drop_table("documents")
    op.drop_table("positions")

    # Remove circular FK before dropping signals
    op.drop_constraint("fk_markets_current_signal", "markets", type_="foreignkey")
    op.drop_table("signals")
    op.drop_table("markets")

    op.execute("DROP EXTENSION IF EXISTS vector")
