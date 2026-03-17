"""Add catalyst_runs and catalyst_queries tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-16

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # catalyst_runs — one row per LLM generation event per market
    # ------------------------------------------------------------------
    op.create_table(
        "catalyst_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column(
            "market_id",
            sa.VARCHAR(255),
            sa.ForeignKey("markets.id"),
            nullable=False,
        ),
        sa.Column("generation", sa.Integer, nullable=False),
        sa.Column(
            "llm_query_id",
            sa.Integer,
            sa.ForeignKey("llm_queries.id"),
            nullable=True,
        ),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_catalyst_runs_market_id", "catalyst_runs", ["market_id"])

    # ------------------------------------------------------------------
    # catalyst_queries — the actual search strings per run
    # ------------------------------------------------------------------
    op.create_table(
        "catalyst_queries",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("catalyst_runs.id"),
            nullable=False,
        ),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_catalyst_queries_run_id", "catalyst_queries", ["run_id"])


def downgrade() -> None:
    op.drop_table("catalyst_queries")
    op.drop_table("catalyst_runs")
