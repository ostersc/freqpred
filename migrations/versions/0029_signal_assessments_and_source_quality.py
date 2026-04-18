"""signal_assessments and source_quality_scores

Revision ID: 0029
Revises: 0028
Create Date: 2026-04-18
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID


revision: str = "0029"
down_revision: Union[str, None] = "0028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "source_quality_scores",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("market_category", sa.VARCHAR(100), nullable=True),
        sa.Column("lookback_days", sa.Integer(), nullable=False),
        sa.Column("weighted_brier", sa.Float(), nullable=False),
        sa.Column("overall_brier", sa.Float(), nullable=False),
        sa.Column("n_signals", sa.Integer(), nullable=False),
        sa.Column("total_doc_uses", sa.Integer(), nullable=False),
        sa.Column(
            "computed_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_source_quality_scores_source_category_computed",
        "source_quality_scores",
        ["source_name", "market_category", "computed_at"],
    )

    op.create_table(
        "signal_assessments",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("signal_id", UUID(as_uuid=True), nullable=False),
        sa.Column("trust_score", sa.Float(), nullable=False),
        sa.Column("size_multiplier", sa.Float(), nullable=False),
        sa.Column("verdict", sa.VARCHAR(20), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("key_factors", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("warnings", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_breakdown", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("similar_market_summary", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("llm_query_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["llm_query_id"], ["llm_queries.id"]),
        sa.ForeignKeyConstraint(["signal_id"], ["signals.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_signal_assessments_signal_created",
        "signal_assessments",
        ["signal_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_signal_assessments_signal_created", table_name="signal_assessments")
    op.drop_table("signal_assessments")

    op.drop_index(
        "ix_source_quality_scores_source_category_computed",
        table_name="source_quality_scores",
    )
    op.drop_table("source_quality_scores")
