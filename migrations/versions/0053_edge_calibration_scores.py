"""edge_calibration_scores

Revision ID: 0053
Revises: 0052
Create Date: 2026-07-11
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID

revision: str = "0053"
down_revision: str | None = "0052"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "edge_calibration_scores",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("edge_band", sa.VARCHAR(10), nullable=False),
        sa.Column("direction", sa.VARCHAR(10), nullable=False),
        sa.Column("series_ticker", sa.Text(), nullable=True),
        sa.Column("n_signals", sa.Integer(), nullable=False),
        sa.Column("n_markets", sa.Integer(), nullable=False),
        sa.Column("hit_rate", sa.Float(), nullable=False),
        sa.Column("avg_market_implied_p", sa.Float(), nullable=False),
        sa.Column("avg_model_implied_p", sa.Float(), nullable=False),
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
        "ix_edge_calibration_scores_band_direction_series_computed",
        "edge_calibration_scores",
        ["edge_band", "direction", "series_ticker", "computed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_edge_calibration_scores_band_direction_series_computed",
        table_name="edge_calibration_scores",
    )
    op.drop_table("edge_calibration_scores")
