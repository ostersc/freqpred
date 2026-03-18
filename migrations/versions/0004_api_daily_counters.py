"""Add api_daily_counters table for per-service daily request tracking.

Replaces the Redis-based quota counter originally scoped for NewsAPI (T15).
A single row per (service, date) pair is upserted on each successful request.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-18
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "api_daily_counters",
        sa.Column("service", sa.Text(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("service", "date"),
    )


def downgrade() -> None:
    op.drop_table("api_daily_counters")
