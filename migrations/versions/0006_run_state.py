"""Add run_state table for persisting run-loop state across restarts.

Revision ID: 0006
Revises: 0005
Create Date: 2026-03-19
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("state", sa.VARCHAR(20), nullable=False),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("run_state")
