"""Add daily_loss_ack_at to run_state table.

When the operator resumes the run loop via /start, this timestamp is stamped
to the current moment.  The daily loss circuit breaker then measures losses
since max(today_start, daily_loss_ack_at), preventing an already-tripped
breaker from immediately re-firing on the next cycle with unchanged P&L data.

Revision ID: 0027
Revises: 0026
Create Date: 2026-04-06
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0027"
down_revision: Union[str, None] = "0026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "run_state",
        sa.Column(
            "daily_loss_ack_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("run_state", "daily_loss_ack_at")
