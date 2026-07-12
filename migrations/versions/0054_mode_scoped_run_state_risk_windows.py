"""Mode-scope run_state risk-window fields (T90).

The drawdown baseline and daily-loss acknowledgement were stored unscoped, so
a baseline captured while running one trading mode was silently reused after
switching to the other — paper and live bankrolls are never on the same scale,
which fired a false-positive 97.9% drawdown halt live. Replace the three
unscoped columns with paper/live pairs.

No data backfill: the old unscoped baseline was already wrong for at least one
mode, so both modes restart from "no baseline yet" (NULL), which is the safe
default — the drawdown check is skipped until /reset_drawdown establishes a
fresh baseline in that mode.

Revision ID: 0054
Revises: 0053
Create Date: 2026-07-12
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0054"
down_revision: str | None = "0053"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "run_state",
        sa.Column("drawdown_reset_at_paper", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "run_state",
        sa.Column("drawdown_reset_bankroll_paper", sa.Float(), nullable=True),
    )
    op.add_column(
        "run_state",
        sa.Column("drawdown_reset_at_live", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "run_state",
        sa.Column("drawdown_reset_bankroll_live", sa.Float(), nullable=True),
    )
    op.add_column(
        "run_state",
        sa.Column("daily_loss_ack_at_paper", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "run_state",
        sa.Column("daily_loss_ack_at_live", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.drop_column("run_state", "drawdown_reset_at")
    op.drop_column("run_state", "drawdown_reset_bankroll")
    op.drop_column("run_state", "daily_loss_ack_at")


def downgrade() -> None:
    op.add_column(
        "run_state",
        sa.Column("drawdown_reset_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "run_state",
        sa.Column("drawdown_reset_bankroll", sa.Float(), nullable=True),
    )
    op.add_column(
        "run_state",
        sa.Column("daily_loss_ack_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.drop_column("run_state", "daily_loss_ack_at_live")
    op.drop_column("run_state", "daily_loss_ack_at_paper")
    op.drop_column("run_state", "drawdown_reset_bankroll_live")
    op.drop_column("run_state", "drawdown_reset_at_live")
    op.drop_column("run_state", "drawdown_reset_bankroll_paper")
    op.drop_column("run_state", "drawdown_reset_at_paper")
