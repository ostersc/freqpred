"""Add drawdown_reset_bankroll to run_state table.

Stores the net bankroll value at the time of the last /reset_drawdown call.
The drawdown circuit breaker and display now compare current net bankroll
against this stored value, giving a real reference point instead of a
synthetic "ATH" reconstructed from losses since reset.

Revision ID: 0019
Revises: 0018
Create Date: 2026-04-02
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "run_state",
        sa.Column("drawdown_reset_bankroll", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run_state", "drawdown_reset_bankroll")
