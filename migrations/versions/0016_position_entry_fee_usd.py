"""Add entry_fee_usd to positions table.

Stores the exchange fee paid at fill time. Used to compute net P&L
(gross_pnl - entry_fee_usd) and effective entry cost.

Revision ID: 0016
Revises: 0015
Create Date: 2026-03-22
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "positions",
        sa.Column(
            "entry_fee_usd",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("positions", "entry_fee_usd")
