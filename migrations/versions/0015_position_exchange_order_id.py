"""Add exchange_order_id to positions table.

Stores the Kalshi exchange order ID for live-mode positions so PositionWatcher
can reconcile pending orders against the exchange.

Revision ID: 0015
Revises: 0014
Create Date: 2026-03-22
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "positions",
        sa.Column("exchange_order_id", sa.VARCHAR(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("positions", "exchange_order_id")
