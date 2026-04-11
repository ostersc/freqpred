"""Widen markets.status from VARCHAR(20) to VARCHAR(50).

Kalshi sends status values like 'fractional_trading_updated' (27 chars)
which exceed the previous 20-char limit.

Revision ID: 0028
Revises: 0027
Create Date: 2026-04-09
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0028"
down_revision: Union[str, None] = "0027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "markets",
        "status",
        existing_type=sa.VARCHAR(20),
        type_=sa.VARCHAR(50),
        existing_nullable=False,
        existing_server_default="active",
    )


def downgrade() -> None:
    op.alter_column(
        "markets",
        "status",
        existing_type=sa.VARCHAR(50),
        type_=sa.VARCHAR(20),
        existing_nullable=False,
        existing_server_default="active",
    )
