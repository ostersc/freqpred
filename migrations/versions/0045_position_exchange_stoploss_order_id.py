"""Add exchange_stoploss_order_id column to positions table.

Stores the Kalshi order ID of a resting stoploss order posted on the exchange
when stoploss_on_exchange=True (T48). NULL for paper mode and pre-T48 rows.

Revision ID: 0045
Revises: 0044
Create Date: 2026-06-07
"""
from alembic import op
import sqlalchemy as sa

revision = "0045"
down_revision = "0044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "positions",
        sa.Column("exchange_stoploss_order_id", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("positions", "exchange_stoploss_order_id")
