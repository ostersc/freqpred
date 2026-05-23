"""Add exchange-confirmed order state columns to positions.

Tracks per-order fill metadata so the live entry path can poll
Kalshi via get_order() rather than inferring fills from net positions.

Revision ID: 0037
Revises: 0036
Create Date: 2026-05-23
"""
from alembic import op
import sqlalchemy as sa

revision = "0037"
down_revision = "0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "positions",
        sa.Column("requested_contracts", sa.Integer(), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column("exchange_order_status", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column(
            "last_exchange_sync_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("positions", "last_exchange_sync_at")
    op.drop_column("positions", "exchange_order_status")
    op.drop_column("positions", "requested_contracts")
