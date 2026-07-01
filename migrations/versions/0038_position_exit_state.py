"""Add exit-side order-state columns to positions.

Tracks exit order metadata so the live exit path can poll Kalshi via
get_order() and handle IOC partial fills without closing the ledger
prematurely.

Revision ID: 0038
Revises: 0037
Create Date: 2026-05-23
"""
import sqlalchemy as sa
from alembic import op

revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "positions",
        sa.Column("exit_order_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column(
            "exit_fee_usd",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "positions",
        sa.Column("exit_requested_contracts", sa.Integer(), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column("exit_filled_contracts", sa.Integer(), nullable=True),
    )
    op.add_column(
        "positions",
        sa.Column(
            "realized_pnl_accumulator",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("positions", "realized_pnl_accumulator")
    op.drop_column("positions", "exit_filled_contracts")
    op.drop_column("positions", "exit_requested_contracts")
    op.drop_column("positions", "exit_fee_usd")
    op.drop_column("positions", "exit_order_id")
