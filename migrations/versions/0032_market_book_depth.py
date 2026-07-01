"""Replace liquidity column with yes_bid_size/yes_ask_size on markets table.

Revision ID: 0032
Revises: 0031
Create Date: 2026-05-01
"""
import sqlalchemy as sa
from alembic import op

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("markets", "liquidity")
    op.add_column("markets", sa.Column("yes_bid_size", sa.Float(), nullable=False, server_default="0"))
    op.add_column("markets", sa.Column("yes_ask_size", sa.Float(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("markets", "yes_bid_size")
    op.drop_column("markets", "yes_ask_size")
    op.add_column("markets", sa.Column("liquidity", sa.Float(), nullable=False, server_default="0"))
