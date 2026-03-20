"""Add mae and mfe columns to positions table.

mae (Maximum Adverse Excursion): most negative unrealized P&L per contract seen while open.
mfe (Maximum Favorable Excursion): most positive unrealized P&L per contract seen while open.
Stored as signed price deltas — multiply by contracts to get dollar impact.

Revision ID: 0009
Revises: 0008
"""

from alembic import op
import sqlalchemy as sa

revision: str = "0009"
down_revision: str = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("positions", sa.Column("mae", sa.Float(), nullable=True))
    op.add_column("positions", sa.Column("mfe", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("positions", "mfe")
    op.drop_column("positions", "mae")
