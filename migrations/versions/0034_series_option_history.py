"""Add series_option_history table for base-rate caching.

Revision ID: 0034
Revises: 0033
Create Date: 2026-05-12
"""
from alembic import op
import sqlalchemy as sa

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "series_option_history",
        sa.Column("series_ticker", sa.Text(), nullable=False),
        sa.Column("option_code", sa.Text(), nullable=False),
        sa.Column("option_label", sa.Text(), nullable=False),
        sa.Column("yes_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("no_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "last_fetched_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("series_ticker", "option_code"),
    )


def downgrade() -> None:
    op.drop_table("series_option_history")
