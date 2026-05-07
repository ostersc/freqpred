"""Add llm_query_id column to signals table.

Revision ID: 0033
Revises: 0032
Create Date: 2026-05-02
"""
from alembic import op
import sqlalchemy as sa

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "signals",
        sa.Column("llm_query_id", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("signals", "llm_query_id")
