"""Add settlement_value column to markets table.

Revision ID: 0031
Revises: 0030
Create Date: 2026-04-29
"""
import sqlalchemy as sa
from alembic import op

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("markets", sa.Column("settlement_value", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("markets", "settlement_value")
