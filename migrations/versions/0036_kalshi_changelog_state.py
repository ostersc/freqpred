"""Add kalshi_changelog_state table for tracking API changelog review status.

Revision ID: 0036
Revises: 0035
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kalshi_changelog_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("last_reviewed_at", sa.Date(), nullable=False),
        sa.Column("unreviewed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "has_unreviewed_breaking_change",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
        sa.Column(
            "last_checked_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # Seed the singleton row. last_reviewed_at = today; we reviewed the full
    # changelog on 2026-05-21 before introducing this monitor.
    op.execute(
        "INSERT INTO kalshi_changelog_state (id, last_reviewed_at) VALUES (1, '2026-05-21')"
    )


def downgrade() -> None:
    op.drop_table("kalshi_changelog_state")
