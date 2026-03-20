"""Add fetcher_cursors table — replaces Redis last-run tracking.

Revision ID: 0007
Revises: 0006
"""

from alembic import op
import sqlalchemy as sa

revision: str = "0007"
down_revision: str = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fetcher_cursors",
        sa.Column("fetcher", sa.Text(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column(
            "last_run_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("fetcher", "key"),
    )


def downgrade() -> None:
    op.drop_table("fetcher_cursors")
