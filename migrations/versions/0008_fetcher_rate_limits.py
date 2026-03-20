"""Add fetcher_rate_limits table — persistent exponential backoff per fetcher.

Revision ID: 0008
Revises: 0007
"""

from alembic import op
import sqlalchemy as sa

revision: str = "0008"
down_revision: str = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fetcher_rate_limits",
        sa.Column("service", sa.Text(), nullable=False),
        sa.Column("skip_cycles_remaining", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skip_cycles_next", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("tripped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("service"),
    )


def downgrade() -> None:
    op.drop_table("fetcher_rate_limits")
