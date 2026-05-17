"""Add factbase_phrase_frequency table for KXTRUMPSAY phrase frequency caching.

Revision ID: 0035
Revises: 0034
Create Date: 2026-05-17
"""
from alembic import op
import sqlalchemy as sa

revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "factbase_phrase_frequency",
        sa.Column("market_id", sa.VARCHAR(255), nullable=False),
        sa.Column("display_phrase", sa.Text(), nullable=False),
        sa.Column("api_query", sa.Text(), nullable=False),
        sa.Column("speaker_slug", sa.Text(), nullable=False, server_default="trump"),
        sa.Column("in_market_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("count_7d", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("count_30d", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("count_365d", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "top_quotes",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
        sa.Column("last_fetched_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["market_id"], ["markets.id"]),
        sa.PrimaryKeyConstraint("market_id"),
    )


def downgrade() -> None:
    op.drop_table("factbase_phrase_frequency")
