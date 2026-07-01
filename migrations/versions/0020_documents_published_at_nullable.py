"""documents_published_at_nullable

Revision ID: 0020
Revises: 0019
Create Date: 2026-04-02 18:44:59.897240

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0020'
down_revision: str | None = '0019'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        'documents',
        'published_at',
        existing_type=postgresql.TIMESTAMP(timezone=True),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        'documents',
        'published_at',
        existing_type=postgresql.TIMESTAMP(timezone=True),
        nullable=False,
    )
