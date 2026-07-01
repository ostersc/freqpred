"""add open_time to markets

Revision ID: ea37ae04fdd1
Revises: 0011
Create Date: 2026-03-21 09:16:21.225869

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ea37ae04fdd1'
down_revision: str | None = '0011'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'markets',
        sa.Column('open_time', postgresql.TIMESTAMP(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('markets', 'open_time')
