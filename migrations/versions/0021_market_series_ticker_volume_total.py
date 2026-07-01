"""market_series_ticker_volume_total

Revision ID: 0021
Revises: 0020
Create Date: 2026-04-04

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0021'
down_revision: str | None = '0020'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('markets', sa.Column('series_ticker', sa.Text(), nullable=True))
    op.add_column(
        'markets',
        sa.Column('volume_total', sa.Float(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_column('markets', 'volume_total')
    op.drop_column('markets', 'series_ticker')
