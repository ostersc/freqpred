"""add market_ask_at_signal to signals

Revision ID: 0018
Revises: 0017
Create Date: 2026-03-27 17:58:03.527653

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0018'
down_revision: str | None = '0017'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('signals', sa.Column('market_ask_at_signal', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('signals', 'market_ask_at_signal')
