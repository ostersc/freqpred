"""add status, result, last_price, liquidity to markets

Revision ID: 0013
Revises: ea37ae04fdd1
Create Date: 2026-03-21

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0013'
down_revision: Union[str, None] = 'ea37ae04fdd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('markets', sa.Column('status', sa.VARCHAR(20), nullable=False, server_default='active'))
    op.add_column('markets', sa.Column('result', sa.VARCHAR(10), nullable=True))
    op.add_column('markets', sa.Column('last_price', sa.Float(), nullable=False, server_default='0'))
    op.add_column('markets', sa.Column('liquidity', sa.Float(), nullable=False, server_default='0'))


def downgrade() -> None:
    op.drop_column('markets', 'liquidity')
    op.drop_column('markets', 'last_price')
    op.drop_column('markets', 'result')
    op.drop_column('markets', 'status')
