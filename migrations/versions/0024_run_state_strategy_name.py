"""run_state_strategy_name

Revision ID: 0024
Revises: 0023
Create Date: 2026-04-06

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: Union[str, None] = "0023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "run_state",
        sa.Column("strategy_name", sa.VARCHAR(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run_state", "strategy_name")
