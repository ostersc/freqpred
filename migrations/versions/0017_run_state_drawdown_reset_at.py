"""Add drawdown_reset_at to run_state table.

Stores the timestamp of the last /reset_drawdown call. The drawdown circuit
breaker only considers positions closed after this timestamp, so calling
/reset_drawdown starts a fresh drawdown window from the current moment.

Revision ID: 0017
Revises: 0016
Create Date: 2026-03-25
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "run_state",
        sa.Column(
            "drawdown_reset_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("run_state", "drawdown_reset_at")
