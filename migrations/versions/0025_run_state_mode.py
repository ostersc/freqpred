"""run_state_mode

Revision ID: 0025
Revises: 0024
Create Date: 2026-04-06

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "run_state",
        sa.Column("mode", sa.VARCHAR(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run_state", "mode")
