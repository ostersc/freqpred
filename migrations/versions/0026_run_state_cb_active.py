"""run_state_cb_active

Revision ID: 0026
Revises: 0025
Create Date: 2026-04-06

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: Union[str, None] = "0025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "run_state",
        sa.Column("cb_active", sa.Boolean, nullable=False, server_default="false"),
    )
    op.add_column(
        "run_state",
        sa.Column("cb_reason", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run_state", "cb_reason")
    op.drop_column("run_state", "cb_active")
