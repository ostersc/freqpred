"""runtime_config_overrides

Revision ID: 0023
Revises: 0022
Create Date: 2026-04-05

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from alembic import op

revision: str = "0023"
down_revision: Union[str, None] = "0022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "runtime_config_overrides",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("strategy_name", sa.VARCHAR(255), nullable=False),
        sa.Column("overrides", JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "updated_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_unique_constraint(
        "uq_runtime_config_overrides_strategy_name",
        "runtime_config_overrides",
        ["strategy_name"],
    )


def downgrade() -> None:
    op.drop_table("runtime_config_overrides")
