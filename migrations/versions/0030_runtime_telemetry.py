"""runtime_telemetry

Revision ID: 0030
Revises: 0029
Create Date: 2026-04-19
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

revision: str = "0030"
down_revision: str | None = "0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "service_heartbeats",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("service_name", sa.VARCHAR(length=100), nullable=False),
        sa.Column("last_success_at", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_error_at", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_error_message", sa.Text(), nullable=True),
        sa.Column("details", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("service_name"),
    )
    op.create_index(
        "ix_service_heartbeats_service_name",
        "service_heartbeats",
        ["service_name"],
        unique=True,
    )

    op.create_table(
        "runtime_events",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("service_name", sa.VARCHAR(length=100), nullable=False),
        sa.Column("category", sa.VARCHAR(length=50), nullable=False),
        sa.Column("level", sa.VARCHAR(length=20), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_runtime_events_category_created_at",
        "runtime_events",
        ["category", "created_at"],
    )
    op.create_index(
        "ix_runtime_events_service_created_at",
        "runtime_events",
        ["service_name", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_runtime_events_service_created_at", table_name="runtime_events")
    op.drop_index("ix_runtime_events_category_created_at", table_name="runtime_events")
    op.drop_table("runtime_events")

    op.drop_index("ix_service_heartbeats_service_name", table_name="service_heartbeats")
    op.drop_table("service_heartbeats")

