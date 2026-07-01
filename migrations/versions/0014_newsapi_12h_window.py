"""Change api_daily_counters to track per 12-hour window.

NewsAPI developer accounts allow 50 requests per 12-hour window (not 100 per day).
Adds an ``hour_slot`` column (0 = 00:00–11:59 UTC, 1 = 12:00–23:59 UTC) and
updates the primary key from (service, date) to (service, date, hour_slot).

Revision ID: 0014
Revises: 0013
Create Date: 2026-03-22
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add hour_slot column — existing rows get slot 0.
    op.add_column(
        "api_daily_counters",
        sa.Column("hour_slot", sa.SmallInteger(), nullable=False, server_default="0"),
    )
    # Drop the old (service, date) primary key and replace with (service, date, hour_slot).
    # Must happen before the INSERT below so the new PK allows a second row per date.
    op.drop_constraint("api_daily_counters_pkey", "api_daily_counters", type_="primary")
    op.create_primary_key(
        "api_daily_counters_pkey", "api_daily_counters", ["service", "date", "hour_slot"]
    )
    # Split today's counts evenly across both windows.  Historical dates stay in slot 0
    # (we don't know which half of those days the requests were in, and the data is stale).
    # For today, insert a slot=1 row with floor(count/2), then reduce slot=0 to the remainder.
    op.execute(
        "INSERT INTO api_daily_counters (service, date, hour_slot, request_count) "
        "SELECT service, date, 1, request_count / 2 "
        "FROM api_daily_counters "
        "WHERE date = (NOW() AT TIME ZONE 'UTC')::date AND hour_slot = 0"
    )
    op.execute(
        "UPDATE api_daily_counters "
        "SET request_count = request_count - (request_count / 2) "
        "WHERE date = (NOW() AT TIME ZONE 'UTC')::date AND hour_slot = 0"
    )


def downgrade() -> None:
    op.drop_constraint("api_daily_counters_pkey", "api_daily_counters", type_="primary")
    # Collapse any duplicate (service, date) rows that exist after removing hour_slot.
    op.execute(
        "DELETE FROM api_daily_counters a USING api_daily_counters b "
        "WHERE a.service = b.service AND a.date = b.date AND a.hour_slot > b.hour_slot"
    )
    op.create_primary_key(
        "api_daily_counters_pkey", "api_daily_counters", ["service", "date"]
    )
    op.drop_column("api_daily_counters", "hour_slot")
