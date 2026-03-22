"""Per-service 12-hour window request quota tracking backed by Postgres.

NewsAPI developer accounts allow 50 requests per 12-hour window (not 100/day).
Uses the ``api_daily_counters`` table with (service, date, hour_slot) as the
composite primary key, where hour_slot is 0 (00:00–11:59 UTC) or 1 (12:00–23:59 UTC).
Upserts are atomic so concurrent processes see consistent counts.
"""
from __future__ import annotations

from datetime import date, datetime, timezone


from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def current_window(now: datetime | None = None) -> tuple[date, int]:
    """Return (date, hour_slot) for the current 12-hour UTC window.

    hour_slot is 0 for 00:00–11:59 UTC and 1 for 12:00–23:59 UTC.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    return now.date(), now.hour // 12


async def get_window_count(
    session: AsyncSession, service: str, for_date: date, hour_slot: int
) -> int:
    """Return the request count for *service* in the given window (0 if no row yet)."""
    result = await session.execute(
        text(
            "SELECT request_count FROM api_daily_counters "
            "WHERE service = :service AND date = :date AND hour_slot = :slot"
        ),
        {"service": service, "date": for_date, "slot": hour_slot},
    )
    row = result.fetchone()
    return row[0] if row else 0


async def increment_window_count(
    session: AsyncSession, service: str, for_date: date, hour_slot: int
) -> None:
    """Atomically increment the counter for *service* in the given window by 1."""
    await session.execute(
        text(
            "INSERT INTO api_daily_counters (service, date, hour_slot, request_count) "
            "VALUES (:service, :date, :slot, 1) "
            "ON CONFLICT (service, date, hour_slot) DO UPDATE "
            "SET request_count = api_daily_counters.request_count + 1"
        ),
        {"service": service, "date": for_date, "slot": hour_slot},
    )
