"""Per-service daily request quota tracking backed by Postgres.

Uses the ``api_daily_counters`` table with an atomic upsert so concurrent
processes see consistent counts. The caller is responsible for committing
the session.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_daily_count(session: AsyncSession, service: str, for_date: date) -> int:
    """Return the request count for *service* on *for_date* (0 if no row yet)."""
    result = await session.execute(
        text(
            "SELECT request_count FROM api_daily_counters "
            "WHERE service = :service AND date = :date"
        ),
        {"service": service, "date": for_date},
    )
    row = result.fetchone()
    return row[0] if row else 0


async def increment_daily_count(
    session: AsyncSession, service: str, for_date: date
) -> None:
    """Atomically increment the counter for *service* on *for_date* by 1."""
    await session.execute(
        text(
            "INSERT INTO api_daily_counters (service, date, request_count) "
            "VALUES (:service, :date, 1) "
            "ON CONFLICT (service, date) DO UPDATE "
            "SET request_count = api_daily_counters.request_count + 1"
        ),
        {"service": service, "date": for_date},
    )
