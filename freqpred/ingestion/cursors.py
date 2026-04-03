"""Fetcher cursor persistence — tracks last-run timestamps in Postgres.

Replaces the Redis-based ingestion:last_run:* key pattern.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_cursor(session: AsyncSession, fetcher: str, key: str) -> datetime | None:
    """Return the last-run datetime for (fetcher, key), or None if never run."""
    result = await session.execute(
        text(
            "SELECT last_run_at FROM fetcher_cursors WHERE fetcher = :fetcher AND key = :key"
        ),
        {"fetcher": fetcher, "key": key},
    )
    row = result.one_or_none()
    if row is None:
        return None
    dt: datetime = row[0]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


async def set_cursor(
    session: AsyncSession, fetcher: str, key: str, last_run_at: datetime
) -> None:
    """Upsert the last-run timestamp for (fetcher, key)."""
    await session.execute(
        text(
            """
            INSERT INTO fetcher_cursors (fetcher, key, last_run_at)
            VALUES (:fetcher, :key, :last_run_at)
            ON CONFLICT (fetcher, key) DO UPDATE SET last_run_at = EXCLUDED.last_run_at
            """
        ),
        {"fetcher": fetcher, "key": key, "last_run_at": last_run_at},
    )


async def delete_cursors(
    session: AsyncSession, fetcher: str, keys: list[str] | set[str]
) -> None:
    """Delete all cursor rows for the given fetcher+keys.

    Used to clean up per-market cursors when markets close or become
    uninteresting so stale rows do not accumulate in fetcher_cursors.
    No-op if *keys* is empty.
    """
    if not keys:
        return
    await session.execute(
        text(
            "DELETE FROM fetcher_cursors WHERE fetcher = :fetcher AND key = ANY(:keys)"
        ),
        {"fetcher": fetcher, "keys": list(keys)},
    )
