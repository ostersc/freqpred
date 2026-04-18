"""Background scheduler for source-quality snapshots."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.metrics.calibration import refresh_source_quality_scores
from freqpred.metrics.models import SourceQualityScoreRow

log = structlog.get_logger(__name__)


def _seconds_until_next(time_str: str, tz: ZoneInfo) -> float:
    hour, minute = (int(p) for p in time_str.split(":"))
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def refresh_source_quality_scores_if_due(
    session: AsyncSession,
    *,
    lookback_days: int = 90,
    now: datetime | None = None,
) -> int:
    """Refresh daily source-quality rows only when today's snapshot is missing."""
    current = now or datetime.now(UTC)
    latest_result = await session.execute(
        select(func.max(SourceQualityScoreRow.computed_at))
    )
    latest = latest_result.scalar_one()
    if latest is not None and latest.date() >= current.date():
        return 0
    return await refresh_source_quality_scores(session, lookback_days=lookback_days)


async def run_source_quality_scheduler(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    refresh_time: str = "07:00",
    refresh_timezone: str = "America/New_York",
    lookback_days: int = 90,
) -> None:
    """Refresh source-quality snapshots once per day on their own scheduler."""
    tz = ZoneInfo(refresh_timezone)
    log.info(
        "source_quality_scheduler.started",
        refresh_time=refresh_time,
        refresh_timezone=refresh_timezone,
        lookback_days=lookback_days,
    )

    async def _run_refresh(reason: str) -> None:
        async with session_factory() as session:
            try:
                rows_written = await refresh_source_quality_scores_if_due(
                    session,
                    lookback_days=lookback_days,
                )
                await session.commit()
                log.info(
                    "source_quality_scheduler.refreshed",
                    reason=reason,
                    rows_written=rows_written,
                )
            except Exception:
                await session.rollback()
                raise

    try:
        await _run_refresh("startup")
    except Exception:
        log.exception("source_quality_scheduler.startup_error")

    while True:
        wait = _seconds_until_next(refresh_time, tz)
        log.debug("source_quality_scheduler.sleeping", seconds=round(wait))
        await asyncio.sleep(wait)

        try:
            await _run_refresh("scheduled")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("source_quality_scheduler.error")
