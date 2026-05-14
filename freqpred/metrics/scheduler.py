"""Background scheduler for source-quality snapshots and series history."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.metrics.calibration import refresh_source_quality_scores
from freqpred.metrics.models import SourceQualityScoreRow

if TYPE_CHECKING:
    from freqpred.markets.kalshi import KalshiClient

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
    kalshi_client: "KalshiClient | None" = None,
    series_lookback_days: int = 7,
    telemetry: object | None = None,
) -> None:
    """Refresh source-quality snapshots and series history once per day."""
    tz = ZoneInfo(refresh_timezone)
    log.info(
        "source_quality_scheduler.started",
        refresh_time=refresh_time,
        refresh_timezone=refresh_timezone,
        lookback_days=lookback_days,
    )

    async def _run_refresh(reason: str) -> None:
        async with session_factory() as session:
            # Source quality — its own try/except so series history still runs on failure
            sq_error: Exception | None = None
            try:
                rows_written = await refresh_source_quality_scores_if_due(
                    session,
                    lookback_days=lookback_days,
                )
                log.info(
                    "source_quality_scheduler.refreshed",
                    reason=reason,
                    rows_written=rows_written,
                )
                if telemetry is not None:
                    from freqpred.runtime.telemetry import SERVICE_SOURCE_QUALITY_SCHEDULER  # noqa: PLC0415

                    await telemetry.mark_success(
                        SERVICE_SOURCE_QUALITY_SCHEDULER,
                        details={"reason": reason, "rows_written": rows_written},
                    )
            except Exception as exc:
                sq_error = exc
                log.exception("source_quality_scheduler.error", reason=reason)
                if telemetry is not None:
                    from freqpred.runtime.telemetry import SERVICE_SOURCE_QUALITY_SCHEDULER  # noqa: PLC0415

                    await telemetry.mark_error(SERVICE_SOURCE_QUALITY_SCHEDULER, str(exc))

            # Series history — independent heartbeat
            if kalshi_client is not None:
                try:
                    from freqpred.metrics.series_history import refresh_series_history  # noqa: PLC0415
                    from freqpred.runtime.telemetry import SERVICE_SERIES_HISTORY_SCHEDULER  # noqa: PLC0415

                    series_rows = await refresh_series_history(
                        session,
                        kalshi_client,
                        lookback_days=series_lookback_days,
                    )
                    log.info(
                        "series_history.scheduler.refreshed",
                        reason=reason,
                        rows_upserted=series_rows,
                    )
                    if telemetry is not None:
                        await telemetry.mark_success(
                            SERVICE_SERIES_HISTORY_SCHEDULER,
                            details={"reason": reason, "rows_upserted": series_rows},
                        )
                except Exception as exc:
                    log.exception("series_history.scheduler.error", reason=reason)
                    if telemetry is not None:
                        from freqpred.runtime.telemetry import SERVICE_SERIES_HISTORY_SCHEDULER  # noqa: PLC0415

                        await telemetry.mark_error(SERVICE_SERIES_HISTORY_SCHEDULER, str(exc))

            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise

            # Re-raise source quality error after commit attempt so caller sees it
            if sq_error is not None:
                raise sq_error

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
            log.exception("source_quality_scheduler.cycle_error")
