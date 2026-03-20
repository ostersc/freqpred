"""Persistent exponential backoff for ingestion fetchers.

State is stored in the ``fetcher_rate_limits`` table so it survives restarts.

Backoff logic:
  - On rate-limit hit: skip_cycles_remaining = skip_cycles_next;
    skip_cycles_next doubles (capped at _MAX_SKIP_CYCLES).
  - At cycle start: all skip_cycles_remaining > 0 are decremented by 1.
  - On successful call: skip_cycles_next resets to 1, skip_cycles_remaining to 0.

Example progression for a service that keeps tripping:
  trip 1 → skip 1 cycle  (skip_cycles_next becomes 2)
  trip 2 → skip 2 cycles (skip_cycles_next becomes 4)
  trip 3 → skip 4 cycles (skip_cycles_next becomes 8)
  ...
  success → reset to skip_cycles_next = 1
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.models import FetcherRateLimitRow

log = structlog.get_logger(__name__)

_MAX_SKIP_CYCLES = 32


async def tick_and_load(session: AsyncSession) -> dict[str, bool]:
    """Decrement all backed-off services by 1 cycle, then return their state.

    Called once at the start of each ingestion cycle. Returns a dict mapping
    service name → is_backed_off (True means skip this cycle).

    Only services that have a row in the table are included in the result.
    Services with no row are not backed off (default / never tripped state).
    """
    # Decrement all services that are still waiting.
    await session.execute(
        update(FetcherRateLimitRow)
        .where(FetcherRateLimitRow.skip_cycles_remaining > 0)
        .values(skip_cycles_remaining=FetcherRateLimitRow.skip_cycles_remaining - 1)
    )
    await session.flush()

    result = await session.execute(
        select(FetcherRateLimitRow.service, FetcherRateLimitRow.skip_cycles_remaining)
    )
    state: dict[str, bool] = {
        service: (remaining > 0)
        for service, remaining in result.all()
    }

    for service, backed_off in state.items():
        if backed_off:
            log.info(
                "fetcher_backoff.skipping_cycle",
                service=service,
            )

    return state


async def record_rate_limit(session: AsyncSession, service: str) -> int:
    """Record a rate-limit hit for *service*.

    Sets skip_cycles_remaining = skip_cycles_next and doubles skip_cycles_next
    (capped at _MAX_SKIP_CYCLES). Commits. Returns skip_cycles_remaining so
    the caller can log the value.
    """
    result = await session.execute(
        select(FetcherRateLimitRow).where(FetcherRateLimitRow.service == service)
    )
    row = result.scalar_one_or_none()

    if row is None:
        skip = 1
        next_skip = 2
        session.add(FetcherRateLimitRow(
            service=service,
            skip_cycles_remaining=skip,
            skip_cycles_next=next_skip,
            tripped_at=datetime.now(UTC),
        ))
    else:
        skip = min(row.skip_cycles_next, _MAX_SKIP_CYCLES)
        next_skip = min(row.skip_cycles_next * 2, _MAX_SKIP_CYCLES)
        row.skip_cycles_remaining = skip
        row.skip_cycles_next = next_skip
        row.tripped_at = datetime.now(UTC)

    await session.commit()

    log.warning(
        "fetcher_backoff.rate_limit_recorded",
        service=service,
        skip_cycles=skip,
        next_skip_cycles=next_skip,
    )
    return skip


async def record_success(session: AsyncSession, service: str) -> None:
    """Record a successful call for *service*, resetting backoff to initial state.

    No-op if the service has no row (never tripped) or is already at defaults.
    """
    result = await session.execute(
        select(FetcherRateLimitRow).where(FetcherRateLimitRow.service == service)
    )
    row = result.scalar_one_or_none()

    if row is None or (row.skip_cycles_next == 1 and row.skip_cycles_remaining == 0):
        return

    row.skip_cycles_remaining = 0
    row.skip_cycles_next = 1
    row.last_success_at = datetime.now(UTC)
    await session.commit()

    log.info("fetcher_backoff.reset_on_success", service=service)
