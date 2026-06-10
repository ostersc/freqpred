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
  trip 3 → skip 4 cycles (skip_cycles_next becomes 8 — the cap)
  trip 4+ → skip 8 cycles, logged at error level
  success → reset to skip_cycles_next = 1

The cap is deliberately low (8 cycles = 4 hours at the default 30-min cadence).
A service that keeps tripping at the cap is persistently failing, not
transiently rate-limited — that needs surfacing, not ever-longer silence.
GDELT once escalated to 32-cycle skips (16 hours) and the source was
effectively dead for days before anyone noticed.
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.models import FetcherRateLimitRow

log = structlog.get_logger(__name__)

_MAX_SKIP_CYCLES = 8


async def tick_and_load(
    session: AsyncSession,
    services: frozenset[str] | None = None,
) -> dict[str, bool]:
    """Decrement backed-off services by 1 cycle, then return their state.

    Called once at the start of each ingestion cycle. Returns a dict mapping
    service name → is_backed_off (True means skip this cycle).

    Only services that have a row in the table are included in the result.
    Services with no row are not backed off (default / never tripped state).

    Args:
        services: If provided, only tick and return state for these service
                  names. Pass the set of services owned by the calling
                  scheduler so that multiple schedulers running at different
                  intervals do not drain each other's backoff counters.
    """
    # Decrement services that are still waiting (optionally scoped).
    tick_stmt = (
        update(FetcherRateLimitRow)
        .where(FetcherRateLimitRow.skip_cycles_remaining > 0)
        .values(skip_cycles_remaining=FetcherRateLimitRow.skip_cycles_remaining - 1)
    )
    if services is not None:
        tick_stmt = tick_stmt.where(FetcherRateLimitRow.service.in_(services))
    await session.execute(tick_stmt)
    await session.flush()

    load_stmt = select(
        FetcherRateLimitRow.service, FetcherRateLimitRow.skip_cycles_remaining
    )
    if services is not None:
        load_stmt = load_stmt.where(FetcherRateLimitRow.service.in_(services))
    result = await session.execute(load_stmt)
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

    if skip >= _MAX_SKIP_CYCLES:
        # Repeated trips at the cap mean the service is persistently failing,
        # not transiently rate-limited.
        log.error(
            "fetcher_backoff.at_cap",
            service=service,
            skip_cycles=skip,
            max_skip_cycles=_MAX_SKIP_CYCLES,
        )
    else:
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
