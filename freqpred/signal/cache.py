"""Retrieval hash check and signal deduplication."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.models import FactbasePhraseRow
from freqpred.signal.models import SignalRow

log = structlog.get_logger(__name__)

# Cooldown applied when the last scheduled signal had low confidence.
# Prevents burning Sonnet budget re-analyzing markets the LLM already
# said "not confident" on just because new docs arrived.
_LOW_CONF_THRESHOLD = 0.60       # below this → short cooldown
_LOW_CONF_COOLDOWN_HOURS = 4.0
_VERY_LOW_CONF_THRESHOLD = 0.50  # below this → long cooldown
_VERY_LOW_CONF_COOLDOWN_HOURS = 12.0


async def should_skip(
    session: AsyncSession,
    current_signal_id: str | None,
    new_hash: str,
) -> bool:
    """Return True if *new_hash* matches the current signal's retrieval hash.

    This means no new evidence has arrived since the last analysis — the
    caller should skip the LLM call and return without creating a new Signal.

    Returns False when:
    - *current_signal_id* is None (no prior signal for this market)
    - The current signal's hash differs from *new_hash* (new evidence exists)
    - The referenced signal row is not found in the DB
    """
    if current_signal_id is None:
        return False

    try:
        signal_uuid = uuid.UUID(str(current_signal_id))
    except (ValueError, AttributeError):
        log.warning("cache.invalid_signal_id", current_signal_id=current_signal_id)
        return False

    result = await session.execute(
        select(SignalRow.retrieval_hash).where(SignalRow.id == signal_uuid)
    )
    current_hash = result.scalar_one_or_none()
    if current_hash is None:
        log.warning("cache.signal_not_found", signal_id=str(current_signal_id))
        return False

    match = current_hash == new_hash
    log.debug(
        "cache.hash_check",
        signal_id=str(current_signal_id),
        match=match,
    )
    return match


def scheduled_skip_decision(
    last_hash: str,
    last_created_at: datetime,
    new_hash: str,
    max_interval_hours: float,
    fb_last_fetched_at: datetime | None,
    now: datetime,
) -> bool:
    """Pure decision core of :func:`should_skip_scheduled`.

    Skips only when ALL of the following hold:
    - The last scheduled signal is less than ``max_interval_hours`` old
      (temporal-reasoning rerun not yet due).
    - The retrieval hash hasn't changed (same RAG doc set).
    - FactBase data hasn't been refreshed since the last scheduled signal.

    Takes the prior-signal state and clock explicitly so the replay harness
    and tests can evaluate the decision without a DB session.
    """
    age_hours = (now - last_created_at).total_seconds() / 3600

    if age_hours >= max_interval_hours:
        return False

    if last_hash != new_hash:
        return False

    if fb_last_fetched_at is not None and fb_last_fetched_at > last_created_at:
        return False

    return True


async def should_skip_scheduled(
    session: AsyncSession,
    market_id: str,
    new_hash: str,
    max_interval_hours: float = 24.0,
    _now: datetime | None = None,
) -> bool:
    """Return True if a scheduled LLM call should be skipped.

    Gathers the last scheduled signal and FactBase freshness from the DB, then
    delegates to :func:`scheduled_skip_decision`. Returns False (→ run the LLM)
    the moment any one condition fails, so the pipeline reacts immediately to
    new evidence or new FactBase data, and still re-runs at least once every
    ``max_interval_hours`` for temporal reasoning.

    ``_now`` pins the clock for deterministic tests/replay; defaults to the
    real wall-clock.
    """
    result = await session.execute(
        select(SignalRow.retrieval_hash, SignalRow.created_at)
        .where(SignalRow.market_id == market_id, SignalRow.trigger == "scheduled")
        .order_by(SignalRow.created_at.desc())
        .limit(1)
    )
    row = result.one_or_none()
    if row is None:
        return False

    last_hash, last_created_at = row

    fb_result = await session.execute(
        select(FactbasePhraseRow.last_fetched_at)
        .where(FactbasePhraseRow.market_id == market_id)
    )
    fb_row = fb_result.one_or_none()
    fb_last_fetched_at = fb_row[0] if fb_row is not None else None

    return scheduled_skip_decision(
        last_hash,
        last_created_at,
        new_hash,
        max_interval_hours,
        fb_last_fetched_at,
        _now if _now is not None else datetime.now(tz=UTC),
    )


def cooldown_decision(confidence: float, created_at: datetime, now: datetime) -> float:
    """Pure decision core of :func:`scheduled_cooldown_remaining`.

    Cooldown tiers:
      confidence < 0.50 → 12-hour cooldown
      confidence < 0.60 → 4-hour cooldown
      confidence ≥ 0.60 → no cooldown
    """
    age_hours = (now - created_at).total_seconds() / 3600

    if confidence < _VERY_LOW_CONF_THRESHOLD:
        remaining = _VERY_LOW_CONF_COOLDOWN_HOURS - age_hours
    elif confidence < _LOW_CONF_THRESHOLD:
        remaining = _LOW_CONF_COOLDOWN_HOURS - age_hours
    else:
        return 0.0

    return max(0.0, remaining)


async def scheduled_cooldown_remaining(
    session: AsyncSession,
    market_id: str,
    _now: datetime | None = None,
) -> float:
    """Return hours remaining in low-confidence cooldown for a market.

    Looks up the most recent *scheduled* signal for *market_id* and delegates
    to :func:`cooldown_decision`. If that signal's confidence was below the
    low-confidence threshold and it was created recently, returns the hours
    left to wait before re-analyzing. Returns 0.0 when no cooldown is active
    (safe to call the LLM).

    ``_now`` pins the clock for deterministic tests/replay; defaults to the
    real wall-clock.
    """
    result = await session.execute(
        select(SignalRow.confidence, SignalRow.created_at)
        .where(SignalRow.market_id == market_id, SignalRow.trigger == "scheduled")
        .order_by(SignalRow.created_at.desc())
        .limit(1)
    )
    row = result.one_or_none()
    if row is None:
        return 0.0

    confidence, created_at = row
    return cooldown_decision(
        confidence, created_at, _now if _now is not None else datetime.now(tz=UTC)
    )
