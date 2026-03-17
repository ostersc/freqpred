"""Retrieval hash check and signal deduplication."""
from __future__ import annotations

import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.signal.models import SignalRow

log = structlog.get_logger(__name__)


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
