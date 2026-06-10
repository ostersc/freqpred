"""Unit tests for freqpred/ingestion/backoff.py — escalation cap behavior."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.ingestion.backoff import _MAX_SKIP_CYCLES, record_rate_limit


def _session_with_row(row) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = row
    session.execute.return_value = result
    session.add = MagicMock()
    return session


def _row(skip_cycles_next: int) -> MagicMock:
    row = MagicMock()
    row.skip_cycles_next = skip_cycles_next
    row.skip_cycles_remaining = 0
    return row


@pytest.mark.asyncio
async def test_first_trip_skips_one_cycle() -> None:
    session = _session_with_row(None)

    skip = await record_rate_limit(session, "gdelt")

    assert skip == 1
    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_backoff_doubles_until_cap() -> None:
    row = _row(skip_cycles_next=4)
    session = _session_with_row(row)

    skip = await record_rate_limit(session, "gdelt")

    assert skip == 4
    assert row.skip_cycles_next == 8


@pytest.mark.asyncio
async def test_backoff_never_exceeds_cap() -> None:
    """skip_cycles_next must clamp at _MAX_SKIP_CYCLES — repeated trips do not
    escalate further. GDELT once reached 32-cycle (16h) skips this way."""
    row = _row(skip_cycles_next=_MAX_SKIP_CYCLES)
    session = _session_with_row(row)

    skip = await record_rate_limit(session, "gdelt")

    assert skip == _MAX_SKIP_CYCLES
    assert row.skip_cycles_next == _MAX_SKIP_CYCLES


@pytest.mark.asyncio
async def test_legacy_state_above_cap_is_clamped() -> None:
    """Rows written before the cap was lowered (e.g. skip_cycles_next=32) must
    clamp down to the cap on the next trip, not perpetuate the old value."""
    row = _row(skip_cycles_next=32)
    session = _session_with_row(row)

    skip = await record_rate_limit(session, "gdelt")

    assert skip == _MAX_SKIP_CYCLES
    assert row.skip_cycles_next == _MAX_SKIP_CYCLES


def test_cap_is_eight() -> None:
    """4 hours at the default 30-min cadence. If this changes, re-check the
    staleness thresholds in build_freshness_specs (fetcher specs assume a
    backed-off fetcher still retries well within the 24h stale window)."""
    assert _MAX_SKIP_CYCLES == 8
