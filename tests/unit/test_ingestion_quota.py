"""Unit tests for freqpred.ingestion.quota."""
from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.ingestion.quota import current_window, get_window_count, increment_window_count

_SERVICE = "newsapi"
_DATE = date(2026, 3, 18)
_SLOT = 0


def _make_session(scalar_result=None) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.fetchone.return_value = scalar_result
    session.execute.return_value = result
    return session


@pytest.mark.asyncio
async def test_get_window_count_returns_zero_when_no_row():
    session = _make_session(scalar_result=None)
    count = await get_window_count(session, _SERVICE, _DATE, _SLOT)
    assert count == 0


@pytest.mark.asyncio
async def test_get_window_count_returns_row_value():
    session = _make_session(scalar_result=(47,))
    count = await get_window_count(session, _SERVICE, _DATE, _SLOT)
    assert count == 47


@pytest.mark.asyncio
async def test_increment_window_count_executes_upsert():
    session = AsyncMock()
    await increment_window_count(session, _SERVICE, _DATE, _SLOT)
    session.execute.assert_called_once()
    sql = str(session.execute.call_args[0][0])
    assert "ON CONFLICT" in sql
    assert "request_count" in sql


def test_current_window_midnight_slot():
    now = datetime(2026, 3, 18, 3, 0, 0, tzinfo=UTC)
    d, slot = current_window(now)
    assert d == date(2026, 3, 18)
    assert slot == 0


def test_current_window_noon_slot():
    now = datetime(2026, 3, 18, 14, 0, 0, tzinfo=UTC)
    d, slot = current_window(now)
    assert d == date(2026, 3, 18)
    assert slot == 1


def test_current_window_boundary_noon():
    now = datetime(2026, 3, 18, 12, 0, 0, tzinfo=UTC)
    _, slot = current_window(now)
    assert slot == 1
