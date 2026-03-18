"""Unit tests for freqpred.ingestion.quota."""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.ingestion.quota import get_daily_count, increment_daily_count

_SERVICE = "newsapi"
_DATE = date(2026, 3, 18)


def _make_session(scalar_result=None) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.fetchone.return_value = scalar_result
    session.execute.return_value = result
    return session


@pytest.mark.asyncio
async def test_get_daily_count_returns_zero_when_no_row():
    session = _make_session(scalar_result=None)
    count = await get_daily_count(session, _SERVICE, _DATE)
    assert count == 0


@pytest.mark.asyncio
async def test_get_daily_count_returns_row_value():
    session = _make_session(scalar_result=(47,))
    count = await get_daily_count(session, _SERVICE, _DATE)
    assert count == 47


@pytest.mark.asyncio
async def test_increment_daily_count_executes_upsert():
    session = AsyncMock()
    await increment_daily_count(session, _SERVICE, _DATE)
    session.execute.assert_called_once()
    sql = str(session.execute.call_args[0][0])
    assert "ON CONFLICT" in sql
    assert "request_count" in sql
