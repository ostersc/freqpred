"""Unit tests for P&L time-series functions (ledger.py) and GET /api/pnl/time-series."""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# Register all ORM models so relationship forward-refs resolve.
import freqpred.alerts.models  # noqa: F401
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.metrics.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.runtime.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
import freqpred.strategy.models  # noqa: F401
from freqpred.dashboard.api.app import create_app
from freqpred.dashboard.api.routes import get_db
from freqpred.trading.ledger import get_llm_spend_time_series, get_pnl_time_series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_result(rows: list) -> MagicMock:
    r = MagicMock()
    r.all.return_value = rows
    return r


def _make_app(session: object, bankroll_usd: float = 1000.0) -> object:
    sf = MagicMock()
    app = create_app(session_factory=sf, daily_cap_usd=10.0, bankroll_usd=bankroll_usd)

    async def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app


def _mode_result(mode: str = "paper") -> MagicMock:
    row = MagicMock()
    row.mode = mode
    r = MagicMock()
    r.scalar_one_or_none.return_value = row
    return r


# ---------------------------------------------------------------------------
# get_pnl_time_series — unit tests (no DB, no HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pnl_empty_returns_empty_list() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([]))
    result = await get_pnl_time_series(session, mode="paper")
    assert result == []


@pytest.mark.asyncio
async def test_pnl_single_day_single_trade() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([
        (date(2026, 1, 1), 10.0, 1),
    ]))
    result = await get_pnl_time_series(session, mode="paper")
    assert len(result) == 1
    assert result[0]["date"] == "2026-01-01"
    assert result[0]["daily_pnl"] == pytest.approx(10.0)
    assert result[0]["cumulative_pnl"] == pytest.approx(10.0)
    assert result[0]["trade_count"] == 1


@pytest.mark.asyncio
async def test_pnl_cumulative_accumulates_correctly() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([
        (date(2026, 1, 1), 5.0, 1),
        (date(2026, 1, 2), -3.0, 2),
    ]))
    result = await get_pnl_time_series(session, mode="paper")
    assert len(result) == 2
    assert result[0]["cumulative_pnl"] == pytest.approx(5.0)
    assert result[1]["cumulative_pnl"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_pnl_negative_day_reduces_cumulative() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([
        (date(2026, 1, 1), 10.0, 2),
        (date(2026, 1, 2), -15.0, 1),
    ]))
    result = await get_pnl_time_series(session, mode="paper")
    assert result[1]["cumulative_pnl"] == pytest.approx(-5.0)
    assert result[1]["daily_pnl"] == pytest.approx(-15.0)


@pytest.mark.asyncio
async def test_pnl_open_positions_excluded() -> None:
    """The query includes status='closed' — open positions never appear in results.
    The mock returns only closed-position rows to simulate this filter."""
    session = MagicMock()
    # Only one closed row (open rows would never be returned by the filtered query)
    session.execute = AsyncMock(return_value=_all_result([
        (date(2026, 1, 1), 8.0, 1),
    ]))
    result = await get_pnl_time_series(session, mode="paper")
    # Only closed position pnl appears
    assert len(result) == 1
    assert result[0]["trade_count"] == 1


@pytest.mark.asyncio
async def test_pnl_lookback_days_applies_cutoff() -> None:
    """lookback_days adds a WHERE filter; here the mock simulates the DB returning
    only in-window rows."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([
        (date(2026, 5, 10), 5.0, 1),
        (date(2026, 5, 11), 3.0, 1),
    ]))
    result = await get_pnl_time_series(session, mode="paper", lookback_days=7)
    assert len(result) == 2
    assert result[-1]["cumulative_pnl"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# get_llm_spend_time_series — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_cumulative_correct() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([
        (date(2026, 1, 1), 0.01),
        (date(2026, 1, 2), 0.02),
    ]))
    result = await get_llm_spend_time_series(session)
    assert len(result) == 2
    assert result[0]["cumulative_spend"] == pytest.approx(0.01, abs=1e-6)
    assert result[1]["cumulative_spend"] == pytest.approx(0.03, abs=1e-6)


@pytest.mark.asyncio
async def test_llm_no_lookback_returns_all() -> None:
    """With lookback_days=None the function issues no date cutoff — mock returns all rows."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=_all_result([
        (date(2025, 12, 1), 0.005),
        (date(2026, 1, 1), 0.01),
    ]))
    result = await get_llm_spend_time_series(session, lookback_days=None)
    assert len(result) == 2
    assert result[1]["cumulative_spend"] == pytest.approx(0.015, abs=1e-6)


# ---------------------------------------------------------------------------
# GET /api/pnl/time-series — endpoint tests via FastAPI TestClient
# ---------------------------------------------------------------------------


def _build_endpoint_session(
    pnl_rows: list,
    llm_rows: list,
    available_rows: list | None = None,
) -> MagicMock:
    """Build a mock session whose execute() cycles through:
    1. _get_mode (scalar_one_or_none)
    2. get_pnl_time_series execute → .all()
    3. get_llm_spend_time_series execute → .all()
    4-10. 7× distinct available_* queries → .all() each
    """
    if available_rows is None:
        available_rows = []

    call_idx = 0
    results = [
        _mode_result("paper"),      # 1. _get_mode
        _all_result(pnl_rows),      # 2. pnl time series
        _all_result(llm_rows),      # 3. llm spend time series
        _all_result(available_rows),  # 4. available_strategies
        _all_result(available_rows),  # 5. available_directions
        _all_result(available_rows),  # 6. available_market_ids
        _all_result(available_rows),  # 7. available_models (join)
        _all_result(available_rows),  # 8. available_prompt_versions (join)
        _all_result(available_rows),  # 9. available_categories (join)
        _all_result(available_rows),  # 10. available_series_tickers (join)
    ]

    async def _execute(stmt: object) -> MagicMock:
        nonlocal call_idx
        r = results[call_idx] if call_idx < len(results) else _all_result([])
        call_idx += 1
        return r

    session = MagicMock()
    session.execute = _execute
    return session


@pytest.mark.asyncio
async def test_endpoint_returns_200() -> None:
    pnl_rows = [(date(2026, 1, 1), 10.0, 2)]
    llm_rows = [(date(2026, 1, 1), 0.01)]
    session = _build_endpoint_session(pnl_rows, llm_rows)
    app = _make_app(session, bankroll_usd=500.0)

    with TestClient(app) as client:
        resp = client.get("/api/pnl/time-series")

    assert resp.status_code == 200
    body = resp.json()
    assert "pnl_series" in body
    assert "llm_series" in body
    assert "initial_bankroll" in body
    assert len(body["pnl_series"]) == 1
    assert body["pnl_series"][0]["daily_pnl"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_endpoint_no_positions() -> None:
    session = _build_endpoint_session(pnl_rows=[], llm_rows=[])
    app = _make_app(session, bankroll_usd=1000.0)

    with TestClient(app) as client:
        resp = client.get("/api/pnl/time-series")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_trades"] == 0
    assert body["all_time_pnl"] == pytest.approx(0.0)
    assert body["pnl_series"] == []
    assert body["llm_series"] == []


@pytest.mark.asyncio
async def test_endpoint_available_filters_stable() -> None:
    """available_* arrays come from the unfiltered set regardless of active filters."""
    strategy_row = MagicMock()
    strategy_row.__iter__ = lambda self: iter(["TestStrategy"])

    pnl_rows = [(date(2026, 1, 1), 5.0, 1)]
    llm_rows = []

    call_idx = 0
    mode_r = _mode_result("paper")
    pnl_r = _all_result(pnl_rows)
    llm_r = _all_result(llm_rows)

    results = [mode_r, pnl_r, llm_r] + [_all_result([("TestStrategy",)])] * 7

    async def _execute(stmt: object) -> MagicMock:
        nonlocal call_idx
        r = results[call_idx] if call_idx < len(results) else _all_result([])
        call_idx += 1
        return r

    session = MagicMock()
    session.execute = _execute
    app = _make_app(session)

    with TestClient(app) as client:
        # Pass a strategy filter — available_strategies should still include all strategies
        resp = client.get("/api/pnl/time-series?strategy=TestStrategy")

    assert resp.status_code == 200
    body = resp.json()
    assert "TestStrategy" in body["available_strategies"]


@pytest.mark.asyncio
async def test_endpoint_initial_bankroll_included() -> None:
    session = _build_endpoint_session(pnl_rows=[], llm_rows=[])
    app = _make_app(session, bankroll_usd=2500.0)

    with TestClient(app) as client:
        resp = client.get("/api/pnl/time-series")

    assert resp.status_code == 200
    body = resp.json()
    assert body["initial_bankroll"] == pytest.approx(2500.0)
