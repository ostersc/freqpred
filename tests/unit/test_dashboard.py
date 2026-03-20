"""Unit tests for the dashboard API endpoints.

Uses FastAPI's TestClient with dependency_overrides to inject a mock session —
no real DB or Redis required.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# Register all ORM models so relationship forward-refs resolve.
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models        # noqa: F401
import freqpred.markets.models    # noqa: F401
import freqpred.rag.models        # noqa: F401
import freqpred.signal.models     # noqa: F401

from freqpred.dashboard.api.app import create_app
from freqpred.dashboard.api.routes import get_db


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_app(session: AsyncMock, daily_cap: float = 10.0) -> object:
    """Create app with the real session factory replaced by a mock."""
    sf = MagicMock()
    app = create_app(session_factory=sf, daily_cap_usd=daily_cap)

    async def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app


def _execute_side_effects(*results: MagicMock) -> AsyncMock:
    """Return an AsyncMock whose side_effect cycles through the provided results."""
    return AsyncMock(side_effect=list(results))


def _scalar_result(value: object) -> MagicMock:
    """Mock result that supports .scalar_one() → value."""
    r = MagicMock()
    r.scalar_one.return_value = value
    return r


def _scalars_result(rows: list) -> MagicMock:
    """Mock result that supports .scalars().all() → rows."""
    r = MagicMock()
    r.scalars.return_value.all.return_value = rows
    return r


def _all_result(rows: list) -> MagicMock:
    """Mock result that supports .all() → rows."""
    r = MagicMock()
    r.all.return_value = rows
    return r


def _make_signal_row(**kw) -> MagicMock:
    row = MagicMock()
    row.id = kw.get("id", uuid.uuid4())
    row.market_id = kw.get("market_id", "MARKET-1")
    row.estimated_probability = kw.get("estimated_probability", 0.65)
    row.confidence = kw.get("confidence", 0.80)
    row.edge = kw.get("edge", 0.05)
    row.market_mid_at_signal = kw.get("market_mid_at_signal", 0.60)
    row.direction = kw.get("direction", "YES")
    row.reasoning = kw.get("reasoning", "test reasoning")
    row.sources = kw.get("sources", [])
    row.retrieval_hash = kw.get("retrieval_hash", "abc123")
    row.model_used = kw.get("model_used", "claude-sonnet-4-6")
    row.prompt_version = kw.get("prompt_version", "signal-v1")
    row.trigger = kw.get("trigger", "scheduled")
    row.created_at = kw.get("created_at", datetime(2026, 1, 1, tzinfo=UTC))
    row.social_sentiment_summary = kw.get("social_sentiment_summary", None)
    return row


def _make_position_row(**kw) -> MagicMock:
    row = MagicMock()
    row.id = kw.get("id", uuid.uuid4())
    row.market_id = kw.get("market_id", "MARKET-1")
    row.signal_id = kw.get("signal_id", uuid.uuid4())
    row.strategy_name = kw.get("strategy_name", "ConservativeDefault")
    row.strategy_version = kw.get("strategy_version", "1.0")
    row.signal_confidence = kw.get("signal_confidence", 0.80)
    row.signal_edge = kw.get("signal_edge", 0.05)
    row.signal_estimated_prob = kw.get("signal_estimated_prob", 0.65)
    row.direction = kw.get("direction", "YES")
    row.contracts = kw.get("contracts", 10)
    row.entry_price = kw.get("entry_price", 0.60)
    row.entry_time = kw.get("entry_time", datetime(2026, 1, 1, tzinfo=UTC))
    row.mode = kw.get("mode", "paper")
    row.status = kw.get("status", "open")
    row.exit_price = kw.get("exit_price", None)
    row.exit_time = kw.get("exit_time", None)
    row.resolution = kw.get("resolution", None)
    row.pnl = kw.get("pnl", None)
    row.pnl_pct = kw.get("pnl_pct", None)
    row.created_at = kw.get("created_at", datetime(2026, 1, 1, tzinfo=UTC))
    return row


# ---------------------------------------------------------------------------
# /api/signals
# ---------------------------------------------------------------------------


def test_signals_endpoint_returns_paginated_list() -> None:
    row1 = _make_signal_row()
    row2 = _make_signal_row()

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(2),          # count query
        _scalars_result([row1, row2]),  # data query
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/signals?limit=20&offset=0")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["total"] == 2
    assert data["limit"] == 20
    assert data["offset"] == 0
    # Required fields present
    item = data["items"][0]
    assert "estimated_probability" in item
    assert "edge" in item
    assert "direction" in item
    assert "created_at" in item


def test_signals_endpoint_filters_by_market_id() -> None:
    row = _make_signal_row(market_id="MKTX-42")

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(1),
        _scalars_result([row]),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/signals?market_id=MKTX-42")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["market_id"] == "MKTX-42"


def test_signals_endpoint_unknown_id_returns_404() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_scalar_result(None))  # scalar_one_or_none

    # Patch scalar_one_or_none to return None
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result_mock)

    client = TestClient(_make_app(session))
    resp = client.get(f"/api/signals/{uuid.uuid4()}")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Signal not found"


def test_signals_invalid_uuid_returns_404() -> None:
    session = AsyncMock()
    client = TestClient(_make_app(session))
    resp = client.get("/api/signals/not-a-uuid")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/positions
# ---------------------------------------------------------------------------


def test_positions_endpoint_filters_by_status() -> None:
    open_pos = _make_position_row(status="open")

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(1),
        _scalars_result([open_pos]),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/positions?status=open")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["status"] == "open"


def test_positions_endpoint_all_statuses() -> None:
    rows = [
        _make_position_row(status="open"),
        _make_position_row(status="closed"),
    ]

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(2),
        _scalars_result(rows),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/positions?status=all")

    assert resp.status_code == 200
    assert resp.json()["total"] == 2


def test_positions_unknown_id_returns_404() -> None:
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    client = TestClient(_make_app(session))
    resp = client.get(f"/api/positions/{uuid.uuid4()}")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Position not found"


# ---------------------------------------------------------------------------
# /api/calibration
# ---------------------------------------------------------------------------


def test_calibration_endpoint_returns_brier_score() -> None:
    # compute_calibration does its own session.execute calls inside.
    # We mock the session to return a single resolved sample.
    rows = [(0.7, 0.5, 1)]  # (estimated_prob, mid, resolution)

    result_mock = MagicMock()
    result_mock.all.return_value = rows

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    client = TestClient(_make_app(session))
    resp = client.get("/api/calibration")

    assert resp.status_code == 200
    data = resp.json()
    assert "brier_score" in data
    assert "n_samples" in data
    assert "buckets" in data
    assert data["n_samples"] == 1
    assert len(data["buckets"]) == 10


def test_calibration_endpoint_no_samples() -> None:
    result_mock = MagicMock()
    result_mock.all.return_value = []

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    client = TestClient(_make_app(session))
    resp = client.get("/api/calibration")

    assert resp.status_code == 200
    data = resp.json()
    assert data["n_samples"] == 0


# ---------------------------------------------------------------------------
# /api/llm/cost
# ---------------------------------------------------------------------------


def test_llm_cost_endpoint_returns_daily_spend() -> None:
    # Calls: get_daily_spend_usd (1 execute), weekly sum (1 execute), by_type (1 execute)
    today_result = _scalar_result(0.42)          # today_usd
    weekly_result = _scalar_result(2.10)         # weekly_usd
    by_type_result = _all_result([("market_analysis", 0.42)])  # by_query_type

    session = AsyncMock()
    session.execute = _execute_side_effects(
        today_result, weekly_result, by_type_result
    )

    client = TestClient(_make_app(session, daily_cap=10.0))
    resp = client.get("/api/llm/cost")

    assert resp.status_code == 200
    data = resp.json()
    assert data["today_usd"] == pytest.approx(0.42)
    assert data["daily_cap_usd"] == pytest.approx(10.0)
    assert "pct_used" in data
    assert "by_query_type" in data
    assert data["pct_used"] == pytest.approx(4.2)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_200_when_db_reachable() -> None:
    # health calls: open positions count, get_daily_spend_usd
    open_count_result = _scalar_result(3)
    spend_result = _scalar_result(1.50)

    session = AsyncMock()
    session.execute = _execute_side_effects(open_count_result, spend_result)

    client = TestClient(_make_app(session))
    resp = client.get("/api/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db"] == "connected"
    assert data["open_positions"] == 3


def test_health_endpoint_db_error_returns_degraded() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("db connection lost"))

    client = TestClient(_make_app(session))
    resp = client.get("/api/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["db"] == "error"
