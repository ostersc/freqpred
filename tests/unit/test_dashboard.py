"""Unit tests for the dashboard API endpoints.

Uses FastAPI's TestClient with dependency_overrides to inject a mock session —
no real DB or Redis required.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Register all ORM models so relationship forward-refs resolve.
import freqpred.alerts.models     # noqa: F401
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models        # noqa: F401
import freqpred.markets.models    # noqa: F401
import freqpred.metrics.models    # noqa: F401
import freqpred.rag.models        # noqa: F401
import freqpred.runtime.models    # noqa: F401
import freqpred.signal.models     # noqa: F401
import freqpred.strategy.models   # noqa: F401

from freqpred.dashboard.api.app import create_app
from freqpred.dashboard.api.routes import get_db
from freqpred.markets.kalshi import KalshiAPIError
from freqpred.runtime.telemetry import RuntimeTelemetry, build_freshness_specs
from freqpred.strategy.config import StrategyConfig
from freqpred.trading.order_manager import PositionNotFoundError, PositionNotOpenError


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_strategy_config(**overrides: object) -> StrategyConfig:
    defaults = dict(
        name="TestStrategy",
        min_edge=0.15,
        min_confidence=0.70,
        max_exposure_per_market=0.10,
        kelly_fraction=0.25,
        categories=["politics"],
        min_volume_24h=500.0,
        max_days_to_close=30.0,
        min_days_to_close=1.0,
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)  # type: ignore[arg-type]


def _make_app(
    session: AsyncMock,
    daily_cap: float = 10.0,
    risk_config: object | None = None,
    bankroll_usd: float = 1000.0,
    signal_pipeline: object | None = None,
    order_manager: object | None = None,
    runtime_telemetry: object | None = None,
) -> object:
    """Create app with the real session factory replaced by a mock."""
    sf = MagicMock()
    app = create_app(
        session_factory=sf,
        daily_cap_usd=daily_cap,
        risk_config=risk_config,
        bankroll_usd=bankroll_usd,
        signal_pipeline=signal_pipeline,
        order_manager=order_manager,
        runtime_telemetry=runtime_telemetry,
    )

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


def _mode_result(mode: str = "paper") -> MagicMock:
    """Mock run_state row result for _get_mode (scalar_one_or_none → row with .mode)."""
    row = MagicMock()
    row.mode = mode
    r = MagicMock()
    r.scalar_one_or_none.return_value = row
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


def _make_signal_assessment_row(**kw) -> MagicMock:
    row = MagicMock()
    row.signal_id = kw.get("signal_id", uuid.uuid4())
    row.trust_score = kw.get("trust_score", 0.64)
    row.size_multiplier = kw.get("size_multiplier", 1.06)
    row.verdict = kw.get("verdict", "size_up")
    row.reasoning = kw.get("reasoning", "Strong recent sources and good family history.")
    row.key_factors = kw.get("key_factors", ["Reliable source mix"])
    row.warnings = kw.get("warnings", ["Exact-match history is still a small sample."])
    row.source_breakdown = kw.get(
        "source_breakdown",
        [
            {
                "source_name": "Reuters",
                "document_share": 0.7,
                "delta_vs_overall": -0.03,
            },
        ],
    )
    row.similar_market_summary = kw.get(
        "similar_market_summary",
        {
            "available": True,
            "family_match": {"resolved_signals": 18, "family_signal_delta_vs_overall": -0.02},
        },
    )
    row.llm_query_id = kw.get("llm_query_id", 42)
    row.created_at = kw.get("created_at", datetime(2026, 1, 2, tzinfo=UTC))
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
    row.exit_reason = kw.get("exit_reason", None)
    row.resolution = kw.get("resolution", None)
    row.pnl = kw.get("pnl", None)
    row.pnl_pct = kw.get("pnl_pct", None)
    row.created_at = kw.get("created_at", datetime(2026, 1, 1, tzinfo=UTC))
    return row


def _make_source_quality_row(**kw) -> MagicMock:
    row = MagicMock()
    row.source_name = kw.get("source_name", "Reuters")
    row.market_category = kw.get("market_category", "politics")
    row.lookback_days = kw.get("lookback_days", 90)
    row.weighted_brier = kw.get("weighted_brier", 0.141)
    row.overall_brier = kw.get("overall_brier", 0.167)
    row.n_signals = kw.get("n_signals", 24)
    row.total_doc_uses = kw.get("total_doc_uses", 60)
    row.computed_at = kw.get("computed_at", datetime(2026, 1, 4, tzinfo=UTC))
    return row


# ---------------------------------------------------------------------------
# /api/signals
# ---------------------------------------------------------------------------


def _signals_list_result(rows: list) -> MagicMock:
    """Mock result for the signals list query.

    Each element must be a 6-tuple:
    (SignalRow, question, series_ticker, rag_hit_count, has_factbase, has_assessment)
    """
    r = MagicMock()
    r.all.return_value = rows
    return r


def test_signals_endpoint_returns_paginated_list() -> None:
    row1 = _make_signal_row()
    row2 = _make_signal_row()

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(2),                                              # count query
        _signals_list_result([(row1, "Q1", None, 2, 0, 0), (row2, "Q2", "KXTEST", 0, 1, 1)]),  # data query
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/signals?limit=20&offset=0")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["total"] == 2
    assert data["limit"] == 20
    assert data["offset"] == 0
    item = data["items"][0]
    assert "estimated_probability" in item
    assert "edge" in item
    assert "direction" in item
    assert "created_at" in item
    assert item["market_question"] == "Q1"


def test_signals_endpoint_filters_by_market_id() -> None:
    row = _make_signal_row(market_id="MKTX-42")

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(1),
        _signals_list_result([(row, "Will X happen?", None, 0, 0, 0)]),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/signals?market_id=MKTX-42")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["market_id"] == "MKTX-42"


def test_signals_endpoint_market_question_none_when_market_missing() -> None:
    row = _make_signal_row()

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(1),
        _signals_list_result([(row, None, None, 0, 0, 0)]),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/signals")

    assert resp.status_code == 200
    assert resp.json()["items"][0]["market_question"] is None


def test_signals_endpoint_unknown_id_returns_404() -> None:
    result_mock = MagicMock()
    result_mock.one_or_none.return_value = None
    session = AsyncMock()
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


def test_signal_detail_returns_assessment_null_when_absent() -> None:
    signal_id = uuid.uuid4()
    signal_row = _make_signal_row(id=signal_id)

    signal_result = MagicMock()
    signal_result.one_or_none.return_value = (signal_row, "Will X happen?", None)

    docs_result = _all_result([])
    assessment_result = MagicMock()
    assessment_result.scalar_one_or_none.return_value = None
    factbase_result = _scalar_result(0)

    session = AsyncMock()
    session.execute = _execute_side_effects(signal_result, docs_result, assessment_result, factbase_result)

    client = TestClient(_make_app(session))
    resp = client.get(f"/api/signals/{signal_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["assessment"] is None
    assert body["document_links"] == []


def test_signal_detail_returns_serialized_assessment_when_present() -> None:
    signal_id = uuid.uuid4()
    signal_row = _make_signal_row(id=signal_id)
    assessment_row = _make_signal_assessment_row(signal_id=signal_id)

    signal_result = MagicMock()
    signal_result.one_or_none.return_value = (signal_row, "Will X happen?", None)

    docs_result = _all_result([])
    assessment_result = MagicMock()
    assessment_result.scalar_one_or_none.return_value = assessment_row
    factbase_result = _scalar_result(0)

    session = AsyncMock()
    session.execute = _execute_side_effects(signal_result, docs_result, assessment_result, factbase_result)

    client = TestClient(_make_app(session))
    resp = client.get(f"/api/signals/{signal_id}")

    assert resp.status_code == 200
    assessment = resp.json()["assessment"]
    assert assessment["trust_score"] == pytest.approx(0.64)
    assert assessment["size_multiplier"] == pytest.approx(1.06)
    assert assessment["reasoning"] == "Strong recent sources and good family history."
    assert assessment["llm_query_id"] == 42


# ---------------------------------------------------------------------------
# /api/positions
# ---------------------------------------------------------------------------


def test_positions_endpoint_filters_by_status() -> None:
    open_pos = _make_position_row(status="open")

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _mode_result(),              # _get_mode
        _scalar_result(1),
        _all_result([(open_pos, 0.20, 0.18, 0.22, 0.20, None, 0)]),   # (PositionRow, mid_price, yes_bid, yes_ask, last_price, series_ticker, has_factbase)
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
        _mode_result(),              # _get_mode
        _scalar_result(2),
        _all_result([(r, 0.50, 0.48, 0.52, 0.50, None, 0) for r in rows]),   # (PositionRow, mid_price, yes_bid, yes_ask, last_price, series_ticker, has_factbase)
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


def test_position_detail_includes_entry_signal_assessment() -> None:
    position_id = uuid.uuid4()
    signal_id = uuid.uuid4()
    position_row = _make_position_row(id=position_id, signal_id=signal_id)
    market_row = MagicMock()
    market_row.question = "Will X happen?"
    market_row.mid_price = 0.55
    market_row.yes_bid = 0.54
    market_row.yes_ask = 0.56
    market_row.last_price = 0.55
    market_row.series_ticker = None
    signal_row = _make_signal_row(id=signal_id)
    assessment_row = _make_signal_assessment_row(signal_id=signal_id)

    position_result = MagicMock()
    position_result.scalar_one_or_none.return_value = position_row
    market_result = MagicMock()
    market_result.scalar_one_or_none.return_value = market_row
    signal_result = MagicMock()
    signal_result.one_or_none.return_value = (signal_row, "Will X happen?")
    docs_result = _all_result([])
    assessment_result = MagicMock()
    assessment_result.scalar_one_or_none.return_value = assessment_row
    factbase_result = _scalar_result(0)
    market_signals_result = _scalars_result([signal_row])

    session = AsyncMock()
    session.execute = _execute_side_effects(
        position_result,
        market_result,
        signal_result,
        docs_result,
        assessment_result,
        factbase_result,
        market_signals_result,
    )

    client = TestClient(_make_app(session))
    resp = client.get(f"/api/positions/{position_id}/detail")

    assert resp.status_code == 200
    entry_assessment = resp.json()["entry_signal"]["assessment"]
    assert entry_assessment is not None
    assert entry_assessment["verdict"] == "size_up"
    assert entry_assessment["llm_query_id"] == 42


# ---------------------------------------------------------------------------
# /api/calibration
# ---------------------------------------------------------------------------


def test_calibration_endpoint_returns_brier_score() -> None:
    # compute_calibration does its own session.execute calls inside.
    # We mock the session to return a single resolved sample.
    rows = [(0.7, 0.5, 1)]  # (estimated_prob, mid, resolution)

    result_mock = MagicMock()
    result_mock.all.return_value = rows
    categories_result = _all_result([("politics",), ("economics",)])

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _mode_result(), result_mock, categories_result,
        _all_result([]), _all_result([]), _all_result([]), _all_result([]),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/calibration")

    assert resp.status_code == 200
    data = resp.json()
    assert "brier_score" in data
    assert "n_samples" in data
    assert "buckets" in data
    assert data["available_categories"] == ["politics", "economics"]
    assert data["n_samples"] == 1
    assert len(data["buckets"]) == 10


def test_calibration_endpoint_no_samples() -> None:
    result_mock = MagicMock()
    result_mock.all.return_value = []
    categories_result = _all_result([])

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _mode_result(), result_mock, categories_result,
        _all_result([]), _all_result([]), _all_result([]), _all_result([]),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/calibration")

    assert resp.status_code == 200
    data = resp.json()
    assert data["n_samples"] == 0


def test_calibration_endpoint_accepts_category_filter() -> None:
    rows = [(0.7, 0.5, 1)]
    result_mock = MagicMock()
    result_mock.all.return_value = rows
    categories_result = _all_result([("Mentions",)])

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _mode_result(), result_mock, categories_result,
        _all_result([]), _all_result([]), _all_result([]), _all_result([]),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/calibration?category=Mentions&lookback_days=30")

    assert resp.status_code == 200
    data = resp.json()
    assert data["available_categories"] == ["Mentions"]
    assert data["n_samples"] == 1


def test_source_quality_returns_rows_from_latest_snapshot() -> None:
    latest_at = datetime(2026, 1, 4, tzinfo=UTC)
    row1 = _make_source_quality_row(source_name="Reuters", computed_at=latest_at)
    row2 = _make_source_quality_row(source_name="AP", computed_at=latest_at, weighted_brier=0.152)

    latest_result = MagicMock()
    latest_result.scalar_one.return_value = latest_at
    rows_result = _scalars_result([row1, row2])

    session = AsyncMock()
    session.execute = _execute_side_effects(latest_result, rows_result)

    client = TestClient(_make_app(session))
    resp = client.get("/api/metrics/source-quality")

    assert resp.status_code == 200
    data = resp.json()
    assert [item["source_name"] for item in data["items"]] == ["Reuters", "AP"]
    assert datetime.fromisoformat(data["items"][0]["computed_at"].replace("Z", "+00:00")) == latest_at


def test_source_quality_filters_by_category() -> None:
    latest_at = datetime(2026, 1, 4, tzinfo=UTC)
    row = _make_source_quality_row(source_name="Reuters", market_category="economics", computed_at=latest_at)

    latest_result = MagicMock()
    latest_result.scalar_one.return_value = latest_at
    rows_result = _scalars_result([row])

    session = AsyncMock()
    session.execute = _execute_side_effects(latest_result, rows_result)

    client = TestClient(_make_app(session))
    resp = client.get("/api/metrics/source-quality?category=economics")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["market_category"] == "economics"


def test_source_quality_filters_by_lookback_days() -> None:
    latest_at = datetime(2026, 1, 4, tzinfo=UTC)
    row = _make_source_quality_row(source_name="Reuters", lookback_days=30, computed_at=latest_at)

    latest_result = MagicMock()
    latest_result.scalar_one.return_value = latest_at
    rows_result = _scalars_result([row])

    session = AsyncMock()
    session.execute = _execute_side_effects(latest_result, rows_result)

    client = TestClient(_make_app(session))
    resp = client.get("/api/metrics/source-quality?lookback_days=30")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["source_name"] == "Reuters"


def test_source_quality_returns_empty_list_when_no_snapshot_and_no_live_data_exist() -> None:
    latest_result = MagicMock()
    latest_result.scalar_one.return_value = None
    distinct_categories_result = _all_result([])

    session = AsyncMock()
    session.execute = _execute_side_effects(latest_result, distinct_categories_result)

    empty_report = MagicMock()
    empty_report.n_samples = 0

    with patch("freqpred.dashboard.api.routes.compute_calibration", return_value=empty_report):
        client = TestClient(_make_app(session))
        resp = client.get("/api/metrics/source-quality")

    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_source_quality_computes_live_when_requested_lookback_snapshot_missing() -> None:
    latest_result = MagicMock()
    latest_result.scalar_one.return_value = None
    distinct_categories_result = _all_result([("Mentions",)])

    session = AsyncMock()
    session.execute = _execute_side_effects(latest_result, distinct_categories_result)

    live_score = MagicMock()
    live_score.source_name = "TVArchive"
    live_score.weighted_brier_score = 0.14
    live_score.n_signals = 12
    live_score.total_doc_appearances = 33

    live_report = MagicMock()
    live_report.n_samples = 20
    live_report.brier_score = 0.19

    with (
        patch("freqpred.dashboard.api.routes.compute_calibration", side_effect=[live_report, live_report]),
        patch("freqpred.dashboard.api.routes.compute_source_brier_scores", return_value=[live_score]),
    ):
        client = TestClient(_make_app(session))
        resp = client.get("/api/metrics/source-quality?lookback_days=7")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    assert {item["market_category"] for item in data["items"]} == {None, "Mentions"}
    assert {item["source_name"] for item in data["items"]} == {"TVArchive"}
    assert all(item["weighted_brier"] == pytest.approx(0.14) for item in data["items"])
    assert all(item["overall_brier"] == pytest.approx(0.19) for item in data["items"])


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
    # health calls: _get_mode, open positions count, get_daily_spend_usd
    open_count_result = _scalar_result(3)
    spend_result = _scalar_result(1.50)

    session = AsyncMock()
    session.execute = _execute_side_effects(_mode_result(), open_count_result, spend_result)

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


# ---------------------------------------------------------------------------
# /api/strategy/config
# ---------------------------------------------------------------------------


def _run_state_result(strategy_name: str | None) -> MagicMock:
    """Mock run_state query result for get_strategy_name."""
    row = MagicMock()
    row.strategy_name = strategy_name
    result = MagicMock()
    result.scalar_one_or_none.return_value = row if strategy_name is not None else None
    return result


def _overrides_result(overrides: dict) -> MagicMock:
    """Mock runtime_config_overrides query result for load_overrides."""
    row = MagicMock()
    row.overrides = overrides
    result = MagicMock()
    result.scalar_one_or_none.return_value = row if overrides else None
    return result


def test_get_strategy_config_returns_all_fields() -> None:
    cfg = _make_strategy_config()
    mock_strategy = MagicMock()
    mock_strategy.config = cfg

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _run_state_result("TestStrategy"),   # get_strategy_name
        _overrides_result({}),               # load_overrides
    )

    with patch("freqpred.strategy.loader.load_strategy", return_value=mock_strategy):
        client = TestClient(_make_app(session))
        resp = client.get("/api/strategy/config")

    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "TestStrategy"
    assert data["min_edge"] == pytest.approx(0.15)
    assert data["min_confidence"] == pytest.approx(0.70)
    assert data["kelly_fraction"] == pytest.approx(0.25)
    assert data["stoploss"] == pytest.approx(-0.15)
    assert data["categories"] == ["politics"]
    assert "trailing_stop" in data
    assert "max_exposure_per_market" in data
    assert "block_reentry_after_stoploss" in data
    assert data["assessment_scale_min"] == pytest.approx(0.80)
    assert data["assessment_scale_max"] == pytest.approx(1.20)
    assert data["similar_market_min_signals"] == 10
    assert data["similar_market_min_trades"] == 5


def test_get_strategy_config_no_active_run_returns_503() -> None:
    session = AsyncMock()
    # run_state row exists but no strategy_name set
    session.execute = _execute_side_effects(_run_state_result(None))

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy/config")
    assert resp.status_code == 503


def test_put_strategy_config_updates_mutable_fields() -> None:
    cfg = _make_strategy_config()
    mock_strategy = MagicMock()
    mock_strategy.config = cfg

    no_override_row = MagicMock()
    no_override_row.scalar_one_or_none.return_value = None

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _run_state_result("TestStrategy"),   # get_strategy_name
        _overrides_result({}),               # load_overrides (inside _load_active)
        _overrides_result({}),               # load_overrides (inside PUT logic)
        no_override_row,                     # save_overrides select
    )
    session.add = MagicMock()
    session.commit = AsyncMock()

    with patch("freqpred.strategy.loader.load_strategy", return_value=mock_strategy):
        client = TestClient(_make_app(session))
        resp = client.put("/api/strategy/config", json={"min_edge": 0.22})

    assert resp.status_code == 200
    assert resp.json()["min_edge"] == pytest.approx(0.22)


def test_put_strategy_config_updates_assessment_fields() -> None:
    cfg = _make_strategy_config()
    mock_strategy = MagicMock()
    mock_strategy.config = cfg

    no_override_row = MagicMock()
    no_override_row.scalar_one_or_none.return_value = None

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _run_state_result("TestStrategy"),
        _overrides_result({}),
        _overrides_result({}),
        no_override_row,
    )
    session.add = MagicMock()
    session.commit = AsyncMock()

    with patch("freqpred.strategy.loader.load_strategy", return_value=mock_strategy):
        client = TestClient(_make_app(session))
        resp = client.put(
            "/api/strategy/config",
            json={
                "assessment_scale_min": 0.75,
                "assessment_scale_max": 1.30,
                "similar_market_min_signals": 12,
                "similar_market_min_trades": 6,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["assessment_scale_min"] == pytest.approx(0.75)
    assert data["assessment_scale_max"] == pytest.approx(1.30)
    assert data["similar_market_min_signals"] == 12
    assert data["similar_market_min_trades"] == 6


def test_put_strategy_config_rejects_immutable_fields() -> None:
    cfg = _make_strategy_config()
    mock_strategy = MagicMock()
    mock_strategy.config = cfg

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _run_state_result("TestStrategy"),
        _overrides_result({}),
    )

    with patch("freqpred.strategy.loader.load_strategy", return_value=mock_strategy):
        client = TestClient(_make_app(session))
        resp = client.put("/api/strategy/config", json={"categories": ["sports"]})

    assert resp.status_code == 422
    assert "immutable" in resp.json()["detail"].lower()


def test_put_strategy_config_rejects_name_field() -> None:
    cfg = _make_strategy_config()
    mock_strategy = MagicMock()
    mock_strategy.config = cfg

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _run_state_result("TestStrategy"),
        _overrides_result({}),
    )

    with patch("freqpred.strategy.loader.load_strategy", return_value=mock_strategy):
        client = TestClient(_make_app(session))
        resp = client.put("/api/strategy/config", json={"name": "OtherStrategy"})

    assert resp.status_code == 422
    assert "immutable" in resp.json()["detail"].lower()


def test_put_strategy_config_no_active_run_returns_503() -> None:
    session = AsyncMock()
    session.execute = _execute_side_effects(_run_state_result(None))

    client = TestClient(_make_app(session))
    resp = client.put("/api/strategy/config", json={"min_edge": 0.20})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /api/system/health
# ---------------------------------------------------------------------------


def _make_system_health_session(
    run_state: str = "running",
    mode: str = "paper",
    cb_active: bool = False,
    cb_reason: str | None = None,
    daily_loss_ack_at: datetime | None = None,
    daily_pnl: float = 0.0,
    all_time_pnl: float = 0.0,
    llm_spend: float = 1.0,
    pending_count: int = 0,
    oldest_pending_entry_time: datetime | None = None,
    open_count: int = 3,
    llm_errors: int = 0,
    kalshi_errors: int = 0,
    heartbeat_rows: list | None = None,
) -> AsyncMock:
    """Return a mock session that services all queries made by GET /api/system/health."""
    # Query order in the route:
    # 0. _get_mode → scalar_one_or_none (RunStateRow.mode)
    # 1. select(RunStateRow) → scalar_one_or_none  (run_state + cb_active + cb_reason)
    # 2. get_net_bankroll → scalar_one (all-time closed PnL SUM)
    # 3. daily_pnl SUM
    # 4. get_daily_spend_usd → scalar_one
    # 5. pending positions COUNT
    # 6. oldest pending entry_time MIN
    # 7. open positions COUNT
    # 8. LLM errors COUNT
    # 9. Kalshi runtime errors COUNT
    # 10. service_heartbeats list (only when runtime telemetry is provided)

    run_state_row = MagicMock()
    run_state_row.state = run_state
    run_state_row.mode = mode
    run_state_row.cb_active = cb_active
    run_state_row.cb_reason = cb_reason
    run_state_row.daily_loss_ack_at = daily_loss_ack_at
    run_state_row.drawdown_reset_at = None
    run_state_row.drawdown_reset_bankroll = None

    run_state_result = MagicMock()
    run_state_result.scalar_one_or_none.return_value = run_state_row

    all_time_pnl_result = _scalar_result(all_time_pnl)  # get_net_bankroll
    daily_pnl_result = _scalar_result(daily_pnl)
    llm_spend_result = _scalar_result(llm_spend)
    pending_result = _scalar_result(pending_count)
    pending_min_result = _scalar_result(oldest_pending_entry_time)
    open_result = _scalar_result(open_count)
    llm_errors_result = _scalar_result(llm_errors)
    kalshi_errors_result = _scalar_result(kalshi_errors)
    heartbeats_result = _scalars_result(heartbeat_rows or [])

    session = AsyncMock()
    session.execute = _execute_side_effects(
        run_state_result,    # _get_mode
        run_state_result,    # select(RunStateRow) for CB state
        all_time_pnl_result, # get_net_bankroll
        daily_pnl_result,
        llm_spend_result,
        pending_result,
        pending_min_result,
        open_result,
        llm_errors_result,
        kalshi_errors_result,
        heartbeats_result,
    )
    return session


def test_system_health_returns_run_state() -> None:
    session = _make_system_health_session(run_state="paused")
    client = TestClient(_make_app(session))
    resp = client.get("/api/system/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_state"] == "paused"
    assert data["mode"] == "paper"
    assert data["db_ok"] is True
    assert "circuit_breakers" in data
    assert "websocket" in data
    assert "api_errors" in data
    assert "uptime_seconds" in data


def test_system_health_circuit_breaker_fields_no_halt() -> None:
    session = _make_system_health_session(daily_pnl=0.0, llm_spend=1.0)
    client = TestClient(_make_app(session, daily_cap=10.0, bankroll_usd=1000.0))
    resp = client.get("/api/system/health")

    assert resp.status_code == 200
    cb = resp.json()["circuit_breakers"]
    assert cb["trading_halted"] is False
    assert cb["reason"] is None
    assert cb["daily_loss_pct"] == pytest.approx(0.0)
    assert cb["llm_budget_used_usd"] == pytest.approx(1.0)
    assert cb["llm_budget_cap_usd"] == pytest.approx(10.0)
    # Without an ack, window starts at today's UTC midnight and ack_at is null.
    assert cb["daily_loss_ack_at"] is None
    assert "daily_loss_window_start" in cb


def test_system_health_daily_loss_window_honors_ack_at() -> None:
    """The dashboard must mirror risk.py: when daily_loss_ack_at is set, it bounds
    the loss window and is surfaced in the response so the UI can show why the
    percentage differs from the naive midnight-to-now total."""
    ack_at = datetime.now(UTC).replace(microsecond=0)
    session = _make_system_health_session(daily_loss_ack_at=ack_at)
    client = TestClient(_make_app(session, bankroll_usd=1000.0))
    resp = client.get("/api/system/health")

    assert resp.status_code == 200
    cb = resp.json()["circuit_breakers"]
    # ack_at is surfaced as-is
    assert cb["daily_loss_ack_at"] is not None
    assert datetime.fromisoformat(cb["daily_loss_ack_at"]) == ack_at
    # Window starts at max(today_start, ack_at) — since ack_at is "now", that's ack_at.
    window_start = datetime.fromisoformat(cb["daily_loss_window_start"])
    assert window_start == ack_at


def test_system_health_circuit_breaker_daily_loss_triggers_halt() -> None:
    # CB was persisted by the run loop as active.
    session = _make_system_health_session(
        cb_active=True,
        cb_reason="Circuit breaker: daily loss 150.00 exceeds 15% of bankroll (150.00)",
        daily_pnl=-150.0,
    )
    client = TestClient(_make_app(session, bankroll_usd=1000.0))
    resp = client.get("/api/system/health")

    assert resp.status_code == 200
    cb = resp.json()["circuit_breakers"]
    assert cb["trading_halted"] is True
    assert cb["reason"] is not None
    assert "daily loss" in cb["reason"].lower()


def test_system_health_websocket_stubbed_unknown() -> None:
    session = _make_system_health_session()
    client = TestClient(_make_app(session))
    resp = client.get("/api/system/health")

    ws = resp.json()["websocket"]
    assert ws["status"] == "unknown"
    assert ws["connected"] is None
    assert ws["subscribed_markets"] is None
    assert ws["last_message_at"] is None
    assert ws["last_reconcile_at"] is None


def test_system_health_llm_errors_returned() -> None:
    session = _make_system_health_session(llm_errors=3)
    client = TestClient(_make_app(session))
    resp = client.get("/api/system/health")

    assert resp.json()["api_errors"]["llm_errors_last_hour"] == 3
    assert resp.json()["api_errors"]["kalshi_errors_last_hour"] == 0
    assert resp.json()["api_errors"]["consecutive_llm_errors"] is None


def test_system_health_returns_real_websocket_fields() -> None:
    telemetry = RuntimeTelemetry(
        session_factory=MagicMock(),
        freshness_specs=build_freshness_specs(
            ingestion_interval_seconds=1800,
            realtime_interval_seconds=300,
            signal_interval_seconds=1800,
            market_watcher_interval_seconds=300,
        ),
    )
    telemetry._websocket_connected = True
    telemetry._websocket_subscribed_markets = 2
    telemetry._websocket_last_message_at = datetime.now(UTC) - timedelta(minutes=5)
    telemetry._websocket_last_reconcile_at = datetime.now(UTC) - timedelta(minutes=10)

    ws_row = MagicMock()
    ws_row.service_name = "position_watcher_last_message"
    ws_row.last_success_at = telemetry._websocket_last_message_at
    ws_row.last_error_at = None
    ws_row.last_error_message = None

    session = _make_system_health_session(heartbeat_rows=[ws_row])
    client = TestClient(_make_app(session, runtime_telemetry=telemetry))
    resp = client.get("/api/system/health")

    assert resp.status_code == 200
    ws = resp.json()["websocket"]
    assert ws["status"] == "ok"
    assert ws["connected"] is True
    assert ws["subscribed_markets"] == 2
    assert ws["last_message_at"] is not None
    assert ws["last_reconcile_at"] is not None


def test_system_health_marks_stale_services() -> None:
    telemetry = RuntimeTelemetry(
        session_factory=MagicMock(),
        freshness_specs=build_freshness_specs(
            ingestion_interval_seconds=1800,
            realtime_interval_seconds=300,
            signal_interval_seconds=1800,
            market_watcher_interval_seconds=300,
        ),
    )
    old = datetime.now(UTC) - timedelta(hours=2)
    row = MagicMock()
    row.service_name = "ingestion_scheduler"
    row.last_success_at = old
    row.last_error_at = None
    row.last_error_message = None

    session = _make_system_health_session(heartbeat_rows=[row])
    client = TestClient(_make_app(session, runtime_telemetry=telemetry))
    resp = client.get("/api/system/health")

    assert resp.status_code == 200
    services = {svc["service_name"]: svc for svc in resp.json()["services"]}
    assert services["ingestion_scheduler"]["status"] == "stale"


def test_system_health_db_error_returns_degraded_db_ok_false() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("db down"))
    client = TestClient(_make_app(session))
    resp = client.get("/api/system/health")

    assert resp.status_code == 200
    assert resp.json()["db_ok"] is False


# ---------------------------------------------------------------------------
# /api/llm/queries
# ---------------------------------------------------------------------------


def _make_llm_query_row(**kw) -> MagicMock:
    row = MagicMock()
    row.id = kw.get("id", 1)
    row.timestamp = kw.get("timestamp", datetime(2026, 1, 1, tzinfo=UTC))
    row.query_type = kw.get("query_type", "market_analysis")
    row.market_id = kw.get("market_id", "MARKET-1")
    row.model_used = kw.get("model_used", "claude-sonnet-4-6")
    row.tokens_total = kw.get("tokens_total", 500)
    row.cost_usd = kw.get("cost_usd", 0.01)
    row.latency_ms = kw.get("latency_ms", 1200)
    row.success = kw.get("success", True)
    row.prompt = kw.get("prompt", "Test prompt text")
    row.response = kw.get("response", "Test response text")
    row.error_message = kw.get("error_message", None)
    return row


def test_llm_queries_empty_list() -> None:
    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(0),          # count
        _scalars_result([]),        # rows
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/llm/queries")

    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["limit"] == 50
    assert data["offset"] == 0


def test_llm_queries_returns_rows_ordered_by_timestamp() -> None:
    row1 = _make_llm_query_row(id=2, timestamp=datetime(2026, 1, 2, tzinfo=UTC))
    row2 = _make_llm_query_row(id=1, timestamp=datetime(2026, 1, 1, tzinfo=UTC))

    session = AsyncMock()
    session.execute = _execute_side_effects(
        _scalar_result(2),
        _scalars_result([row1, row2]),
    )

    client = TestClient(_make_app(session))
    resp = client.get("/api/llm/queries")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["total"] == 2
    # First row should be the more recent one (id=2)
    assert data["items"][0]["id"] == 2
    assert data["items"][0]["query_type"] == "market_analysis"
    assert data["items"][0]["cost_usd"] == pytest.approx(0.01)
    assert data["items"][0]["success"] is True


def test_llm_queries_detail_returns_prompt_and_response() -> None:
    row = _make_llm_query_row(
        id=42,
        prompt="What is the probability?",
        response="The probability is 0.7.",
    )

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    client = TestClient(_make_app(session))
    resp = client.get("/api/llm/queries/42")

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 42
    assert data["prompt"] == "What is the probability?"
    assert data["response"] == "The probability is 0.7."
    assert data["error_message"] is None


def test_llm_queries_detail_unknown_id_returns_404() -> None:
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    client = TestClient(_make_app(session))
    resp = client.get("/api/llm/queries/9999")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/markets
# ---------------------------------------------------------------------------


def _make_market_row(**kw) -> MagicMock:
    row = MagicMock()
    row.id = kw.get("id", "KLSHI-MARKET-1")
    row.question = kw.get("question", "Will X happen?")
    row.status = kw.get("status", "active")
    row.yes_bid = kw.get("yes_bid", 0.55)
    row.yes_ask = kw.get("yes_ask", 0.60)
    row.mid_price = kw.get("mid_price", 0.575)
    row.volume_24h = kw.get("volume_24h", 1000.0)
    row.close_time = kw.get("close_time", datetime(2026, 6, 1, tzinfo=UTC))
    row.last_fetched_at = kw.get("last_fetched_at", datetime(2026, 1, 1, tzinfo=UTC))
    row.current_signal_id = kw.get("current_signal_id", None)
    return row


def test_markets_list_returns_paginated_results() -> None:
    row1 = _make_market_row(id="M1", question="Will A happen?")
    row2 = _make_market_row(id="M2", question="Will B happen?")

    count_r = _scalar_result(2)
    data_r = _scalars_result([row1, row2])

    session = AsyncMock()
    session.execute = _execute_side_effects(count_r, data_r)

    client = TestClient(_make_app(session))
    resp = client.get("/api/markets?status=open&limit=50&offset=0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["id"] == "M1"
    assert body["items"][1]["id"] == "M2"


def test_markets_list_search_filters_by_question() -> None:
    row = _make_market_row(id="M1", question="Will inflation drop?")

    count_r = _scalar_result(1)
    data_r = _scalars_result([row])

    session = AsyncMock()
    session.execute = _execute_side_effects(count_r, data_r)

    client = TestClient(_make_app(session))
    resp = client.get("/api/markets?search=inflation")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["question"] == "Will inflation drop?"


def test_market_detail_includes_current_signal() -> None:
    sig_id = uuid.uuid4()
    market_row = _make_market_row(id="M1", current_signal_id=sig_id)
    sig_row = _make_signal_row(id=sig_id, market_id="M1")

    # GET /api/markets/{id}: scalar_one_or_none → market_row
    market_r = MagicMock()
    market_r.scalar_one_or_none.return_value = market_row

    # Signal join query: one_or_none → (sig_row, question)
    sig_r = MagicMock()
    sig_r.one_or_none.return_value = (sig_row, "Will X happen?")

    session = AsyncMock()
    session.execute = _execute_side_effects(market_r, sig_r)

    client = TestClient(_make_app(session))
    resp = client.get("/api/markets/M1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "M1"
    assert body["current_signal"] is not None
    assert body["current_signal"]["market_id"] == "M1"


def test_market_detail_returns_404_for_unknown_id() -> None:
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    client = TestClient(_make_app(session))
    resp = client.get("/api/markets/DOES-NOT-EXIST")

    assert resp.status_code == 404


def test_analyze_market_triggers_signal_pipeline() -> None:
    market_row = _make_market_row(id="M1")

    # No current signal → cooldown check skipped, pipeline called
    market_r = MagicMock()
    market_r.scalar_one_or_none.return_value = market_row

    # After pipeline.analyze(), reload the saved SignalRow
    new_sig_id = uuid.uuid4()
    saved_sig = _make_signal_row(id=new_sig_id, market_id="M1")
    saved_sig_r = MagicMock()
    saved_sig_r.scalar_one_or_none.return_value = saved_sig

    session = AsyncMock()
    session.execute = _execute_side_effects(market_r, saved_sig_r)

    # Mock pipeline.analyze() to return a Signal-like object
    from freqpred.signal.models import Signal  # noqa: PLC0415

    mock_signal = Signal(
        id=str(new_sig_id),
        market_id="M1",
        estimated_probability=0.70,
        confidence=0.80,
        edge=0.10,
        market_mid_at_signal=0.60,
        direction="YES",
        reasoning="test",
        sources=[],
        retrieval_hash="hash1",
        model_used="claude-sonnet-4-6",
        prompt_version="signal-v1",
        trigger="manual",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_context="ctx",
    )
    mock_pipeline = AsyncMock()
    mock_pipeline.analyze = AsyncMock(return_value=mock_signal)

    client = TestClient(_make_app(session, signal_pipeline=mock_pipeline))
    resp = client.post("/api/markets/M1/analyze")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is False
    assert body["signal"]["market_id"] == "M1"
    mock_pipeline.analyze.assert_called_once()


def test_analyze_market_429_within_cooldown() -> None:
    sig_id = uuid.uuid4()
    market_row = _make_market_row(id="M1", current_signal_id=sig_id)

    # Signal created just 10 seconds ago → within 60 s cooldown
    recent_sig = _make_signal_row(
        id=sig_id,
        market_id="M1",
        created_at=datetime.now(UTC).replace(second=datetime.now(UTC).second - 10)
        if datetime.now(UTC).second >= 10
        else datetime.now(UTC),
    )
    # Set created_at explicitly to "now minus 5 seconds"
    from datetime import timedelta  # noqa: PLC0415
    recent_sig.created_at = datetime.now(UTC) - timedelta(seconds=5)

    market_r = MagicMock()
    market_r.scalar_one_or_none.return_value = market_row

    cooldown_sig_r = MagicMock()
    cooldown_sig_r.scalar_one_or_none.return_value = recent_sig

    session = AsyncMock()
    session.execute = _execute_side_effects(market_r, cooldown_sig_r)

    mock_pipeline = AsyncMock()
    mock_pipeline.analyze = AsyncMock()

    client = TestClient(_make_app(session, signal_pipeline=mock_pipeline))
    resp = client.post("/api/markets/M1/analyze")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is True
    # Pipeline should NOT have been called
    mock_pipeline.analyze.assert_not_called()


def test_analyze_market_503_when_no_pipeline() -> None:
    market_row = _make_market_row(id="M1")
    market_r = MagicMock()
    market_r.scalar_one_or_none.return_value = market_row

    session = AsyncMock()
    session.execute = AsyncMock(return_value=market_r)

    # No pipeline passed → analyze endpoint should 503
    client = TestClient(_make_app(session, signal_pipeline=None))
    resp = client.post("/api/markets/M1/analyze")

    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /api/strategy-decisions
# ---------------------------------------------------------------------------


def _decisions_query_result(
    rows: list[tuple],
) -> MagicMock:
    """Mock result for the main data query.

    Returns (PositionRow, market_result, market_question, best_prior_ask) tuples.
    """
    r = MagicMock()
    r.all.return_value = rows
    return r


def _distinct_scalars_result(values: list) -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.all.return_value = values
    return r


def _decisions_mock_session(
    rows: list[tuple],
    *,
    total: int | None = None,
    distinct_strategies: list[str] | None = None,
    distinct_exit_reasons: list[str] | None = None,
) -> AsyncMock:
    """Build an AsyncMock session with the 5 execute calls made by /strategy-decisions."""
    session = AsyncMock()
    session.execute = _execute_side_effects(
        _mode_result(),                                                       # _get_mode
        _scalar_result(total if total is not None else len(rows)),            # count
        _decisions_query_result(rows),                                        # data query
        _distinct_scalars_result(                                             # distinct strategies
            distinct_strategies
            if distinct_strategies is not None
            else sorted({r[0].strategy_name for r in rows}),
        ),
        _distinct_scalars_result(                                             # distinct exit reasons
            distinct_exit_reasons
            if distinct_exit_reasons is not None
            else sorted(
                {r[0].exit_reason for r in rows if r[0].exit_reason is not None},
            ),
        ),
    )
    return session


def _make_closed_position(**kw) -> MagicMock:
    """Convenience wrapper for _make_position_row that defaults to a closed position."""
    defaults = dict(
        status="closed",
        exit_price=0.25,
        exit_time=datetime(2026, 2, 1, tzinfo=UTC),
        exit_reason="stoploss",
        pnl=-2.5,
        pnl_pct=-0.5,
    )
    defaults.update(kw)
    return _make_position_row(**defaults)


def test_decisions_empty_when_no_closed_positions() -> None:
    session = _decisions_mock_session([], total=0)

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["limit"] == 50
    assert data["offset"] == 0
    assert data["distinct_strategies"] == []
    assert data["distinct_exit_reasons"] == []


def test_decisions_counterfactual_yes_resolved_yes() -> None:
    """YES entered 0.50, exited 0.25, resolved yes → counterfactual +0.50, exit Δ −0.75."""
    row = _make_closed_position(
        direction="YES",
        contracts=10,
        entry_price=0.50,
        exit_price=0.25,
    )
    session = _decisions_mock_session([(row, "yes", "Will X happen?", None)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["market_result"] == "yes"
    assert item["counterfactual_pnl_per_contract"] == pytest.approx(0.50)
    assert item["counterfactual_pnl_usd"] == pytest.approx(5.0)
    assert item["exit_delta_per_contract"] == pytest.approx(-0.75)
    assert item["exit_delta_usd"] == pytest.approx(-7.5)


def test_decisions_counterfactual_yes_resolved_no() -> None:
    """YES entered 0.50, exited 0.25, resolved no → counterfactual −0.50, exit Δ +0.25."""
    row = _make_closed_position(
        direction="YES",
        contracts=10,
        entry_price=0.50,
        exit_price=0.25,
    )
    session = _decisions_mock_session([(row, "no", "Will X happen?", None)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["counterfactual_pnl_per_contract"] == pytest.approx(-0.50)
    assert item["counterfactual_pnl_usd"] == pytest.approx(-5.0)
    assert item["exit_delta_per_contract"] == pytest.approx(0.25)
    assert item["exit_delta_usd"] == pytest.approx(2.5)


def test_decisions_counterfactual_no_resolved_no() -> None:
    """NO entered 0.40, exited 0.60, resolved no → counterfactual +0.60, exit Δ −0.40."""
    row = _make_closed_position(
        direction="NO",
        contracts=5,
        entry_price=0.40,
        exit_price=0.60,
    )
    session = _decisions_mock_session([(row, "no", "Q?", None)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["counterfactual_pnl_per_contract"] == pytest.approx(0.60)
    assert item["counterfactual_pnl_usd"] == pytest.approx(3.0)
    assert item["exit_delta_per_contract"] == pytest.approx(-0.40)
    assert item["exit_delta_usd"] == pytest.approx(-2.0)


def test_decisions_counterfactual_no_resolved_yes() -> None:
    """NO entered 0.40, exited 0.60, resolved yes → counterfactual −0.40, exit Δ +0.60."""
    row = _make_closed_position(
        direction="NO",
        contracts=5,
        entry_price=0.40,
        exit_price=0.60,
    )
    session = _decisions_mock_session([(row, "yes", "Q?", None)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["counterfactual_pnl_per_contract"] == pytest.approx(-0.40)
    assert item["counterfactual_pnl_usd"] == pytest.approx(-2.0)
    assert item["exit_delta_per_contract"] == pytest.approx(0.60)
    assert item["exit_delta_usd"] == pytest.approx(3.0)


def test_decisions_null_when_unresolved() -> None:
    """market.result is None → all counterfactual fields are None."""
    row = _make_closed_position(
        direction="YES",
        entry_price=0.50,
        exit_price=0.25,
    )
    session = _decisions_mock_session([(row, None, "Q?", None)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["market_result"] is None
    assert item["counterfactual_pnl_per_contract"] is None
    assert item["counterfactual_pnl_usd"] is None
    assert item["exit_delta_per_contract"] is None
    assert item["exit_delta_usd"] is None


def test_decisions_market_resolved_exit_has_zero_delta() -> None:
    """If exit_price equals the resolution value, exit_delta == 0."""
    # YES position held to resolution: exit_price = 1.0, resolved yes
    row = _make_closed_position(
        direction="YES",
        contracts=10,
        entry_price=0.50,
        exit_price=1.0,
        exit_reason="market_resolved",
    )
    session = _decisions_mock_session([(row, "yes", "Q?", None)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["exit_reason"] == "market_resolved"
    assert item["counterfactual_pnl_per_contract"] == pytest.approx(0.50)
    assert item["exit_delta_per_contract"] == pytest.approx(0.0)
    assert item["exit_delta_usd"] == pytest.approx(0.0)


def test_decisions_entry_efficiency_populated() -> None:
    """best_prior_ask = 0.15, entry_price = 0.40 → entry_efficiency −0.25/contract."""
    row = _make_closed_position(
        direction="YES",
        contracts=4,
        entry_price=0.40,
        exit_price=0.50,
    )
    session = _decisions_mock_session([(row, "yes", "Q?", 0.15)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["best_prior_ask"] == pytest.approx(0.15)
    assert item["entry_efficiency_per_contract"] == pytest.approx(-0.25)
    assert item["entry_efficiency_usd"] == pytest.approx(-1.0)


def test_decisions_entry_efficiency_null_when_no_prior_signal() -> None:
    """No qualifying prior signal → best_prior_ask/entry efficiency all None."""
    row = _make_closed_position(direction="NO", entry_price=0.40, exit_price=0.30)
    session = _decisions_mock_session([(row, "no", "Q?", None)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["best_prior_ask"] is None
    assert item["entry_efficiency_per_contract"] is None
    assert item["entry_efficiency_usd"] is None


def test_decisions_entry_efficiency_positive_for_no_side() -> None:
    """NO side: best_prior_ask 0.30, entry 0.40 → entry_efficiency −0.10/contract."""
    row = _make_closed_position(
        direction="NO",
        contracts=2,
        entry_price=0.40,
        exit_price=0.50,
    )
    session = _decisions_mock_session([(row, "no", "Q?", 0.30)])

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions")

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["best_prior_ask"] == pytest.approx(0.30)
    assert item["entry_efficiency_per_contract"] == pytest.approx(-0.10)
    assert item["entry_efficiency_usd"] == pytest.approx(-0.20)


def test_decisions_filter_query_params_accepted() -> None:
    """All filter params should be accepted without 422."""
    row = _make_closed_position(
        strategy_name="ConservativeDefault",
        exit_reason="force_exit:time_based",
        market_id="KXTRUMPSAY-26APR13-COMM",
    )
    session = _decisions_mock_session(
        [(row, "yes", "Q?", None)],
        distinct_strategies=["ConservativeDefault", "TestStrategy"],
        distinct_exit_reasons=["force_exit:time_based", "stoploss"],
    )

    client = TestClient(_make_app(session))
    resp = client.get(
        "/api/strategy-decisions",
        params={
            "strategy": "ConservativeDefault",
            "exit_reason": "force_exit",
            "ticker_prefix": "KXTRUMPSAY",
            "date_from": "2026-01-01T00:00:00",
            "date_to": "2026-12-31T23:59:59",
            "limit": 25,
            "offset": 0,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 25
    assert data["offset"] == 0
    assert len(data["items"]) == 1
    assert data["items"][0]["strategy_name"] == "ConservativeDefault"
    # Dropdowns reflect the full closed-position set, not just the filtered slice.
    assert "ConservativeDefault" in data["distinct_strategies"]
    assert "TestStrategy" in data["distinct_strategies"]
    assert "force_exit:time_based" in data["distinct_exit_reasons"]
    assert "stoploss" in data["distinct_exit_reasons"]


def test_decisions_pagination_limits_and_offset() -> None:
    """Total returned independently of the current page size."""
    rows = [
        (_make_closed_position(direction="YES", entry_price=0.5, exit_price=0.4), "yes", "Q1", None),
        (_make_closed_position(direction="NO", entry_price=0.3, exit_price=0.6), "no", "Q2", None),
    ]
    session = _decisions_mock_session(rows, total=17)

    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions?limit=2&offset=4")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 17
    assert data["limit"] == 2
    assert data["offset"] == 4
    assert len(data["items"]) == 2


def test_decisions_invalid_limit_rejected() -> None:
    """limit > 200 should return 422."""
    session = AsyncMock()
    client = TestClient(_make_app(session))
    resp = client.get("/api/strategy-decisions?limit=500")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/positions/{id}/force-exit
# ---------------------------------------------------------------------------


def test_force_exit_endpoint_closes_open_position() -> None:
    """Success: order manager closes position; response reflects closed state."""
    pos_id = uuid.uuid4()
    mock_om = AsyncMock()
    mock_om.force_exit = AsyncMock(return_value=None)

    # The route does a fresh SELECT after force_exit() commits.
    closed_row = _make_position_row(id=pos_id, status="closed", exit_reason="force_exit:manual")
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_scalar_result(closed_row))

    client = TestClient(_make_app(session, order_manager=mock_om))
    resp = client.post(f"/api/positions/{pos_id}/force-exit")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "closed"
    assert data["exit_reason"] == "force_exit:manual"
    mock_om.force_exit.assert_awaited_once_with(str(pos_id))


def test_force_exit_endpoint_404_position_not_found() -> None:
    """PositionNotFoundError from OM → 404."""
    pos_id = uuid.uuid4()
    mock_om = AsyncMock()
    mock_om.force_exit = AsyncMock(side_effect=PositionNotFoundError("not found"))

    session = AsyncMock()
    client = TestClient(_make_app(session, order_manager=mock_om))
    resp = client.post(f"/api/positions/{pos_id}/force-exit")

    assert resp.status_code == 404


def test_force_exit_endpoint_409_already_closed() -> None:
    """PositionNotOpenError from OM → 409."""
    pos_id = uuid.uuid4()
    mock_om = AsyncMock()
    mock_om.force_exit = AsyncMock(side_effect=PositionNotOpenError("already closed"))

    session = AsyncMock()
    client = TestClient(_make_app(session, order_manager=mock_om))
    resp = client.post(f"/api/positions/{pos_id}/force-exit")

    assert resp.status_code == 409


def test_force_exit_endpoint_409_other_precondition_failure() -> None:
    """ValueError (e.g. LIVE_TRADING_ENABLED guard) from OM → 409."""
    pos_id = uuid.uuid4()
    mock_om = AsyncMock()
    mock_om.force_exit = AsyncMock(
        side_effect=ValueError("LIVE_TRADING_ENABLED must be 'true' for live force exits")
    )

    session = AsyncMock()
    client = TestClient(_make_app(session, order_manager=mock_om))
    resp = client.post(f"/api/positions/{pos_id}/force-exit")

    assert resp.status_code == 409


def test_force_exit_endpoint_503_no_order_manager() -> None:
    """No order_manager wired → 503."""
    pos_id = uuid.uuid4()
    session = AsyncMock()
    client = TestClient(_make_app(session, order_manager=None))
    resp = client.post(f"/api/positions/{pos_id}/force-exit")

    assert resp.status_code == 503


def test_force_exit_endpoint_502_exchange_error() -> None:
    """KalshiAPIError from OM → 502."""
    pos_id = uuid.uuid4()
    mock_om = AsyncMock()
    mock_om.force_exit = AsyncMock(side_effect=KalshiAPIError(500, "exchange down"))

    session = AsyncMock()
    client = TestClient(_make_app(session, order_manager=mock_om))
    resp = client.post(f"/api/positions/{pos_id}/force-exit")

    assert resp.status_code == 502
