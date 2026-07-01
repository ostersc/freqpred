"""Unit tests for system health API freshness telemetry (T68).

Tests verify that:
- When RuntimeTelemetry provides websocket state, /api/system/health returns real values.
- When service heartbeats are stale, the response marks them accordingly.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

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
from freqpred.runtime.telemetry import FreshnessSpec, RuntimeTelemetry, ServiceFreshnessState
from freqpred.strategy.config import StrategyConfig


def _make_strategy_config() -> StrategyConfig:
    return StrategyConfig(
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


def _scalar_result(value: object) -> MagicMock:
    r = MagicMock()
    r.scalar_one.return_value = value
    return r


def _execute_side_effects(*results: MagicMock) -> AsyncMock:
    return AsyncMock(side_effect=list(results))


def _make_session(
    *,
    run_state: str = "running",
    mode: str = "paper",
    cb_active: bool = False,
    pending_count: int = 0,
    open_count: int = 0,
    llm_errors: int = 0,
    kalshi_errors: int = 0,
    service_heartbeats: list | None = None,
) -> AsyncMock:
    run_state_row = MagicMock()
    run_state_row.state = run_state
    run_state_row.mode = mode
    run_state_row.cb_active = cb_active
    run_state_row.cb_reason = None
    run_state_row.daily_loss_ack_at = None

    rs_result = MagicMock()
    rs_result.scalar_one_or_none.return_value = run_state_row

    hb_result = MagicMock()
    hb_result.scalars.return_value.all.return_value = service_heartbeats or []

    changelog_result = MagicMock()
    changelog_result.scalar_one_or_none.return_value = None

    pending_detail_result = MagicMock()
    pending_detail_result.all.return_value = []

    session = AsyncMock()
    session.execute = _execute_side_effects(
        rs_result,                 # _get_mode
        rs_result,                 # RunStateRow for CB state
        _scalar_result(1000.0),    # get_net_bankroll
        _scalar_result(0.0),       # daily_pnl
        _scalar_result(1.0),       # llm_spend
        _scalar_result(pending_count),   # pending count
        _scalar_result(None),            # oldest pending entry_time
        pending_detail_result,           # pending orders detail
        _scalar_result(open_count),      # open count
        _scalar_result(llm_errors),      # llm errors
        _scalar_result(kalshi_errors),   # kalshi errors
        changelog_result,                # kalshi_changelog_state
        hb_result,                       # list_service_heartbeats (only with runtime_telemetry)
    )
    return session


def _make_app(
    session: AsyncMock,
    *,
    runtime_telemetry: RuntimeTelemetry | None = None,
) -> object:
    sf = MagicMock()
    app = create_app(
        session_factory=sf,
        daily_cap_usd=10.0,
        bankroll_usd=1000.0,
        runtime_telemetry=runtime_telemetry,
    )

    async def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app


# ---------------------------------------------------------------------------
# test_system_health_returns_real_websocket_fields
# ---------------------------------------------------------------------------


def test_system_health_returns_real_websocket_fields() -> None:
    """When RuntimeTelemetry provides websocket state, the health response reflects it."""
    session_factory = MagicMock()
    telemetry = RuntimeTelemetry(
        session_factory=session_factory,
        freshness_specs={},
    )
    last_msg = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)
    last_reconcile = datetime(2026, 4, 18, 11, 55, 0, tzinfo=UTC)

    # Inject in-memory state
    telemetry._websocket_connected = True
    telemetry._websocket_subscribed_markets = 3
    telemetry._websocket_last_message_at = last_msg
    telemetry._websocket_last_reconcile_at = last_reconcile

    session = _make_session()
    app = _make_app(session, runtime_telemetry=telemetry)
    client = TestClient(app)

    resp = client.get("/api/system/health")
    assert resp.status_code == 200

    ws = resp.json()["websocket"]
    assert ws["connected"] is True
    assert ws["subscribed_markets"] == 3
    assert ws["last_message_at"] is not None
    assert ws["last_reconcile_at"] is not None


# ---------------------------------------------------------------------------
# test_system_health_marks_stale_services
# ---------------------------------------------------------------------------


def test_system_health_marks_stale_services() -> None:
    """When service heartbeats are stale, the response services list reflects it."""
    session_factory = MagicMock()
    freshness_specs = {
        "signal_loop": FreshnessSpec(
            service_name="signal_loop",
            label="Signal loop",
            stale_after_seconds=300,
        ),
        "ingestion_scheduler": FreshnessSpec(
            service_name="ingestion_scheduler",
            label="Ingestion scheduler",
            stale_after_seconds=3600,
        ),
    }
    telemetry = RuntimeTelemetry(
        session_factory=session_factory,
        freshness_specs=freshness_specs,
    )

    stale_service = ServiceFreshnessState(
        service_name="signal_loop",
        label="Signal loop",
        status="stale",
        last_success_at=datetime(2026, 4, 18, 8, 0, tzinfo=UTC),
        last_error_at=None,
        last_error_message=None,
        stale_after_seconds=300,
        age_seconds=7200,
        alertable=True,
    )
    ok_service = ServiceFreshnessState(
        service_name="ingestion_scheduler",
        label="Ingestion scheduler",
        status="ok",
        last_success_at=datetime(2026, 4, 18, 11, 0, tzinfo=UTC),
        last_error_at=None,
        last_error_message=None,
        stale_after_seconds=3600,
        age_seconds=120,
        alertable=True,
    )
    telemetry.evaluate_service_states = MagicMock(return_value=[stale_service, ok_service])  # type: ignore[method-assign]

    session = _make_session()
    app = _make_app(session, runtime_telemetry=telemetry)
    client = TestClient(app)

    resp = client.get("/api/system/health")
    assert resp.status_code == 200

    services = resp.json()["services"]
    assert len(services) == 2

    signal_svc = next(s for s in services if s["service_name"] == "signal_loop")
    assert signal_svc["status"] == "stale"
    assert signal_svc["age_seconds"] == 7200

    ingestion_svc = next(s for s in services if s["service_name"] == "ingestion_scheduler")
    assert ingestion_svc["status"] == "ok"


def test_system_health_returns_pending_orders_detail() -> None:
    """pending_orders_detail surfaces each pending row, oldest first."""
    import uuid as _uuid
    older_id = _uuid.uuid4()
    newer_id = _uuid.uuid4()
    now = datetime.now(UTC)
    from datetime import timedelta
    older_entry = now - timedelta(seconds=600)
    newer_entry = now - timedelta(seconds=120)

    pending_detail_result = MagicMock()
    pending_detail_result.all.return_value = [
        (older_id, "MKT-A", 10, 0, "resting", older_entry, now),
        (newer_id, "MKT-B", 5, 0, "resting", newer_entry, now),
    ]

    rs_row = MagicMock()
    rs_row.state = "running"
    rs_row.mode = "paper"
    rs_row.cb_active = False
    rs_row.cb_reason = None
    rs_row.daily_loss_ack_at = None
    rs_result = MagicMock()
    rs_result.scalar_one_or_none.return_value = rs_row

    hb_result = MagicMock()
    hb_result.scalars.return_value.all.return_value = []

    changelog_result = MagicMock()
    changelog_result.scalar_one_or_none.return_value = None

    session = AsyncMock()
    session.execute = _execute_side_effects(
        rs_result,                       # _get_mode
        rs_result,                       # RunStateRow
        _scalar_result(1000.0),          # get_net_bankroll
        _scalar_result(0.0),             # daily_pnl
        _scalar_result(0.0),             # llm_spend
        _scalar_result(2),               # pending count
        _scalar_result(older_entry),     # oldest pending entry_time
        pending_detail_result,           # pending orders detail
        _scalar_result(0),               # open count
        _scalar_result(0),               # llm errors
        _scalar_result(0),               # kalshi errors
        changelog_result,                # changelog
        hb_result,                       # heartbeats
    )

    telemetry = RuntimeTelemetry(session_factory=MagicMock(), freshness_specs={})
    app = _make_app(session, runtime_telemetry=telemetry)
    client = TestClient(app)

    resp = client.get("/api/system/health")
    assert resp.status_code == 200
    data = resp.json()
    detail = data["pending_orders_detail"]
    assert len(detail) == 2
    # oldest first
    assert detail[0]["market_id"] == "MKT-A"
    assert detail[1]["market_id"] == "MKT-B"
    assert detail[0]["age_seconds"] >= detail[1]["age_seconds"]


def test_system_health_without_telemetry_returns_empty_services() -> None:
    """Without RuntimeTelemetry, services list is empty and websocket is unknown."""
    session = _make_session()
    session.execute = _execute_side_effects(
        *[_scalar_result(v) for v in [
            MagicMock(scalar_one_or_none=MagicMock(return_value=MagicMock(
                state="running", mode="paper", cb_active=False, cb_reason=None,
                daily_loss_ack_at=None,
            ))),
        ]],
    )

    session2 = _make_session()
    app = _make_app(session2, runtime_telemetry=None)
    client = TestClient(app)

    resp = client.get("/api/system/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["services"] == []
    assert data["websocket"]["status"] == "unknown"
    assert data["websocket"]["connected"] is None
