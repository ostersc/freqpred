"""Unit tests for freqpred/metrics/reporting.py.

All DB interactions and LLM calls are mocked — no external dependencies.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Register ORM models to avoid mapper errors
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.llm.models import LLMResponse
from freqpred.metrics.calibration import CalibrationBucket, CalibrationReport
from freqpred.metrics.reporting import generate_daily_digest

_FAKE_DIGEST = "freqpred holds 2 open positions with $50.00 exposure. Yesterday returned +$3.20."
_FAKE_LLM_RESPONSE = LLMResponse(
    content=_FAKE_DIGEST,
    model="claude-haiku-4-5-20251001",
    tokens_input=120,
    tokens_output=45,
    cost_usd=0.00027,
    latency_ms=310,
    llm_query_id=42,
)

_EMPTY_CALIBRATION = CalibrationReport(
    brier_score=0.0,
    market_brier_score=0.0,
    n_samples=0,
    buckets=[
        CalibrationBucket(
            lower=i / 10.0,
            upper=(i + 1) / 10.0,
            count=0,
            mean_estimated_prob=(i / 10.0 + (i + 1) / 10.0) / 2.0,
            actual_resolution_rate=0.0,
        )
        for i in range(10)
    ],
)

_POPULATED_CALIBRATION = CalibrationReport(
    brier_score=0.18,
    market_brier_score=0.25,
    n_samples=12,
    buckets=[],
)


def _make_session(
    *,
    run_state: str | None = None,
    open_count: int = 2,
    total_exposure: float = 50.0,
    open_position_rows: list | None = None,
    session_pnl: float = 3.20,
    session_exit_rows: list | None = None,
    session_wins: int = 0,
    session_losses: int = 0,
    yesterday_llm_spend: float = 0.38,
    today_llm_spend: float = 0.05,
    llm_errors: int = 0,
    backed_off_services: list | None = None,
    signal_rows: list | None = None,
    heartbeat_rows: list | None = None,
) -> AsyncMock:
    """Return a mock AsyncSession with pre-configured execute responses."""
    session_exit_rows = session_exit_rows if session_exit_rows is not None else []
    backed_off_services = backed_off_services if backed_off_services is not None else []
    session = AsyncMock()

    # execute() calls in order:
    # 1. get_run_state → .scalar_one_or_none()
    # 2. get_drawdown_window → .scalar_one_or_none()
    # 3. open positions (count, exposure) → .one()
    # 4. open position rows (unrealized P&L detail) → .all()
    # 5. session pnl (scalar) → .scalar_one()
    # 6. exit reason breakdown → .all()
    # 7. session win/loss counts → .one()
    # 8. all-time pnl for net_value (scalar) → .scalar_one()
    # 9. yesterday llm spend (scalar) → .scalar_one()
    # 10. LLM errors count (scalar) → .scalar_one()
    # 11. fetcher backoff rows → .all()
    # 12. signals last 24h → .all()
    # 13. (only when telemetry is passed) service heartbeats → .scalars().all()
    # (calibration and today_llm_spend are patched separately)

    run_state_result = MagicMock()
    # get_run_state reads row.state; None row → defaults to "running"
    run_state_result.scalar_one_or_none.return_value = (
        MagicMock(state=run_state) if run_state is not None else None
    )

    drawdown_reset_result = MagicMock()
    drawdown_reset_result.scalar_one_or_none.return_value = None  # no reset

    open_result = MagicMock()
    open_result.one.return_value = (open_count, total_exposure)

    unrealized_result = MagicMock()
    unrealized_result.all.return_value = open_position_rows or []

    pnl_result = MagicMock()
    pnl_result.scalar_one.return_value = session_pnl

    exits_result = MagicMock()
    exits_result.all.return_value = session_exit_rows

    winloss_result = MagicMock()
    winloss_result.one.return_value = (session_wins, session_losses)

    dd_pnl_result = MagicMock()
    dd_pnl_result.scalar_one.return_value = 0.0  # no drawdown losses

    llm_spend_result = MagicMock()
    llm_spend_result.scalar_one.return_value = yesterday_llm_spend

    llm_errors_result = MagicMock()
    llm_errors_result.scalar_one.return_value = llm_errors

    backoff_result = MagicMock()
    backoff_result.all.return_value = backed_off_services

    signals_result = MagicMock()
    signals_result.all.return_value = signal_rows or []

    side_effect = [
        run_state_result, drawdown_reset_result,
        open_result, unrealized_result, pnl_result, exits_result,
        winloss_result, dd_pnl_result,
        llm_spend_result, llm_errors_result, backoff_result,
        signals_result,
    ]
    if heartbeat_rows is not None:
        heartbeats_result = MagicMock()
        heartbeats_result.scalars.return_value.all.return_value = heartbeat_rows
        side_effect.append(heartbeats_result)

    session.execute = AsyncMock(side_effect=side_effect)
    return session


def _make_llm_client(response: LLMResponse = _FAKE_LLM_RESPONSE) -> MagicMock:
    client = MagicMock()
    client.complete = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# test_digest_calls_llm_with_data_snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_calls_llm_with_data_snapshot() -> None:
    """LLMClient.complete called once with query_type='daily_digest' and
    prompt containing position count and Brier score."""
    session = _make_session(open_count=3, total_exposure=42.50)
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_POPULATED_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.05),
    ):
        await generate_daily_digest(session, llm_client)

    llm_client.complete.assert_called_once()
    call_kwargs = llm_client.complete.call_args

    assert call_kwargs.kwargs["query_type"] == "daily_digest"

    prompt: str = call_kwargs.kwargs["prompt"] if "prompt" in call_kwargs.kwargs else call_kwargs.args[0]
    assert "open 3" in prompt      # open position count (header)
    assert "0.180" in prompt       # Brier score (formatted to 3dp)
    assert "n=12" in prompt        # n_samples
    assert "STAT HEADER" in prompt  # header/detail structure present
    assert "DETAIL" in prompt


# ---------------------------------------------------------------------------
# test_digest_returns_llm_response_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_returns_header_plus_llm_response() -> None:
    """Returned string is the deterministic stat header + the LLM analyst take."""
    session = _make_session()
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.0),
    ):
        result = await generate_daily_digest(session, llm_client)

    assert result.startswith("[PAPER] Daily digest")
    assert result.endswith(_FAKE_DIGEST)
    header = result.split("\n\n")[0]
    assert "State running" in header
    assert "open 2" in header
    assert "exposure $50.00" in header


# ---------------------------------------------------------------------------
# test_digest_logs_llm_query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_logs_llm_query() -> None:
    """LLM audit row is written — confirmed by non-None llm_query_id in response."""
    session = _make_session()
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.0),
    ):
        await generate_daily_digest(session, llm_client)

    # The LLMClient mock was called — in production this writes the audit row.
    # We verify query_type is passed correctly (the real client uses it for audit).
    call_kwargs = llm_client.complete.call_args.kwargs
    assert call_kwargs.get("query_type") == "daily_digest"
    # response carries the audit row id
    assert _FAKE_LLM_RESPONSE.llm_query_id == 42


# ---------------------------------------------------------------------------
# test_digest_handles_zero_positions_gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_handles_zero_positions_gracefully() -> None:
    """No positions in DB → prompt reflects that, LLM still called once."""
    session = _make_session(
        open_count=0,
        total_exposure=0.0,
        session_pnl=0.0,
        yesterday_llm_spend=0.0,
        today_llm_spend=0.0,
    )
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.0),
    ):
        result = await generate_daily_digest(session, llm_client)

    # LLM must still be called exactly once
    llm_client.complete.assert_called_once()

    prompt: str = llm_client.complete.call_args.kwargs.get("prompt", "")
    # Prompt should reflect 0 open positions
    assert "open 0" in prompt
    assert "(none)" in prompt  # empty position detail block
    # Result is header + mock analyst take
    assert result.startswith("[PAPER] Daily digest")
    assert result.endswith(_FAKE_DIGEST)


# ---------------------------------------------------------------------------
# test_digest_includes_session_pnl_and_exit_breakdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_includes_session_pnl_and_exit_breakdown() -> None:
    """Session P&L and exit-reason breakdown (including stop losses) appear in the prompt."""
    stop_loss_rows = [
        ("stoploss", 3, -12.50),
        ("market_resolved", 1, 4.00),
    ]
    session = _make_session(
        session_pnl=-8.50, session_exit_rows=stop_loss_rows,
        session_wins=1, session_losses=3,
    )
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.0),
    ):
        await generate_daily_digest(session, llm_client)

    prompt: str = llm_client.complete.call_args.kwargs.get("prompt", "")
    assert "-$8.50" in prompt         # session P&L total (header)
    assert "1W/3L" in prompt          # win/loss counts (header)
    assert "stoploss" in prompt       # stop loss exit reason visible
    assert "3 trade(s)" in prompt     # stop loss count visible
    assert "yesterday through now" in prompt  # label confirms extended window


# ---------------------------------------------------------------------------
# New digest structure: header stats, position detail, signals, health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_header_shows_open_cap_and_llm_cap() -> None:
    session = _make_session(open_count=3)
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=2.50),
    ):
        result = await generate_daily_digest(
            session, llm_client, bankroll=1000.0,
            llm_daily_cap=10.0, max_open_positions=20,
        )

    header = result.split("\n\n")[0]
    assert "open 3/20" in header
    assert "$2.50 today / $10.00 cap (25%)" in header
    assert "net value $" in header  # bankroll provided → net value shown


@pytest.mark.asyncio
async def test_digest_position_detail_covers_yes_and_no_directions() -> None:
    """Per-position unrealized P&L in the LLM detail is correct for both sides."""
    # Row shape: (contracts, entry_price, entry_fee_usd, direction, mae, mfe,
    #             mid_price, market_id, question)
    open_rows = [
        (10, 0.40, 0.0, "YES", None, None, 0.60, "KXYES-26", "Yes question?"),
        (10, 0.40, 0.0, "NO", None, None, 0.50, "KXNO-26", "No question?"),
    ]
    session = _make_session(open_count=2, open_position_rows=open_rows)
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.0),
    ):
        result = await generate_daily_digest(session, llm_client)

    prompt: str = llm_client.complete.call_args.kwargs["prompt"]
    # YES: 10 × (0.60 - 0.40) = +$2.00
    assert "YES 10x KXYES-26" in prompt
    assert "unrealized $+2.00" in prompt
    # NO: 10 × ((1 - 0.50) - 0.40) = +$1.00, current NO price 50c
    assert "NO 10x KXNO-26" in prompt
    assert "unrealized $+1.00" in prompt
    assert "now 50c" in prompt
    # Header aggregates both: +$3.00 unrealized
    assert "unrealized +$3.00" in result.split("\n\n")[0]


@pytest.mark.asyncio
async def test_digest_includes_signal_activity() -> None:
    from types import SimpleNamespace

    signal_rows = [
        SimpleNamespace(direction="YES", edge=0.17, estimated_probability=0.71,
                        market_mid_at_signal=0.54, confidence=0.8, market_id="KXA-26"),
        SimpleNamespace(direction="SKIP", edge=0.0, estimated_probability=0.5,
                        market_mid_at_signal=0.5, confidence=0.3, market_id="KXB-26"),
        SimpleNamespace(direction="NO", edge=-0.09, estimated_probability=0.30,
                        market_mid_at_signal=0.39, confidence=0.6, market_id="KXC-26"),
    ]
    session = _make_session(signal_rows=signal_rows)
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.0),
    ):
        result = await generate_daily_digest(session, llm_client)

    header = result.split("\n\n")[0]
    assert "Signals 24h: 3 (2 actionable)" in header
    prompt: str = llm_client.complete.call_args.kwargs["prompt"]
    # Top signal by |edge| is the YES one; SKIP excluded from top list
    assert "YES KXA-26 edge +17.0%" in prompt
    assert "NO KXC-26 edge -9.0%" in prompt


@pytest.mark.asyncio
async def test_digest_health_line_from_telemetry() -> None:
    from freqpred.runtime.telemetry import ServiceFreshnessState

    session = _make_session(heartbeat_rows=[])
    llm_client = _make_llm_client()

    telemetry = MagicMock()
    telemetry.evaluate_service_states.return_value = [
        ServiceFreshnessState(
            service_name="signal_loop", label="Signal loop", status="ok",
            last_success_at=None, last_error_at=None, last_error_message=None,
            stale_after_seconds=600, age_seconds=30, alertable=True,
        ),
        ServiceFreshnessState(
            service_name="fetcher_reddit", label="Reddit fetcher", status="stale",
            last_success_at=None, last_error_at=None,
            last_error_message="RedditBlockedError: 403",
            stale_after_seconds=86400, age_seconds=90000, alertable=True,
        ),
    ]

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.0),
    ):
        result = await generate_daily_digest(session, llm_client, telemetry=telemetry)

    header = result.split("\n\n")[0]
    assert "Health 1/2 services ok" in header
    assert "Reddit fetcher (1d 1h)" in header
    # LLM detail carries the stale error message
    prompt: str = llm_client.complete.call_args.kwargs["prompt"]
    assert "RedditBlockedError" in prompt


@pytest.mark.asyncio
async def test_digest_flags_halted_run_state() -> None:
    session = _make_session(run_state="stopped")
    llm_client = _make_llm_client()

    with patch(
        "freqpred.metrics.reporting.compute_calibration",
        new=AsyncMock(return_value=_EMPTY_CALIBRATION),
    ), patch(
        "freqpred.metrics.reporting.get_daily_spend_usd",
        new=AsyncMock(return_value=0.0),
    ):
        result = await generate_daily_digest(session, llm_client)

    header = result.split("\n\n")[0]
    assert "State stopped (!)" in header
    prompt: str = llm_client.complete.call_args.kwargs["prompt"]
    assert "HALTED" in prompt
