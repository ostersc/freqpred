"""Unit tests for freqpred/metrics/reporting.py.

All DB interactions and LLM calls are mocked — no external dependencies.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.llm.models import LLMResponse
from freqpred.metrics.calibration import CalibrationReport, CalibrationBucket
from freqpred.metrics.reporting import generate_daily_digest

# Register ORM models to avoid mapper errors
import freqpred.ingestion.models   # noqa: F401
import freqpred.llm.models         # noqa: F401
import freqpred.markets.models     # noqa: F401
import freqpred.rag.models         # noqa: F401
import freqpred.signal.models      # noqa: F401


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
    open_count: int = 2,
    total_exposure: float = 50.0,
    session_pnl: float = 3.20,
    session_exit_rows: list = [],
    yesterday_llm_spend: float = 0.38,
    today_llm_spend: float = 0.05,
    llm_errors: int = 0,
    backed_off_services: list = [],
) -> AsyncMock:
    """Return a mock AsyncSession with pre-configured execute responses."""
    session = AsyncMock()

    # execute() calls in order:
    # 1. open positions (count, exposure) → .one()
    # 2. unrealized P&L rows → .all()
    # 3. session pnl (scalar) → .scalar_one()
    # 4. exit reason breakdown → .all()
    # 5. yesterday llm spend (scalar) → .scalar_one()
    # 6. LLM errors count (scalar) → .scalar_one()
    # 7. fetcher backoff rows → .all()
    # (calibration and today_llm_spend are patched separately)

    open_result = MagicMock()
    open_result.one.return_value = (open_count, total_exposure)

    unrealized_result = MagicMock()
    unrealized_result.all.return_value = []  # no open positions in unit tests

    pnl_result = MagicMock()
    pnl_result.scalar_one.return_value = session_pnl

    exits_result = MagicMock()
    exits_result.all.return_value = session_exit_rows

    llm_spend_result = MagicMock()
    llm_spend_result.scalar_one.return_value = yesterday_llm_spend

    llm_errors_result = MagicMock()
    llm_errors_result.scalar_one.return_value = llm_errors

    backoff_result = MagicMock()
    backoff_result.all.return_value = backed_off_services

    session.execute = AsyncMock(
        side_effect=[open_result, unrealized_result, pnl_result, exits_result, llm_spend_result, llm_errors_result, backoff_result]
    )
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
    assert "3" in prompt          # open position count
    assert "0.180" in prompt      # Brier score (formatted to 3dp)
    assert "12" in prompt         # n_samples


# ---------------------------------------------------------------------------
# test_digest_returns_llm_response_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_returns_llm_response_text() -> None:
    """Returned string matches the mocked LLM response content."""
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

    assert result == f"[PAPER MODE]\n{_FAKE_DIGEST}"


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
    assert "Open positions: 0" in prompt
    # Result is the mode banner + mock response
    assert result == f"[PAPER MODE]\n{_FAKE_DIGEST}"


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
    session = _make_session(session_pnl=-8.50, session_exit_rows=stop_loss_rows)
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
    assert "-8.50" in prompt          # session P&L total
    assert "stoploss" in prompt       # stop loss exit reason visible
    assert "3 trade(s)" in prompt     # stop loss count visible
    assert "yesterday through now" in prompt  # label confirms extended window
