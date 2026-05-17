"""Unit tests for freqpred/trading/risk.py.

All DB calls are mocked — no external dependencies.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.config import RiskConfig
from freqpred.markets.models import Market
from freqpred.signal.models import Signal
from freqpred.trading.risk import RiskDecision, RiskEngine, TradingCircuitBreakerError

# Ensure ORM relationships resolve
import freqpred.ingestion.models   # noqa: F401
import freqpred.llm.models         # noqa: F401
import freqpred.markets.models     # noqa: F401
import freqpred.rag.models         # noqa: F401
import freqpred.signal.models      # noqa: F401

from datetime import timedelta

NOW = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
BANKROLL = 2000.0


def _make_config(**overrides: object) -> RiskConfig:
    defaults = dict(
        max_position_pct=0.05,
        max_daily_loss_pct=0.15,
        max_total_exposure_pct=0.40,
        min_edge_floor=0.10,
        max_open_positions=20,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)  # type: ignore[arg-type]


def _make_signal(edge: float = 0.15) -> Signal:
    return Signal(
        id=str(uuid.uuid4()),
        market_id="MKT-X",
        estimated_probability=0.60,
        confidence=0.80,
        edge=edge,
        market_mid_at_signal=0.45,
        direction="YES",
        reasoning="test",
        sources=[],
        retrieval_hash="x" * 64,
        model_used="claude",
        prompt_version="v1",
        trigger="manual",
        created_at=NOW,
        raw_context="",
    )


def _make_session(
    open_count: int = 0,
    total_exposure: float = 0.0,
    daily_pnl: float = 0.0,
    all_pnl: float = 0.0,
    market_exposure: float = 0.0,
    recent_stoploss_count: int | None = None,
) -> MagicMock:
    """Return a mock AsyncSession whose execute() returns canned scalar results.

    Query order matches check_position:
      (optional) stoploss count  — only when recent_stoploss_count is provided
      1. market_exposure (per-market cumulative)
      2. open_count
      3. total_exposure (portfolio-wide)
      4. daily_pnl

    Pass recent_stoploss_count when the test enables stoploss_cooldown_hours or
    block_reentry_after_stoploss, so the mock has the right value queued.
    """
    call_returns: list[object] = []
    if recent_stoploss_count is not None:
        call_returns.append(recent_stoploss_count)
    call_returns += [market_exposure, open_count, total_exposure, daily_pnl]

    session = MagicMock()

    async def _execute(stmt: object) -> MagicMock:
        result = MagicMock()
        result.scalar_one.return_value = call_returns.pop(0)
        return result

    session.execute = _execute
    return session


def _make_circuit_session(daily_pnl: float = 0.0) -> MagicMock:
    """Session for circuit-breaker calls (1 query: daily P&L).

    The drawdown check no longer queries the DB — it uses the
    ``drawdown_reset_bankroll`` value passed directly to ``check_circuit_breakers``.
    """
    call_returns = [daily_pnl]

    session = MagicMock()

    async def _execute(stmt: object) -> MagicMock:
        result = MagicMock()
        result.scalar_one.return_value = call_returns.pop(0)
        return result

    session.execute = _execute
    return session


# ---------------------------------------------------------------------------
# check_position tests
# ---------------------------------------------------------------------------


MARKET_ID = "MKT-X"
MAX_MARKET_EXPOSURE = BANKROLL * 0.05  # $100 (5% of $2000)


@pytest.mark.asyncio
async def test_blocks_when_edge_below_floor() -> None:
    engine = RiskEngine(_make_config(min_edge_floor=0.10))
    signal = _make_signal(edge=0.05)
    session = MagicMock()  # should not be called — edge check is first

    decision = await engine.check_position(
        session, signal, requested_size=100.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
    )

    assert decision.allowed is False
    assert "edge" in decision.reason
    assert decision.capped_size == 0.0


@pytest.mark.asyncio
async def test_caps_size_at_max_position_pct() -> None:
    # 5% of 2000 = 100; requesting 200 → should be capped at 100
    engine = RiskEngine(_make_config(max_position_pct=0.05))
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=0, total_exposure=0.0, daily_pnl=0.0)

    decision = await engine.check_position(
        session, signal, requested_size=200.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
    )

    assert decision.allowed is True
    assert decision.capped_size == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_blocks_when_market_exposure_at_limit() -> None:
    # Market already has $100 open, limit is $100 → no capacity remaining
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=1, total_exposure=100.0, daily_pnl=0.0, market_exposure=100.0)

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=100.0,
    )

    assert decision.allowed is False
    assert "market" in decision.reason
    assert MARKET_ID in decision.reason


@pytest.mark.asyncio
async def test_caps_size_to_remaining_market_capacity() -> None:
    # Market limit $100, already $70 open → only $30 remaining; $60 request → capped to $30
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=1, total_exposure=70.0, daily_pnl=0.0, market_exposure=70.0)

    decision = await engine.check_position(
        session, signal, requested_size=60.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=100.0,
    )

    assert decision.allowed is True
    assert decision.capped_size == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_blocks_when_max_open_positions_reached() -> None:
    engine = RiskEngine(_make_config(max_open_positions=20))
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=20, total_exposure=0.0, daily_pnl=0.0)

    decision = await engine.check_position(
        session, signal, requested_size=100.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
    )

    assert decision.allowed is False
    assert "open positions" in decision.reason


@pytest.mark.asyncio
async def test_blocks_when_total_exposure_exceeded() -> None:
    # 40% of 2000 = 800; mock exposure = 820 → should block
    engine = RiskEngine(_make_config(max_total_exposure_pct=0.40))
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=5, total_exposure=820.0, daily_pnl=0.0)

    decision = await engine.check_position(
        session, signal, requested_size=100.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
    )

    assert decision.allowed is False
    assert "exposure" in decision.reason


@pytest.mark.asyncio
async def test_blocks_when_total_exposure_exactly_at_limit() -> None:
    # 40% of 2000 = 800; existing exposure = 800 → >= check should block
    engine = RiskEngine(_make_config(max_total_exposure_pct=0.40))
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=5, total_exposure=800.0, daily_pnl=0.0)

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
    )

    assert decision.allowed is False
    assert "exposure" in decision.reason


@pytest.mark.asyncio
async def test_caps_size_to_remaining_total_exposure_capacity() -> None:
    # 40% of 2000 = 800; existing exposure = 750 → only $50 headroom.
    # Requesting $100 should be capped to $50 (not blocked outright).
    engine = RiskEngine(_make_config(max_total_exposure_pct=0.40, max_position_pct=0.10))
    signal = _make_signal(edge=0.20)
    # market_exposure=0 so per-market cap doesn't interfere; MAX_MARKET_EXPOSURE large enough
    session = _make_session(open_count=5, total_exposure=750.0, daily_pnl=0.0, market_exposure=0.0)

    decision = await engine.check_position(
        session, signal, requested_size=100.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=500.0,
    )

    assert decision.allowed is True
    assert decision.capped_size == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_allows_valid_position() -> None:
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=5, total_exposure=200.0, daily_pnl=0.0)

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
    )

    assert decision.allowed is True
    assert decision.reason == ""
    assert decision.capped_size == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_blocks_when_daily_loss_exceeded() -> None:
    # 15% of 2000 = 300; daily loss = -320 → block
    engine = RiskEngine(_make_config(max_daily_loss_pct=0.15))
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=0, total_exposure=0.0, daily_pnl=-320.0)

    decision = await engine.check_position(
        session, signal, requested_size=100.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
    )

    assert decision.allowed is False
    assert "daily loss" in decision.reason


# ---------------------------------------------------------------------------
# check_circuit_breakers tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_fires_on_daily_loss() -> None:
    # 15% of 2000 = 300; daily loss = -320 → should raise (paper or live)
    engine = RiskEngine(_make_config(max_daily_loss_pct=0.15))
    session = _make_circuit_session(daily_pnl=-320.0)

    with pytest.raises(TradingCircuitBreakerError, match="daily loss"):
        await engine.check_circuit_breakers(session, bankroll=BANKROLL, mode="paper")


@pytest.mark.asyncio
async def test_circuit_breaker_does_not_refire_after_ack() -> None:
    """After /start, losses before ack_at don't immediately re-trip the breaker.

    The mock returns 0.0 pnl to represent the post-ack window (no new losses).
    The breaker should not fire.
    """
    engine = RiskEngine(_make_config(max_daily_loss_pct=0.15))
    # Post-ack window has no new losses
    session = _make_circuit_session(daily_pnl=0.0)
    ack_at = NOW  # acknowledged right now

    # Should NOT raise — the loss window starts at ack_at, so daily_pnl = 0
    await engine.check_circuit_breakers(
        session, bankroll=BANKROLL, mode="paper", daily_loss_ack_at=ack_at
    )


@pytest.mark.asyncio
async def test_circuit_breaker_refires_on_new_losses_after_ack() -> None:
    """New losses after ack_at that exceed the threshold still trip the breaker."""
    engine = RiskEngine(_make_config(max_daily_loss_pct=0.15))
    # New losses since ack_at: -320 (exceeds 15% of 2000 = 300)
    session = _make_circuit_session(daily_pnl=-320.0)
    ack_at = NOW - timedelta(hours=1)

    with pytest.raises(TradingCircuitBreakerError, match="daily loss"):
        await engine.check_circuit_breakers(
            session, bankroll=BANKROLL, mode="paper", daily_loss_ack_at=ack_at
        )


@pytest.mark.asyncio
async def test_check_position_skips_pre_ack_losses() -> None:
    """check_position also respects daily_loss_ack_at: zero post-ack losses → allowed."""
    engine = RiskEngine(_make_config(max_daily_loss_pct=0.15))
    signal = _make_signal(edge=0.20)
    # daily_pnl=0 represents post-ack window only
    session = _make_session(open_count=0, total_exposure=0.0, daily_pnl=0.0)
    ack_at = NOW

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
        daily_loss_ack_at=ack_at,
    )

    assert decision.allowed is True


@pytest.mark.asyncio
async def test_circuit_breaker_fires_on_drawdown() -> None:
    # baseline=2000, current=1300 → drawdown = 700/2000 = 35% > 30%
    engine = RiskEngine(_make_config())
    session = _make_circuit_session(daily_pnl=0.0)

    with pytest.raises(TradingCircuitBreakerError, match="drawdown"):
        await engine.check_circuit_breakers(
            session, bankroll=1300.0, mode="live", drawdown_reset_bankroll=2000.0
        )


@pytest.mark.asyncio
async def test_circuit_breaker_fires_on_drawdown_paper_mode() -> None:
    # Drawdown CB fires in paper mode too.
    engine = RiskEngine(_make_config())
    session = _make_circuit_session(daily_pnl=0.0)

    with pytest.raises(TradingCircuitBreakerError, match="drawdown"):
        await engine.check_circuit_breakers(
            session, bankroll=1300.0, mode="paper", drawdown_reset_bankroll=2000.0
        )


@pytest.mark.asyncio
async def test_circuit_breaker_silent_when_within_limits() -> None:
    engine = RiskEngine(_make_config())
    session = _make_circuit_session(daily_pnl=-10.0)

    # No drawdown_reset_bankroll supplied → drawdown check is skipped; should not raise
    await engine.check_circuit_breakers(session, bankroll=BANKROLL, mode="live")


@pytest.mark.asyncio
async def test_circuit_breaker_reset_clears_drawdown() -> None:
    """When reset_bankroll == current bankroll, drawdown is 0% — no fire."""
    engine = RiskEngine(_make_config())
    session = _make_circuit_session(daily_pnl=0.0)

    # Should NOT raise — drawdown = (1300 - 1300) / 1300 = 0%
    await engine.check_circuit_breakers(
        session, bankroll=1300.0, mode="paper", drawdown_reset_bankroll=1300.0
    )


@pytest.mark.asyncio
async def test_circuit_breaker_still_fires_if_losses_since_reset_exceed_limit() -> None:
    """Losses from reset baseline that exceed the 30% limit still trigger the CB."""
    engine = RiskEngine(_make_config())
    session = _make_circuit_session(daily_pnl=0.0)

    # baseline=2000, current=1300 → drawdown = 35% > 30%
    with pytest.raises(TradingCircuitBreakerError, match="drawdown"):
        await engine.check_circuit_breakers(
            session, bankroll=1300.0, mode="paper", drawdown_reset_bankroll=2000.0
        )


# ---------------------------------------------------------------------------
# stoploss re-entry guard tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocks_reentry_during_cooldown() -> None:
    """Position is blocked when a stoploss exit exists within cooldown window."""
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    # recent_stoploss_count=1 → there was a stoploss within the cooldown window
    session = _make_session(recent_stoploss_count=1)

    decision = await engine.check_position(
        session, signal, requested_size=100.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
        stoploss_cooldown_hours=4.0,
    )

    assert decision.allowed is False
    assert "cooldown" in decision.reason
    assert MARKET_ID in decision.reason


@pytest.mark.asyncio
async def test_allows_reentry_after_cooldown_window() -> None:
    """Position is allowed when no stoploss exits exist within the cooldown window."""
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    # recent_stoploss_count=0 → no stoploss within cooldown window
    session = _make_session(recent_stoploss_count=0)

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
        stoploss_cooldown_hours=4.0,
    )

    assert decision.allowed is True


@pytest.mark.asyncio
async def test_blocks_reentry_permanently_when_flag_set() -> None:
    """Position is permanently blocked when block_reentry_after_stoploss=True."""
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    session = _make_session(recent_stoploss_count=1)

    decision = await engine.check_position(
        session, signal, requested_size=100.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
        block_reentry_after_stoploss=True,
    )

    assert decision.allowed is False
    assert "block_reentry_after_stoploss" in decision.reason


@pytest.mark.asyncio
async def test_blocks_reentry_after_signal_loss_exit() -> None:
    """Signal exit with a loss triggers the same cooldown as a stoploss."""
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    # Count=1 simulates a signal exit with pnl < 0 found in the window.
    session = _make_session(recent_stoploss_count=1)

    decision = await engine.check_position(
        session, signal, requested_size=100.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
        stoploss_cooldown_hours=48.0,
    )

    assert decision.allowed is False
    assert "cooldown" in decision.reason


@pytest.mark.asyncio
async def test_allows_reentry_after_signal_win_exit() -> None:
    """Signal exit that was profitable does not trigger the cooldown."""
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    # Count=0 simulates no loss exits (profitable signal exit not counted).
    session = _make_session(recent_stoploss_count=0)

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
        stoploss_cooldown_hours=48.0,
    )

    assert decision.allowed is True


@pytest.mark.asyncio
async def test_cooldown_disabled_when_zero() -> None:
    """No stoploss DB query fired when stoploss_cooldown_hours=0 and flag is False."""
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    # Do NOT pass recent_stoploss_count — if the check fires it will pop from an
    # empty list and raise IndexError, proving the guard was incorrectly triggered.
    session = _make_session(open_count=0, total_exposure=0.0, daily_pnl=0.0)

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
        stoploss_cooldown_hours=0.0,
        block_reentry_after_stoploss=False,
    )

    assert decision.allowed is True


# ---------------------------------------------------------------------------
# Helper factories for new public method tests
# ---------------------------------------------------------------------------


def _make_capacity_session(open_count: int, total_exposure: float) -> MagicMock:
    """Session for check_entry_capacity: open_count then total_exposure."""
    call_returns: list[object] = [open_count, total_exposure]
    session = MagicMock()

    async def _execute(stmt: object) -> MagicMock:
        result = MagicMock()
        result.scalar_one.return_value = call_returns.pop(0)
        return result

    session.execute = _execute
    return session


def _make_stoploss_session(stoploss_count: int) -> MagicMock:
    """Session for _check_stoploss_reentry: single stoploss count query."""
    call_returns: list[object] = [stoploss_count]
    session = MagicMock()

    async def _execute(stmt: object) -> MagicMock:
        result = MagicMock()
        result.scalar_one.return_value = call_returns.pop(0)
        return result

    session.execute = _execute
    return session


def _make_market(yes_bid: float = 0.40, yes_ask: float = 0.50) -> Market:
    return Market(
        id=MARKET_ID,
        platform="kalshi",
        question="Will X happen?",
        category="Politics",
        close_time=NOW,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        mid_price=(yes_bid + yes_ask) / 2,
        volume_24h=1000.0,
        open_interest=500.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
    )


# ---------------------------------------------------------------------------
# check_entry_capacity tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_entry_capacity_max_positions_blocked() -> None:
    engine = RiskEngine(_make_config(max_open_positions=5))
    session = _make_capacity_session(open_count=5, total_exposure=0.0)

    blocked, reason = await engine.check_entry_capacity(session, BANKROLL, mode="paper")

    assert blocked is True
    assert "open positions" in reason


@pytest.mark.asyncio
async def test_check_entry_capacity_total_exposure_blocked() -> None:
    # 40% of 2000 = 800; existing exposure = 850 → blocked
    engine = RiskEngine(_make_config(max_total_exposure_pct=0.40))
    session = _make_capacity_session(open_count=3, total_exposure=850.0)

    blocked, reason = await engine.check_entry_capacity(session, BANKROLL, mode="paper")

    assert blocked is True
    assert "exposure" in reason


@pytest.mark.asyncio
async def test_check_entry_capacity_ok() -> None:
    engine = RiskEngine(_make_config(max_open_positions=20, max_total_exposure_pct=0.40))
    session = _make_capacity_session(open_count=3, total_exposure=200.0)

    blocked, reason = await engine.check_entry_capacity(session, BANKROLL, mode="paper")

    assert blocked is False
    assert reason == ""


# ---------------------------------------------------------------------------
# pre_signal_gate tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_signal_gate_spread_too_wide() -> None:
    engine = RiskEngine(_make_config())
    market = _make_market(yes_bid=0.40, yes_ask=0.60)  # spread = 0.20
    session = MagicMock()  # should not be called — spread fails first

    blocked, reason = await engine.pre_signal_gate(
        session, market, mode="paper",
        effective_max_spread=0.10,
        block_reentry_after_stoploss=False,
        stoploss_cooldown_hours=0.0,
    )

    assert blocked is True
    assert "spread" in reason
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_pre_signal_gate_stoploss_block_reentry() -> None:
    engine = RiskEngine(_make_config())
    market = _make_market(yes_bid=0.45, yes_ask=0.50)  # spread = 0.05, within 0.10 max
    session = _make_stoploss_session(stoploss_count=1)

    blocked, reason = await engine.pre_signal_gate(
        session, market, mode="paper",
        effective_max_spread=0.10,
        block_reentry_after_stoploss=True,
        stoploss_cooldown_hours=0.0,
    )

    assert blocked is True
    assert "block_reentry_after_stoploss" in reason


@pytest.mark.asyncio
async def test_pre_signal_gate_stoploss_cooldown_active() -> None:
    engine = RiskEngine(_make_config())
    market = _make_market(yes_bid=0.45, yes_ask=0.50)
    session = _make_stoploss_session(stoploss_count=2)

    blocked, reason = await engine.pre_signal_gate(
        session, market, mode="paper",
        effective_max_spread=0.10,
        block_reentry_after_stoploss=False,
        stoploss_cooldown_hours=4.0,
    )

    assert blocked is True
    assert "cooldown" in reason


@pytest.mark.asyncio
async def test_pre_signal_gate_stoploss_cooldown_expired() -> None:
    engine = RiskEngine(_make_config())
    market = _make_market(yes_bid=0.45, yes_ask=0.50)
    session = _make_stoploss_session(stoploss_count=0)

    blocked, reason = await engine.pre_signal_gate(
        session, market, mode="paper",
        effective_max_spread=0.10,
        block_reentry_after_stoploss=False,
        stoploss_cooldown_hours=4.0,
    )

    assert blocked is False
    assert reason == ""


@pytest.mark.asyncio
async def test_pre_signal_gate_ok() -> None:
    """No spread violation, no stoploss guards configured → gate passes."""
    engine = RiskEngine(_make_config())
    market = _make_market(yes_bid=0.45, yes_ask=0.50)
    session = MagicMock()  # should not be called when both guards are disabled

    blocked, reason = await engine.pre_signal_gate(
        session, market, mode="paper",
        effective_max_spread=0.10,
        block_reentry_after_stoploss=False,
        stoploss_cooldown_hours=0.0,
    )

    assert blocked is False
    assert reason == ""
    session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# check_position regression: behavior unchanged after helper extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_position_still_blocks_on_max_positions() -> None:
    """check_position still blocks when max positions hit — via _check_global_capacity."""
    engine = RiskEngine(_make_config(max_open_positions=3))
    signal = _make_signal(edge=0.20)
    session = _make_session(open_count=3, total_exposure=0.0, daily_pnl=0.0)

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
    )

    assert decision.allowed is False
    assert "open positions" in decision.reason


@pytest.mark.asyncio
async def test_check_position_still_blocks_on_stoploss_cooldown() -> None:
    """check_position still blocks stoploss cooldown — via _check_stoploss_reentry."""
    engine = RiskEngine(_make_config())
    signal = _make_signal(edge=0.20)
    session = _make_session(recent_stoploss_count=1)

    decision = await engine.check_position(
        session, signal, requested_size=50.0, bankroll=BANKROLL,
        market_id=MARKET_ID, max_market_exposure=MAX_MARKET_EXPOSURE,
        stoploss_cooldown_hours=4.0,
    )

    assert decision.allowed is False
    assert "cooldown" in decision.reason
