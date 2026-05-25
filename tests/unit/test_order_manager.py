"""Unit tests for freqpred/trading/order_manager.py.

All DB interactions and ledger writes are mocked — no external dependencies.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import Market, Order, Position
from freqpred.metrics.models import SignalAssessment
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig
from freqpred.trading.order_manager import OrderManager, PositionNotFoundError, PositionNotOpenError
from freqpred.trading.risk import RiskDecision, RiskEngine, TradingCircuitBreakerError

# Ensure ORM relationships resolve
import freqpred.ingestion.models   # noqa: F401
import freqpred.llm.models         # noqa: F401
import freqpred.markets.models     # noqa: F401
import freqpred.rag.models         # noqa: F401
import freqpred.signal.models      # noqa: F401

NOW = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
MARKET_ID = "MKT-TEST"
SIGNAL_ID = str(uuid.uuid4())
BANKROLL = 10_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_market(yes_bid: float = 0.50, yes_ask: float = 0.54) -> Market:
    return Market(
        id=MARKET_ID,
        platform="kalshi",
        question="Will X happen?",
        category="politics",
        close_time=NOW + timedelta(days=10),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        mid_price=(yes_bid + yes_ask) / 2,
        volume_24h=1000.0,
        open_interest=5000.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
    )


def _make_signal(
    direction: str = "YES",
    edge: float = 0.15,
    estimated_probability: float = 0.65,
) -> Signal:
    return Signal(
        id=SIGNAL_ID,
        market_id=MARKET_ID,
        estimated_probability=estimated_probability,
        confidence=0.80,
        edge=edge,
        market_mid_at_signal=0.53,
        direction=direction,
        reasoning="test",
        sources=[],
        retrieval_hash="a" * 64,
        model_used="claude-sonnet-4-6",
        prompt_version="v1",
        trigger="manual",
        created_at=NOW,
        raw_context="",
    )


def _make_strategy(
    should_trade_result: bool = True,
    position_size_result: float = 100.0,
) -> IPredictionStrategy:
    """Return a minimal concrete strategy stub."""

    class _Stub(IPredictionStrategy):
        config = StrategyConfig(
            name="TestStrategy",
            min_edge=0.10,
            min_confidence=0.70,
            max_exposure_per_market=0.10,
            kelly_fraction=0.25,
            categories=["politics"],
            min_volume_24h=0.0,
            max_days_to_close=90,
            min_days_to_close=1,
        )

        def should_trade(self, signal: Signal, market: Market) -> bool:
            return should_trade_result

        def position_size(self, signal: Signal, bankroll: float, existing_market_exposure: float = 0.0) -> float:
            return position_size_result

    return _Stub()


def _make_position(
    contracts: int = 100,
    entry_price: float = 0.56,
    direction: str = "YES",
) -> Position:
    return Position(
        id=str(uuid.uuid4()),
        market_id=MARKET_ID,
        signal_id=SIGNAL_ID,
        strategy_name="TestStrategy",
        strategy_version="1.0",
        signal_confidence=0.80,
        signal_edge=0.15,
        signal_estimated_prob=0.65,
        direction=direction,
        contracts=contracts,
        entry_price=entry_price,
        entry_time=NOW,
        mode="paper",
        status="open",
    )


def _make_order_manager(
    risk: RiskEngine | None = None,
    bankroll: float = BANKROLL,
    mode: str = "paper",
    kalshi_client: object = None,
    llm_client: object | None = None,
    judgment_model: str | None = None,
) -> tuple[OrderManager, MagicMock]:
    """Return (OrderManager, session_factory_mock)."""
    session_factory = MagicMock()
    # Make session_factory() return an async context manager
    mock_session = AsyncMock()
    # session.execute() returns a result whose .scalar_one() gives 0.0
    # (existing market exposure query). This is the default — tests that need
    # non-zero existing exposure should override mock_session.execute.
    _exposure_result = MagicMock()
    _exposure_result.scalar_one.return_value = 0.0
    mock_session.execute = AsyncMock(return_value=_exposure_result)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    session_factory.return_value = session_ctx

    if risk is None:
        risk = MagicMock(spec=RiskEngine)
        risk.check_circuit_breakers = AsyncMock(return_value=None)
        risk.check_position = AsyncMock(
            return_value=RiskDecision(allowed=True, reason="", capped_size=100.0)
        )

    om = OrderManager(
        risk=risk,
        session_factory=session_factory,
        bankroll=bankroll,
        mode=mode,
        strategy_version="1.0",
        kalshi_client=kalshi_client,
        llm_client=llm_client,
        judgment_model=judgment_model,
    )
    return om, session_factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_get_net_bankroll():
    """Patch ledger.get_net_bankroll to return BANKROLL in all tests.

    Individual tests that want a different net bankroll should override
    with their own patch inside the test body.
    """
    with patch(
        "freqpred.trading.order_manager.ledger.get_net_bankroll",
        new_callable=AsyncMock,
        return_value=BANKROLL,
    ):
        yield


@pytest.fixture(autouse=True)
def _patch_get_drawdown_window():
    """Patch get_drawdown_window so order_manager.submit() doesn't need a real DB."""
    with patch(
        "freqpred.alerts.run_state.get_drawdown_window",
        new_callable=AsyncMock,
        return_value=(None, None),
    ):
        yield


@pytest.fixture(autouse=True)
def _patch_get_daily_loss_ack_at():
    """Patch get_daily_loss_ack_at so order_manager.submit() doesn't need a real DB."""
    with patch(
        "freqpred.alerts.run_state.get_daily_loss_ack_at",
        new_callable=AsyncMock,
        return_value=None,
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_returns_none_when_strategy_declines() -> None:
    """should_trade() returns False → None, risk and ledger not called."""
    om, _ = _make_order_manager()
    strategy = _make_strategy(should_trade_result=False)

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(), _make_market(), strategy)

    assert result is None
    mock_ledger.assert_not_called()
    om._risk.check_position.assert_not_called()


@pytest.mark.asyncio
async def test_submit_returns_none_when_risk_blocks() -> None:
    """risk.check_position returns allowed=False → None, no ledger write."""
    risk = MagicMock(spec=RiskEngine)
    risk.check_circuit_breakers = AsyncMock(return_value=None)
    risk.check_position = AsyncMock(
        return_value=RiskDecision(
            allowed=False, reason="edge below floor", capped_size=0.0
        )
    )
    om, _ = _make_order_manager(risk=risk)
    strategy = _make_strategy(should_trade_result=True, position_size_result=200.0)

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(), _make_market(), strategy)

    assert result is None
    mock_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_submit_returns_none_when_spread_too_wide() -> None:
    """Spread > min_edge / 2 → None before strategy or risk checks."""
    om, _ = _make_order_manager()
    strategy = _make_strategy(should_trade_result=True)
    # spread = 0.60 - 0.50 = 0.10 > min_edge/2 = 0.05
    wide_market = _make_market(yes_bid=0.50, yes_ask=0.60)

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(), wide_market, strategy)

    assert result is None
    mock_ledger.assert_not_called()
    om._risk.check_position.assert_not_called()


@pytest.mark.asyncio
async def test_submit_respects_custom_max_spread() -> None:
    """Explicit max_spread on config overrides the min_edge/2 default."""

    class _WideAllowed(IPredictionStrategy):
        config = StrategyConfig(
            name="WideSpread",
            min_edge=0.10,
            min_confidence=0.70,
            max_exposure_per_market=0.10,
            kelly_fraction=0.25,
            categories=["politics"],
            min_volume_24h=0.0,
            max_days_to_close=90,
            min_days_to_close=1,
            max_spread=0.15,  # explicit override — wider than min_edge/2
        )

        def should_trade(self, signal: Signal, market: Market) -> bool:
            return True

        def position_size(self, signal: Signal, bankroll: float, existing_market_exposure: float = 0.0) -> float:
            return 100.0

    om, _ = _make_order_manager()
    strategy = _WideAllowed()
    # spread = 0.60 - 0.50 = 0.10 — would fail default (0.05) but passes custom (0.15)
    market = _make_market(yes_bid=0.50, yes_ask=0.60)
    expected_position = _make_position(entry_price=0.60)

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ) as mock_ledger:
        result = await om.submit(_make_signal(direction="YES"), market, strategy)

    assert result is expected_position
    mock_ledger.assert_called_once()


@pytest.mark.asyncio
async def test_submit_opens_position_when_all_clear() -> None:
    """Happy path: all checks pass → Position returned, ledger.open_position called."""
    expected_position = _make_position(contracts=178, entry_price=0.56)
    om, _ = _make_order_manager()
    # $100 / $0.56 = 178 contracts (floor)
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ) as mock_ledger:
        result = await om.submit(_make_signal(direction="YES"), _make_market(yes_bid=0.52, yes_ask=0.56), strategy)

    assert result is expected_position
    mock_ledger.assert_called_once()
    call_kwargs = mock_ledger.call_args.kwargs
    assert call_kwargs["direction"] == "YES"
    assert call_kwargs["entry_price"] == pytest.approx(0.56)
    assert call_kwargs["contracts"] == math.floor(100.0 / 0.56)
    assert call_kwargs["mode"] == "paper"


@pytest.mark.asyncio
async def test_contracts_floored_to_integer() -> None:
    """$55 / $0.54 = floor(101.85) = 101 contracts."""
    risk = MagicMock(spec=RiskEngine)
    risk.check_circuit_breakers = AsyncMock(return_value=None)
    risk.check_position = AsyncMock(
        return_value=RiskDecision(allowed=True, reason="", capped_size=55.0)
    )
    om, _ = _make_order_manager(risk=risk)
    strategy = _make_strategy(position_size_result=55.0)

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=_make_position(contracts=101, entry_price=0.54),
    ) as mock_ledger:
        await om.submit(_make_signal(direction="YES"), _make_market(yes_ask=0.54), strategy)

    call_kwargs = mock_ledger.call_args.kwargs
    assert call_kwargs["contracts"] == 101


@pytest.mark.asyncio
async def test_returns_none_when_contracts_below_one() -> None:
    """Tiny edge → floor < 1 → return None without ledger write."""
    risk = MagicMock(spec=RiskEngine)
    risk.check_circuit_breakers = AsyncMock(return_value=None)
    risk.check_position = AsyncMock(
        return_value=RiskDecision(allowed=True, reason="", capped_size=0.40)
    )
    om, _ = _make_order_manager(risk=risk)
    strategy = _make_strategy(position_size_result=0.40)

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(), _make_market(yes_bid=0.52, yes_ask=0.56), strategy)

    assert result is None
    mock_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_uses_yes_ask_as_entry_price_for_yes_direction() -> None:
    """YES direction → entry_price = market.yes_ask."""
    risk = MagicMock(spec=RiskEngine)
    risk.check_circuit_breakers = AsyncMock(return_value=None)
    risk.check_position = AsyncMock(
        return_value=RiskDecision(allowed=True, reason="", capped_size=100.0)
    )
    om, _ = _make_order_manager(risk=risk)
    strategy = _make_strategy(position_size_result=100.0)
    market = _make_market(yes_bid=0.56, yes_ask=0.60)

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=_make_position(entry_price=0.60),
    ) as mock_ledger:
        await om.submit(_make_signal(direction="YES"), market, strategy)

    assert mock_ledger.call_args.kwargs["entry_price"] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_uses_no_price_for_no_direction() -> None:
    """NO direction → entry_price = 1 - market.yes_bid."""
    risk = MagicMock(spec=RiskEngine)
    risk.check_circuit_breakers = AsyncMock(return_value=None)
    risk.check_position = AsyncMock(
        return_value=RiskDecision(allowed=True, reason="", capped_size=100.0)
    )
    om, _ = _make_order_manager(risk=risk)
    strategy = _make_strategy(position_size_result=100.0)
    market = _make_market(yes_bid=0.52, yes_ask=0.56)
    # NO price = 1 - 0.52 = 0.48

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=_make_position(direction="NO", entry_price=0.48),
    ) as mock_ledger:
        await om.submit(_make_signal(direction="NO"), market, strategy)

    call_kwargs = mock_ledger.call_args.kwargs
    assert call_kwargs["entry_price"] == pytest.approx(0.48)
    assert call_kwargs["direction"] == "NO"


@pytest.mark.asyncio
async def test_risk_uses_net_bankroll_not_initial() -> None:
    """OrderManager uses net bankroll (initial + closed P&L) for all risk calls.

    With initial bankroll $10k but $4k of realized losses, net bankroll = $6k.
    risk.check_circuit_breakers and risk.check_position should receive $6k.
    """
    net_bankroll = 6_000.0
    risk = MagicMock(spec=RiskEngine)
    risk.check_circuit_breakers = AsyncMock(return_value=None)
    risk.check_position = AsyncMock(
        return_value=RiskDecision(allowed=True, reason="", capped_size=100.0)
    )
    om, _ = _make_order_manager(risk=risk, bankroll=BANKROLL)
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)
    expected_position = _make_position()

    with patch(
        "freqpred.trading.order_manager.ledger.get_net_bankroll",
        new_callable=AsyncMock,
        return_value=net_bankroll,
    ), patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ):
        await om.submit(_make_signal(), _make_market(), strategy)

    risk.check_circuit_breakers.assert_called_once()
    # check_circuit_breakers(session, net_bankroll, mode=...) — bankroll is 2nd positional
    assert risk.check_circuit_breakers.call_args.args[1] == net_bankroll

    risk.check_position.assert_called_once()
    # check_position(session, signal, raw_size, net_bankroll, ...) — bankroll is 4th positional
    assert risk.check_position.call_args.args[3] == net_bankroll


@pytest.mark.asyncio
async def test_circuit_breaker_errors_propagate() -> None:
    """TradingCircuitBreakerError from check_circuit_breakers propagates up."""
    risk = MagicMock(spec=RiskEngine)
    risk.check_circuit_breakers = AsyncMock(
        side_effect=TradingCircuitBreakerError("daily loss limit hit")
    )
    risk.check_position = AsyncMock()
    om, _ = _make_order_manager(risk=risk)
    strategy = _make_strategy(should_trade_result=True)

    with pytest.raises(TradingCircuitBreakerError):
        await om.submit(_make_signal(), _make_market(), strategy)

    risk.check_position.assert_not_called()


# ---------------------------------------------------------------------------
# Live mode tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_mode_blocked_without_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """mode='live', LIVE_TRADING_ENABLED not set → returns None, logs live_blocked."""
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    mock_kalshi = AsyncMock()
    mock_kalshi.place_order = AsyncMock()
    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(), _make_market(), strategy)

    assert result is None
    mock_kalshi.place_order.assert_not_called()
    mock_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_live_mode_calls_place_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """mode='live', LIVE_TRADING_ENABLED=true → KalshiClient.place_order() called."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    expected_position = _make_position(contracts=178, entry_price=0.56)
    expected_position.exchange_order_id = "ORD-123"

    filled_order = Order(
        market_id=MARKET_ID,
        direction="YES",
        contracts=178,
        price=0.56,
        mode="live",
        exchange_order_id="ORD-123",
        status="resting",
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.place_order = AsyncMock(return_value=filled_order)

    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ):
        result = await om.submit(_make_signal(direction="YES"), _make_market(yes_bid=0.52, yes_ask=0.56), strategy)

    assert result is expected_position
    mock_kalshi.place_order.assert_called_once()
    submitted_order: Order = mock_kalshi.place_order.call_args.args[0]
    assert submitted_order.direction == "YES"
    assert submitted_order.contracts == math.floor(100.0 / 0.56)
    assert submitted_order.price == pytest.approx(0.56)
    assert submitted_order.mode == "live"


@pytest.mark.asyncio
async def test_live_mode_records_pending_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """mode='live' → ledger.open_position() called with status='pending' and exchange_order_id set."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    filled_order = Order(
        market_id=MARKET_ID,
        direction="YES",
        contracts=178,
        price=0.56,
        mode="live",
        exchange_order_id="ORD-456",
        status="resting",
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.place_order = AsyncMock(return_value=filled_order)

    pending_position = _make_position(contracts=178, entry_price=0.56)
    pending_position.status = "pending"
    pending_position.exchange_order_id = "ORD-456"

    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=pending_position,
    ) as mock_ledger:
        await om.submit(_make_signal(direction="YES"), _make_market(yes_bid=0.52, yes_ask=0.56), strategy)

    mock_ledger.assert_called_once()
    kwargs = mock_ledger.call_args.kwargs
    assert kwargs["status"] == "pending"
    assert kwargs["exchange_order_id"] == "ORD-456"
    assert kwargs["mode"] == "live"


@pytest.mark.asyncio
async def test_paper_mode_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paper mode submits with status='open' and no exchange_order_id."""
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    expected_position = _make_position(contracts=178, entry_price=0.56)
    om, _ = _make_order_manager(mode="paper")
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ) as mock_ledger:
        result = await om.submit(_make_signal(direction="YES"), _make_market(yes_bid=0.52, yes_ask=0.56), strategy)

    assert result is expected_position
    kwargs = mock_ledger.call_args.kwargs
    assert kwargs["status"] == "open"
    assert kwargs.get("exchange_order_id") is None
    assert kwargs["mode"] == "paper"


@pytest.mark.asyncio
async def test_live_mode_risk_check_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """RiskEngine.check_position() returning allowed=False prevents place_order() call."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    risk = MagicMock(spec=RiskEngine)
    risk.check_circuit_breakers = AsyncMock(return_value=None)
    risk.check_position = AsyncMock(
        return_value=RiskDecision(allowed=False, reason="exposure limit", capped_size=0.0)
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.place_order = AsyncMock()

    om, _ = _make_order_manager(risk=risk, mode="live", kalshi_client=mock_kalshi)
    strategy = _make_strategy(should_trade_result=True, position_size_result=200.0)

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(), _make_market(), strategy)

    assert result is None
    mock_kalshi.place_order.assert_not_called()
    mock_ledger.assert_not_called()


# ---------------------------------------------------------------------------
# Incremental sizing — existing market exposure
# ---------------------------------------------------------------------------


def _make_real_strategy() -> IPredictionStrategy:
    """Return a strategy that uses the real Kelly position_size() (not a stub).

    This exercises the actual incremental sizing logic in IPredictionStrategy.
    """
    class _RealSizing(IPredictionStrategy):
        config = StrategyConfig(
            name="RealSizingStrategy",
            min_edge=0.10,
            min_confidence=0.70,
            max_exposure_per_market=0.10,
            kelly_fraction=0.25,
            categories=["politics"],
            min_volume_24h=0.0,
            max_days_to_close=90,
            min_days_to_close=1,
        )

        def should_trade(self, signal: Signal, market: Market) -> bool:
            return True

    return _RealSizing()


@pytest.mark.asyncio
async def test_no_new_position_when_existing_exposure_covers_ideal() -> None:
    """When existing open exposure >= Kelly ideal, position_size returns 0 → no trade."""
    om, sf = _make_order_manager()
    strategy = _make_real_strategy()
    signal = _make_signal(edge=0.15, estimated_probability=0.65)
    market = _make_market(yes_bid=0.52, yes_ask=0.56)

    # Compute what Kelly would want for this signal on a fresh market.
    ideal_total = strategy.position_size(signal, BANKROLL, existing_market_exposure=0.0)
    assert ideal_total > 0.0, "sanity: signal should want a non-zero position"

    # Mock the DB to report existing exposure = ideal_total (already fully deployed).
    mock_session = sf.return_value.__aenter__.return_value
    exposure_result = MagicMock()
    exposure_result.scalar_one.return_value = ideal_total
    mock_session.execute = AsyncMock(return_value=exposure_result)

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(signal, market, strategy)

    assert result is None
    mock_ledger.assert_not_called()
    # Risk engine should not even be consulted — early exit before that.
    om._risk.check_position.assert_not_called()


@pytest.mark.asyncio
async def test_existing_exposure_passed_to_position_size() -> None:
    """OrderManager queries open exposure and passes it to position_size()."""
    om, sf = _make_order_manager()
    signal = _make_signal(edge=0.15, estimated_probability=0.65)
    market = _make_market(yes_bid=0.52, yes_ask=0.56)

    # Mock DB: report $25.00 of existing same-direction exposure, no opposite-side positions.
    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = 25.0
    _count = MagicMock(); _count.scalar_one.return_value = 0
    _avg = MagicMock(); _avg.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(side_effect=[_exp, _count, _avg])

    # Use a spy strategy to capture the args passed to position_size.
    captured_args: list[tuple] = []

    class _SpyStrategy(IPredictionStrategy):
        config = StrategyConfig(
            name="SpyStrategy",
            min_edge=0.10,
            min_confidence=0.70,
            max_exposure_per_market=0.10,
            kelly_fraction=0.25,
            categories=["politics"],
            min_volume_24h=0.0,
            max_days_to_close=90,
            min_days_to_close=1,
        )

        def should_trade(self, signal: Signal, market: Market) -> bool:
            return True

        def position_size(self, signal: Signal, bankroll: float, existing_market_exposure: float = 0.0) -> float:
            captured_args.append((bankroll, existing_market_exposure))
            return 50.0  # arbitrary positive value so the flow continues

    strategy = _SpyStrategy()
    expected_position = _make_position()

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ):
        await om.submit(signal, market, strategy)

    assert len(captured_args) == 1
    _, existing_exposure = captured_args[0]
    assert existing_exposure == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_legacy_two_arg_position_size_override_still_works() -> None:
    om, _ = _make_order_manager()
    signal = _make_signal()
    market = _make_market()

    class _LegacyTwoArg(IPredictionStrategy):
        config = StrategyConfig(
            name="LegacyTwoArg",
            min_edge=0.10,
            min_confidence=0.70,
            max_exposure_per_market=0.10,
            kelly_fraction=0.25,
            categories=["politics"],
            min_volume_24h=0.0,
            max_days_to_close=90,
            min_days_to_close=1,
        )

        def should_trade(self, signal: Signal, market: Market) -> bool:
            return True

        def position_size(self, signal: Signal, bankroll: float) -> float:
            return 50.0

    expected_position = _make_position()
    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ):
        result = await om.submit(signal, market, _LegacyTwoArg())

    assert result is expected_position


@pytest.mark.asyncio
async def test_assessment_enabled_keeps_legacy_three_arg_override_working() -> None:
    llm_client = MagicMock()
    om, sf = _make_order_manager(
        llm_client=llm_client,
        judgment_model="claude-opus-4-6",
    )
    signal = _make_signal()
    market = _make_market()
    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = 25.0
    _count = MagicMock(); _count.scalar_one.return_value = 0
    _avg = MagicMock(); _avg.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(side_effect=[_exp, _count, _avg])
    assessment = SignalAssessment(
        signal_id=signal.id,
        trust_score=0.65,
        size_multiplier=1.06,
        verdict="size_up",
        reasoning="test",
    )
    captured_existing: list[float] = []

    class _LegacyThreeArg(IPredictionStrategy):
        config = StrategyConfig(
            name="LegacyThreeArg",
            min_edge=0.10,
            min_confidence=0.70,
            max_exposure_per_market=0.10,
            kelly_fraction=0.25,
            categories=["politics"],
            min_volume_24h=0.0,
            max_days_to_close=90,
            min_days_to_close=1,
        )

        def should_trade(self, signal: Signal, market: Market) -> bool:
            return True

        def position_size(
            self,
            signal: Signal,
            bankroll: float,
            existing_market_exposure: float = 0.0,
        ) -> float:
            captured_existing.append(existing_market_exposure)
            return 50.0

    expected_position = _make_position()
    with patch(
        "freqpred.metrics.assessment.assess_signal_context",
        new_callable=AsyncMock,
        return_value=assessment,
    ) as mock_assess, patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ):
        result = await om.submit(signal, market, _LegacyThreeArg())

    assert result is expected_position
    assert captured_existing == [25.0]
    mock_assess.assert_awaited_once()


@pytest.mark.asyncio
async def test_assessment_enabled_passes_assessment_to_supported_strategy() -> None:
    llm_client = MagicMock()
    om, sf = _make_order_manager(
        llm_client=llm_client,
        judgment_model="claude-opus-4-6",
    )
    signal = _make_signal()
    market = _make_market()
    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = 10.0
    _count = MagicMock(); _count.scalar_one.return_value = 0
    _avg = MagicMock(); _avg.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(side_effect=[_exp, _count, _avg])
    assessment = SignalAssessment(
        signal_id=signal.id,
        trust_score=0.70,
        size_multiplier=1.08,
        verdict="size_up",
        reasoning="test",
    )
    captured_assessments: list[SignalAssessment | None] = []

    class _AssessmentAware(IPredictionStrategy):
        config = StrategyConfig(
            name="AssessmentAware",
            min_edge=0.10,
            min_confidence=0.70,
            max_exposure_per_market=0.10,
            kelly_fraction=0.25,
            categories=["politics"],
            min_volume_24h=0.0,
            max_days_to_close=90,
            min_days_to_close=1,
        )

        def should_trade(self, signal: Signal, market: Market) -> bool:
            return True

        def position_size(
            self,
            signal: Signal,
            bankroll: float,
            existing_market_exposure: float = 0.0,
            assessment: SignalAssessment | None = None,
        ) -> float:
            captured_assessments.append(assessment)
            return 50.0

    expected_position = _make_position()
    with patch(
        "freqpred.metrics.assessment.assess_signal_context",
        new_callable=AsyncMock,
        return_value=assessment,
    ), patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ):
        result = await om.submit(signal, market, _AssessmentAware())

    assert result is expected_position
    assert captured_assessments == [assessment]


@pytest.mark.asyncio
async def test_doubledown_allowed_when_edge_increases() -> None:
    """Higher-edge signal produces larger ideal → positive incremental → trade opens."""
    om, sf = _make_order_manager()
    strategy = _make_real_strategy()
    market = _make_market(yes_bid=0.52, yes_ask=0.56)

    # First signal: moderate edge.
    sig_low = _make_signal(edge=0.15, estimated_probability=0.65)
    low_ideal = strategy.position_size(sig_low, BANKROLL, existing_market_exposure=0.0)
    assert low_ideal > 0.0

    # Second signal: higher edge — should justify more total exposure.
    sig_high = _make_signal(edge=0.30, estimated_probability=0.80)
    high_ideal = strategy.position_size(sig_high, BANKROLL, existing_market_exposure=0.0)
    assert high_ideal > low_ideal, "sanity: higher edge → bigger ideal"

    # DB reports existing exposure = low_ideal (same direction, no opposite-side positions).
    # avg_entry=0.60 > yes_ask=0.56, so price-improvement gate passes.
    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = low_ideal
    _count = MagicMock(); _count.scalar_one.return_value = 0
    _avg = MagicMock(); _avg.scalar_one_or_none.return_value = 0.60
    mock_session.execute = AsyncMock(side_effect=[_exp, _count, _avg])

    expected_position = _make_position()
    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ) as mock_ledger:
        result = await om.submit(sig_high, market, strategy)

    assert result is expected_position
    mock_ledger.assert_called_once()


@pytest.mark.asyncio
async def test_reentry_blocked_when_conviction_drops() -> None:
    """Lower-conviction signal → smaller ideal → no incremental → no new position."""
    om, sf = _make_order_manager()
    strategy = _make_real_strategy()
    market = _make_market(yes_bid=0.52, yes_ask=0.56)

    # First signal with high edge.
    sig_high = _make_signal(edge=0.30, estimated_probability=0.80)
    high_ideal = strategy.position_size(sig_high, BANKROLL, existing_market_exposure=0.0)

    # Second signal with lower edge.
    sig_low = _make_signal(edge=0.15, estimated_probability=0.65)

    # DB reports existing exposure = high_ideal (same direction, no opposite-side positions).
    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = high_ideal
    _count = MagicMock(); _count.scalar_one.return_value = 0
    _avg = MagicMock(); _avg.scalar_one_or_none.return_value = 0.40
    mock_session.execute = AsyncMock(side_effect=[_exp, _count, _avg])

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(sig_low, market, strategy)

    assert result is None
    mock_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_reentry_blocked_at_worse_price() -> None:
    """Price-improvement gate: new entry price >= avg existing entry → return None."""
    om, sf = _make_order_manager()
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)
    market = _make_market(yes_bid=0.50, yes_ask=0.54)  # yes_ask = 0.54

    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = 20.0
    _count = MagicMock(); _count.scalar_one.return_value = 0   # same-direction existing position
    _avg = MagicMock(); _avg.scalar_one_or_none.return_value = 0.50  # avg entry better than 0.54
    mock_session.execute = AsyncMock(side_effect=[_exp, _count, _avg])

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(direction="YES"), market, strategy)

    assert result is None
    mock_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_reentry_blocked_at_equal_price() -> None:
    """Price-improvement gate: entry price == avg existing entry → return None."""
    om, sf = _make_order_manager()
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)
    market = _make_market(yes_bid=0.50, yes_ask=0.54)  # yes_ask = 0.54

    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = 20.0
    _count = MagicMock(); _count.scalar_one.return_value = 0
    _avg = MagicMock(); _avg.scalar_one_or_none.return_value = 0.54  # equal → blocked
    mock_session.execute = AsyncMock(side_effect=[_exp, _count, _avg])

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(direction="YES"), market, strategy)

    assert result is None
    mock_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_reentry_allowed_at_better_price() -> None:
    """Price-improvement gate: new entry price < avg existing entry → trade proceeds."""
    om, sf = _make_order_manager()
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)
    market = _make_market(yes_bid=0.50, yes_ask=0.46)  # yes_ask 0.46 < avg_entry 0.54

    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = 20.0
    _count = MagicMock(); _count.scalar_one.return_value = 0   # same-direction existing
    _avg = MagicMock(); _avg.scalar_one_or_none.return_value = 0.54  # avg worse than new ask
    mock_session.execute = AsyncMock(side_effect=[_exp, _count, _avg])

    expected_position = _make_position()
    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ) as mock_ledger:
        result = await om.submit(_make_signal(direction="YES"), market, strategy)

    assert result is expected_position
    mock_ledger.assert_called_once()


@pytest.mark.asyncio
async def test_price_gate_skipped_when_no_existing_position() -> None:
    """Price-improvement gate is not applied when existing_market_exposure == 0."""
    om, sf = _make_order_manager()
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)
    market = _make_market(yes_bid=0.50, yes_ask=0.54)

    # Default mock returns scalar_one=0.0 (no existing exposure).
    expected_position = _make_position()
    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected_position,
    ) as mock_ledger:
        result = await om.submit(_make_signal(direction="YES"), market, strategy)

    assert result is expected_position
    mock_ledger.assert_called_once()


@pytest.mark.asyncio
async def test_yes_position_blocks_no_entry() -> None:
    """Opposite-side guard: open YES position → NO signal entry is blocked."""
    om, sf = _make_order_manager()
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)
    market = _make_market(yes_bid=0.85, yes_ask=0.88)

    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = 20.0   # total exposure > 0
    _count = MagicMock(); _count.scalar_one.return_value = 1  # one YES position (opposite of NO signal)
    mock_session.execute = AsyncMock(side_effect=[_exp, _count])

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(direction="NO"), market, strategy)

    assert result is None
    mock_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_no_position_blocks_yes_entry() -> None:
    """Opposite-side guard: open NO position → YES signal entry is blocked."""
    om, sf = _make_order_manager()
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)
    market = _make_market(yes_bid=0.12, yes_ask=0.15)

    mock_session = sf.return_value.__aenter__.return_value
    _exp = MagicMock(); _exp.scalar_one.return_value = 10.0   # total exposure > 0
    _count = MagicMock(); _count.scalar_one.return_value = 1  # one NO position (opposite of YES signal)
    mock_session.execute = AsyncMock(side_effect=[_exp, _count])

    with patch("freqpred.trading.order_manager.ledger.open_position") as mock_ledger:
        result = await om.submit(_make_signal(direction="YES"), market, strategy)

    assert result is None
    mock_ledger.assert_not_called()


# ---------------------------------------------------------------------------
# force_exit tests
# ---------------------------------------------------------------------------


def _make_force_exit_session(pos_row: MagicMock | None = None, mkt_row: MagicMock | None = None) -> AsyncMock:
    """Return a mock AsyncSession for force_exit() that yields (pos_row, mkt_row)."""
    session = AsyncMock()
    result = MagicMock()
    result.one_or_none.return_value = (pos_row, mkt_row) if pos_row is not None else None
    session.execute = AsyncMock(return_value=result)
    return session


def _make_force_exit_sf(session: AsyncMock) -> MagicMock:
    sf = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    sf.return_value = ctx
    return sf


def _mock_pos_row(
    *,
    pos_id: uuid.UUID | None = None,
    status: str = "open",
    direction: str = "YES",
    market_id: str = MARKET_ID,
    mode: str = "paper",
    contracts: int = 10,
) -> MagicMock:
    row = MagicMock()
    row.id = pos_id or uuid.uuid4()
    row.status = status
    row.direction = direction
    row.market_id = market_id
    row.mode = mode
    row.contracts = contracts
    return row


def _mock_mkt_row(
    *,
    mid_price: float = 0.60,
    yes_bid: float = 0.55,
    yes_ask: float = 0.65,
    result: str | None = None,
) -> MagicMock:
    row = MagicMock()
    row.mid_price = mid_price
    row.yes_bid = yes_bid
    row.yes_ask = yes_ask
    row.result = result
    return row


def _make_pending_row(
    *,
    pos_id: uuid.UUID | None = None,
    market_id: str = "MKT-PEND",
    contracts: int = 10,
    requested_contracts: int | None = 10,
    exchange_order_id: str | None = "ORD-PEND",
    strategy_name: str = "TestStrategy",
    created_at: datetime | None = None,
) -> MagicMock:
    row = MagicMock(spec=[
        "id", "market_id", "contracts", "requested_contracts",
        "exchange_order_id", "status", "strategy_name", "created_at",
        "exchange_order_status", "last_exchange_sync_at", "entry_price",
    ])
    row.id = pos_id or uuid.uuid4()
    row.market_id = market_id
    row.contracts = contracts
    row.requested_contracts = requested_contracts
    row.exchange_order_id = exchange_order_id
    row.status = "pending"
    row.strategy_name = strategy_name
    row.created_at = created_at or NOW
    row.exchange_order_status = "resting"
    row.last_exchange_sync_at = None
    row.entry_price = 0.50
    return row


def _make_pending_session(rows: list[MagicMock]) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_reconcile_pending_order_partial_fill() -> None:
    """Kalshi 'partial' with fills < requested → DB row.status='open', contracts=filled."""
    row = _make_pending_row(contracts=10, requested_contracts=10)
    session = _make_pending_session([row])

    partial_order = Order(
        market_id=row.market_id,
        direction="YES",
        contracts=3,
        price=0.50,
        mode="live",
        exchange_order_id="ORD-PEND",
        status="partial",
        requested_count=10,
        filled_yes_count=3,
        filled_no_count=0,
        remaining_count=7,
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.get_order = AsyncMock(return_value=partial_order)

    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    await om.reconcile_pending_orders(session, _now=NOW)

    assert row.status == "open"
    assert row.contracts == 3
    assert row.exchange_order_status == "partial"
    assert row.last_exchange_sync_at == NOW
    session.commit.assert_awaited()


@pytest.mark.asyncio
async def test_reconcile_pending_order_cancelled_on_exchange() -> None:
    """Kalshi 'canceled' with 0 fills → DB row.status='cancelled'."""
    row = _make_pending_row()
    session = _make_pending_session([row])

    cancelled_order = Order(
        market_id=row.market_id,
        direction="YES",
        contracts=0,
        price=0.50,
        mode="live",
        exchange_order_id="ORD-PEND",
        status="canceled",
        requested_count=10,
        filled_yes_count=0,
        filled_no_count=0,
        remaining_count=10,
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.get_order = AsyncMock(return_value=cancelled_order)

    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    await om.reconcile_pending_orders(session, _now=NOW)

    assert row.status == "cancelled"
    assert row.exchange_order_status == "canceled"


@pytest.mark.asyncio
async def test_reconcile_pending_skips_null_exchange_order_id() -> None:
    """Pending live rows with NULL exchange_order_id are filtered out at query time.

    The query uses .where(exchange_order_id.is_not(None)) so the legacy row
    never makes it into the loop. We verify get_order is never called.
    """
    session = _make_pending_session([])  # filtered query returns no rows
    mock_kalshi = AsyncMock()
    mock_kalshi.get_order = AsyncMock()

    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    await om.reconcile_pending_orders(session, _now=NOW)

    mock_kalshi.get_order.assert_not_called()


@pytest.mark.asyncio
async def test_live_timeout_cancels_resting_order() -> None:
    """Pending order age > timeout AND still pending after status check → cancel_order called."""
    timeout = 60.0
    aged_created = NOW - timedelta(seconds=120)
    row = _make_pending_row(created_at=aged_created)
    session = _make_pending_session([row])

    resting_order = Order(
        market_id=row.market_id,
        direction="YES",
        contracts=0,
        price=0.50,
        mode="live",
        exchange_order_id="ORD-PEND",
        status="resting",
        requested_count=10,
        filled_yes_count=0,
        filled_no_count=0,
        remaining_count=10,
    )
    cancelled_order = Order(
        market_id=row.market_id,
        direction="YES",
        contracts=0,
        price=0.50,
        mode="live",
        exchange_order_id="ORD-PEND",
        status="canceled",
        requested_count=10,
        filled_yes_count=0,
        filled_no_count=0,
        remaining_count=10,
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.get_order = AsyncMock(return_value=resting_order)
    mock_kalshi.cancel_order = AsyncMock(return_value=cancelled_order)

    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    om._default_pending_timeout_seconds = timeout
    await om.reconcile_pending_orders(session, _now=NOW)

    mock_kalshi.cancel_order.assert_awaited_once_with("ORD-PEND")
    assert row.status == "cancelled"


@pytest.mark.asyncio
async def test_timeout_uses_position_created_at_not_service_uptime() -> None:
    """Timeout is computed against row.created_at, not the OrderManager uptime.

    This protects against the "restart resets the timer" bug.
    """
    # Created 10 minutes before _now; timeout 60s → should expire.
    row = _make_pending_row(created_at=NOW - timedelta(minutes=10))
    session = _make_pending_session([row])

    resting_order = Order(
        market_id=row.market_id,
        direction="YES",
        contracts=0,
        price=0.50,
        mode="live",
        exchange_order_id="ORD-PEND",
        status="resting",
        requested_count=10,
        filled_yes_count=0,
        filled_no_count=0,
        remaining_count=10,
    )
    cancelled_order = Order(
        market_id=row.market_id,
        direction="YES",
        contracts=0,
        price=0.50,
        mode="live",
        exchange_order_id="ORD-PEND",
        status="canceled",
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.get_order = AsyncMock(return_value=resting_order)
    mock_kalshi.cancel_order = AsyncMock(return_value=cancelled_order)

    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    om._default_pending_timeout_seconds = 60.0
    await om.reconcile_pending_orders(session, _now=NOW)

    mock_kalshi.cancel_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_reconcile_catches_expired_timeout_from_prior_run() -> None:
    """A row aged past timeout from prior process run → startup pass cancels it."""
    row = _make_pending_row(created_at=NOW - timedelta(hours=1))
    session = _make_pending_session([row])

    resting_order = Order(
        market_id=row.market_id, direction="YES", contracts=0, price=0.5, mode="live",
        exchange_order_id="ORD-PEND", status="resting",
        requested_count=10, filled_yes_count=0, filled_no_count=0, remaining_count=10,
    )
    cancelled_order = Order(
        market_id=row.market_id, direction="YES", contracts=0, price=0.5, mode="live",
        exchange_order_id="ORD-PEND", status="canceled",
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.get_order = AsyncMock(return_value=resting_order)
    mock_kalshi.cancel_order = AsyncMock(return_value=cancelled_order)

    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    om._default_pending_timeout_seconds = 900.0
    await om.reconcile_pending_orders(session, _now=NOW)

    mock_kalshi.cancel_order.assert_awaited_once_with("ORD-PEND")


@pytest.mark.asyncio
async def test_live_entry_records_requested_and_filled_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_submit_live persists requested_contracts via ledger.open_position kwargs."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    # $100 raw size / $0.55 entry → floor(181.81) = 181 contracts
    expected_contracts = math.floor(100.0 / 0.55)
    mock_kalshi = AsyncMock()
    filled_order = Order(
        market_id=MARKET_ID, direction="YES", contracts=expected_contracts,
        price=0.55, mode="live",
        exchange_order_id="ORD-OK", status="resting",
        requested_count=expected_contracts,
        filled_yes_count=0, filled_no_count=0, remaining_count=expected_contracts,
    )
    mock_kalshi.place_order = AsyncMock(return_value=filled_order)
    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)

    expected = _make_position(contracts=expected_contracts, entry_price=0.55)
    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        return_value=expected,
    ) as mock_ledger:
        await om.submit(_make_signal(direction="YES"), _make_market(yes_ask=0.55, yes_bid=0.52), strategy)

    call_kwargs = mock_ledger.call_args.kwargs
    assert call_kwargs["requested_contracts"] == expected_contracts
    assert call_kwargs["exchange_order_status"] == "resting"
    assert "last_exchange_sync_at" in call_kwargs


@pytest.mark.asyncio
async def test_live_entry_orphan_cancels_on_ledger_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DB write fails after place_order succeeds → cancel_order is invoked."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    filled_order = Order(
        market_id=MARKET_ID, direction="YES", contracts=10, price=0.55, mode="live",
        exchange_order_id="ORD-ORPHAN", status="resting",
        requested_count=10, filled_yes_count=0, filled_no_count=0, remaining_count=10,
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.place_order = AsyncMock(return_value=filled_order)
    mock_kalshi.cancel_order = AsyncMock(return_value=filled_order)
    om, _ = _make_order_manager(mode="live", kalshi_client=mock_kalshi)
    strategy = _make_strategy(should_trade_result=True, position_size_result=100.0)

    with patch(
        "freqpred.trading.order_manager.ledger.open_position",
        new_callable=AsyncMock,
        side_effect=RuntimeError("DB write blew up"),
    ):
        with pytest.raises(RuntimeError):
            await om.submit(_make_signal(direction="YES"), _make_market(yes_ask=0.55, yes_bid=0.52), strategy)

    mock_kalshi.cancel_order.assert_awaited_once_with("ORD-ORPHAN")


@pytest.mark.asyncio
async def test_force_exit_paper_success() -> None:
    """Paper force_exit: closes position at current mid price for YES direction."""
    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="open", direction="YES", mode="paper")
    mkt = _mock_mkt_row(mid_price=0.60)
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    closed_pos = _make_position()
    closed_pos.status = "closed"
    closed_pos.exit_price = 0.60
    closed_pos.exit_reason = "force_exit:manual"

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="paper",
    )

    with patch(
        "freqpred.trading.order_manager.ledger.close_position",
        new_callable=AsyncMock,
        return_value=closed_pos,
    ) as mock_close:
        result = await om.force_exit(str(pos_id))

    assert result is closed_pos
    mock_close.assert_called_once()
    assert mock_close.call_args.args[1] == str(pos_id)
    assert mock_close.call_args.kwargs["exit_price"] == pytest.approx(0.60)
    assert mock_close.call_args.kwargs["exit_reason"] == "force_exit:manual"


@pytest.mark.asyncio
async def test_force_exit_wrong_mode_not_found() -> None:
    """Position not found for this mode → PositionNotFoundError."""
    session = _make_force_exit_session()  # one_or_none returns None
    sf = _make_force_exit_sf(session)

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="paper",
    )

    with pytest.raises(PositionNotFoundError):
        await om.force_exit(str(uuid.uuid4()))


@pytest.mark.asyncio
async def test_force_exit_already_closed_raises() -> None:
    """Position status != 'open' → PositionNotOpenError."""
    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="closed", direction="YES", mode="paper")
    mkt = _mock_mkt_row()
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="paper",
    )

    with pytest.raises(PositionNotOpenError):
        await om.force_exit(str(pos_id))


@pytest.mark.asyncio
async def test_force_exit_on_pending_cancels_instead_of_selling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live force_exit on status='pending' → cancel_order, not place_order(sell)."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="pending", direction="YES", mode="live")
    pos.exchange_order_id = "ORD-PEND-FX"
    pos.requested_contracts = 5
    pos.exchange_order_status = "resting"
    pos.last_exchange_sync_at = None
    pos.entry_price = 0.5
    mkt = _mock_mkt_row()
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    cancelled_order = Order(
        market_id=pos.market_id, direction="YES", contracts=0, price=0.5, mode="live",
        exchange_order_id="ORD-PEND-FX", status="canceled",
        requested_count=5, filled_yes_count=0, filled_no_count=0, remaining_count=5,
    )
    mock_kalshi = AsyncMock()
    mock_kalshi.cancel_order = AsyncMock(return_value=cancelled_order)
    mock_kalshi.place_order = AsyncMock()

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
        kalshi_client=mock_kalshi,
    )

    result = await om.force_exit(str(pos_id))

    mock_kalshi.cancel_order.assert_awaited_once_with("ORD-PEND-FX")
    mock_kalshi.place_order.assert_not_called()
    assert pos.status == "cancelled"
    assert result is not None


@pytest.mark.asyncio
async def test_force_exit_live_blocked_without_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live force_exit raises ValueError when LIVE_TRADING_ENABLED is not set."""
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)

    sf = MagicMock()
    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
    )

    with pytest.raises(ValueError, match="LIVE_TRADING_ENABLED"):
        await om.force_exit(str(uuid.uuid4()))

    sf.assert_not_called()


@pytest.mark.asyncio
async def test_force_exit_live_no_kalshi_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live force_exit with kalshi_client=None raises ValueError (wiring guard)."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="open", direction="YES", mode="live")
    mkt = _mock_mkt_row(result=None)
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
        kalshi_client=None,
    )

    with pytest.raises(ValueError, match="KalshiClient"):
        await om.force_exit(str(pos_id))


@pytest.mark.asyncio
async def test_force_exit_live_non_executable_price_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """YES direction with yes_bid=0.0 → limit_price=0.0 → ValueError (market closed to trading)."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="open", direction="YES", mode="live")
    mkt = _mock_mkt_row(yes_bid=0.0, yes_ask=1.0, result=None)
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    exchange_pos = MagicMock()
    exchange_pos.market_id = MARKET_ID
    exchange_pos.direction = "YES"
    exchange_pos.contracts = 5

    mock_kalshi = AsyncMock()
    mock_kalshi.get_positions = AsyncMock(return_value=[exchange_pos])
    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
        kalshi_client=mock_kalshi,
    )

    with pytest.raises(ValueError, match="No executable exit price"):
        await om.force_exit(str(pos_id))

    mock_kalshi.get_positions.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_exit_live_no_exchange_position_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """No matching live position on exchange → ValueError, ledger not written."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="open", direction="YES", mode="live")
    mkt = _mock_mkt_row(yes_bid=0.55, yes_ask=0.65, result=None)
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    mock_kalshi = AsyncMock()
    mock_kalshi.get_positions = AsyncMock(return_value=[])

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
        kalshi_client=mock_kalshi,
    )

    with patch("freqpred.trading.order_manager.ledger.close_position") as mock_close:
        with pytest.raises(ValueError, match="reconciliation required"):
            await om.force_exit(str(pos_id))

    mock_close.assert_not_called()


@pytest.mark.asyncio
async def test_force_exit_live_exchange_failure_leaves_position_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """KalshiAPIError from place_order propagates; ledger.close_position not called."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="open", direction="YES", mode="live")
    mkt = _mock_mkt_row(yes_bid=0.55, yes_ask=0.65, result=None)
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    exchange_pos = MagicMock()
    exchange_pos.market_id = MARKET_ID
    exchange_pos.direction = "YES"
    exchange_pos.contracts = 5

    mock_kalshi = AsyncMock()
    mock_kalshi.get_positions = AsyncMock(return_value=[exchange_pos])
    mock_kalshi.place_order = AsyncMock(side_effect=KalshiAPIError(500, "internal error"))

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
        kalshi_client=mock_kalshi,
    )

    with patch("freqpred.trading.order_manager.ledger.close_position") as mock_close:
        with pytest.raises(KalshiAPIError):
            await om.force_exit(str(pos_id))

    mock_close.assert_not_called()


@pytest.mark.asyncio
async def test_force_exit_live_uses_reconciled_net_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit order uses exchange net contracts (5), not PositionRow.contracts (10)."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="open", direction="YES", mode="live", contracts=10)
    mkt = _mock_mkt_row(yes_bid=0.55, yes_ask=0.65, result=None)
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    exchange_pos = MagicMock()
    exchange_pos.market_id = MARKET_ID
    exchange_pos.direction = "YES"
    exchange_pos.contracts = 5

    filled = Order(
        market_id=MARKET_ID, direction="YES", contracts=5, price=0.55, mode="live", status="executed",
    )

    mock_kalshi = AsyncMock()
    mock_kalshi.get_positions = AsyncMock(return_value=[exchange_pos])
    mock_kalshi.place_order = AsyncMock(return_value=filled)

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
        kalshi_client=mock_kalshi,
    )

    closed_pos = _make_position()
    with patch(
        "freqpred.trading.order_manager.ledger.partial_close_position",
        new_callable=AsyncMock,
        return_value=closed_pos,
    ):
        await om.force_exit(str(pos_id))

    placed: Order = mock_kalshi.place_order.call_args.args[0]
    assert placed.contracts == 5
    assert placed.action == "sell"
    assert placed.time_in_force == "fill_or_kill"
    assert pos.contracts == 5


@pytest.mark.asyncio
async def test_force_exit_live_resolved_market_closes_at_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolved live exits use reconciled size and settlement payout without ordering."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    pos_id = uuid.uuid4()
    pos = _mock_pos_row(
        pos_id=pos_id,
        status="open",
        direction="YES",
        mode="live",
        contracts=10,
    )
    mkt = _mock_mkt_row(result="yes")
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    exchange_pos = MagicMock()
    exchange_pos.market_id = MARKET_ID
    exchange_pos.direction = "YES"
    exchange_pos.contracts = 5

    mock_kalshi = AsyncMock()
    mock_kalshi.get_positions = AsyncMock(return_value=[exchange_pos])
    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
        kalshi_client=mock_kalshi,
    )

    closed_pos = _make_position()
    with patch(
        "freqpred.trading.order_manager.ledger.close_position",
        new_callable=AsyncMock,
        return_value=closed_pos,
        ) as mock_close:
        await om.force_exit(str(pos_id))

    mock_kalshi.get_positions.assert_awaited_once()
    mock_kalshi.place_order.assert_not_called()
    assert pos.contracts == 5
    assert mock_close.call_args.kwargs["exit_price"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_force_exit_live_resolved_market_without_exchange_position_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved live exits still require a matching exchange position for reconciliation."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    pos_id = uuid.uuid4()
    pos = _mock_pos_row(pos_id=pos_id, status="open", direction="YES", mode="live")
    mkt = _mock_mkt_row(result="yes")
    session = _make_force_exit_session(pos, mkt)
    sf = _make_force_exit_sf(session)

    mock_kalshi = AsyncMock()
    mock_kalshi.get_positions = AsyncMock(return_value=[])

    om = OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=sf,
        bankroll=BANKROLL,
        mode="live",
        kalshi_client=mock_kalshi,
    )

    with patch("freqpred.trading.order_manager.ledger.close_position") as mock_close:
        with pytest.raises(ValueError, match="reconciliation required"):
            await om.force_exit(str(pos_id))

    mock_close.assert_not_called()


@pytest.mark.asyncio
async def test_force_exit_live_resolved_market_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settlement check is case-insensitive: 'YES', 'No', 'yes' all work correctly."""
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")

    cases = [
        ("YES", "YES", 1.0),   # uppercase YES result, YES position → wins
        ("No",  "NO",  1.0),   # mixed-case No result, NO position → wins
        ("yes", "NO",  0.0),   # YES wins, NO position → loses
    ]

    for result_str, direction, expected_price in cases:
        pos_id = uuid.uuid4()
        pos = _mock_pos_row(pos_id=pos_id, status="open", direction=direction, mode="live")
        mkt = _mock_mkt_row(result=result_str)
        session = _make_force_exit_session(pos, mkt)
        sf = _make_force_exit_sf(session)

        exchange_pos = MagicMock()
        exchange_pos.market_id = MARKET_ID
        exchange_pos.direction = direction
        exchange_pos.contracts = 4

        mock_kalshi = AsyncMock()
        mock_kalshi.get_positions = AsyncMock(return_value=[exchange_pos])

        om = OrderManager(
            risk=MagicMock(spec=RiskEngine),
            session_factory=sf,
            bankroll=BANKROLL,
            mode="live",
            kalshi_client=mock_kalshi,
        )

        closed_pos = _make_position()
        with patch(
            "freqpred.trading.order_manager.ledger.close_position",
            new_callable=AsyncMock,
            return_value=closed_pos,
        ) as mock_close:
            await om.force_exit(str(pos_id))

        assert mock_close.call_args.kwargs["exit_price"] == pytest.approx(expected_price), \
            f"result={result_str!r}, direction={direction}: expected {expected_price}"
        mock_kalshi.get_positions.assert_awaited_once()
        mock_kalshi.place_order.assert_not_called()
