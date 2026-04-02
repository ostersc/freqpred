"""Unit tests for freqpred/trading/order_manager.py.

All DB interactions and ledger writes are mocked — no external dependencies.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.markets.models import Market, Order, Position
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig
from freqpred.trading.order_manager import OrderManager
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

        def position_size(self, signal: Signal, bankroll: float) -> float:
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
) -> tuple[OrderManager, MagicMock]:
    """Return (OrderManager, session_factory_mock)."""
    session_factory = MagicMock()
    # Make session_factory() return an async context manager
    mock_session = AsyncMock()
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

        def position_size(self, signal: Signal, bankroll: float) -> float:
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
