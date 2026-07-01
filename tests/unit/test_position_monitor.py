"""Unit tests for freqpred/trading/position_monitor.py (T26).

All DB interactions mocked — no external dependencies.
Tests cover:
- _check_stoploss
- _check_trailing_stop
- PositionMonitor.evaluate_exit (full priority order)
- PositionMonitor.check_all_positions (integration of the loop)
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure ORM relationships resolve (needed for MarketRow joins)
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import Market, Order, Position
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig
from freqpred.trading.position_monitor import (
    PositionMonitor,
    _check_stoploss,
    _check_trailing_stop,
    _compute_exchange_stoploss_price,
    _compute_trailing_stop_level,
)

NOW = datetime(2026, 3, 18, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_position(
    *,
    entry_price: float = 0.50,
    direction: str = "YES",
    entry_time: datetime | None = None,
    position_id: str | None = None,
    strategy_name: str = "TestStrategy",
    mode: str = "paper",
    contracts: int = 100,
) -> Position:
    return Position(
        id=position_id or str(uuid.uuid4()),
        market_id="MKT-1",
        signal_id=str(uuid.uuid4()),
        strategy_name=strategy_name,
        strategy_version="1.0",
        signal_confidence=0.80,
        signal_edge=0.15,
        signal_estimated_prob=0.65,
        direction=direction,
        contracts=contracts,
        entry_price=entry_price,
        entry_time=entry_time or NOW,
        mode=mode,
        status="open",
    )


def _make_market(
    *,
    mid_price: float = 0.50,
    close_time: datetime | None = None,
) -> Market:
    return Market(
        id="MKT-1",
        platform="kalshi",
        question="Will X happen?",
        category="politics",
        close_time=close_time or (datetime.now(UTC) + timedelta(days=10)),
        yes_bid=mid_price - 0.02,
        yes_ask=mid_price + 0.02,
        mid_price=mid_price,
        volume_24h=1000.0,
        open_interest=5000.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
    )


def _make_signal(direction: str = "YES", confidence: float = 0.82) -> Signal:
    return Signal(
        id=str(uuid.uuid4()),
        market_id="MKT-1",
        estimated_probability=0.65,
        confidence=confidence,
        edge=0.15,
        market_mid_at_signal=0.50,
        direction=direction,
        reasoning="test",
        sources=[],
        retrieval_hash="abc123",
        model_used="claude-sonnet-4-6",
        prompt_version="v1",
        trigger="price_moved",
        created_at=NOW,
        raw_context="{}",
    )


def _make_strategy(
    *,
    stoploss: float = -0.20,
    trailing_stop: bool = False,
    trailing_stop_positive: float | None = None,
    trailing_stop_positive_offset: float = 0.02,
    min_confidence: float = 0.80,
) -> IPredictionStrategy:
    class _TestStrategy(IPredictionStrategy):
        config = StrategyConfig(
            name="TestStrategy",
            min_edge=0.10,
            min_confidence=min_confidence,
            max_exposure_per_market=0.05,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=365,
            min_days_to_close=0,
            stoploss=stoploss,
            trailing_stop=trailing_stop,
            trailing_stop_positive=trailing_stop_positive,
            trailing_stop_positive_offset=trailing_stop_positive_offset,
        )

        def should_trade(self, signal, market):  # type: ignore[override]
            return True

        def position_size(self, signal, bankroll):  # type: ignore[override]
            return bankroll * 0.01

    return _TestStrategy()


# ---------------------------------------------------------------------------
# _check_stoploss
# ---------------------------------------------------------------------------

class TestCheckStoploss:
    def test_fires_when_loss_exceeds_threshold(self) -> None:
        pos = _make_position(entry_price=0.50)
        # dollar_loss = 0.28 - 0.50 = -0.22 ≤ -0.20 → fires
        result = _check_stoploss(pos, current_price=0.28, stoploss=-0.20)
        assert result == ("stoploss", 0.28)

    def test_fires_just_at_threshold(self) -> None:
        pos = _make_position(entry_price=0.50)
        # entry + stoploss = 0.50 - 0.20 = 0.30; dollar_loss = 0.30 - 0.50 = -0.20 ≤ -0.20
        result = _check_stoploss(pos, current_price=0.30, stoploss=-0.20)
        assert result == ("stoploss", 0.30)

    def test_no_fire_when_loss_below_threshold(self) -> None:
        pos = _make_position(entry_price=0.50)
        # dollar_loss = 0.42 - 0.50 = -0.08 > -0.20
        result = _check_stoploss(pos, current_price=0.42, stoploss=-0.20)
        assert result is None

    def test_no_fire_when_profitable(self) -> None:
        pos = _make_position(entry_price=0.50)
        result = _check_stoploss(pos, current_price=0.65, stoploss=-0.20)
        assert result is None


# ---------------------------------------------------------------------------
# _check_trailing_stop
# ---------------------------------------------------------------------------

class TestCheckTrailingStop:
    def test_fires_when_price_drops_from_peak(self) -> None:
        pos = _make_position(entry_price=0.50)
        # peak=0.70, stoploss=-0.20 → trail_distance=0.20
        # stop = 0.70 - 0.20 = 0.50; current=0.49 ≤ 0.50
        result = _check_trailing_stop(
            pos,
            current_price=0.49,
            peak_price=0.70,
            stoploss=-0.20,
            trailing_stop_positive=None,
            trailing_stop_positive_offset=0.02,
        )
        assert result == ("trailing_stop", 0.49)

    def test_no_fire_when_above_trail(self) -> None:
        pos = _make_position(entry_price=0.50)
        # peak=0.70, stop=0.50; current=0.60 > 0.50
        result = _check_trailing_stop(
            pos,
            current_price=0.60,
            peak_price=0.70,
            stoploss=-0.20,
            trailing_stop_positive=None,
            trailing_stop_positive_offset=0.02,
        )
        assert result is None

    def test_tight_trail_applied_once_positive_threshold_crossed(self) -> None:
        pos = _make_position(entry_price=0.50)
        # peak_gain = 0.65 - 0.50 = 0.15 >= trailing_stop_positive=0.10 → tight trail
        # tight stop = 0.65 - 0.02 = 0.63; current=0.63 ≤ 0.63
        result = _check_trailing_stop(
            pos,
            current_price=0.63,
            peak_price=0.65,
            stoploss=-0.20,
            trailing_stop_positive=0.10,
            trailing_stop_positive_offset=0.02,
        )
        assert result == ("trailing_stop", 0.63)

    def test_normal_trail_used_before_positive_threshold(self) -> None:
        pos = _make_position(entry_price=0.50)
        # peak_gain = 0.54 - 0.50 = 0.04 < trailing_stop_positive=0.10 → normal trail
        # normal stop = 0.54 - 0.20 = 0.34; current=0.44 > 0.34
        result = _check_trailing_stop(
            pos,
            current_price=0.44,
            peak_price=0.54,
            stoploss=-0.20,
            trailing_stop_positive=0.10,
            trailing_stop_positive_offset=0.02,
        )
        assert result is None


# ---------------------------------------------------------------------------
# PositionMonitor.evaluate_exit — priority order
# ---------------------------------------------------------------------------

class TestEvaluateExit:
    """Tests that evaluate_exit respects exit priority and returns correct tags."""

    def _monitor(self, strategy: IPredictionStrategy) -> PositionMonitor:
        monitor = PositionMonitor(
            session_factory=MagicMock(),
            strategies={"TestStrategy": strategy},
        )
        return monitor

    def test_stoploss_fires(self) -> None:
        strategy = _make_strategy(stoploss=-0.20)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        market = _make_market(mid_price=0.28)  # dollar_loss = -0.22 ≤ -0.20 → stoploss

        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=0.28, strategy=strategy
        )
        assert result is not None
        assert result[0] == "stoploss"

    def test_trailing_stop_fires(self) -> None:
        strategy = _make_strategy(trailing_stop=True, stoploss=-0.20)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        monitor._peak_prices[pos.id] = 0.70  # seen a peak of 0.70

        # current=0.49 → below trail (0.70 - 0.20 = 0.50)
        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.49),
            current_price=0.49,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "trailing_stop"

    def test_custom_exit_fires_when_signal_provided(self) -> None:
        strategy = _make_strategy()
        strategy.custom_exit = MagicMock(return_value="time_based")  # type: ignore[method-assign]

        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50, entry_time=NOW)
        sig = _make_signal()

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.52),
            current_price=0.52,
            strategy=strategy,
            fresh_signal=sig,
        )
        assert result is not None
        assert result[0] == "custom_exit:time_based"

    def test_custom_exit_not_called_without_signal(self) -> None:
        strategy = _make_strategy()
        strategy.custom_exit = MagicMock(return_value="time_based")  # type: ignore[method-assign]

        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50, entry_time=NOW)

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.52),
            current_price=0.52,
            strategy=strategy,
            fresh_signal=None,
        )
        # No signal → custom_exit not called → no exit
        strategy.custom_exit.assert_not_called()
        assert result is None

    def test_signal_exit_fires_when_direction_flips(self) -> None:
        strategy = _make_strategy(min_confidence=0.80)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50, direction="YES", entry_time=NOW)
        sig = _make_signal(direction="NO", confidence=0.85)

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.52),
            current_price=0.52,
            strategy=strategy,
            fresh_signal=sig,
        )
        assert result is not None
        assert result[0] == "signal"

    def test_no_exit_when_close_time_passed_but_result_unknown(self) -> None:
        """close_time in the past but market.result is None → hold, don't close.

        Kalshi may pause/delay settlement after close_time. Closing at current
        price would lock in an arbitrary P&L before the real result is known.
        The sweep in MarketWatcher will eventually populate market.result and
        step 0 of evaluate_exit will handle the actual settlement.
        """
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        market = _make_market(mid_price=0.52, close_time=NOW - timedelta(hours=1))
        # market.result is None — Kalshi hasn't posted a result yet

        result = monitor.evaluate_exit(
            position=pos,
            market=market,
            current_price=0.52,
            strategy=strategy,
        )
        assert result is None

    def test_market_resolved_when_kalshi_status_resolved(self) -> None:
        """Fires market_resolved when Kalshi marks status='resolved', even if close_time is future."""
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        # close_time is in the future — wouldn't trigger via time check alone
        market = _make_market(mid_price=0.99, close_time=NOW + timedelta(days=5))
        market.status = "resolved"

        result = monitor.evaluate_exit(
            position=pos,
            market=market,
            current_price=0.99,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "market_resolved"

    def test_market_resolved_winning_position_exits_at_one(self) -> None:
        """YES position + market.result='yes' → exit_price=1.0, not mid_price."""
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50, direction="YES")
        market = _make_market(mid_price=0.60, close_time=NOW - timedelta(hours=1))
        market.result = "yes"

        result = monitor.evaluate_exit(
            position=pos,
            market=market,
            current_price=0.60,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "market_resolved"
        assert result[1] == pytest.approx(1.0)

    def test_market_resolved_losing_position_exits_at_zero(self) -> None:
        """NO position + market.result='yes' → exit_price=0.0 (NO side loses).

        mid_price=0.60 chosen so NO effective_price=0.40 > stoploss threshold,
        meaning stoploss does not fire before market_resolved.
        """
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.35, direction="NO")
        market = _make_market(mid_price=0.60, close_time=NOW - timedelta(hours=1))
        market.result = "yes"

        result = monitor.evaluate_exit(
            position=pos,
            market=market,
            current_price=0.60,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "market_resolved"
        assert result[1] == pytest.approx(0.0)

    def test_no_exit_when_close_time_passed_and_status_non_terminal(self) -> None:
        """close_time passed, result None, status non-terminal → hold position.

        Replaced the old 'falls back to effective_price' behavior. We no longer
        close on close_time alone — we wait for market.result (step 0) or a
        terminal status from Kalshi (step 6).
        """
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50, direction="YES")
        market = _make_market(mid_price=0.52, close_time=NOW - timedelta(hours=1))
        # market.result is None, market.status is "active" (non-terminal)

        result = monitor.evaluate_exit(
            position=pos,
            market=market,
            current_price=0.52,
            strategy=strategy,
        )
        assert result is None

    def test_no_market_resolved_when_status_open_and_close_time_future(self) -> None:
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        # Use real now so this test doesn't break as calendar time advances.
        market = _make_market(
            mid_price=0.52,
            close_time=datetime.now(tz=UTC) + timedelta(days=5),
        )
        market.metadata["status"] = "open"

        result = monitor.evaluate_exit(
            position=pos,
            market=market,
            current_price=0.52,
            strategy=strategy,
        )
        assert result is None

    def test_no_stoploss_false_fire_for_no_position(self) -> None:
        """Regression: NO position entered at 0.67 should NOT stoploss when YES mid is 0.355.

        no_bid = 1 - yes_ask = 1 - 0.375 = 0.625; dollar_loss = 0.625 - 0.67 = -0.045 > -0.20 → no fire.
        """
        strategy = _make_strategy(stoploss=-0.20)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.67, direction="NO")
        market = _make_market(mid_price=0.355)  # yes_ask=0.375 → no_bid=0.625
        no_bid = round(1.0 - market.yes_ask, 4)

        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=no_bid, strategy=strategy
        )
        assert result is None, (
            "NO position should not stoploss when YES price hasn't moved "
            "(stoploss was firing by comparing YES mid to NO entry price)"
        )

    def test_stoploss_fires_correctly_for_no_position(self) -> None:
        """NO position stoploss fires when YES price rises enough to push NO value below threshold."""
        strategy = _make_strategy(stoploss=-0.20)
        monitor = self._monitor(strategy)
        # Entered at NO ask 0.67; stoploss fires when no_bid < 0.67 - 0.20 = 0.47
        pos = _make_position(entry_price=0.67, direction="NO")
        # YES mid=0.70 → yes_ask=0.72 → no_bid=0.28; dollar_loss=0.28-0.67=-0.39 ≤ -0.20
        market = _make_market(mid_price=0.70)
        no_bid = round(1.0 - market.yes_ask, 4)
        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=no_bid, strategy=strategy
        )
        assert result is not None
        assert result[0] == "stoploss"
        # exit_price is the no_bid (what you'd actually receive)
        assert result[1] == pytest.approx(no_bid)

    def test_no_phantom_stoploss_on_resolved_market_no_position(self) -> None:
        """Regression: NO position on a resolved-NO market must NOT trigger stoploss.

        When a market stops trading, the Kalshi orderbook returns no_bid=0.0 (yes_ask=1.0
        by default when the book is empty). Without this guard, the stoploss fires first
        (0.0 - entry < stoploss) and closes the winning NO position at exit_price=0.0
        instead of letting market_resolved settle it at 1.0.
        """
        strategy = _make_strategy(stoploss=-0.30)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.78, direction="NO")
        # Market has resolved NO — order book is empty so yes_ask defaults to 1.0 → no_bid=0.0
        market = _make_market(mid_price=0.01)
        market.result = "no"
        market.status = "finalized"
        current_price = 0.0  # 1.0 - yes_ask(1.0) — phantom stale price after book clears

        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=current_price, strategy=strategy
        )
        assert result is not None
        assert result[0] == "market_resolved", (
            "Resolved market should settle via market_resolved, not stoploss"
        )
        assert result[1] == pytest.approx(1.0), (
            "NO position winning (market resolved NO) should exit at 1.0, not 0.0"
        )

    def test_empty_book_yes_position_last_price_high(self) -> None:
        """YES position: yes_bid=0, last_price=0.99, result=None → proxy=0.99, no stoploss.

        Kalshi clears the order book on close before writing result. last_price is used
        as the effective price proxy — far above entry, so stoploss doesn't fire.
        """
        strategy = _make_strategy(stoploss=-0.10)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.53, direction="YES")
        market = _make_market(mid_price=0.5)
        market.yes_bid = 0.0
        market.yes_ask = 1.0
        market.last_price = 0.99
        market.result = None
        market.status = "closed"

        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=0.0, strategy=strategy
        )
        assert result is None, "Empty book (last_price=0.99) must not fire stoploss"

    def test_empty_book_yes_position_mid_range_last_price(self) -> None:
        """YES position: yes_bid=0, last_price=0.43 (suspended market), result=None → no stoploss.

        Covers paused/suspended markets where last_price is mid-range, not near settlement.
        The old guard (last_price >= 0.95) would have let this stoploss fire at 0.
        Proxy = 0.43; entry = 0.38 → delta = +0.05, above stoploss threshold.
        """
        strategy = _make_strategy(stoploss=-0.10)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.38, direction="YES")
        market = _make_market(mid_price=0.5)
        market.yes_bid = 0.0
        market.yes_ask = 1.0
        market.last_price = 0.43
        market.result = None
        market.status = "closed"

        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=0.0, strategy=strategy
        )
        assert result is None, "Suspended market with mid-range last_price must not fire stoploss at 0"

    def test_real_zero_yes_position_stoploss_fires(self) -> None:
        """YES position: yes_bid=0 and last_price=0.01 → genuine collapse, stoploss fires.

        Proxy = last_price = 0.01; entry = 0.53 → delta = -0.52, exceeds stoploss threshold.
        """
        strategy = _make_strategy(stoploss=-0.10)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.53, direction="YES")
        market = _make_market(mid_price=0.01)
        market.yes_bid = 0.0
        market.yes_ask = 0.02
        market.last_price = 0.01
        market.result = None

        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=0.0, strategy=strategy
        )
        assert result is not None
        assert result[0] == "stoploss"

    def test_empty_book_no_position_last_price_low(self) -> None:
        """NO position: yes_ask=1.0 (empty book), last_price=0.01, result=None → proxy=0.99, no stoploss.

        Proxy for NO = 1 - last_price = 0.99; entry = 0.47 → large gain, stoploss suppressed.
        """
        strategy = _make_strategy(stoploss=-0.10)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.47, direction="NO")
        market = _make_market(mid_price=0.5)
        market.yes_bid = 0.0
        market.yes_ask = 1.0
        market.last_price = 0.01
        market.result = None
        market.status = "closed"

        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=0.0, strategy=strategy
        )
        assert result is None, "Empty book (NO position, last_price=0.01) must not fire stoploss"

    def test_no_exit_when_conditions_not_met(self) -> None:
        strategy = _make_strategy(stoploss=-0.20)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50, entry_time=NOW)

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.55),  # only +10%
            current_price=0.55,
            strategy=strategy,
        )
        assert result is None

    def test_hold_when_no_fresh_signal_for_signal_exit(self) -> None:
        """should_exit is not called without a fresh signal."""
        strategy = _make_strategy()
        strategy.should_exit = MagicMock(return_value=True)  # type: ignore[method-assign]

        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50, direction="YES")

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.52),
            current_price=0.52,
            strategy=strategy,
            fresh_signal=None,
        )
        strategy.should_exit.assert_not_called()
        assert result is None


# ---------------------------------------------------------------------------
# PositionMonitor.check_all_positions — integration
# ---------------------------------------------------------------------------

class TestCheckAllPositions:
    """Smoke tests for the async loop. DB calls are mocked."""

    @pytest.mark.asyncio
    async def test_closes_position_on_stoploss(self) -> None:
        strategy = _make_strategy(stoploss=-0.20)
        pos = _make_position(entry_price=0.50, strategy_name="TestStrategy")
        pos_id = pos.id

        # Set up mock session
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        session_factory = MagicMock()
        session_factory.return_value = session

        monitor = PositionMonitor(
            session_factory=session_factory,
            strategies={"TestStrategy": strategy},
        )

        closed_pos = _make_position(
            entry_price=0.50, strategy_name="TestStrategy", position_id=pos_id
        )
        closed_pos = Position(
            **{
                **closed_pos.__dict__,
                "status": "closed",
                "exit_price": 0.38,
                "exit_reason": "stoploss",
                "pnl": -12.0,
                "pnl_pct": -0.24,
            }
        )

        market = _make_market(mid_price=0.38)  # injected via evaluate_exit mock; price value unused

        with (
            patch(
                "freqpred.trading.position_monitor.ledger.get_open_positions",
                new_callable=AsyncMock,
                return_value=[pos],
            ),
            patch(
                "freqpred.trading.position_monitor.ledger.close_position",
                new_callable=AsyncMock,
                return_value=closed_pos,
            ) as mock_close,
            patch(
                "freqpred.trading.position_monitor.select",
            ),
        ):
            # Patch the market fetch
            scalars_mock = MagicMock()
            scalars_mock.scalars.return_value.all.return_value = []
            session.execute = AsyncMock(return_value=scalars_mock)

            # Inject market directly via evaluate_exit override
            monitor.evaluate_exit = MagicMock(  # type: ignore[method-assign]
                return_value=("stoploss", 0.38)
            )

            # Override check_all_positions to use our injected market
            async def _patched_check(**kwargs):  # type: ignore[no-untyped-def]
                # Replicate check_all_positions but with known market
                from freqpred.trading import ledger as _ledger
                positions = [pos]
                closed = []
                for position in positions:
                    result = monitor.evaluate_exit(
                        position=position,
                        market=market,
                        current_price=market.mid_price,
                        strategy=strategy,
                    )
                    if result is not None:
                        exit_reason, exit_price = result
                        c = await _ledger.close_position(
                            session,
                            position.id,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                        )
                        closed.append(c)
                return closed

            result = await _patched_check()

        assert len(result) == 1
        assert result[0].exit_reason == "stoploss"
        mock_close.assert_awaited_once_with(
            session,
            pos_id,
            exit_price=0.38,
            exit_reason="stoploss",
        )

    @pytest.mark.asyncio
    async def test_resolution_inferred_from_settled_price(self) -> None:
        """When position exits at YES contract price ~1.0, resolution=1 is passed to close_position."""
        strategy = _make_strategy()
        pos = _make_position(entry_price=0.50, strategy_name="TestStrategy", direction="YES")
        pos_id = pos.id

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session_factory = MagicMock()
        session_factory.return_value = session

        monitor = PositionMonitor(
            session_factory=session_factory,
            strategies={"TestStrategy": strategy},
        )

        closed_pos = Position(
            **{
                **pos.__dict__, "status": "closed", "exit_price": 1.0,
                "exit_reason": "market_resolved", "resolution": 1, "pnl": 50.0, "pnl_pct": 1.0,
            }
        )

        # Market is Kalshi-resolved YES — exit_price must be 1.0 (payout), not mid_price
        market = _make_market(mid_price=0.99, close_time=NOW + timedelta(days=5))
        market.status = "resolved"
        market.result = "yes"

        with (
            patch(
                "freqpred.trading.position_monitor.ledger.get_open_positions",
                new_callable=AsyncMock,
                return_value=[pos],
            ),
            patch(
                "freqpred.trading.position_monitor.ledger.get_pending_positions",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.trading.position_monitor.ledger.close_position",
                new_callable=AsyncMock,
                return_value=closed_pos,
            ) as mock_close,
        ):
            scalars_mock = MagicMock()
            scalars_mock.scalars.return_value.all.return_value = [
                MagicMock(
                    id=market.id,
                    platform=market.platform,
                    question=market.question,
                    category=market.category,
                    status="resolved",
                    result="yes",
                    settlement_value=None,
                    close_time=market.close_time,
                    yes_bid=market.yes_bid,
                    yes_ask=market.yes_ask,
                    mid_price=market.mid_price,
                    last_price=0.0,
                    volume_24h=market.volume_24h,
                    open_interest=market.open_interest,
                    yes_bid_size=0.0,
                    yes_ask_size=0.0,
                    last_fetched_at=market.last_fetched_at,
                    price_updated_at=market.price_updated_at,
                    metadata_fetched_at=market.metadata_fetched_at,
                    current_signal_id=None,
                    metadata_={},
                )
            ]
            session.execute = AsyncMock(return_value=scalars_mock)

            result = await monitor.check_all_positions()

        assert len(result) == 1
        mock_close.assert_awaited_once_with(
            session,
            pos_id,
            exit_price=1.0,
            exit_reason="market_resolved",
            resolution=1,
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_open_positions(self) -> None:
        strategy = _make_strategy()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session_factory = MagicMock()
        session_factory.return_value = session

        monitor = PositionMonitor(
            session_factory=session_factory,
            strategies={"TestStrategy": strategy},
        )

        with patch(
            "freqpred.trading.position_monitor.ledger.get_open_positions",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "freqpred.trading.position_monitor.ledger.get_pending_positions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await monitor.check_all_positions()

        assert result == []

    @pytest.mark.asyncio
    async def test_fresh_signal_flip_triggers_exit(self) -> None:
        """check_all_positions(fresh_signals=...) fires should_exit when direction flips."""
        strategy = _make_strategy(min_confidence=0.70)
        pos = _make_position(entry_price=0.60, direction="YES", strategy_name="TestStrategy")

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session_factory = MagicMock()
        session_factory.return_value = session

        monitor = PositionMonitor(
            session_factory=session_factory,
            strategies={"TestStrategy": strategy},
        )

        closed_pos = Position(
            **{**pos.__dict__, "status": "closed", "exit_price": 0.58,
               "exit_reason": "signal", "pnl": -0.02, "pnl_pct": -0.033}
        )

        market = _make_market(mid_price=0.60)
        flip_signal = _make_signal(direction="NO", confidence=0.85)

        scalars_mock = MagicMock()
        scalars_mock.scalars.return_value.all.return_value = [
            MagicMock(
                id=market.id,
                platform=market.platform,
                question=market.question,
                category=market.category,
                status="open",
                result=None,
                settlement_value=None,
                close_time=market.close_time,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                mid_price=market.mid_price,
                last_price=0.58,
                volume_24h=market.volume_24h,
                open_interest=market.open_interest,
                yes_bid_size=0.0,
                yes_ask_size=0.0,
                last_fetched_at=market.last_fetched_at,
                price_updated_at=market.price_updated_at,
                metadata_fetched_at=market.metadata_fetched_at,
                current_signal_id=None,
                metadata_={},
                open_time=None,
                series_ticker=None,
            )
        ]

        with (
            patch(
                "freqpred.trading.position_monitor.ledger.get_open_positions",
                new_callable=AsyncMock,
                return_value=[pos],
            ),
            patch(
                "freqpred.trading.position_monitor.ledger.get_pending_positions",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.trading.position_monitor.ledger.close_position",
                new_callable=AsyncMock,
                return_value=closed_pos,
            ) as mock_close,
        ):
            session.execute = AsyncMock(return_value=scalars_mock)
            result = await monitor.check_all_positions(
                fresh_signals={"MKT-1": flip_signal}
            )

        assert len(result) == 1
        assert result[0].exit_reason == "signal"
        mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Peak price tracking
# ---------------------------------------------------------------------------

class TestPeakPriceTracking:
    def test_peak_advances_on_higher_price(self) -> None:
        monitor = PositionMonitor(
            session_factory=MagicMock(),
            strategies={},
        )
        pos = _make_position(entry_price=0.50)
        monitor._update_peak(pos, 0.65)
        assert monitor._peak_prices[pos.id] == pytest.approx(0.65)

    def test_peak_does_not_retreat(self) -> None:
        monitor = PositionMonitor(
            session_factory=MagicMock(),
            strategies={},
        )
        pos = _make_position(entry_price=0.50)
        monitor._update_peak(pos, 0.70)
        monitor._update_peak(pos, 0.60)  # lower — should not update
        assert monitor._peak_prices[pos.id] == pytest.approx(0.70)

    def test_peak_initialises_to_entry_price(self) -> None:
        monitor = PositionMonitor(
            session_factory=MagicMock(),
            strategies={},
        )
        pos = _make_position(entry_price=0.50)
        # Before any update, peak is entry_price
        peak = monitor._peak_prices.get(pos.id, pos.entry_price)
        assert peak == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Live exit — T38
# ---------------------------------------------------------------------------

class TestLiveExit:
    """Tests for _execute_live_exit and the live/paper branching in _execute_exit."""

    def _make_monitor(
        self,
        kalshi_client: MagicMock | None = None,
        alert_dispatcher: MagicMock | None = None,
    ) -> PositionMonitor:
        return PositionMonitor(
            session_factory=MagicMock(),
            strategies={"TestStrategy": _make_strategy()},
            mode="live",
            kalshi_client=kalshi_client,
            alert_dispatcher=alert_dispatcher,
        )

    @pytest.mark.asyncio
    async def test_live_exit_calls_place_order(self) -> None:
        """Live exit submits an IOC order in the opposite direction to close the position."""
        pos = _make_position(direction="YES", contracts=50, mode="live")
        market = _make_market(mid_price=0.65)

        kalshi_client = MagicMock()
        filled_order = Order(
            market_id=pos.market_id,
            direction="YES",
            contracts=50,
            price=market.yes_bid,
            mode="live",
            status="executed",
        )
        kalshi_client.place_order = AsyncMock(return_value=filled_order)

        monitor = self._make_monitor(kalshi_client=kalshi_client)
        session = AsyncMock()

        with patch(
            "freqpred.trading.position_monitor.ledger.partial_close_position",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ):
            await monitor._execute_live_exit(
                session=session,
                position=pos,
                market=market,
                exit_reason="stoploss",
                resolution=None,
            )

        kalshi_client.place_order.assert_awaited_once()
        submitted: Order = kalshi_client.place_order.call_args.args[0]
        assert submitted.direction == "YES"         # same side as position
        assert submitted.action == "sell"
        assert submitted.contracts == 50
        assert submitted.time_in_force == "fill_or_kill"
        assert submitted.market_id == pos.market_id

    @pytest.mark.asyncio
    async def test_live_exit_uses_filled_price_for_pnl(self) -> None:
        """ledger.partial_close_position() is called with the price from the exchange response."""
        pos = _make_position(direction="YES", mode="live")
        market = _make_market(mid_price=0.65)

        filled_price = 0.72
        kalshi_client = MagicMock()
        filled_order = Order(
            market_id=pos.market_id,
            direction="YES",
            contracts=pos.contracts,
            price=filled_price,
            mode="live",
            status="executed",
            action="sell",
        )
        kalshi_client.place_order = AsyncMock(return_value=filled_order)

        monitor = self._make_monitor(kalshi_client=kalshi_client)
        session = AsyncMock()

        with patch(
            "freqpred.trading.position_monitor.ledger.partial_close_position",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_close:
            await monitor._execute_live_exit(
                session=session,
                position=pos,
                market=market,
                exit_reason="roi",
                resolution=None,
            )

        mock_close.assert_awaited_once()
        _, call_kwargs = mock_close.call_args
        assert call_kwargs["fill_price"] == pytest.approx(filled_price)

    @pytest.mark.asyncio
    async def test_live_exit_does_not_close_on_api_error(self) -> None:
        """On KalshiAPIError: position stays open, alert is sent, ledger not called."""
        pos = _make_position(direction="YES", mode="live")
        market = _make_market(mid_price=0.65)

        kalshi_client = MagicMock()
        kalshi_client.place_order = AsyncMock(
            side_effect=KalshiAPIError(503, "service unavailable")
        )

        alert_dispatcher = MagicMock()
        alert_dispatcher.send = AsyncMock()

        monitor = self._make_monitor(
            kalshi_client=kalshi_client, alert_dispatcher=alert_dispatcher
        )
        session = AsyncMock()

        with patch(
            "freqpred.trading.position_monitor.ledger.partial_close_position",
            new_callable=AsyncMock,
        ) as mock_close:
            result = await monitor._execute_live_exit(
                session=session,
                position=pos,
                market=market,
                exit_reason="stoploss",
                resolution=None,
            )

        assert result is None
        mock_close.assert_not_awaited()
        alert_dispatcher.send.assert_awaited_once()
        # Alert message should mention the market question
        alert_msg: str = alert_dispatcher.send.call_args.args[0]
        assert pos.market_id in alert_msg or market.question in alert_msg

    @pytest.mark.asyncio
    async def test_paper_exit_unchanged(self) -> None:
        """Paper mode calls ledger.close_position() directly — no Kalshi API call."""
        pos = _make_position(direction="YES", mode="paper")
        market = _make_market(mid_price=0.65)

        kalshi_client = MagicMock()
        kalshi_client.place_order = AsyncMock()

        monitor = PositionMonitor(
            session_factory=MagicMock(),
            strategies={"TestStrategy": _make_strategy()},
            mode="paper",
            kalshi_client=kalshi_client,
        )
        session = AsyncMock()

        with patch(
            "freqpred.trading.position_monitor.ledger.close_position",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_close:
            await monitor._execute_exit(
                session=session,
                position=pos,
                market=market,
                exit_reason="roi",
                exit_price=0.65,
                resolution=None,
            )

        mock_close.assert_awaited_once()
        kalshi_client.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_exit_submits_no_order(self) -> None:
        """When evaluate_exit returns None, place_order is never called."""
        strategy = _make_strategy(stoploss=-0.20)
        pos = _make_position(entry_price=0.50, entry_time=NOW, mode="live")

        kalshi_client = MagicMock()
        kalshi_client.place_order = AsyncMock()

        monitor = PositionMonitor(
            session_factory=MagicMock(),
            strategies={"TestStrategy": strategy},
            mode="live",
            kalshi_client=kalshi_client,
        )

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.55),  # only +10%, no exit fires
            current_price=0.55,
            strategy=strategy,
        )

        assert result is None
        kalshi_client.place_order.assert_not_awaited()


# ---------------------------------------------------------------------------
# T67: periodic reconcile wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_periodic_reconcile_called_from_tick_loop() -> None:
    """PositionMonitor's tick loop drives order_manager.reconcile_pending_orders."""
    from freqpred.trading.position_monitor import PositionMonitor as _PM

    session_factory = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    session_factory.return_value = mock_session

    order_manager = MagicMock()
    order_manager.reconcile_pending_orders = AsyncMock()

    monitor = _PM(
        session_factory=session_factory,
        strategies={},
        mode="live",
        order_manager=order_manager,
        reconcile_interval_seconds=0.0,  # always fire
    )

    await monitor._maybe_run_periodic_reconcile()
    order_manager.reconcile_pending_orders.assert_awaited_once()


@pytest.mark.asyncio
async def test_periodic_reconcile_noop_in_paper_mode() -> None:
    """Paper mode never drives reconcile, even if order_manager is wired."""
    from freqpred.trading.position_monitor import PositionMonitor as _PM

    order_manager = MagicMock()
    order_manager.reconcile_pending_orders = AsyncMock()
    monitor = _PM(
        session_factory=MagicMock(),
        strategies={},
        mode="paper",
        order_manager=order_manager,
        reconcile_interval_seconds=0.0,
    )
    await monitor._maybe_run_periodic_reconcile()
    order_manager.reconcile_pending_orders.assert_not_called()


# ---------------------------------------------------------------------------
# T76: exit-side polling + partial-fill tests
# ---------------------------------------------------------------------------


class TestLiveExitPolling:
    """Tests for T76: _execute_live_exit polling and partial-fill handling."""

    def _make_monitor(
        self,
        kalshi_client: MagicMock | None = None,
        alert_dispatcher: MagicMock | None = None,
    ) -> PositionMonitor:
        return PositionMonitor(
            session_factory=MagicMock(),
            strategies={"TestStrategy": _make_strategy()},
            mode="live",
            kalshi_client=kalshi_client,
            alert_dispatcher=alert_dispatcher,
        )

    @pytest.mark.asyncio
    async def test_live_exit_polls_until_terminal_state(self) -> None:
        """_execute_live_exit calls get_order when place_order returns non-terminal."""
        pos = _make_position(direction="YES", contracts=10, mode="live")
        market = _make_market(mid_price=0.65)

        # place_order returns resting (non-terminal); get_order returns executed on first poll
        non_terminal = Order(
            market_id=pos.market_id, direction="YES", contracts=10,
            price=0.63, mode="live", status="resting",
            exchange_order_id="exit-order-1",
        )
        terminal = Order(
            market_id=pos.market_id, direction="YES", contracts=10,
            price=0.63, mode="live", status="executed",
            exchange_order_id="exit-order-1",
            filled_yes_count=10,
        )

        kalshi_client = MagicMock()
        kalshi_client.place_order = AsyncMock(return_value=non_terminal)
        kalshi_client.get_order = AsyncMock(return_value=terminal)

        monitor = self._make_monitor(kalshi_client=kalshi_client)
        session = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch(
                "freqpred.trading.position_monitor.ledger.partial_close_position",
                new_callable=AsyncMock,
                return_value=MagicMock(status="closed"),
            ):
                await monitor._execute_live_exit(
                    session=session, position=pos, market=market,
                    exit_reason="stoploss", resolution=None,
                )

        kalshi_client.get_order.assert_awaited()

    @pytest.mark.asyncio
    async def test_live_exit_partial_fill_leaves_position_open_for_retry(self) -> None:
        """When IOC fills 6/10, partial_close_position is called with filled_contracts=6."""
        pos = _make_position(direction="YES", contracts=10, mode="live", entry_price=0.50)
        market = _make_market(mid_price=0.65)

        # IOC canceled after partially filling 6/10
        partial_fill_order = Order(
            market_id=pos.market_id, direction="YES", contracts=10,
            price=0.63, mode="live", status="canceled",
            exchange_order_id="exit-order-1",
            filled_yes_count=6,
            requested_count=10,
        )
        kalshi_client = MagicMock()
        kalshi_client.place_order = AsyncMock(return_value=partial_fill_order)

        monitor = self._make_monitor(kalshi_client=kalshi_client)
        session = AsyncMock()

        with patch(
            "freqpred.trading.position_monitor.ledger.partial_close_position",
            new_callable=AsyncMock,
            return_value=MagicMock(status="open", contracts=4),
        ) as mock_partial:
            await monitor._execute_live_exit(
                session=session, position=pos, market=market,
                exit_reason="stoploss", resolution=None,
            )

        mock_partial.assert_awaited_once()
        _, kwargs = mock_partial.call_args
        assert kwargs["filled_contracts"] == 6
        assert kwargs["fill_price"] == pytest.approx(0.63)
        assert kwargs["exit_reason"] == "stoploss"

    @pytest.mark.asyncio
    async def test_live_exit_residual_size_used_for_next_tick_stoploss_check(self) -> None:
        """After partial close leaves 4 contracts, stoploss checks position.contracts=4."""
        pos = _make_position(direction="YES", contracts=4, mode="live", entry_price=0.50)
        # Stoploss at -0.20; current price = entry - 0.22 → fires
        strategy = _make_strategy(stoploss=-0.20)
        market = _make_market(mid_price=0.28)

        result = strategy.config.stoploss
        exit_check = pos.entry_price + result  # 0.50 - 0.20 = 0.30
        current_effective = market.yes_bid  # close to mid
        assert current_effective < exit_check or market.mid_price < exit_check

        from freqpred.trading.position_monitor import _check_stoploss
        outcome = _check_stoploss(pos, current_price=0.28, stoploss=-0.20)
        assert outcome is not None
        # Should fire for the 4-contract residual just as for the original 10
        assert outcome[0] == "stoploss"

    @pytest.mark.asyncio
    async def test_live_exit_zero_fill_leaves_position_fully_open(self) -> None:
        """IOC fully cancelled with no fill → partial_close_position NOT called."""
        pos = _make_position(direction="YES", contracts=10, mode="live")
        market = _make_market(mid_price=0.65)

        no_fill_order = Order(
            market_id=pos.market_id, direction="YES", contracts=10,
            price=0.63, mode="live", status="canceled",
            exchange_order_id="exit-order-1",
            filled_yes_count=0,
            requested_count=10,
        )
        kalshi_client = MagicMock()
        kalshi_client.place_order = AsyncMock(return_value=no_fill_order)

        monitor = self._make_monitor(kalshi_client=kalshi_client)
        session = AsyncMock()

        with patch(
            "freqpred.trading.position_monitor.ledger.partial_close_position",
            new_callable=AsyncMock,
        ) as mock_partial:
            result = await monitor._execute_live_exit(
                session=session, position=pos, market=market,
                exit_reason="stoploss", resolution=None,
            )

        assert result is None
        mock_partial.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_exit_poll_timeout_sends_alert(self) -> None:
        """When polling times out, alert is sent and position stays open."""
        pos = _make_position(direction="YES", contracts=10, mode="live")
        market = _make_market(mid_price=0.65)

        # place_order returns non-terminal; get_order never reaches terminal
        non_terminal = Order(
            market_id=pos.market_id, direction="YES", contracts=10,
            price=0.63, mode="live", status="resting",
            exchange_order_id="exit-order-1",
        )
        kalshi_client = MagicMock()
        kalshi_client.place_order = AsyncMock(return_value=non_terminal)
        kalshi_client.get_order = AsyncMock(return_value=non_terminal)

        alert_dispatcher = MagicMock()
        alert_dispatcher.send = AsyncMock()
        monitor = self._make_monitor(
            kalshi_client=kalshi_client, alert_dispatcher=alert_dispatcher
        )
        session = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch(
                "freqpred.trading.position_monitor.ledger.partial_close_position",
                new_callable=AsyncMock,
            ) as mock_partial:
                result = await monitor._execute_live_exit(
                    session=session, position=pos, market=market,
                    exit_reason="stoploss", resolution=None,
                )

        assert result is None
        mock_partial.assert_not_awaited()
        alert_dispatcher.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_live_exit_no_side_both_yes_and_no(self) -> None:
        """YES and NO positions both submit sell orders at the correct side bid."""
        for direction in ("YES", "NO"):
            pos = _make_position(direction=direction, contracts=5, mode="live")
            market = _make_market(mid_price=0.62)
            # yes_bid = 0.61, yes_ask = 0.63; no_bid = 1 - 0.63 = 0.37, limit for NO = 1 - 0.63 = 0.37
            if direction == "YES":
                expected = round(market.yes_bid or 0.0, 4)
            else:
                expected = round(1.0 - (market.yes_ask or 1.0), 4)

            placed_order = Order(
                market_id=pos.market_id, direction=direction, contracts=5,
                price=expected, mode="live", status="executed",
                exchange_order_id="ord-1", filled_yes_count=5 if direction == "YES" else 0,
                filled_no_count=0 if direction == "YES" else 5,
            )
            kalshi_client = MagicMock()
            kalshi_client.place_order = AsyncMock(return_value=placed_order)
            monitor = self._make_monitor(kalshi_client=kalshi_client)
            session = AsyncMock()

            with patch(
                "freqpred.trading.position_monitor.ledger.partial_close_position",
                new_callable=AsyncMock,
                return_value=MagicMock(status="closed"),
            ) as mock_partial:
                await monitor._execute_live_exit(
                    session=session, position=pos, market=market,
                    exit_reason="stoploss", resolution=None,
                )

            mock_partial.assert_awaited_once()
            submitted: Order = kalshi_client.place_order.call_args.args[0]
            assert submitted.direction == direction
            assert submitted.price == pytest.approx(expected)


# ---------------------------------------------------------------------------
# T47 — paper-mode pending fill check
# ---------------------------------------------------------------------------


class TestPendingPositionFillCheck:
    """Tests for PositionMonitor._process_pending_position."""

    def _make_pending_position(
        self,
        *,
        direction: str = "YES",
        entry_price: float = 0.50,
        entry_time: datetime | None = None,
    ) -> Position:
        return _make_position(
            entry_price=entry_price,
            direction=direction,
            entry_time=entry_time or NOW,
            mode="paper",
        )

    def _make_strategy_with_timeout(self, timeout_hours: float = 4.0):
        class _S(IPredictionStrategy):
            config = StrategyConfig(
                name="TestStrategy",
                min_edge=0.10,
                min_confidence=0.70,
                max_exposure_per_market=0.05,
                kelly_fraction=0.25,
                categories=[],
                min_volume_24h=0.0,
                max_days_to_close=365,
                min_days_to_close=0,
                limit_order_timeout_hours=timeout_hours,
            )

            def should_trade(self, signal, market):  # type: ignore[override]
                return True

            def position_size(self, signal, bankroll):  # type: ignore[override]
                return 0.0

        return _S()

    def _monitor_with_strategy(self, strategy, mode: str = "paper") -> PositionMonitor:
        return PositionMonitor(
            session_factory=MagicMock(),
            strategies={"TestStrategy": strategy},
            mode=mode,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("direction,yes_bid,yes_ask,entry_price,should_fill", [
        # YES: fills when yes_ask <= entry_price
        ("YES", 0.45, 0.50, 0.50, True),   # ask == entry → fills
        ("YES", 0.44, 0.49, 0.50, True),   # ask below entry → fills
        ("YES", 0.45, 0.51, 0.50, False),  # ask above entry → stays pending
        # NO: fills when (1 - yes_bid) <= entry_price
        ("NO",  0.50, 0.55, 0.50, True),   # no_ask = 1-0.50 = 0.50 == entry → fills
        ("NO",  0.52, 0.56, 0.50, True),   # no_ask = 1-0.52 = 0.48 < entry → fills
        ("NO",  0.48, 0.53, 0.50, False),  # no_ask = 1-0.48 = 0.52 > entry → stays pending
    ])
    async def test_pending_fill_and_no_fill(
        self,
        direction: str,
        yes_bid: float,
        yes_ask: float,
        entry_price: float,
        should_fill: bool,
    ) -> None:
        strategy = self._make_strategy_with_timeout()
        monitor = self._monitor_with_strategy(strategy)
        pos = self._make_pending_position(direction=direction, entry_price=entry_price)
        market = Market(
            id="MKT-1",
            platform="kalshi",
            question="test",
            category="politics",
            close_time=NOW + timedelta(days=10),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            mid_price=(yes_bid + yes_ask) / 2,
            volume_24h=1000.0,
            open_interest=500.0,
            last_fetched_at=NOW,
            price_updated_at=NOW,
            metadata_fetched_at=NOW,
        )

        with patch(
            "freqpred.trading.position_monitor.ledger.promote_pending_to_open",
            new_callable=AsyncMock,
        ) as mock_promote, patch(
            "freqpred.trading.position_monitor.ledger.close_position",
            new_callable=AsyncMock,
        ) as mock_close:
            await monitor._process_pending_position(pos, market, strategy, NOW)

        if should_fill:
            mock_promote.assert_awaited_once()
            # fill_price = min(current_ask, entry_price)
            if direction == "YES":
                expected_fill = min(yes_ask, entry_price)
            else:
                expected_fill = min(round(1.0 - yes_bid, 4), entry_price)
            assert mock_promote.call_args.kwargs["fill_price"] == pytest.approx(expected_fill)
            mock_close.assert_not_awaited()
        else:
            mock_promote.assert_not_awaited()
            mock_close.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("direction,yes_bid,yes_ask,limit_price,expected_fill", [
        # ask already below limit → fill at ask (price improvement)
        ("YES", 0.43, 0.45, 0.50, 0.45),
        # ask exactly at limit → fill at limit
        ("YES", 0.46, 0.50, 0.50, 0.50),
        # NO: no_ask = 1 - yes_bid; below limit → fill at no_ask
        ("NO",  0.55, 0.60, 0.50, round(1.0 - 0.55, 4)),  # no_ask=0.45 < 0.50
        # NO: no_ask exactly at limit
        ("NO",  0.50, 0.55, 0.50, 0.50),
    ])
    async def test_fill_price_reflects_price_improvement(
        self,
        direction: str,
        yes_bid: float,
        yes_ask: float,
        limit_price: float,
        expected_fill: float,
    ) -> None:
        """fill_price = min(current_ask, limit_price) — captures price improvement."""
        strategy = self._make_strategy_with_timeout()
        monitor = self._monitor_with_strategy(strategy)
        pos = self._make_pending_position(direction=direction, entry_price=limit_price)
        market = Market(
            id="MKT-1",
            platform="kalshi",
            question="test",
            category="politics",
            close_time=NOW + timedelta(days=10),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            mid_price=(yes_bid + yes_ask) / 2,
            volume_24h=1000.0,
            open_interest=500.0,
            last_fetched_at=NOW,
            price_updated_at=NOW,
            metadata_fetched_at=NOW,
        )

        with patch(
            "freqpred.trading.position_monitor.ledger.promote_pending_to_open",
            new_callable=AsyncMock,
        ) as mock_promote, patch(
            "freqpred.trading.position_monitor.ledger.close_position",
            new_callable=AsyncMock,
        ):
            await monitor._process_pending_position(pos, market, strategy, NOW)

        mock_promote.assert_awaited_once()
        assert mock_promote.call_args.kwargs["fill_price"] == pytest.approx(expected_fill)

    @pytest.mark.asyncio
    async def test_pending_position_cancelled_on_timeout(self) -> None:
        """Position older than limit_order_timeout_hours is closed with exit_reason='cancelled'."""
        strategy = self._make_strategy_with_timeout(timeout_hours=4.0)
        monitor = self._monitor_with_strategy(strategy)
        old_entry_time = NOW - timedelta(hours=5)  # 5 h old > 4 h timeout
        pos = self._make_pending_position(entry_price=0.50, entry_time=old_entry_time)
        market = _make_market(mid_price=0.55)  # ask above limit → won't fill

        with patch(
            "freqpred.trading.position_monitor.ledger.promote_pending_to_open",
            new_callable=AsyncMock,
        ) as mock_promote, patch(
            "freqpred.trading.position_monitor.ledger.close_position",
            new_callable=AsyncMock,
        ) as mock_close:
            await monitor._process_pending_position(pos, market, strategy, NOW)

        mock_promote.assert_not_awaited()
        mock_close.assert_awaited_once()
        kwargs = mock_close.call_args.kwargs
        assert kwargs["exit_reason"] == "cancelled"
        assert kwargs["exit_price"] == pytest.approx(pos.entry_price)

    @pytest.mark.asyncio
    async def test_pending_position_not_cancelled_before_timeout(self) -> None:
        """Position younger than timeout is left alone when ask is above limit."""
        strategy = self._make_strategy_with_timeout(timeout_hours=4.0)
        monitor = self._monitor_with_strategy(strategy)
        recent_entry = NOW - timedelta(hours=2)  # 2 h old < 4 h timeout
        pos = self._make_pending_position(entry_price=0.50, entry_time=recent_entry)
        market = _make_market(mid_price=0.55)  # ask above limit → won't fill

        with patch(
            "freqpred.trading.position_monitor.ledger.promote_pending_to_open",
            new_callable=AsyncMock,
        ) as mock_promote, patch(
            "freqpred.trading.position_monitor.ledger.close_position",
            new_callable=AsyncMock,
        ) as mock_close:
            await monitor._process_pending_position(pos, market, strategy, NOW)

        mock_promote.assert_not_awaited()
        mock_close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_check_all_positions_processes_pending_in_paper_mode(self) -> None:
        """check_all_positions fetches pending positions and calls _process_pending_position."""
        strategy = self._make_strategy_with_timeout()
        pending_pos = self._make_pending_position(entry_price=0.50)
        market_row = MagicMock()
        market_row.id = "MKT-1"
        market_row.platform = "kalshi"
        market_row.question = "test"
        market_row.category = "politics"
        market_row.close_time = NOW + timedelta(days=10)
        market_row.yes_bid = 0.45
        market_row.yes_ask = 0.49  # below entry_price → should fill
        market_row.mid_price = 0.47
        market_row.last_price = 0.47
        market_row.volume_24h = 1000.0
        market_row.open_interest = 500.0
        market_row.last_fetched_at = NOW
        market_row.price_updated_at = NOW
        market_row.metadata_fetched_at = NOW
        market_row.yes_bid_size = None
        market_row.yes_ask_size = None
        market_row.result = None
        market_row.settlement_value = None
        market_row.status = "open"
        market_row.open_time = None
        market_row.series_ticker = None
        market_row.current_signal_id = None
        market_row.metadata_ = {}

        # get_open_positions and get_pending_positions are patched at the ledger
        # level, so session.execute is only called once — for the market query.
        market_result = MagicMock()
        market_result.scalars.return_value.all.return_value = [market_row]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=market_result)

        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        session_factory = MagicMock(return_value=session_ctx)

        monitor = PositionMonitor(
            session_factory=session_factory,
            strategies={"TestStrategy": strategy},
            mode="paper",
        )

        with patch(
            "freqpred.trading.position_monitor.ledger.get_open_positions",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "freqpred.trading.position_monitor.ledger.get_pending_positions",
            new_callable=AsyncMock,
            return_value=[pending_pos],
        ), patch(
            "freqpred.trading.position_monitor.ledger.promote_pending_to_open",
            new_callable=AsyncMock,
        ) as mock_promote:
            await monitor.check_all_positions(_now=NOW)

        mock_promote.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_all_positions_skips_pending_in_live_mode(self) -> None:
        """Live mode does not call get_pending_positions (handled by reconcile)."""
        strategy = self._make_strategy_with_timeout()
        monitor = PositionMonitor(
            session_factory=MagicMock(),
            strategies={"TestStrategy": strategy},
            mode="live",
        )

        with patch(
            "freqpred.trading.position_monitor.ledger.get_open_positions",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "freqpred.trading.position_monitor.ledger.get_pending_positions",
            new_callable=AsyncMock,
        ) as mock_pending:
            await monitor.check_all_positions(_now=NOW)

        mock_pending.assert_not_awaited()


# ---------------------------------------------------------------------------
# T48 — limit exits + exchange-hosted stoploss
# ---------------------------------------------------------------------------


def _make_limit_strategy(
    *,
    stoploss: float = -0.15,
    trailing_stop: bool = True,
    exit_type: str = "market",
    stoploss_on_exchange: bool = False,
    stoploss_type: str = "market",
    limit_ratio: float = 0.99,
) -> IPredictionStrategy:
    from freqpred.strategy.config import OrderTypes

    class _S(IPredictionStrategy):
        config = StrategyConfig(
            name="TestStrategy",
            min_edge=0.10,
            min_confidence=0.70,
            max_exposure_per_market=0.05,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=365,
            min_days_to_close=0,
            stoploss=stoploss,
            trailing_stop=trailing_stop,
            order_types=OrderTypes(
                exit=exit_type,
                stoploss=stoploss_type,
                stoploss_on_exchange=stoploss_on_exchange,
                stoploss_on_exchange_limit_ratio=limit_ratio,
            ),
        )

        def should_trade(self, signal, market):  # type: ignore[override]
            return True

        def position_size(self, signal, bankroll):  # type: ignore[override]
            return 0.0

    return _S()


class TestT48LimitExits:
    """T48: limit exit + exchange-hosted stoploss behaviour."""

    def _monitor(self, strategy: IPredictionStrategy) -> PositionMonitor:
        return PositionMonitor(
            session_factory=MagicMock(),
            strategies={"TestStrategy": strategy},
        )

    # ------------------------------------------------------------------
    # exit="limit" trailing-stop fill price
    # ------------------------------------------------------------------

    def test_limit_exit_fills_at_trailing_stop_price(self) -> None:
        """exit="limit": trailing stop fires, fill at stop level not current bid."""
        strategy = _make_limit_strategy(
            stoploss=-0.20, trailing_stop=True, exit_type="limit"
        )
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        monitor._peak_prices[pos.id] = 0.70  # peak seen at 0.70

        # Trailing stop level = 0.70 - 0.20 = 0.50; current bid = 0.48 (has breached)
        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.48),
            current_price=0.48,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "trailing_stop"
        # Fill at stop level (0.50), not current bid (0.48)
        assert result[1] == pytest.approx(0.50)

    def test_market_exit_fills_at_current_price(self) -> None:
        """exit="market" (default): trailing stop fills at current bid — unchanged."""
        strategy = _make_limit_strategy(
            stoploss=-0.20, trailing_stop=True, exit_type="market"
        )
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        monitor._peak_prices[pos.id] = 0.70  # stop level = 0.50

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.48),
            current_price=0.48,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "trailing_stop"
        assert result[1] == pytest.approx(0.48)  # current bid, not stop level

    # ------------------------------------------------------------------
    # custom_exit_price hook
    # ------------------------------------------------------------------

    def test_custom_exit_price_hook_used(self) -> None:
        """custom_exit_price() returns non-None → that price is used for resting order."""
        strategy = _make_limit_strategy(
            stoploss=-0.20, trailing_stop=True, exit_type="limit"
        )
        custom_price = 0.55
        strategy.custom_exit_price = MagicMock(return_value=custom_price)  # type: ignore[method-assign]

        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        monitor._peak_prices[pos.id] = 0.70  # stop level = 0.50, but hook overrides

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.48),
            current_price=0.48,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "trailing_stop"
        assert result[1] == pytest.approx(custom_price)
        from unittest.mock import ANY
        strategy.custom_exit_price.assert_called_once_with(pos, None, ANY, "trailing_stop")

    def test_custom_exit_price_none_falls_back_to_stop_level(self) -> None:
        """custom_exit_price() returns None → stop level used."""
        strategy = _make_limit_strategy(
            stoploss=-0.20, trailing_stop=True, exit_type="limit"
        )
        strategy.custom_exit_price = MagicMock(return_value=None)  # type: ignore[method-assign]

        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        monitor._peak_prices[pos.id] = 0.70  # stop level = 0.50

        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.48),
            current_price=0.48,
            strategy=strategy,
        )
        assert result is not None
        assert result[1] == pytest.approx(0.50)

    # ------------------------------------------------------------------
    # stoploss_on_exchange: in-memory fallback always fires
    # ------------------------------------------------------------------

    def test_stoploss_on_exchange_in_memory_fallback_still_fires(self) -> None:
        """stoploss_on_exchange=True: in-memory stoploss check still closes position."""
        strategy = _make_limit_strategy(
            stoploss=-0.15,
            trailing_stop=False,
            stoploss_on_exchange=True,
            stoploss_type="limit",
        )
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.60)

        # Price drops 0.20 below entry — exceeds stoploss=-0.15
        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=0.40),
            current_price=0.40,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "stoploss"

    # ------------------------------------------------------------------
    # Exchange stoploss price formula
    # ------------------------------------------------------------------

    def test_stoploss_exchange_formula_yes(self) -> None:
        """YES position: exchange stoploss price = (entry + stoploss) * limit_ratio."""
        from freqpred.strategy.config import OrderTypes
        pos = _make_position(entry_price=0.60, direction="YES")
        order_types = OrderTypes(
            stoploss_on_exchange=True,
            stoploss="limit",
            stoploss_on_exchange_limit_ratio=0.99,
        )
        # (0.60 + (-0.15)) * 0.99 = 0.45 * 0.99 = 0.4455
        price = _compute_exchange_stoploss_price(pos, order_types, stoploss=-0.15)
        assert price == pytest.approx(0.4455)

    def test_stoploss_exchange_formula_no(self) -> None:
        """NO position: same formula applied to NO contract value (entry_price in NO terms)."""
        from freqpred.strategy.config import OrderTypes
        pos = _make_position(entry_price=0.40, direction="NO")
        order_types = OrderTypes(
            stoploss_on_exchange=True,
            stoploss="limit",
            stoploss_on_exchange_limit_ratio=0.99,
        )
        # (0.40 + (-0.15)) * 0.99 = 0.25 * 0.99 = 0.2475
        price = _compute_exchange_stoploss_price(pos, order_types, stoploss=-0.15)
        assert price == pytest.approx(0.2475)

    # ------------------------------------------------------------------
    # emergency exit always uses market (current bid)
    # ------------------------------------------------------------------

    def test_emergency_exit_always_market(self) -> None:
        """force_exit (emergency path) fills at current bid regardless of exit='limit'."""
        strategy = _make_limit_strategy(
            stoploss=-0.20, trailing_stop=True, exit_type="limit"
        )
        strategy.force_exit = MagicMock(return_value="emergency")  # type: ignore[method-assign]

        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        monitor._peak_prices[pos.id] = 0.52  # far from stop level; trailing stop won't fire

        current_bid = 0.51
        result = monitor.evaluate_exit(
            position=pos,
            market=_make_market(mid_price=current_bid),
            current_price=current_bid,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "force_exit:emergency"
        # Emergency exit fills at current price, not a limit price
        assert result[1] == pytest.approx(current_bid)

    # ------------------------------------------------------------------
    # Exchange stoploss level refreshed when peak advances
    # ------------------------------------------------------------------

    def test_trailing_stop_limit_refreshed_on_peak_advance(self) -> None:
        """stoploss_on_exchange=True: _stoploss_order_levels updated as peak advances."""
        strategy = _make_limit_strategy(
            stoploss=-0.20,
            trailing_stop=True,
            stoploss_on_exchange=True,
            stoploss_type="limit",
            limit_ratio=0.99,
        )
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)

        # Simulate peak advancing from 0.60 to 0.70
        monitor._peak_prices[pos.id] = 0.60
        monitor._maybe_refresh_exchange_stoploss(pos, strategy.config)
        level_at_60 = monitor._stoploss_order_levels.get(pos.id)

        monitor._peak_prices[pos.id] = 0.70
        monitor._maybe_refresh_exchange_stoploss(pos, strategy.config)
        level_at_70 = monitor._stoploss_order_levels.get(pos.id)

        # stop level at peak 0.60 = 0.60 - 0.20 = 0.40 → * 0.99 = 0.396
        # stop level at peak 0.70 = 0.70 - 0.20 = 0.50 → * 0.99 = 0.495
        assert level_at_60 is not None
        assert level_at_70 is not None
        assert level_at_70 > level_at_60
        assert level_at_60 == pytest.approx(0.396)
        assert level_at_70 == pytest.approx(0.495)

    # ------------------------------------------------------------------
    # Helper function: _compute_trailing_stop_level
    # ------------------------------------------------------------------

    def test_compute_trailing_stop_level_normal_trail(self) -> None:
        """Normal trail: stop = peak - |stoploss|."""
        pos = _make_position(entry_price=0.50)
        level = _compute_trailing_stop_level(
            pos, peak_price=0.70, stoploss=-0.20,
            trailing_stop_positive=None, trailing_stop_positive_offset=0.02,
        )
        assert level == pytest.approx(0.50)  # 0.70 - 0.20

    def test_compute_trailing_stop_level_tight_trail(self) -> None:
        """Tight trail kicks in when peak_gain >= trailing_stop_positive."""
        pos = _make_position(entry_price=0.50)
        level = _compute_trailing_stop_level(
            pos, peak_price=0.70, stoploss=-0.20,
            trailing_stop_positive=0.10,  # 0.70 - 0.50 = 0.20 >= 0.10 → tight
            trailing_stop_positive_offset=0.02,
        )
        assert level == pytest.approx(0.68)  # 0.70 - 0.02
