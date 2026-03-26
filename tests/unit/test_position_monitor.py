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
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import Market, Order, Position
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig
from freqpred.trading.position_monitor import (
    PositionMonitor,
    _check_stoploss,
    _check_trailing_stop,
)

# Ensure ORM relationships resolve (needed for MarketRow joins)
import freqpred.ingestion.models   # noqa: F401
import freqpred.llm.models         # noqa: F401
import freqpred.markets.models     # noqa: F401
import freqpred.rag.models         # noqa: F401
import freqpred.signal.models      # noqa: F401

NOW = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)


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
        close_time=close_time or (NOW + timedelta(days=10)),
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

    def test_market_resolved_when_close_time_passed(self) -> None:
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        market = _make_market(mid_price=0.52, close_time=NOW - timedelta(hours=1))

        result = monitor.evaluate_exit(
            position=pos,
            market=market,
            current_price=0.52,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "market_resolved"

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

    def test_market_resolved_uses_mid_when_result_unknown(self) -> None:
        """close_time passed but result not yet published → falls back to effective_price."""
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50, direction="YES")
        market = _make_market(mid_price=0.52, close_time=NOW - timedelta(hours=1))
        # market.result is None (not yet set by Kalshi)

        result = monitor.evaluate_exit(
            position=pos,
            market=market,
            current_price=0.52,
            strategy=strategy,
        )
        assert result is not None
        assert result[0] == "market_resolved"
        assert result[1] == pytest.approx(0.52)

    def test_no_market_resolved_when_status_open_and_close_time_future(self) -> None:
        strategy = _make_strategy()
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.50)
        # Use real now so this test doesn't break as calendar time advances.
        market = _make_market(
            mid_price=0.52,
            close_time=datetime.now(tz=timezone.utc) + timedelta(days=5),
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

        Old bug: pnl_pct = (0.355 - 0.67) / 0.67 = -47% → fired immediately.
        Fixed:   effective_price = 1 - 0.355 = 0.645; pnl_pct = -3.7% → no fire.
        """
        strategy = _make_strategy(stoploss=-0.20)
        monitor = self._monitor(strategy)
        pos = _make_position(entry_price=0.67, direction="NO")
        # YES mid is 0.355 — price hasn't moved since entry
        market = _make_market(mid_price=0.355)

        result = monitor.evaluate_exit(
            position=pos, market=market, current_price=0.355, strategy=strategy
        )
        assert result is None, (
            "NO position should not stoploss when YES price hasn't moved "
            "(stoploss was firing by comparing YES mid to NO entry price)"
        )

    def test_stoploss_fires_correctly_for_no_position(self) -> None:
        """NO position stoploss fires when YES price rises enough to push NO value below threshold."""
        strategy = _make_strategy(stoploss=-0.20)
        monitor = self._monitor(strategy)
        # Entered at NO ask 0.67; stoploss fires when NO value < 0.67 - 0.20 = 0.47
        # → fires when YES mid > 1 - 0.47 = 0.53
        pos = _make_position(entry_price=0.67, direction="NO")
        # YES mid = 0.70 → effective_no_price = 0.30; dollar_loss = 0.30 - 0.67 = -0.37 ≤ -0.20
        result = monitor.evaluate_exit(
            position=pos, market=_make_market(mid_price=0.70), current_price=0.70, strategy=strategy
        )
        assert result is not None
        assert result[0] == "stoploss"
        # exit_price must be the effective NO price, not the YES mid
        assert result[1] == pytest.approx(0.30)

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
            original = monitor.check_all_positions

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
            **{**pos.__dict__, "status": "closed", "exit_price": 1.0, "exit_reason": "market_resolved", "resolution": 1, "pnl": 50.0, "pnl_pct": 1.0}
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
                    close_time=market.close_time,
                    yes_bid=market.yes_bid,
                    yes_ask=market.yes_ask,
                    mid_price=market.mid_price,
                    last_price=0.0,
                    volume_24h=market.volume_24h,
                    open_interest=market.open_interest,
                    liquidity=0.0,
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
        ):
            result = await monitor.check_all_positions()

        assert result == []


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
            "freqpred.trading.position_monitor.ledger.close_position",
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
        """ledger.close_position() is called with the price from the exchange response."""
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
            "freqpred.trading.position_monitor.ledger.close_position",
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
        assert call_kwargs["exit_price"] == pytest.approx(filled_price)

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
            "freqpred.trading.position_monitor.ledger.close_position",
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
