"""Unit tests for freqpred/strategy/algo_base.py (T49).

Tests cover:
- tick buffer management
- cache invalidation
- _resample() OHLC correctness
- force_exit() logic (all return paths)
- PositionMonitor.on_tick() delegation
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from freqpred.markets.models import Market, Position
from freqpred.strategy.algo_base import IAlgoStrategy, _Tick
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig
from freqpred.trading.position_monitor import PositionMonitor

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 3, 22, 10, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float) -> datetime:
    return _BASE_TS + timedelta(seconds=offset_seconds)


def _make_market(market_id: str = "MKT-1") -> Market:
    return Market(
        id=market_id,
        platform="kalshi",
        question="Will X happen?",
        category="politics",
        status="active",
        result=None,
        close_time=_ts(86400),
        yes_bid=0.45,
        yes_ask=0.55,
        mid_price=0.50,
        last_price=0.50,
        volume_24h=1000.0,
        open_interest=500,
        liquidity=200.0,
        last_fetched_at=_BASE_TS,
        price_updated_at=_BASE_TS,
        metadata_fetched_at=None,
        current_signal_id=None,
        metadata={},
        open_time=None,
    )


def _make_position(market_id: str = "MKT-1") -> Position:
    return Position(
        id=str(uuid.uuid4()),
        market_id=market_id,
        signal_id=str(uuid.uuid4()),
        strategy_name="TestAlgo",
        strategy_version="1.0",
        direction="YES",
        contracts=10,
        entry_price=0.50,
        entry_time=_BASE_TS,
        status="open",
        mode="paper",
        signal_confidence=0.80,
        signal_edge=0.10,
        signal_estimated_prob=0.60,
    )


# ---------------------------------------------------------------------------
# Concrete IAlgoStrategy implementation for tests
# ---------------------------------------------------------------------------


def _make_algo(
    exit_long_value: bool = False,
    timeframe: str = "1min",
    max_candles: int = 500,
    min_candles: int = 2,
) -> IAlgoStrategy:
    """Return a concrete IAlgoStrategy whose populate_exit_trend uses a fixed value."""

    class _TestAlgo(IAlgoStrategy):
        config = StrategyConfig(
            name="TestAlgo",
            min_edge=0.0,
            min_confidence=0.0,
            max_exposure_per_market=100.0,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=3650.0,
            min_days_to_close=0.0,
        )
        _exit_val = exit_long_value

        def populate_exit_trend(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
            df["exit_long"] = self._exit_val
            return df

        def should_trade(self, signal, market):
            return True

        def position_size(self, signal, bankroll):
            return 1.0

    algo = _TestAlgo()
    algo.timeframe = timeframe
    algo.max_candles = max_candles
    algo.min_candles = min_candles
    return algo


def _ingest(algo: IAlgoStrategy, market_id: str, ts: datetime, bid: float = 0.45, ask: float = 0.55) -> None:
    algo.ingest_tick(market_id, yes_bid=bid, yes_ask=ask, ts=ts)


# ---------------------------------------------------------------------------
# Tick buffer tests
# ---------------------------------------------------------------------------


def test_first_tick_creates_buffer() -> None:
    algo = _make_algo()
    _ingest(algo, "MKT-1", _ts(0))
    assert "MKT-1" in algo._ticks
    assert len(algo._ticks["MKT-1"]) == 1


def test_tick_appended() -> None:
    algo = _make_algo()
    _ingest(algo, "MKT-1", _ts(0))
    _ingest(algo, "MKT-1", _ts(5))
    assert len(algo._ticks["MKT-1"]) == 2


def test_invalidates_cache() -> None:
    algo = _make_algo()
    # Seed cache with a non-None sentinel
    algo._candle_cache["MKT-1"] = MagicMock()
    _ingest(algo, "MKT-1", _ts(0))
    assert algo._candle_cache["MKT-1"] is None


def test_mid_computed_correctly() -> None:
    algo = _make_algo()
    _ingest(algo, "MKT-1", _ts(0), bid=0.40, ask=0.60)
    tick = algo._ticks["MKT-1"][0]
    assert tick.mid_price == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# _resample() edge cases
# ---------------------------------------------------------------------------


def test_returns_none_no_ticks() -> None:
    algo = _make_algo()
    result = algo._resample("MKT-1")
    assert result is None


def test_returns_none_single_partial_bucket() -> None:
    """All ticks within the same 1-min window → only 1 bucket → dropped as partial → None."""
    algo = _make_algo(timeframe="1min")
    # Three ticks all within the same minute
    for i in range(3):
        _ingest(algo, "MKT-1", _ts(i * 10))  # 0s, 10s, 20s — same bucket
    result = algo._resample("MKT-1")
    assert result is None


def test_two_complete_buckets() -> None:
    """Ticks spread across 3 minute-buckets → 2 complete candles after dropping partial."""
    algo = _make_algo(timeframe="1min")
    # Bucket 0 (minute 0): tick at 0s
    _ingest(algo, "MKT-1", _ts(0))
    # Bucket 1 (minute 1): tick at 65s
    _ingest(algo, "MKT-1", _ts(65))
    # Bucket 2 (minute 2) — this is the "current" partial bucket: tick at 130s
    _ingest(algo, "MKT-1", _ts(130))

    result = algo._resample("MKT-1")
    assert result is not None
    assert len(result) == 2


def test_ohlc_values_correct() -> None:
    """Verify open/high/low/close from a known tick sequence."""
    algo = _make_algo(timeframe="1min")
    # Bucket 0: prices 0.50, 0.55, 0.48 (mid_prices for these bids/asks)
    _ingest(algo, "MKT-1", _ts(0), bid=0.49, ask=0.51)   # mid=0.50
    _ingest(algo, "MKT-1", _ts(20), bid=0.54, ask=0.56)  # mid=0.55
    _ingest(algo, "MKT-1", _ts(40), bid=0.47, ask=0.49)  # mid=0.48

    # Bucket 1 (minute 1): partial — will be dropped
    _ingest(algo, "MKT-1", _ts(65), bid=0.60, ask=0.62)  # mid=0.61

    result = algo._resample("MKT-1")
    assert result is not None
    assert len(result) == 1
    row = result.iloc[0]
    assert row["open"] == pytest.approx(0.50)
    assert row["high"] == pytest.approx(0.55)
    assert row["low"] == pytest.approx(0.48)
    assert row["close"] == pytest.approx(0.48)


def test_volume_is_tick_count() -> None:
    algo = _make_algo(timeframe="1min")
    # Bucket 0: 3 ticks
    for i in range(3):
        _ingest(algo, "MKT-1", _ts(i * 10))
    # Bucket 1: 1 tick (partial — dropped)
    _ingest(algo, "MKT-1", _ts(65))

    result = algo._resample("MKT-1")
    assert result is not None
    assert result.iloc[0]["volume"] == 3


def test_spread_column() -> None:
    """spread == yes_ask − yes_bid of last tick in bucket."""
    algo = _make_algo(timeframe="1min")
    _ingest(algo, "MKT-1", _ts(0), bid=0.40, ask=0.60)
    _ingest(algo, "MKT-1", _ts(30), bid=0.44, ask=0.54)   # last tick in bucket 0

    # Bucket 1 (partial — dropped)
    _ingest(algo, "MKT-1", _ts(65), bid=0.45, ask=0.55)

    result = algo._resample("MKT-1")
    assert result is not None
    row = result.iloc[0]
    assert row["yes_bid"] == pytest.approx(0.44)
    assert row["yes_ask"] == pytest.approx(0.54)
    assert row["spread"] == pytest.approx(0.10)


def test_max_candles_trims_output() -> None:
    """max_candles=3 with 6 complete candles (7 buckets total) → 3 rows returned."""
    algo = _make_algo(timeframe="1min", max_candles=3)
    # 7 ticks, one per minute — 7 buckets, drop last → 6 complete candles
    for i in range(7):
        _ingest(algo, "MKT-1", _ts(i * 60))

    result = algo._resample("MKT-1")
    assert result is not None
    assert len(result) == 3


def test_max_candles_trims_raw_buffer() -> None:
    """After trim, raw tick buffer covers only the retained candle range."""
    algo = _make_algo(timeframe="1min", max_candles=3)
    # 7 buckets (drop last → 6 complete), then trim to 3 most recent
    for i in range(7):
        _ingest(algo, "MKT-1", _ts(i * 60))

    result = algo._resample("MKT-1")
    assert result is not None

    # The retained candles start at minute 3 (index 3 out of 0..6 with 7 ticks)
    # Ticks 0..2 fall in candles 0..2 (dropped), ticks 3..6 in candles 3..6.
    # Last bucket (bucket 6, tick at 360s) is dropped as partial.
    # Retained: candles 3, 4, 5 → ticks at 180s, 240s, 300s.
    # Tick at 360s is in the dropped partial bucket — it's included in the cutoff range.
    retained_ts = {t.ts for t in algo._ticks["MKT-1"]}
    cutoff = result.index[0].to_pydatetime()
    for tick in algo._ticks["MKT-1"]:
        assert tick.ts >= cutoff


# ---------------------------------------------------------------------------
# force_exit() tests
# ---------------------------------------------------------------------------


def test_none_empty_buffer() -> None:
    algo = _make_algo(exit_long_value=True)
    market = _make_market()
    position = _make_position()
    assert algo.force_exit(position, market) is None


def test_none_one_candle() -> None:
    """1 complete candle + partial → force_exit returns None (< default min_candles=2)."""
    algo = _make_algo(exit_long_value=True)
    market = _make_market()
    position = _make_position()
    # Bucket 0: 1 tick
    _ingest(algo, market.id, _ts(0))
    # Bucket 1 (partial): 1 tick at 65s
    _ingest(algo, market.id, _ts(65))
    assert algo.force_exit(position, market) is None


def test_min_candles_no_premature_exit_and_fires_at_2x() -> None:
    """Exhaustive min_candles guard test.

    Uses a strategy whose exit condition only fires once 2×min_candles complete
    candles are present.  Verifies:
    - None on every tick before min_candles are reached (framework suppression)
    - None on every tick from min_candles to 2×min_candles-1 (condition not met)
    - "algo_exit" exactly when 2×min_candles complete candles are available
    """
    min_candles = 5

    class _LateExitAlgo(IAlgoStrategy):
        """Exits only once the DataFrame has ≥ 2×min_candles rows."""

        config = StrategyConfig(
            name="LateExit",
            min_edge=0.0,
            min_confidence=0.0,
            max_exposure_per_market=100.0,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=3650.0,
            min_days_to_close=0.0,
        )

        def populate_exit_trend(self, df, metadata):
            df["exit_long"] = len(df) >= min_candles * 2
            return df

        def should_trade(self, s, m):
            return True

        def position_size(self, s, b):
            return 1.0

    algo = _LateExitAlgo()
    algo.min_candles = min_candles
    market = _make_market()
    position = _make_position()
    mkt = market.id

    # Each tick is 65s apart so every tick lands in a new 1-min bucket.
    # N ticks → N-1 complete candles (last bucket dropped as partial).

    # Phase 1: before min_candles — framework suppresses regardless of condition.
    for i in range(min_candles):
        _ingest(algo, mkt, _ts(i * 65))
        algo._candle_cache[mkt] = None  # invalidate so each call recomputes
        assert algo.force_exit(position, market) is None, (
            f"expected None at tick {i} ({i} complete candles, min_candles={min_candles})"
        )

    # Phase 2: min_candles to 2×min_candles-1 complete candles — condition not yet met.
    for i in range(min_candles, min_candles * 2):
        _ingest(algo, mkt, _ts(i * 65))
        algo._candle_cache[mkt] = None
        assert algo.force_exit(position, market) is None, (
            f"expected None at tick {i} ({i} complete candles, exit fires at {min_candles * 2})"
        )

    # Phase 3: exactly 2×min_candles complete candles → condition met → fires.
    _ingest(algo, mkt, _ts(min_candles * 2 * 65))
    algo._candle_cache[mkt] = None
    assert algo.force_exit(position, market) == "algo_exit"


def test_none_exit_long_false() -> None:
    algo = _make_algo(exit_long_value=False)
    market = _make_market()
    position = _make_position()
    # 3 buckets → 2 complete candles
    _ingest(algo, market.id, _ts(0))
    _ingest(algo, market.id, _ts(65))
    _ingest(algo, market.id, _ts(130))
    assert algo.force_exit(position, market) is None


def test_algo_exit_when_true() -> None:
    """exit_long=True with 2+ complete candles → force_exit returns 'algo_exit'."""
    algo = _make_algo(exit_long_value=True)
    market = _make_market()
    position = _make_position()
    _ingest(algo, market.id, _ts(0))
    _ingest(algo, market.id, _ts(65))
    _ingest(algo, market.id, _ts(130))
    assert algo.force_exit(position, market) == "algo_exit"


def test_none_missing_exit_long_column() -> None:
    """populate_exit_trend that omits exit_long → None + warning logged."""

    class _NoColumnAlgo(IAlgoStrategy):
        config = StrategyConfig(
            name="NoCol",
            min_edge=0.0,
            min_confidence=0.0,
            max_exposure_per_market=100.0,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=3650.0,
            min_days_to_close=0.0,
        )

        def populate_exit_trend(self, df, metadata):
            # deliberately omit exit_long
            return df

        def should_trade(self, s, m):
            return True

        def position_size(self, s, b):
            return 1.0

    algo = _NoColumnAlgo()
    market = _make_market()
    position = _make_position()
    _ingest(algo, market.id, _ts(0))
    _ingest(algo, market.id, _ts(65))
    _ingest(algo, market.id, _ts(130))

    with patch("freqpred.strategy.algo_base.logger") as mock_log:
        result = algo.force_exit(position, market)
    assert result is None
    mock_log.warning.assert_called_once()
    assert "missing_exit_long_column" in mock_log.warning.call_args[0][0]


def test_none_hook_raises() -> None:
    """populate_exit_trend that raises → None, exception is swallowed."""

    class _RaisingAlgo(IAlgoStrategy):
        config = StrategyConfig(
            name="Raiser",
            min_edge=0.0,
            min_confidence=0.0,
            max_exposure_per_market=100.0,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=3650.0,
            min_days_to_close=0.0,
        )

        def populate_exit_trend(self, df, metadata):
            raise RuntimeError("boom")

        def should_trade(self, s, m):
            return True

        def position_size(self, s, b):
            return 1.0

    algo = _RaisingAlgo()
    market = _make_market()
    position = _make_position()
    _ingest(algo, market.id, _ts(0))
    _ingest(algo, market.id, _ts(65))
    _ingest(algo, market.id, _ts(130))

    with patch("freqpred.strategy.algo_base.logger") as mock_log:
        result = algo.force_exit(position, market)
    assert result is None
    mock_log.warning.assert_called_once()
    assert "populate_exit_trend_error" in mock_log.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


def test_cache_reused() -> None:
    """force_exit × 2 without new tick → populate_exit_trend called only once."""
    call_count = 0

    class _CountingAlgo(IAlgoStrategy):
        config = StrategyConfig(
            name="Counter",
            min_edge=0.0,
            min_confidence=0.0,
            max_exposure_per_market=100.0,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=3650.0,
            min_days_to_close=0.0,
        )

        def populate_exit_trend(self, df, metadata):
            nonlocal call_count
            call_count += 1
            df["exit_long"] = True
            return df

        def should_trade(self, s, m):
            return True

        def position_size(self, s, b):
            return 1.0

    algo = _CountingAlgo()
    market = _make_market()
    position = _make_position()
    _ingest(algo, market.id, _ts(0))
    _ingest(algo, market.id, _ts(65))
    _ingest(algo, market.id, _ts(130))

    algo.force_exit(position, market)
    algo.force_exit(position, market)

    assert call_count == 1


def test_cache_invalidated_after_tick() -> None:
    """force_exit, ingest_tick, force_exit → populate_exit_trend called twice."""
    call_count = 0

    class _CountingAlgo(IAlgoStrategy):
        config = StrategyConfig(
            name="Counter2",
            min_edge=0.0,
            min_confidence=0.0,
            max_exposure_per_market=100.0,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=3650.0,
            min_days_to_close=0.0,
        )

        def populate_exit_trend(self, df, metadata):
            nonlocal call_count
            call_count += 1
            df["exit_long"] = True
            return df

        def should_trade(self, s, m):
            return True

        def position_size(self, s, b):
            return 1.0

    algo = _CountingAlgo()
    market = _make_market()
    position = _make_position()
    _ingest(algo, market.id, _ts(0))
    _ingest(algo, market.id, _ts(65))
    _ingest(algo, market.id, _ts(130))

    algo.force_exit(position, market)
    # New tick — invalidates cache
    _ingest(algo, market.id, _ts(135))
    algo.force_exit(position, market)

    assert call_count == 2


# ---------------------------------------------------------------------------
# PositionMonitor.on_tick() tests
# ---------------------------------------------------------------------------


def _make_monitor(strategies: dict) -> PositionMonitor:
    return PositionMonitor(
        session_factory=MagicMock(),
        strategies=strategies,
        mode="paper",
    )


def test_on_tick_calls_ingest_for_algo_strategy() -> None:
    algo = _make_algo()
    algo.ingest_tick = MagicMock()  # type: ignore[method-assign]
    monitor = _make_monitor({"algo": algo})

    monitor.on_tick("MKT-1", yes_bid=0.45, yes_ask=0.55, ts=_BASE_TS)

    algo.ingest_tick.assert_called_once_with("MKT-1", 0.45, 0.55, _BASE_TS)


def test_on_tick_noop_for_plain_strategy() -> None:
    """Plain IPredictionStrategy in _strategies → no error raised."""

    class _PlainStrategy(IPredictionStrategy):
        config = StrategyConfig(
            name="Plain",
            min_edge=0.0,
            min_confidence=0.0,
            max_exposure_per_market=100.0,
            kelly_fraction=0.25,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=3650.0,
            min_days_to_close=0.0,
        )

        def should_trade(self, s, m):
            return False

        def position_size(self, s, b):
            return 0.0

    plain = _PlainStrategy()
    monitor = _make_monitor({"plain": plain})
    # Should not raise
    monitor.on_tick("MKT-1", yes_bid=0.45, yes_ask=0.55, ts=_BASE_TS)


def test_on_tick_calls_all_algo_strategies() -> None:
    """Two IAlgoStrategy instances → both get ingest_tick."""
    algo1 = _make_algo()
    algo2 = _make_algo()
    algo1.ingest_tick = MagicMock()  # type: ignore[method-assign]
    algo2.ingest_tick = MagicMock()  # type: ignore[method-assign]
    monitor = _make_monitor({"a1": algo1, "a2": algo2})

    monitor.on_tick("MKT-1", yes_bid=0.40, yes_ask=0.60, ts=_BASE_TS)

    algo1.ingest_tick.assert_called_once_with("MKT-1", 0.40, 0.60, _BASE_TS)
    algo2.ingest_tick.assert_called_once_with("MKT-1", 0.40, 0.60, _BASE_TS)


def test_cannot_instantiate_without_populate_exit_trend() -> None:
    """Direct instantiation of IAlgoStrategy (without implement) → TypeError."""
    with pytest.raises(TypeError):
        IAlgoStrategy()  # type: ignore[abstract]
