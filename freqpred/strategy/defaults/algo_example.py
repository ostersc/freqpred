"""AlgoExampleStrategy — prediction-market-native exit demo for IAlgoStrategy (T49).

Demonstrates how to subclass ``IAlgoStrategy`` to drive exits from WebSocket
tick data using prediction-market-appropriate signals rather than equity TA.

Exit logic (last complete candle) — two PM-native signals:

  1. **Asymmetric ceiling** (structural): price above ``_PRICE_CEILING`` means
     remaining upside is capped while downside remains fully binary.  Exit to
     shed that tail risk.

  2. **Safe-zone displacement + choppiness on a liquid book** (thesis-aware): the *safe zone*
     spans from the entry price to the signal's estimated probability (plus a
     margin on each side).  While price stays inside this zone, the market is
     consistent with our thesis — hold and let the LLM re-evaluate on its
     schedule.  Once price moves *outside* the safe zone AND the market is
     choppy (high intra-candle range) AND the book is tight enough to be
     quoting a real price, exit at the top of the recent range:

     - **Below the safe zone**: the market is pricing in information we missed.
       Exit near the range top to minimize loss before it gets worse.
     - **Above the safe zone**: profit beyond our thesis.  If the market is
       choppy up there, take the win before it reverses.  If it's trending
       smoothly, the trailing stop (framework-level) provides the safety net.

     The safe-zone margin is ``config.min_edge`` — the same threshold used
     to decide whether to enter.  If 10¢ of edge was worth trading, 10¢ of
     movement beyond your thesis is where you should start worrying:
       ``safe_low  = min(entry, p_est) - min_edge``
       ``safe_high = max(entry, p_est) + min_edge``
     Prices are bounded to [0, 1] so candle range is already in dollar terms —
     no normalization needed.

     The liquidity term uses ``config.effective_max_spread`` (``max_spread``,
     else ``min_edge / 2``) — a book too wide to open a position on is too wide
     to read a thesis-invalidation out of.  It is applied per-candle, so one
     hollow candle breaks the 3-candle run rather than being averaged away.

  3. **Trailing stop** (confirmation): handled by the framework
     (``config.trailing_stop``), not implemented here.

Position-aware metadata:
  ``IAlgoStrategy.force_exit()`` passes ``entry_price`` and ``p_est``
  (direction-corrected) through the metadata dict.  The OHLC DataFrame is
  already direction-corrected, so ``close`` and ``entry_price`` are in the
  same frame of reference.

Entries are still LLM-signal-driven (inherited from IPredictionStrategy).
This class only adds the algo exit layer on top.

Usage in config.yaml::

    strategy: AlgoExampleStrategy
    strategy_path: freqpred/strategy/defaults/algo_example.py
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from freqpred.strategy.algo_base import IAlgoStrategy
from freqpred.strategy.config import StrategyConfig

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    import pandas as pd

    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal

# Price ceiling above which asymmetric risk logic activates.
# At 0.85 the max remaining upside is 15¢ but full binary downside remains.
_PRICE_CEILING = 0.85

# Rolling window (in candles) for choppiness measurement and range-top detection.
_LOOKBACK = 10

# Average absolute (high - low) in dollar terms over the lookback window above this = choppy.
# Prediction market prices are bounded to [0, 1] so the candle range is already in dollar
# terms — no need to normalize by close.  0.05 = 5¢ average range per 1-min candle.
_CHOPPINESS_THRESHOLD = 0.05

# Require at least this many complete candles for indicators to settle.
_MIN_CANDLES = 25


class AlgoExampleStrategy(IAlgoStrategy):
    """PM-native exits: asymmetric ceiling + thesis-aware displacement/choppiness."""

    timeframe: str = "1min"
    max_candles: int = 500
    min_candles: int = _MIN_CANDLES

    config = StrategyConfig(
        name="AlgoExampleStrategy",
        min_edge=0.10,
        min_confidence=0.60,
        max_exposure_per_market=50.0,
        kelly_fraction=0.25,
        categories=[],
        min_volume_24h=100.0,
        max_days_to_close=30.0,
        min_days_to_close=0.5,
    )

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Per-candle oscillation: wide range but small body (price moved but reversed).

        range_ = high - low (total swing)
        body   = |close - open| (net directional move)
        A candle is choppy when range_ exceeds the threshold AND at least half the
        range was reversed (body < range_ / 2).  Trending candles do not meet this.
        """
        df["range_"] = df["high"] - df["low"]
        df["body"] = (df["close"] - df["open"]).abs()
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Exit on asymmetric ceiling or thesis-displacement + choppiness for 3 consecutive candles.

        Signal 1 — Asymmetric ceiling:
          Price above _PRICE_CEILING → limited upside, full binary downside.
          Fires immediately (no persistence required).

        Signal 2 — Safe-zone displacement + choppiness, on a liquid book:
          Safe zone = [min(entry, p_est) - min_edge, max(entry, p_est) + min_edge].
          Each candle must independently exceed the choppiness threshold *and* be
          quoted within ``config.effective_max_spread``.
          The 3-candle persistence window (not a rolling average) provides smoothing.

          The liquidity term separates the two things a wide intra-candle range
          can mean.  On a tight book a 5c swing is real two-sided repricing —
          the consensus moving against the thesis, which is what this exit is
          for.  On a hollow book the same swing is one lot crossing the gap
          between a stale bid and a stale ask: the price never traded and nobody
          disagreed with us.  Acting on the second case sells the bottom of a
          liquidity hole.  The threshold is the same tightness the entry gate
          demands, so a book too wide to open on is too wide to read a
          thesis-invalidation out of.

          The gate is deliberately *not* applied to Signal 1: the ceiling is a
          structural statement about the price level, not a reaction to
          fluctuation, so choppiness has no bearing on it.  The framework-level
          guard in ``PositionMonitor._execute_exit`` still defers the resulting
          order while the book is too wide to cross, so a ceiling exit cannot
          dump into a hollow book either.
        """
        entry = metadata["entry_price"]
        p_est = metadata["p_est"]

        # Signal 1: structural ceiling — immediate exit, no persistence needed
        near_ceiling = df["close"] > _PRICE_CEILING

        # Signal 2: thesis-aware displacement for 3 consecutive candles
        safe_low = min(entry, p_est) - self.config.min_edge
        safe_high = max(entry, p_est) + self.config.min_edge
        outside_safe = (df["close"] < safe_low) | (df["close"] > safe_high)

        # Liquidity: spread is direction-invariant (ask - bid is the same width
        # in YES or NO terms), so this needs no inversion for NO positions.
        max_spread = self.config.effective_max_spread
        liquid = df["spread"] <= max_spread

        # Choppiness: wide range but small body (oscillation, not trend).
        # Sustained for 3 consecutive candles (3 min) before firing.
        per_candle_choppy = (df["range_"] > _CHOPPINESS_THRESHOLD) & (df["body"] < df["range_"] * 0.5)
        per_candle_actionable = per_candle_choppy & liquid
        sustained_choppy = per_candle_actionable.rolling(3, min_periods=3).min() == 1

        displacement_exit = outside_safe & sustained_choppy

        df["exit_long"] = (near_ceiling | displacement_exit).fillna(False)

        last = df.iloc[-1]
        logger.debug(
            "algo_exit_eval",
            market_id=metadata["market_id"],
            close=round(last["close"], 4),
            entry=round(entry, 4),
            p_est=round(p_est, 4),
            safe_low=round(safe_low, 4),
            safe_high=round(safe_high, 4),
            chop_thresh=_CHOPPINESS_THRESHOLD,
            spread=round(last["spread"], 4),
            max_spread=max_spread,
            near_ceiling=bool(near_ceiling.iloc[-1]),
            outside_safe=bool(outside_safe.iloc[-1]),
            range_=round(last["range_"], 4),
            body=round(last["body"], 4),
            per_candle_choppy=bool(per_candle_choppy.iloc[-1]),
            liquid=bool(liquid.iloc[-1]),
            sustained_choppy=bool(sustained_choppy.iloc[-1]),
            exit=bool(df["exit_long"].iloc[-1]),
        )
        return df

    def should_trade(self, signal: Signal, market: Market) -> bool:
        return super().should_trade(signal, market)
