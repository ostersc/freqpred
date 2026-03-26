"""AlgoExampleStrategy — prediction-market-native exit demo for IAlgoStrategy (T49).

Demonstrates how to subclass ``IAlgoStrategy`` to drive exits from WebSocket
tick data using prediction-market-appropriate signals rather than equity TA.

Exit logic (last complete candle) — two PM-native signals:

  1. **Asymmetric ceiling** (structural): price above ``_PRICE_CEILING`` means
     remaining upside is capped while downside remains fully binary.  Exit to
     shed that tail risk.

  2. **Safe-zone displacement + choppiness** (thesis-aware): the *safe zone*
     spans from the entry price to the signal's estimated probability (plus a
     margin on each side).  While price stays inside this zone, the market is
     consistent with our thesis — hold and let the LLM re-evaluate on its
     schedule.  Once price moves *outside* the safe zone AND the market is
     choppy (high intra-candle range), exit at the top of the recent range:

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
     At extreme prices (0.25, 0.75) the available range in one direction is
     small, so a modest absolute move consumes a large fraction of it —
     exactly when you should care about choppiness.

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

# Average (high - low) / close over the lookback window above this = choppy.
# 0.05 ≈ 2.5¢ range per candle on a 50¢ contract, or 4¢ on an 80¢ contract.
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

    def populate_indicators(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Choppiness (normalized candle range) and rolling high for range-top timing."""
        df["range_pct"] = (df["high"] - df["low"]) / df["close"].clip(lower=0.01)
        df["choppiness"] = df["range_pct"].rolling(_LOOKBACK, min_periods=1).mean()
        df["rolling_high"] = df["close"].rolling(_LOOKBACK, min_periods=1).max()
        return df

    def populate_exit_trend(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Exit on asymmetric ceiling or thesis-displacement + choppiness.

        Signal 1 — Asymmetric ceiling:
          Price above _PRICE_CEILING → limited upside, full binary downside.

        Signal 2 — Safe-zone displacement + choppiness:
          Safe zone = [min(entry, p_est) - min_edge, max(entry, p_est) + min_edge].
          Outside this zone, the market has moved beyond our thesis range.
          If it's also choppy (high candle range) AND price is near the top
          of the recent range, exit:
            - Below safe zone: minimize loss at best available price.
            - Above safe zone: lock in unexplained profit before reversal.
        """
        entry = metadata["entry_price"]
        p_est = metadata["p_est"]

        # Signal 1: structural ceiling
        near_ceiling = df["close"] > _PRICE_CEILING

        # Signal 2: thesis-aware displacement
        safe_low = min(entry, p_est) - self.config.min_edge
        safe_high = max(entry, p_est) + self.config.min_edge
        outside_safe = (df["close"] < safe_low) | (df["close"] > safe_high)

        choppy = df["choppiness"] > _CHOPPINESS_THRESHOLD
        near_range_top = df["close"] >= df["rolling_high"] * 0.98

        displacement_exit = outside_safe & choppy & near_range_top

        df["exit_long"] = (near_ceiling | displacement_exit).fillna(False)

        # Debug: log last candle's key values for exit decision.
        last = df.iloc[-1]
        logger.debug(
            "algo_exit_eval",
            market_id=metadata["market_id"],
            close=round(last["close"], 4),
            entry=round(entry, 4),
            p_est=round(p_est, 4),
            safe_low=round(safe_low, 4),
            safe_high=round(safe_high, 4),
            choppiness=round(last["choppiness"], 4),
            chop_thresh=_CHOPPINESS_THRESHOLD,
            rolling_high=round(last["rolling_high"], 4),
            near_ceiling=bool(near_ceiling.iloc[-1]),
            outside_safe=bool(outside_safe.iloc[-1]),
            choppy=bool(choppy.iloc[-1]),
            near_top=bool(near_range_top.iloc[-1]),
            exit=bool(df["exit_long"].iloc[-1]),
        )
        return df

    def should_trade(self, signal: "Signal", market: "Market") -> bool:
        return (
            signal.edge >= self.config.min_edge
            and signal.confidence >= self.config.min_confidence
        )
