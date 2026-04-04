"""PoliticsEdgeStrategy: US politics markets with PM-native exits.

Uses 5-min candles (politics markets move slowly) with the same
thesis-aware exit logic as AlgoExampleStrategy:

  1. **Safe-zone displacement + choppiness** — outside thesis range + choppy
     + near range top → exit at best available price.  Safe-zone margin is
     ``config.min_edge`` (0.15 for this strategy).
  2. **Trailing stop** — framework-level (disabled by default for politics;
     these markets are sticky and reversals are often noise).
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

# Politics markets move slowly — 5-min candles give meaningful structure
# without excessive noise from thin order books.
_TIMEFRAME = "5min"

# Rolling window (in candles) for choppiness and range-top detection.
# 10 × 5min = 50 minutes of lookback.
_LOOKBACK = 10

# Average (high - low) / close over the lookback window above this = choppy.
# Slightly higher than the 1-min default (0.05) because 5-min candles
# naturally have wider ranges.
_CHOPPINESS_THRESHOLD = 0.06

# Require at least this many complete 5-min candles for indicators to settle.
# 25 × 5min = ~2 hours of data before any TA exit fires.
_MIN_CANDLES = 25


class PoliticsEdgeStrategy(IAlgoStrategy):
    """US politics markets with conservative Kelly sizing and PM-native exits.

    LLM signal controls entry (min_edge=0.15, min_confidence=0.60).

    Exit priority (per framework rules):
      1–2. Hard stoploss / trailing stop  (config below)
      3.   force_exit — PM-native: thesis-aware displacement/choppiness (this class)
      4–5. custom_exit / should_exit  (inherited no-ops)
    """

    timeframe: str = _TIMEFRAME
    max_candles: int = 500
    min_candles: int = _MIN_CANDLES

    config = StrategyConfig(
        name="PoliticsEdgeStrategy",
        min_edge=0.15,
        min_confidence=0.60,
        max_exposure_per_market=0.20,
        kelly_fraction=0.25,
        categories=["Politics", "Elections", "Mentions"],
        min_volume_24h=1000.0,
        max_days_to_close=7,
        min_days_to_close=0.25,
        stoploss=-0.30,
        stoploss_cooldown_hours=48.0,
        trailing_stop=False,
        trailing_stop_positive=None,
        trailing_stop_positive_offset=0.02,
    )

    def __init__(self) -> None:
        super().__init__()

    def populate_indicators(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Choppiness (normalized candle range) and rolling high for range-top timing."""
        df["range_pct"] = (df["high"] - df["low"]) / df["close"].clip(lower=0.01)
        df["choppiness"] = df["range_pct"].rolling(_LOOKBACK, min_periods=1).mean()
        df["rolling_high"] = df["close"].rolling(_LOOKBACK, min_periods=1).max()
        return df

    def populate_exit_trend(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Exit on thesis-displacement + choppiness.

        Signal — Safe-zone displacement + choppiness:
          Safe zone = [min(entry, p_est) - min_edge, max(entry, p_est) + min_edge].
          Outside + choppy + near range top → exit at best available price.
        """
        entry = metadata["entry_price"]
        p_est = metadata["p_est"]

        # Signal: thesis-aware displacement
        safe_low = min(entry, p_est) - self.config.min_edge
        safe_high = max(entry, p_est) + self.config.min_edge
        outside_safe = (df["close"] < safe_low) | (df["close"] > safe_high)

        choppy = df["choppiness"] > _CHOPPINESS_THRESHOLD
        near_range_top = df["close"] >= df["rolling_high"] * 0.98

        displacement_exit = outside_safe & choppy & near_range_top

        df["exit_long"] = (displacement_exit).fillna(False)

        # Debug: log last candle's key values for exit decision.
        last = df.iloc[-1]
        logger.debug(
            "politics_exit_eval",
            market_id=metadata["market_id"],
            close=round(last["close"], 4),
            entry=round(entry, 4),
            p_est=round(p_est, 4),
            safe_low=round(safe_low, 4),
            safe_high=round(safe_high, 4),
            choppiness=round(last["choppiness"], 4),
            chop_thresh=_CHOPPINESS_THRESHOLD,
            rolling_high=round(last["rolling_high"], 4),
            outside_safe=bool(outside_safe.iloc[-1]),
            choppy=bool(choppy.iloc[-1]),
            near_top=bool(near_range_top.iloc[-1]),
            exit=bool(df["exit_long"].iloc[-1]),
        )
        return df

    def is_market_interesting(self, market: "Market") -> bool:
        return "Trump" in market.question and super().is_market_interesting(market)

    def should_trade(self, signal: "Signal", market: "Market") -> bool:
        return (
            signal.edge >= self.config.min_edge
            and signal.confidence >= self.config.min_confidence
        )
