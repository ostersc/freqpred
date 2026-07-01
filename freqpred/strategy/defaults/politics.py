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

from freqpred.ingestion.fetchers.factbase import FactbasePhraseCache
from freqpred.strategy.algo_base import IAlgoStrategy
from freqpred.strategy.config import OrderTypes, StrategyConfig

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    import pandas as pd

    from freqpred.markets.models import Market

# Politics markets move slowly — 5-min candles give meaningful structure
# without excessive noise from thin order books.
_TIMEFRAME = "5min"

# Intra-candle oscillation (high - low) above this threshold = choppy for that candle.
# Prediction market prices are bounded to [0, 1] so the candle range is already
# in dollar terms — no need to normalize by close (which inflates the metric at
# low prices).  0.05 = 5¢ average range per 5-min candle.
_CHOPPINESS_THRESHOLD = 0.05


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
    min_candles: int = 3

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
        max_edge=0.40,
        min_mid_price=0.10,
        stoploss=-1.0,
        stoploss_cooldown_hours=48.0,
        trailing_stop=False,
        trailing_stop_positive=None,
        trailing_stop_positive_offset=0.02,
        factbase_series_allowlist=["KXTRUMPSAY", "KXTRUMPSAYMONTH", "KXTRUMPSAYNICKNAME", "KXTRUMPSAYTRUMP"],
        # Resting limit entries at estimated_probability - min_edge (0.15 below model estimate).
        # Only fills when market comes to our price — guarantees edge at fill.
        order_types=OrderTypes(entry="limit", exit="limit"),
        limit_order_timeout_hours=2.0,
    )

    def __init__(self, phrase_cache: FactbasePhraseCache | None = None) -> None:
        super().__init__()
        self._phrase_cache = phrase_cache

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Per-candle oscillation: wide range but small body (price moved but reversed).

        range_ = high - low (total swing)
        body   = |close - open| (net directional move)
        A candle is choppy when range_ exceeds the threshold AND at least half the
        range was reversed (body < range_ / 2).  Trending candles — where the close
        is near the high or low — do not meet this criterion.
        """
        range_ = df["high"] - df["low"]
        body = (df["close"] - df["open"]).abs()
        df["range_"] = range_
        df["body"] = body
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Exit when outside thesis range AND choppy for 3 consecutive candles (15 min).

        Safe zone = [min(entry, p_est) - min_edge, max(entry, p_est) + min_edge].
        Each candle must independently exceed the choppiness threshold — no rolling
        average.  The 3-candle persistence window provides the smoothing.
        """
        entry = metadata["entry_price"]
        p_est = metadata["p_est"]

        safe_low = max(0.0, min(entry, p_est) - self.config.min_edge)
        safe_high = min(1.0, max(entry, p_est) + self.config.min_edge)
        outside_safe = (df["close"] < safe_low) | (df["close"] > safe_high)

        # Choppiness: wide range but small body (oscillation, not trend).
        # Sustained for 3 consecutive candles (15 min) before firing.
        per_candle_choppy = (df["range_"] > _CHOPPINESS_THRESHOLD) & (df["body"] < df["range_"] * 0.5)
        sustained_choppy = per_candle_choppy.rolling(3, min_periods=3).min() == 1

        df["exit_long"] = (outside_safe & sustained_choppy).fillna(False)

        last = df.iloc[-1]
        logger.debug(
            "politics_exit_eval",
            market_id=metadata["market_id"],
            close=round(last["close"], 4),
            entry=round(entry, 4),
            p_est=round(p_est, 4),
            safe_low=round(safe_low, 4),
            safe_high=round(safe_high, 4),
            range_=round(last["range_"], 4),
            body=round(last["body"], 4),
            chop_thresh=_CHOPPINESS_THRESHOLD,
            outside_safe=bool(outside_safe.iloc[-1]),
            per_candle_choppy=bool(per_candle_choppy.iloc[-1]),
            sustained_choppy=bool(sustained_choppy.iloc[-1]),
            exit=bool(df["exit_long"].iloc[-1]),
        )
        return df

    def is_market_interesting(self, market: Market) -> bool:
        if not market.series_ticker or market.series_ticker not in self.config.factbase_series_allowlist:
            return False
        if self._phrase_cache is not None and not self._phrase_cache.is_ready(market.id):
            return False
        return super().is_market_interesting(market)

