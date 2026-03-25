"""PoliticsEdgeStrategy: US politics markets with conservative Kelly + TA exits."""
from __future__ import annotations

from typing import TYPE_CHECKING

from freqpred.strategy.algo_base import IAlgoStrategy
from freqpred.strategy.config import StrategyConfig

if TYPE_CHECKING:
    import pandas as pd

    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal

# Politics markets move slowly — 5-min candles give meaningful structure
# without excessive noise from thin order books.
_TIMEFRAME = "5min"

# EMA21 is the slowest indicator; require enough candles for it to warm up.
# 22 × 5min = 110 minutes of data before any TA exit fires.
_EMA_SHORT = 9
_EMA_LONG = 21
_RSI_PERIOD = 14
_MIN_CANDLES = _EMA_LONG + 1  # 22

# Higher RSI threshold than equity markets — politics prices are stickier
# and an RSI of 70 is common even without a real reversal.
_RSI_OVERBOUGHT = 78.0


class PoliticsEdgeStrategy(IAlgoStrategy):
    """US politics markets with conservative Kelly sizing and TA-driven exits.

    LLM signal controls entry (min_edge=0.18, min_confidence=0.70).

    Exit priority (per framework rules):
      1–3. Hard stoploss / trailing stop / minimal ROI  (config below)
      4.   force_exit — TA: RSI overbought OR EMA bearish cross (this class)
      5–6. custom_exit / should_exit  (inherited no-ops)
    """

    timeframe: str = _TIMEFRAME
    max_candles: int = 500
    min_candles: int = _MIN_CANDLES

    config = StrategyConfig(
        name="PoliticsEdgeStrategy",
        min_edge=0.18,
        min_confidence=0.60,
        max_exposure_per_market=0.05,
        kelly_fraction=0.25,
        categories=["politics"],
        min_volume_24h=1000.0,
        max_days_to_close=90,
        min_days_to_close=0.25,
        stoploss=-0.30,
        stoploss_cooldown_hours=48.0,
        minimal_roi={"0": 0.40, "1440": 0.25, "10080": 0.10},
        trailing_stop=False,
        trailing_stop_positive=None,
        trailing_stop_positive_offset=0.02,
    )

    def __init__(self) -> None:
        super().__init__()

    def populate_indicators(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """RSI(14), EMA(9), EMA(21) on 5-min close prices."""
        import pandas_ta as ta  # noqa: PLC0415

        df["rsi"] = ta.rsi(df["close"], length=_RSI_PERIOD)
        df["ema_short"] = ta.ema(df["close"], length=_EMA_SHORT)
        df["ema_long"] = ta.ema(df["close"], length=_EMA_LONG)
        return df

    def populate_exit_trend(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Exit on RSI overbought OR bearish EMA crossover.

        By the time this is called, min_candles (22) is guaranteed by
        IAlgoStrategy.force_exit(), so both indicators are fully warmed up.
        """
        import pandas_ta as ta  # noqa: PLC0415

        rsi_exit = df["rsi"] > _RSI_OVERBOUGHT
        ema_cross_exit = ta.cross(df["ema_short"], df["ema_long"], above=False)

        df["exit_long"] = (rsi_exit | ema_cross_exit).fillna(False)
        return df

    def should_trade(self, signal: "Signal", market: "Market") -> bool:
        return (
            signal.edge >= self.config.min_edge
            and signal.confidence >= self.config.min_confidence
        )

    def position_size(self, signal: "Signal", bankroll: float) -> float:
        kelly = signal.edge / (1 - signal.estimated_probability)
        return bankroll * kelly * self.config.kelly_fraction
