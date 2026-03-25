"""AlgoExampleStrategy — EMA crossover + RSI exit demo for IAlgoStrategy (T49).

Demonstrates how to subclass ``IAlgoStrategy`` to drive exits from WebSocket
tick data using ``pandas_ta`` — the same TA library used in modern freqtrade
strategies.

Exit logic (last complete candle):
  - **RSI(14) > 70** (overbought): take profit
  - **EMA9 crosses below EMA21** (bearish crossover): trend reversal exit

Both conditions are OR'd — either one fires the exit.

Entries are still LLM-signal-driven (inherited from IPredictionStrategy).
This class only adds the algo exit layer on top.

Usage in config.yaml::

    strategy: AlgoExampleStrategy
    strategy_path: freqpred/strategy/defaults/algo_example.py
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from freqpred.strategy.algo_base import IAlgoStrategy
from freqpred.strategy.config import StrategyConfig

if TYPE_CHECKING:
    import pandas as pd

    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal

_RSI_PERIOD = 14
_EMA_SHORT = 9
_EMA_LONG = 21
_RSI_OVERBOUGHT = 70.0

# EMA21 is the slowest indicator — require at least this many complete candles
# before the strategy will act.  Exits during the warm-up period are suppressed
# by IAlgoStrategy.force_exit() when len(df) < min_candles.
_MIN_CANDLES = _EMA_LONG + 1  # 22


class AlgoExampleStrategy(IAlgoStrategy):
    """EMA crossover + RSI exit strategy using pandas_ta indicators."""

    timeframe: str = "1min"
    max_candles: int = 500
    min_candles: int = _MIN_CANDLES  # suppress exits until EMA21 has warmed up

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
        """Add RSI(14), EMA(9), and EMA(21) via pandas_ta."""
        import pandas_ta as ta  # noqa: PLC0415

        df["rsi"] = ta.rsi(df["close"], length=_RSI_PERIOD)
        df["ema_short"] = ta.ema(df["close"], length=_EMA_SHORT)
        df["ema_long"] = ta.ema(df["close"], length=_EMA_LONG)
        return df

    def populate_exit_trend(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Exit when RSI overbought OR bearish EMA crossover fires.

        By the time this is called, ``min_candles`` is guaranteed to have been
        met by ``IAlgoStrategy.force_exit()``, so both indicators are warmed up.
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

