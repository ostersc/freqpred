"""IAlgoStrategy — DataFrame-driven exits via WebSocket tick data (T49).

Strategies that want indicator-based exits subclass ``IAlgoStrategy`` instead
of (or in addition to) ``IPredictionStrategy``.  LLM still controls entries;
this base only adds exits via ``populate_exit_trend``.

Usage::

    class MyAlgo(IAlgoStrategy):
        config = StrategyConfig(name="my_algo", ...)
        timeframe = "5min"

        def populate_indicators(self, df, metadata):
            df["ema9"] = df["close"].ewm(span=9).mean()
            return df

        def populate_exit_trend(self, df, metadata):
            df["exit_long"] = df["ema9"] < df["close"].shift(1)
            return df

        def should_trade(self, signal, market): ...
        def position_size(self, signal, bankroll): ...
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING


def _timeframe_seconds(timeframe: str) -> int:
    """Parse a timeframe string (e.g. '5min', '1h') to seconds."""
    if timeframe.endswith("min"):
        return int(timeframe[:-3]) * 60
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 3600
    if timeframe.endswith("s"):
        return int(timeframe[:-1])
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")

import structlog

from freqpred.strategy.base import IPredictionStrategy

if TYPE_CHECKING:
    import pandas as pd

    from freqpred.markets.models import Market, Position

logger = structlog.get_logger(__name__)


@dataclass
class _Tick:
    ts: datetime
    mid_price: float
    yes_bid: float
    yes_ask: float


class IAlgoStrategy(IPredictionStrategy):
    """Base class for indicator-driven exits from WebSocket tick data.

    Subclasses must implement ``populate_exit_trend``; ``populate_indicators``
    has a passthrough default.  ``force_exit`` is implemented here and returns
    ``"algo_exit"`` when the last complete candle has ``exit_long == True``.

    ``pandas`` is imported lazily inside ``_resample()`` — not at module level.
    """

    #: pandas resample rule, e.g. "1min", "5min", "15min"
    timeframe: str = "1min"
    #: rolling window of complete candles retained per market
    max_candles: int = 500
    #: minimum complete candles required before force_exit() will act.
    #: set to the lookback of your slowest indicator so exits are suppressed
    #: during the indicator warm-up period.
    min_candles: int = 2

    def __init__(self) -> None:
        # raw tick buffer: one _Tick per WebSocket tick, per market
        self._ticks: dict[str, list[_Tick]] = {}
        # candle cache keyed by (market_id, direction).
        # YES and NO positions receive direction-corrected candles so they must be
        # cached separately — a NO position's DataFrame has inverted OHLC.
        self._candle_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
        # tracks the UTC epoch (integer seconds) of the current open candle bucket
        # per market; cache is only invalidated when this changes.
        self._current_bucket: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def populate_indicators(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Add indicator columns to the OHLC DataFrame.  Default: passthrough."""
        return df

    @abstractmethod
    def populate_exit_trend(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Add ``exit_long`` (bool) column.  ``exit_long=True`` → trigger exit."""
        ...

    # ------------------------------------------------------------------
    # Tick ingestion
    # ------------------------------------------------------------------

    def ingest_tick(
        self,
        market_id: str,
        yes_bid: float,
        yes_ask: float,
        ts: datetime,
    ) -> None:
        """Append a raw tick to the buffer and invalidate the candle cache."""
        mid_price = (yes_bid + yes_ask) / 2.0
        tick = _Tick(ts=ts, mid_price=mid_price, yes_bid=yes_bid, yes_ask=yes_ask)
        if market_id not in self._ticks:
            self._ticks[market_id] = []
        self._ticks[market_id].append(tick)
        # Only invalidate the candle cache when the tick crosses into a new candle
        # bucket.  Ticks within the same window don't change the complete-candle set
        # (_resample always drops the partial current bucket), so recomputing on
        # every tick would be wasted work.
        bucket = int(ts.timestamp()) // _timeframe_seconds(self.timeframe) * _timeframe_seconds(self.timeframe)
        if self._current_bucket.get(market_id) != bucket:
            self._current_bucket[market_id] = bucket
            self._candle_cache[(market_id, "YES")] = None
            self._candle_cache[(market_id, "NO")] = None

    # ------------------------------------------------------------------
    # force_exit override
    # ------------------------------------------------------------------

    def force_exit(self, position: "Position", market: "Market") -> str | None:
        """Return ``"algo_exit"`` when the last complete candle has exit_long=True.

        Returns None when:
        - buffer is empty
        - fewer than 2 complete candles are available
        - ``exit_long`` column is missing (logs a warning)
        - ``populate_exit_trend`` raises (logs a warning, exception swallowed)
        """
        market_id = market.id
        direction = position.direction
        cache_key = (market_id, direction)
        cached = self._candle_cache.get(cache_key)

        if cached is None:
            # Cache is invalidated — recompute from raw tick buffer.
            df = self._resample(market_id)
            if df is None or len(df) < self.min_candles:
                return None
            # Correct OHLC to the position's perspective so that indicators
            # always see "contract value" — rising = profitable.  For NO
            # positions the contract value is (1 - YES_price), so we invert.
            if direction == "NO":
                df = _invert_ohlc(df)
            # Direction-correct p_est: for NO positions the effective estimate
            # is (1 - p_est), same frame as the inverted OHLC.
            p_est = position.signal_estimated_prob
            if direction == "NO":
                p_est = 1.0 - p_est
            metadata = {
                "market_id": market_id,
                "entry_price": position.entry_price,
                "p_est": p_est,
            }
            try:
                df = self.populate_indicators(df, metadata)
                df = self.populate_exit_trend(df, metadata)
            except Exception:
                logger.warning(
                    "algo_strategy.populate_exit_trend_error",
                    market_id=market_id,
                )
                return None
            if "exit_long" not in df.columns:
                logger.warning(
                    "algo_strategy.missing_exit_long_column",
                    market_id=market_id,
                )
                return None
            self._candle_cache[cache_key] = df
            cached = df

        if bool(cached["exit_long"].iloc[-1]):
            return "algo_exit"
        return None

    # ------------------------------------------------------------------
    # Internal resampling
    # ------------------------------------------------------------------

    def _resample(self, market_id: str) -> "pd.DataFrame | None":
        """Resample the raw tick buffer into a complete-candle OHLC DataFrame.

        - Only complete buckets are returned (the partial current bucket is dropped).
        - Result is trimmed to ``max_candles`` rows.
        - The raw tick buffer is trimmed to match the retained candle range.
        - Returns ``None`` if there are no complete candles.

        ``pandas`` is imported here (deferred — not at module level).
        """
        import pandas as pd  # noqa: PLC0415

        ticks = self._ticks.get(market_id)
        if not ticks:
            return None

        # Build a DataFrame indexed by timestamp.
        index = pd.DatetimeIndex([t.ts for t in ticks])
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")

        df = pd.DataFrame(
            {
                "mid_price": [t.mid_price for t in ticks],
                "yes_bid": [t.yes_bid for t in ticks],
                "yes_ask": [t.yes_ask for t in ticks],
            },
            index=index,
        )

        # Resample mid_price to OHLC + tick count, plus last bid/ask per bucket.
        ohlc = df["mid_price"].resample(self.timeframe).ohlc()
        volume = df["mid_price"].resample(self.timeframe).count().rename("volume")
        yes_bid_last = df["yes_bid"].resample(self.timeframe).last().rename("yes_bid")
        yes_ask_last = df["yes_ask"].resample(self.timeframe).last().rename("yes_ask")

        result = pd.concat([ohlc, volume, yes_bid_last, yes_ask_last], axis=1)
        result["spread"] = result["yes_ask"] - result["yes_bid"]

        # Drop empty buckets (no ticks landed in that window).
        result = result.dropna(subset=["open"])

        # Drop the last bucket — it may be a partial (current) candle.
        if len(result) < 1:
            return None
        result = result.iloc[:-1]

        if len(result) == 0:
            return None

        # Trim to max_candles (keep the most recent).
        if len(result) > self.max_candles:
            result = result.iloc[-self.max_candles :]

        # Trim the raw tick buffer so it covers only the retained candle range.
        cutoff = result.index[0].to_pydatetime()  # tz-aware UTC
        self._ticks[market_id] = [t for t in ticks if _ts_gte(t.ts, cutoff)]

        return result


# ---------------------------------------------------------------------------
# OHLC direction helper
# ---------------------------------------------------------------------------


def _invert_ohlc(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return a copy of *df* with OHLC columns flipped to the NO-contract perspective.

    In a binary market ``no_price = 1 - yes_price``.  The high/low columns
    swap because when YES is at its low, NO is at its high and vice versa::

        no_open  = 1 - yes_open
        no_close = 1 - yes_close
        no_high  = 1 - yes_low
        no_low   = 1 - yes_high

    Non-OHLC columns (volume, spread, yes_bid, yes_ask) are left unchanged.
    """
    result = df.copy()
    result["open"] = 1.0 - df["open"]
    result["close"] = 1.0 - df["close"]
    result["high"] = 1.0 - df["low"]
    result["low"] = 1.0 - df["high"]
    return result


# ---------------------------------------------------------------------------
# Timezone-safe timestamp comparison helper
# ---------------------------------------------------------------------------


def _ts_gte(ts: datetime, cutoff: datetime) -> bool:
    """Return True if ts >= cutoff, handling mixed tz-aware / naive datetimes."""
    if ts.tzinfo is None and cutoff.tzinfo is not None:
        from datetime import timezone
        ts = ts.replace(tzinfo=timezone.utc)
    elif ts.tzinfo is not None and cutoff.tzinfo is None:
        from datetime import timezone
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    return ts >= cutoff
