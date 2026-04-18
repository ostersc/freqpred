"""StrategyConfig dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    name: str
    min_edge: float
    min_confidence: float
    max_exposure_per_market: float
    kelly_fraction: float
    categories: list[str]
    min_volume_24h: float
    max_days_to_close: float
    min_days_to_close: float

    # Exit management — all thresholds are absolute 0-1 price-scale dollars.
    # e.g. stoploss=-0.10 means "exit if price drops 10 cents from entry".
    stoploss: float = -0.15
    trailing_stop: bool = False
    trailing_stop_positive: float | None = None      # switch to tight trail once up this many cents
    trailing_stop_positive_offset: float = 0.02      # tight trail distance (cents) below peak

    # Price range filter: skip markets the market has already decided.
    # Markets trading below min_mid_price or above max_mid_price are excluded
    # from ingestion and signal generation. None = no filter on that bound.
    min_mid_price: float | None = 0.05
    max_mid_price: float | None = 0.95

    # Liquidity filter: reject entry if yes_ask - yes_bid exceeds this threshold.
    # None = auto-compute as min_edge / 2 (spread must consume < half your edge).
    max_spread: float | None = None

    # Re-entry guards after a stoploss or trailing_stop exit.
    # block_reentry_after_stoploss takes precedence: if True, the market is
    # permanently blocked from re-entry once any stoploss/trailing_stop has fired,
    # regardless of stoploss_cooldown_hours.
    # If False and stoploss_cooldown_hours > 0, re-entry is blocked for that many
    # hours after the most recent stoploss/trailing_stop exit on this market.
    block_reentry_after_stoploss: bool = False
    stoploss_cooldown_hours: float = 4.0  # set to 0.0 to disable cooldown

    # Assessment-based sizing controls. The Opus judgment model outputs a
    # trust_score, which the framework maps to this multiplier range.
    assessment_scale_min: float = 0.80
    assessment_scale_max: float = 1.20
    similar_market_min_signals: int = 10
    similar_market_min_trades: int = 5
