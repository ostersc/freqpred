"""FreshMarketStrategy: politics + tech markets created in the last 4 hours.

Targets newly listed markets before they reach efficient pricing.
Short-dated only (max 7 days), low volume threshold to catch markets early.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig

if TYPE_CHECKING:
    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal

_FRESHNESS_HOURS = 24
_MAX_SPREAD = 0.15  # yes_ask - yes_bid; wider than this = illiquid/untradeable


class FreshMarketStrategy(IPredictionStrategy):
    config = StrategyConfig(
        name="FreshMarketStrategy",
        min_edge=0.15,
        min_confidence=0.68,
        max_exposure_per_market=0.04,
        kelly_fraction=0.20,
        categories=["politics", "technology"],
        min_volume_24h=0.0,
        max_days_to_close=7.0,
        min_days_to_close=0.25,
        stoploss=-0.20,
        minimal_roi={"0": 0.30, "1440": 0.15},
        trailing_stop=True,
        trailing_stop_positive=0.10,
        trailing_stop_positive_offset=0.02,
    )

    def is_market_interesting(self, market: Market) -> bool:
        if market.open_time is None:
            return False
        now = datetime.now(tz=timezone.utc)
        freshness_cutoff = now - timedelta(hours=_FRESHNESS_HOURS)
        spread = market.yes_ask - market.yes_bid
        return (
            super().is_market_interesting(market)
            and market.open_time >= freshness_cutoff
            and spread <= _MAX_SPREAD
        )

    def should_trade(self, signal: Signal, market: Market) -> bool:
        return (
            signal.edge >= self.config.min_edge
            and signal.confidence >= self.config.min_confidence
        )

    def position_size(self, signal: Signal, bankroll: float) -> float:
        kelly = signal.edge / (1 - signal.estimated_probability)
        return bankroll * kelly * self.config.kelly_fraction
