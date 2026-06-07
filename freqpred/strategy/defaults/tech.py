"""TechNewsStrategy: technology/fintech markets, shorter-dated."""
from __future__ import annotations

from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import OrderTypes, StrategyConfig

if False:  # TYPE_CHECKING
    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal


class TechNewsStrategy(IPredictionStrategy):
    config = StrategyConfig(
        name="TechNewsStrategy",
        min_edge=0.15,
        min_confidence=0.68,
        max_exposure_per_market=0.20,  # kelly_fraction × 0.20 = 4% of bankroll max
        kelly_fraction=0.20,
        categories=["Science and Technology"],
        min_volume_24h=500.0,
        max_days_to_close=30,
        min_days_to_close=1,
        stoploss=-0.15,
        trailing_stop=True,
        trailing_stop_positive=0.10,
        trailing_stop_positive_offset=0.02,
        order_types=OrderTypes(entry="limit", exit="limit"),
    )

    def should_trade(self, signal: Signal, market: Market) -> bool:
        return super().should_trade(signal, market)

