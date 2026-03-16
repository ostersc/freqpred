"""PoliticsEdgeStrategy: US politics markets with conservative Kelly."""
from __future__ import annotations

from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig

if False:  # TYPE_CHECKING
    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal


class PoliticsEdgeStrategy(IPredictionStrategy):
    config = StrategyConfig(
        name="PoliticsEdgeStrategy",
        min_edge=0.18,
        min_confidence=0.70,
        max_exposure_per_market=0.05,
        kelly_fraction=0.25,
        categories=["politics"],
        min_volume_24h=1000.0,
        max_days_to_close=90,
        min_days_to_close=1,
    )

    def should_trade(self, signal: Signal, market: Market) -> bool:
        return (
            signal.edge >= self.config.min_edge
            and signal.confidence >= self.config.min_confidence
        )

    def position_size(self, signal: Signal, bankroll: float) -> float:
        kelly = signal.edge / (1 - signal.estimated_probability)
        return bankroll * kelly * self.config.kelly_fraction
