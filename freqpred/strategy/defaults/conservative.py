"""ConservativeDefault: high-confidence only, tiny sizing — safe starting point."""
from __future__ import annotations

from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig

if False:  # TYPE_CHECKING
    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal


class ConservativeDefault(IPredictionStrategy):
    config = StrategyConfig(
        name="ConservativeDefault",
        min_edge=0.20,
        min_confidence=0.80,
        max_exposure_per_market=0.02,
        kelly_fraction=0.10,
        categories=["politics", "technology", "fintech"],
        min_volume_24h=2000.0,
        max_days_to_close=60,
        min_days_to_close=2,
    )

    def should_trade(self, signal: Signal, market: Market) -> bool:
        return (
            signal.edge >= self.config.min_edge
            and signal.confidence >= self.config.min_confidence
        )

    def position_size(self, signal: Signal, bankroll: float) -> float:
        kelly = signal.edge / (1 - signal.estimated_probability)
        return bankroll * kelly * self.config.kelly_fraction
