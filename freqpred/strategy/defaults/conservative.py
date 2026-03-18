"""ConservativeDefault: high-confidence only, tiny sizing — safe starting point."""
from __future__ import annotations

from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig

if False:  # TYPE_CHECKING
    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal


class ConservativeDefault(IPredictionStrategy):
    """High-confidence, small-sizing strategy that trades all categories.

    Designed as a safe starting point: requires strong edge and confidence
    before trading, and uses a tiny Kelly fraction to limit exposure.

    Parameters:
        min_edge=0.12        — must see at least 12% edge over market price
        min_confidence=0.80  — LLM confidence must be >= 80%
        kelly_fraction=0.15  — use 15% of full Kelly sizing
        max_exposure_per_market=0.02  — never risk more than 2% of bankroll per market
    """

    config = StrategyConfig(
        name="ConservativeDefault",
        min_edge=0.12,
        min_confidence=0.80,
        max_exposure_per_market=0.02,
        kelly_fraction=0.15,
        categories=[],  # empty = all categories
        min_volume_24h=500.0,
        max_days_to_close=60,
        min_days_to_close=2,
    )

    def should_trade(self, signal: Signal, market: Market) -> bool:
        return (
            signal.edge >= self.config.min_edge
            and signal.confidence >= self.config.min_confidence
        )

    def position_size(self, signal: Signal, bankroll: float) -> float:
        """Kelly-fractional sizing, capped at max_exposure_per_market."""
        kelly = signal.edge / (1 - signal.estimated_probability)
        raw = bankroll * kelly * self.config.kelly_fraction
        cap = bankroll * self.config.max_exposure_per_market
        return min(raw, cap)
