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
    max_days_to_close: int
    min_days_to_close: int
