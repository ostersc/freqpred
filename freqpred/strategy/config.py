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

    # Exit management
    stoploss: float = -0.20
    minimal_roi: dict[str, float] = field(default_factory=lambda: {
        "0": 0.30,       # exit at 30% profit immediately
        "1440": 0.15,    # exit at 15% profit after 1 day
        "10080": 0.05,   # exit at 5% profit after 1 week
    })
    trailing_stop: bool = False
    trailing_stop_positive: float | None = None      # switch to tight trail at this profit %
    trailing_stop_positive_offset: float = 0.02      # tight trail distance once profitable
