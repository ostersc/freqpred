"""Market, Order, and Position dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Market:
    # Identity (never changes after creation)
    id: str
    platform: str  # "kalshi"
    question: str
    category: str
    close_time: datetime

    # Price snapshot (changes frequently)
    yes_bid: float
    yes_ask: float
    mid_price: float
    volume_24h: float
    open_interest: float

    # Cache control
    last_fetched_at: datetime
    price_updated_at: datetime
    metadata_fetched_at: datetime

    # Signal linkage
    current_signal_id: str | None = None

    metadata: dict = field(default_factory=dict)


@dataclass
class Order:
    market_id: str
    direction: str   # "YES" | "NO"
    contracts: int
    price: float
    mode: str        # "paper" | "live"
    id: str | None = None
    status: str = "pending"


@dataclass
class Position:
    id: str
    market_id: str
    signal_id: str

    # Strategy attribution
    strategy_name: str
    strategy_version: str

    # Signal snapshot at time of trade
    signal_confidence: float
    signal_edge: float
    signal_estimated_prob: float

    # Order details
    direction: str   # "YES" | "NO"
    contracts: int
    entry_price: float
    entry_time: datetime
    mode: str        # "paper" | "live"

    # Lifecycle
    status: str = "pending"  # "pending" | "open" | "closed" | "cancelled"

    # Filled after resolution
    exit_price: float | None = None
    exit_time: datetime | None = None
    resolution: int | None = None   # 1 = YES won, 0 = NO won
    pnl: float | None = None
    pnl_pct: float | None = None
