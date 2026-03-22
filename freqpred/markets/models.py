"""Market, Order, and Position dataclasses and SQLAlchemy ORM models.

Also contains Pydantic schemas for validating raw Kalshi API responses.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import (
    JSON,
    VARCHAR,
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from freqpred.db import Base

# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------


class MarketRow(Base):
    """ORM model for the ``markets`` table."""

    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(VARCHAR(255), primary_key=True)
    platform: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    close_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    open_time: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Market state
    status: Mapped[str] = mapped_column(VARCHAR(20), nullable=False, server_default="active")
    result: Mapped[str | None] = mapped_column(VARCHAR(10), nullable=True)

    # Price snapshot
    yes_bid: Mapped[float] = mapped_column(Float, nullable=False)
    yes_ask: Mapped[float] = mapped_column(Float, nullable=False)
    mid_price: Mapped[float] = mapped_column(Float, nullable=False)
    last_price: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    volume_24h: Mapped[float] = mapped_column(Float, nullable=False)
    open_interest: Mapped[float] = mapped_column(Float, nullable=False)
    liquidity: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")

    # Cache control
    last_fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    price_updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    metadata_fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )

    # Signal linkage (use_alter avoids circular FK creation order)
    current_signal_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("signals.id", use_alter=True, name="fk_markets_current_signal"),
        nullable=True,
    )

    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default="now()",
    )

    # Relationships (back-populated in other models)
    signals: Mapped[list["SignalRow"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "SignalRow",
        back_populates="market",
        foreign_keys="SignalRow.market_id",
        lazy="raise",
    )
    positions: Mapped[list["PositionRow"]] = relationship(
        "PositionRow",
        back_populates="market",
        lazy="raise",
    )


class PositionRow(Base):
    """ORM model for the ``positions`` table."""

    __tablename__ = "positions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    market_id: Mapped[str] = mapped_column(
        VARCHAR(255), ForeignKey("markets.id"), nullable=False
    )
    signal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signals.id"), nullable=False
    )

    # Strategy attribution
    strategy_name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    strategy_version: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)

    # Signal snapshot
    signal_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    signal_edge: Mapped[float] = mapped_column(Float, nullable=False)
    signal_estimated_prob: Mapped[float] = mapped_column(Float, nullable=False)

    # Order details
    direction: Mapped[str] = mapped_column(VARCHAR(10), nullable=False)
    contracts: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    mode: Mapped[str] = mapped_column(VARCHAR(10), nullable=False)

    # Lifecycle
    status: Mapped[str] = mapped_column(VARCHAR(20), nullable=False, default="pending")

    # Filled after resolution
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_time: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    exit_reason: Mapped[str | None] = mapped_column(VARCHAR(100), nullable=True)
    resolution: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Exchange order reference (live mode only)
    exchange_order_id: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)

    # Excursion tracking — signed price deltas (multiply by contracts for $ impact)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)  # max adverse excursion
    mfe: Mapped[float | None] = mapped_column(Float, nullable=True)  # max favorable excursion

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default="now()",
    )

    market: Mapped["MarketRow"] = relationship("MarketRow", back_populates="positions")
    signal: Mapped["SignalRow"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "SignalRow", back_populates="positions"
    )


# ---------------------------------------------------------------------------
# Dataclasses (domain models)
# ---------------------------------------------------------------------------


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
    created_at: datetime = field(default_factory=lambda: datetime(1970, 1, 1, tzinfo=timezone.utc))
    open_time: datetime | None = None
    status: str = "active"
    result: str | None = None
    last_price: float = 0.0
    liquidity: float = 0.0


# ---------------------------------------------------------------------------
# Pydantic schemas — Kalshi API response validation
# ---------------------------------------------------------------------------


class KalshiMarketSchema(BaseModel):
    """Pydantic schema for a single market object from the Kalshi REST API.

    Used to validate raw API payloads before converting to the Market dataclass.
    Dollar-denominated price fields use Kalshi's fixed-point string format
    (e.g. "0.5600"). The validator normalises them to float.

    The Kalshi v2 API uses ``_fp`` (fixed-point) suffix for volume/OI fields.
    ``populate_by_name=True`` lets tests pass plain field names too.
    """

    model_config = ConfigDict(populate_by_name=True)

    ticker: str
    event_ticker: str = ""
    title: str = ""
    subtitle: str = ""
    yes_sub_title: str = ""
    rules_primary: str = ""
    rules_secondary: str = ""
    status: str = ""
    result: str = ""
    close_time: str  # ISO-8601 string; converted downstream
    open_time: str = ""  # when the market opened for trading; converted downstream
    yes_bid_dollars: str = "0.0000"
    yes_ask_dollars: str = "0.0000"
    no_bid_dollars: str = "0.0000"
    no_ask_dollars: str = "0.0000"
    last_price_dollars: str = "0.0000"
    liquidity_dollars: str = "0.0000"
    volume_24h: float = Field(default=0.0, alias="volume_24h_fp")
    open_interest: float = Field(default=0.0, alias="open_interest_fp")

    @field_validator(
        "yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars",
        "last_price_dollars", "liquidity_dollars",
        mode="before",
    )
    @classmethod
    def coerce_dollar_string(cls, v: object) -> str:
        """Accept numeric values as well as strings."""
        if v is None:
            return "0.0000"
        return str(v)

    @property
    def yes_bid(self) -> float:
        return float(self.yes_bid_dollars)

    @property
    def yes_ask(self) -> float:
        return float(self.yes_ask_dollars)

    @property
    def last_price(self) -> float:
        return float(self.last_price_dollars)

    @property
    def liquidity(self) -> float:
        return float(self.liquidity_dollars)

    @property
    def mid_price(self) -> float:
        b, a = self.yes_bid, self.yes_ask
        return (b + a) / 2.0 if (b + a) > 0 else 0.0


class KalshiMarketsResponse(BaseModel):
    """Pydantic schema for GET /markets response envelope."""

    markets: list[KalshiMarketSchema] = Field(default_factory=list)
    cursor: str = ""


class KalshiSingleMarketResponse(BaseModel):
    """Pydantic schema for GET /markets/{ticker} response envelope."""

    market: KalshiMarketSchema


class KalshiSeriesSchema(BaseModel):
    """Pydantic schema for a single series object from GET /series."""

    ticker: str
    category: str = ""
    title: str = ""


class KalshiSeriesResponse(BaseModel):
    """Pydantic schema for GET /series response envelope."""

    series: list[KalshiSeriesSchema] = Field(default_factory=list)


@dataclass
class Order:
    market_id: str
    direction: str   # "YES" | "NO"
    contracts: int
    price: float
    mode: str        # "paper" | "live"
    id: str | None = None
    exchange_order_id: str | None = None
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
    exit_reason: str | None = None  # "stoploss" | "trailing_stop" | "roi" | "signal" | "custom_exit:<tag>" | "market_resolved"
    resolution: int | None = None   # 1 = YES won, 0 = NO won
    pnl: float | None = None
    pnl_pct: float | None = None
    mae: float | None = None  # max adverse excursion (signed price delta)
    mfe: float | None = None  # max favorable excursion (signed price delta)
    exchange_order_id: str | None = None  # Kalshi order ID (live mode only)
