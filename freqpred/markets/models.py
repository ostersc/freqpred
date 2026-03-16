"""Market, Order, and Position dataclasses and SQLAlchemy ORM models."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

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

    # Price snapshot
    yes_bid: Mapped[float] = mapped_column(Float, nullable=False)
    yes_ask: Mapped[float] = mapped_column(Float, nullable=False)
    mid_price: Mapped[float] = mapped_column(Float, nullable=False)
    volume_24h: Mapped[float] = mapped_column(Float, nullable=False)
    open_interest: Mapped[float] = mapped_column(Float, nullable=False)

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
    resolution: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

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
