"""Signal dataclass and SQLAlchemy ORM model."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import ARRAY, Float, ForeignKey, Text, VARCHAR
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from freqpred.db import Base

# ---------------------------------------------------------------------------
# SQLAlchemy ORM model
# ---------------------------------------------------------------------------


class SignalRow(Base):
    """ORM model for the ``signals`` table."""

    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    market_id: Mapped[str] = mapped_column(
        VARCHAR(255), ForeignKey("markets.id"), nullable=False
    )

    # Estimate
    estimated_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    edge: Mapped[float] = mapped_column(Float, nullable=False)
    market_mid_at_signal: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(VARCHAR(10), nullable=False)

    # Context
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list] = mapped_column(ARRAY(Text), nullable=False, default=list)
    social_sentiment_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_hash: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)

    # Provenance
    model_used: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    trigger: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )
    raw_context: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    market: Mapped["MarketRow"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "MarketRow",
        back_populates="signals",
        foreign_keys=[market_id],
    )
    positions: Mapped[list["PositionRow"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "PositionRow", back_populates="signal"
    )
    document_links: Mapped[list["DocumentMarketLinkRow"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "DocumentMarketLinkRow", back_populates="signal"
    )


# ---------------------------------------------------------------------------
# Dataclass (domain model)
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    id: str
    market_id: str

    # Estimate
    estimated_probability: float
    confidence: float
    edge: float
    market_mid_at_signal: float
    direction: str   # "YES" | "NO" | "SKIP"

    # Context
    reasoning: str
    sources: list[str]
    retrieval_hash: str

    # Provenance
    model_used: str
    prompt_version: str
    trigger: str     # "scheduled" | "price_moved" | "new_evidence" | "manual"
    created_at: datetime
    raw_context: str

    social_sentiment_summary: str | None = None
