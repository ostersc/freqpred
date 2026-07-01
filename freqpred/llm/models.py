"""LLMQuery audit log dataclass and SQLAlchemy ORM model."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, Text, VARCHAR
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from freqpred.db import Base

# ---------------------------------------------------------------------------
# SQLAlchemy ORM model
# ---------------------------------------------------------------------------


class LLMQueryRow(Base):
    """ORM model for the ``llm_queries`` table (auto-increment int PK)."""

    __tablename__ = "llm_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # When & why
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    strategy: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    query_type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    market_id: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    signal_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), nullable=True
    )

    # Full request/response
    model_used: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)

    # Cost
    tokens_input: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_output: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_total: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)

    # Extracted outputs
    confidence_extracted: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_extracted: Mapped[str | None] = mapped_column(VARCHAR(10), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Error handling
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )


# ---------------------------------------------------------------------------
# Dataclass (domain model)
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Return value from LLMClient.complete()."""

    content: str
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: int
    llm_query_id: int
    thinking: str | None = None
    thinking_tokens: int | None = None


@dataclass
class LLMQuery:
    # auto-increment PK (int, not UUID)
    id: int | None

    # When & why
    timestamp: datetime
    strategy: str
    query_type: str  # "market_analysis" | "social_summarization" | "movement_prediction" | "daily_digest"
    model_used: str
    prompt_version: str

    # Full request/response
    prompt: str
    response: str

    # Cost
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost_usd: float

    # Extracted outputs
    latency_ms: int
    success: bool

    market_id: str | None = None
    signal_id: str | None = None
    confidence_extracted: float | None = None
    decision_extracted: str | None = None
    error_message: str | None = None
