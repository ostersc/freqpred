"""ORM models and dataclasses for signal assessment metrics."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Integer, Text, VARCHAR
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import PrimaryKeyConstraint

from freqpred.db import Base


class SourceQualityScoreRow(Base):
    """Daily rolling source-quality snapshot for one source/category pair."""

    __tablename__ = "source_quality_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    market_category: Mapped[str | None] = mapped_column(VARCHAR(100), nullable=True)
    lookback_days: Mapped[int] = mapped_column(Integer, nullable=False)
    weighted_brier: Mapped[float] = mapped_column(Float, nullable=False)
    overall_brier: Mapped[float] = mapped_column(Float, nullable=False)
    n_signals: Mapped[int] = mapped_column(Integer, nullable=False)
    total_doc_uses: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )


class SignalAssessmentRow(Base):
    """Append-only persisted trade-sizing assessment for one signal."""

    __tablename__ = "signal_assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    signal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signals.id"), nullable=False
    )
    trust_score: Mapped[float] = mapped_column(Float, nullable=False)
    size_multiplier: Mapped[float] = mapped_column(Float, nullable=False)
    verdict: Mapped[str] = mapped_column(VARCHAR(20), nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    key_factors: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    source_breakdown: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    similar_market_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    llm_query_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("llm_queries.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )


class SeriesOptionHistoryRow(Base):
    """Cached YES/NO settlement counts for a series option (or the aggregate row)."""

    __tablename__ = "series_option_history"

    series_ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    option_code: Mapped[str] = mapped_column(Text, primary_key=True)
    option_label: Mapped[str] = mapped_column(Text, nullable=False)
    yes_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    no_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )


@dataclass
class SignalAssessment:
    """Runtime assessment result used for trade sizing and audit."""

    signal_id: str
    trust_score: float
    size_multiplier: float
    verdict: str
    reasoning: str
    key_factors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_breakdown: list[dict[str, Any]] = field(default_factory=list)
    similar_market_summary: dict[str, Any] = field(default_factory=dict)
    llm_query_id: int | None = None
    created_at: datetime | None = None
