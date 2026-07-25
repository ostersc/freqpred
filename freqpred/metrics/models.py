"""ORM models and dataclasses for signal assessment metrics."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import (
    VARCHAR,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

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


class EdgeCalibrationScoreRow(Base):
    """Daily rolling edge-band calibration snapshot for one band/direction/series.

    series_ticker=NULL is the global (all-series) rollup row for that band+direction.
    """

    __tablename__ = "edge_calibration_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    edge_band: Mapped[str] = mapped_column(VARCHAR(10), nullable=False)
    direction: Mapped[str] = mapped_column(VARCHAR(10), nullable=False)
    series_ticker: Mapped[str | None] = mapped_column(Text, nullable=True)
    # NULL = the all-versions rollup (the historical shape, and the fallback when
    # a version cohort is too thin). Non-NULL rows are scoped to one signal prompt
    # version, because measured performance is strongly version-dependent: on
    # KXTRUMPSAY the NO-side profit edge runs -0.240 (signal-v7), -0.067 (v4),
    # +0.120 (v9), +0.133 (v11), so an all-versions pool describes a model
    # production no longer runs.
    prompt_version: Mapped[str | None] = mapped_column(VARCHAR(100), nullable=True)
    n_signals: Mapped[int] = mapped_column(Integer, nullable=False)
    n_markets: Mapped[int] = mapped_column(Integer, nullable=False)
    hit_rate: Mapped[float] = mapped_column(Float, nullable=False)
    avg_market_implied_p: Mapped[float] = mapped_column(Float, nullable=False)
    avg_model_implied_p: Mapped[float] = mapped_column(Float, nullable=False)
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


class MarketCandleRow(Base):
    """One OHLC candlestick for one market at one period interval.

    Kalshi's candlestick endpoint is the only source of a real intra-hold price
    path — the rest of the system persists only point observations
    (``signals.market_*_at_signal``) and censored extremes (``positions.mae``).

    **Retention is a rolling window (~67 days measured 2026-07-25).** Candles for
    markets that settled before the cutoff return 404 and are gone permanently,
    which is why this table exists rather than fetching on demand.

    The primary key is the natural composite (market_id, period_interval,
    end_period_ts) rather than a UUID, so re-fetching an overlapping window is an
    idempotent upsert instead of a duplicate. ``series_option_history`` sets the
    same precedent.

    There is deliberately no FK to ``markets``: the backfill tool must work for
    any ticker, including markets this system never ingested.

    Price fields are nullable because a period with no trades has no ``price``
    OHLC, and an empty book yields a 0.0 bid. **A 0.0 bid means "no bid", not a
    price of zero** — anything reading these must not treat it as a tradeable
    level, or every stoploss counterfactual will trigger on an empty book.
    """

    __tablename__ = "market_candles"

    market_id: Mapped[str] = mapped_column(VARCHAR(255), primary_key=True)
    period_interval: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    end_period_ts: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), primary_key=True
    )
    series_ticker: Mapped[str] = mapped_column(Text, nullable=False)

    price_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_mean: Mapped[float | None] = mapped_column(Float, nullable=True)

    yes_bid_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    yes_bid_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    yes_bid_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    yes_bid_close: Mapped[float | None] = mapped_column(Float, nullable=True)

    yes_ask_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    yes_ask_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    yes_ask_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    yes_ask_close: Mapped[float | None] = mapped_column(Float, nullable=True)

    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    open_interest: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )


class CandleFetchCursorRow(Base):
    """Per-(market, interval) record of what has already been fetched.

    Without this the scheduler cannot tell "we fetched this window and the market
    was quiet" from "we never fetched this window", and would re-request empty
    ranges forever against a rate-limited API.
    """

    __tablename__ = "candle_fetch_cursors"

    market_id: Mapped[str] = mapped_column(VARCHAR(255), primary_key=True)
    period_interval: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    covered_from: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    covered_to: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    candle_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Set when the API returned 404 — the market settled before the retention
    # cutoff. Permanent: never retry, the data does not come back.
    expired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
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
