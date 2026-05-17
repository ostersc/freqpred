"""Catalyst data models: CatalystRun and CatalystQuery — ORM + dataclasses.

Also contains ApiDailyCounterRow for per-service daily request tracking.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import Boolean, Date, ForeignKey, Integer, JSON, SmallInteger, Text, VARCHAR
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import PrimaryKeyConstraint
from sqlalchemy import text as sa_text

from freqpred.db import Base

# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------


class CatalystRunRow(Base):
    """ORM model for the ``catalyst_runs`` table.

    One row per generation event per market. The ingestion scheduler always
    reads catalyst queries from the latest run where ``is_active=True``.
    """

    __tablename__ = "catalyst_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    market_id: Mapped[str] = mapped_column(
        VARCHAR(255), ForeignKey("markets.id"), nullable=False, index=True
    )
    # Monotonically increasing per market: 1 = first run, 2 = first re-run, etc.
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    # FK to the LLMQuery that produced this run — audit trail.
    # Nullable because we set it after the LLM response is logged.
    llm_query_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("llm_queries.id"), nullable=True
    )
    # Set to False when market closes or no strategy is interested.
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )

    queries: Mapped[list["CatalystQueryRow"]] = relationship(
        "CatalystQueryRow", back_populates="run", lazy="raise"
    )


class CatalystQueryRow(Base):
    """ORM model for the ``catalyst_queries`` table.

    Each row is one search string derived by the Catalyst Generator for a
    specific market generation. The ingestion scheduler runs fetchers against
    these strings.
    """

    __tablename__ = "catalyst_queries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("catalyst_runs.id"), nullable=False, index=True
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    tv_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )

    run: Mapped["CatalystRunRow"] = relationship(
        "CatalystRunRow", back_populates="queries"
    )


class ApiDailyCounterRow(Base):
    """ORM model for the ``api_daily_counters`` table.

    One row per (service, date, hour_slot) triple. hour_slot is 0 for
    00:00–11:59 UTC and 1 for 12:00–23:59 UTC. Incremented atomically via
    INSERT ... ON CONFLICT DO UPDATE in quota.py.
    """

    __tablename__ = "api_daily_counters"
    __table_args__ = (PrimaryKeyConstraint("service", "date", "hour_slot"),)

    service: Mapped[str] = mapped_column(Text, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    hour_slot: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class FetcherCursorRow(Base):
    """ORM model for the ``fetcher_cursors`` table.

    Tracks the last time a fetcher ran for a given key (typically a market ID).
    Used to implement adaptive per-market fetch intervals without Redis.
    """

    __tablename__ = "fetcher_cursors"
    __table_args__ = (PrimaryKeyConstraint("fetcher", "key"),)

    fetcher: Mapped[str] = mapped_column(Text, nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    last_run_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)


class FactbasePhraseRow(Base):
    """ORM model for the ``factbase_phrase_frequency`` table.

    One row per market. Stores the Haiku-extracted search terms and the
    pre-computed frequency counts for each time window. ``api_query`` is
    the Lucene OR query sent to the FactBase API; ``display_phrase`` is
    the human-readable label used in prompts.
    """

    __tablename__ = "factbase_phrase_frequency"

    market_id: Mapped[str] = mapped_column(
        VARCHAR(255), ForeignKey("markets.id"), primary_key=True
    )
    display_phrase: Mapped[str] = mapped_column(Text, nullable=False)
    api_query: Mapped[str] = mapped_column(Text, nullable=False)
    speaker_slug: Mapped[str] = mapped_column(Text, nullable=False, server_default="trump")
    in_market_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    count_7d: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    count_30d: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    count_365d: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    top_quotes: Mapped[list] = mapped_column(
        JSON, nullable=False, server_default=sa_text("'[]'::json")
    )
    last_fetched_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )


class FetcherRateLimitRow(Base):
    """ORM model for the ``fetcher_rate_limits`` table.

    Persists exponential backoff state per fetcher service across restarts.
    Managed by backoff.py.

    skip_cycles_remaining: how many more cycles to skip before retrying.
    skip_cycles_next:       what to set skip_cycles_remaining to on the next
                            rate-limit trip (doubles each time, capped at 32).
    """

    __tablename__ = "fetcher_rate_limits"

    service: Mapped[str] = mapped_column(Text, primary_key=True)
    skip_cycles_remaining: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skip_cycles_next: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    tripped_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )


# ---------------------------------------------------------------------------
# Dataclasses (domain models)
# ---------------------------------------------------------------------------


@dataclass
class CatalystRun:
    id: str
    market_id: str
    generation: int
    is_active: bool
    created_at: datetime
    llm_query_id: int | None = None


@dataclass
class CatalystQuery:
    id: str
    run_id: str
    query_text: str
    created_at: datetime
    tv_query: str | None = None
