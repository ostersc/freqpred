"""ORM models for runtime telemetry and events."""
from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import VARCHAR, Boolean, Date, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from freqpred.db import Base


class ServiceHeartbeatRow(Base):
    """Latest success/error heartbeat for one long-running service."""

    __tablename__ = "service_heartbeats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    service_name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False, unique=True)
    last_success_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    last_error_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    last_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default="now()",
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )


class KalshiChangelogStateRow(Base):
    """Singleton row (id=1) tracking Kalshi changelog review state.

    ``last_reviewed_at`` is the operator-confirmed date through which all
    changelog entries have been reviewed.  Update it via an Alembic migration
    after reviewing and addressing any flagged entries.
    """

    __tablename__ = "kalshi_changelog_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_reviewed_at: Mapped[date] = mapped_column(Date, nullable=False)
    unreviewed_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    has_unreviewed_breaking_change: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )


class RuntimeEventRow(Base):
    """Timestamped runtime event for ops counters and diagnosis."""

    __tablename__ = "runtime_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    service_name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    category: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    level: Mapped[str] = mapped_column(VARCHAR(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default="now()",
    )

