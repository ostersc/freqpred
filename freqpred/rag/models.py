"""RAG data models: Document and DocumentMarketLink — dataclasses and ORM models."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Float, ForeignKey, Text, UniqueConstraint, VARCHAR
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from freqpred.db import Base

# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------


class DocumentRow(Base):
    """ORM model for the ``documents`` table."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Identity & deduplication
    source_url: Mapped[str] = mapped_column(
        Text, nullable=False, unique=True, index=True
    )
    content_hash: Mapped[str] = mapped_column(VARCHAR(64), nullable=False)

    # Content
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    source_name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)

    # Classification
    category: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    tags: Mapped[list] = mapped_column(ARRAY(Text), nullable=False, default=list)

    # Temporal
    published_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )

    # Vector search — pgvector VECTOR(384) for all-MiniLM-L6-v2
    embedding: Mapped[list] = mapped_column(Vector(384), nullable=False)
    embedding_model: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )

    # Relationships
    market_links: Mapped[list["DocumentMarketLinkRow"]] = relationship(
        "DocumentMarketLinkRow", back_populates="document"
    )


class DocumentMarketLinkRow(Base):
    """ORM model for the ``document_market_links`` join table."""

    __tablename__ = "document_market_links"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    market_id: Mapped[str] = mapped_column(
        VARCHAR(255), ForeignKey("markets.id"), nullable=False
    )
    signal_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True
    )
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    linked_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )

    document: Mapped["DocumentRow"] = relationship(
        "DocumentRow", back_populates="market_links"
    )
    signal: Mapped["SignalRow | None"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "SignalRow", back_populates="document_links"
    )


# ---------------------------------------------------------------------------
# Dataclasses (domain models)
# ---------------------------------------------------------------------------


@dataclass
class Document:
    id: str

    # Identity & deduplication
    source_url: str   # unique constraint — prevents duplicate storage
    content_hash: str

    # Content
    title: str
    body: str
    source_type: str  # "news" | "reddit" | "twitter" | "kalshi_comment" | "manifold"
    source_name: str  # e.g. "Reuters", "r/politics", "Kalshi"

    # Classification
    category: str
    tags: list[str]

    # Temporal
    published_at: datetime
    fetched_at: datetime

    # Vector search
    embedding: list[float]
    embedding_model: str

    summary: str | None = None


@dataclass
class DocumentMarketLink:
    document_id: str
    market_id: str
    relevance_score: float
    linked_at: datetime
    signal_id: str | None = None
