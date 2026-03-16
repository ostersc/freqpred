"""RAG data models: Document and DocumentMarketLink."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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
