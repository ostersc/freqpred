"""Document store: dedup, embed (Voyage AI), and upsert into Postgres.

Flow for upsert_document:
1. Strip HTML from body.
2. Compute SHA-256 content_hash.
3. SELECT existing row by source_url.
4. If existing row has same content_hash → return as-is (no re-embed).
5. Otherwise embed the cleaned body (new doc or content changed).
6. INSERT … ON CONFLICT (source_url) DO UPDATE — atomic upsert.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.rag.embedder import LocalEmbedder
from freqpred.rag.models import Document, DocumentMarketLinkRow, DocumentRow

log = structlog.get_logger()

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class DocumentSkipped(Exception):
    """Raised when a document is intentionally skipped (e.g. empty body after cleaning)."""

# Truncate body before embedding to keep token count reasonable.
# all-MiniLM-L6-v2 has a 512-token limit; ~2000 chars ≈ 400 tokens.
_MAX_EMBED_CHARS = 2_000


# ---------------------------------------------------------------------------
# Input type
# ---------------------------------------------------------------------------


@dataclass
class RawDocument:
    """A document as returned by a fetcher, before embedding."""

    source_url: str
    title: str
    body: str          # may contain HTML
    source_type: str   # "news" | "reddit" | "twitter" | "kalshi_comment"
    source_name: str   # e.g. "Reuters", "r/politics"
    category: str
    tags: list[str]
    published_at: datetime | None
    fetched_at: datetime
    summary: str | None = None


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Minimal HTMLParser subclass that collects raw text nodes."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join("".join(self._parts).split())


def _strip_html(text: str) -> str:
    """Remove HTML tags from *text* and return the plain-text content."""
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


def _sanitize(text: str) -> str:
    """Strip null bytes that PostgreSQL's UTF-8 encoding rejects."""
    return text.replace("\x00", "")


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Domain conversion
# ---------------------------------------------------------------------------


def _row_to_domain(row: DocumentRow) -> Document:
    return Document(
        id=str(row.id),
        source_url=row.source_url,
        content_hash=row.content_hash,
        title=row.title,
        body=row.body,
        summary=row.summary,
        source_type=row.source_type,
        source_name=row.source_name,
        category=row.category,
        tags=list(row.tags),
        published_at=row.published_at,
        fetched_at=row.fetched_at,
        embedding=list(row.embedding),
        embedding_model=row.embedding_model,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def upsert_document(
    session: AsyncSession,
    embedder: LocalEmbedder,
    raw_doc: RawDocument,
) -> Document:
    """Embed and upsert a document, skipping re-embedding if content unchanged.

    Args:
        session:  An open async SQLAlchemy session (caller manages commit).
        embedder: Voyage AI embedder instance.
        raw_doc:  The raw fetched document.

    Returns:
        The persisted Document domain object.
    """
    body_clean = _sanitize(_strip_html(raw_doc.body))

    if not body_clean.strip():
        log.debug("store.upsert_document.skip", source_url=raw_doc.source_url, reason="empty_body")
        raise DocumentSkipped(raw_doc.source_url)

    content_hash = _sha256(body_clean)
    embed_text = body_clean[:_MAX_EMBED_CHARS]

    # Check for an existing row with the same URL.
    result = await session.execute(
        select(DocumentRow).where(DocumentRow.source_url == raw_doc.source_url)
    )
    existing = result.scalar_one_or_none()

    if existing is not None and existing.content_hash == content_hash:
        log.debug(
            "store.upsert_document.skip",
            source_url=raw_doc.source_url,
            reason="content_hash_unchanged",
        )
        return _row_to_domain(existing)

    # New doc or content changed — generate embedding.
    log.debug(
        "store.upsert_document.embed",
        source_url=raw_doc.source_url,
        is_update=(existing is not None),
    )
    embedding = await embedder.embed_text(embed_text)

    doc_id = uuid.uuid4() if existing is None else existing.id

    stmt = (
        pg_insert(DocumentRow)
        .values(
            id=doc_id,
            source_url=raw_doc.source_url,
            content_hash=content_hash,
            title=_sanitize(raw_doc.title),
            body=body_clean,
            summary=_sanitize(raw_doc.summary) if raw_doc.summary else raw_doc.summary,
            source_type=raw_doc.source_type,
            source_name=raw_doc.source_name,
            category=raw_doc.category,
            tags=raw_doc.tags,
            published_at=raw_doc.published_at,
            fetched_at=raw_doc.fetched_at,
            embedding=embedding,
            embedding_model=_EMBEDDING_MODEL,
        )
        .on_conflict_do_update(
            index_elements=["source_url"],
            set_={
                "content_hash": content_hash,
                "title": _sanitize(raw_doc.title),
                "body": body_clean,
                "summary": _sanitize(raw_doc.summary) if raw_doc.summary else raw_doc.summary,
                "source_type": raw_doc.source_type,
                "source_name": raw_doc.source_name,
                "category": raw_doc.category,
                "tags": raw_doc.tags,
                "published_at": raw_doc.published_at,
                "fetched_at": raw_doc.fetched_at,
                "embedding": embedding,
                "embedding_model": _EMBEDDING_MODEL,
            },
        )
        .returning(DocumentRow)
    )

    row_result = await session.execute(stmt)
    row = row_result.scalar_one()
    await session.flush()

    return _row_to_domain(row)


async def link_document_to_market(
    session: AsyncSession,
    document_id: str,
    market_id: str,
) -> None:
    """Write an ingestion-time DocumentMarketLink (signal_id=None) if one does not exist.

    Uses INSERT ... ON CONFLICT DO NOTHING against the partial unique index
    (document_id, market_id) WHERE signal_id IS NULL, so repeated ingestion
    cycles are idempotent.
    """
    stmt = (
        pg_insert(DocumentMarketLinkRow)
        .values(
            id=uuid.uuid4(),
            document_id=uuid.UUID(document_id),
            market_id=market_id,
            signal_id=None,
            relevance_score=0.0,
            linked_at=datetime.now(tz=timezone.utc),
        )
        .on_conflict_do_nothing()
    )
    await session.execute(stmt)
