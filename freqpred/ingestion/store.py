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
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select, text as sa_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.rag.embedder import LocalEmbedder
from freqpred.rag.models import Document, DocumentMarketLinkRow, DocumentRow

if TYPE_CHECKING:
    from freqpred.llm.client import LLMClient

log = structlog.get_logger()

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class DocumentSkipped(Exception):
    """Raised when a document is intentionally skipped (e.g. empty body after cleaning)."""


from enum import Enum


class UpsertStatus(str, Enum):
    INSERTED = "inserted"   # brand-new document
    UPDATED = "updated"     # existing URL, content changed
    DEDUPED = "deduped"     # existing URL, content unchanged — no DB write

# Truncate body before embedding to keep token count reasonable.
# all-MiniLM-L6-v2 has a 512-token limit; ~2000 chars ≈ 400 tokens.
_MAX_EMBED_CHARS = 2_000

# LLM summarization thresholds.
# Bodies longer than _SUMMARY_THRESHOLD are candidates for LLM summarization.
# _MIN_BM25_SCORE is the ts_rank floor against the market question's first line;
# documents scoring below this are off-topic and not worth summarizing.
_SUMMARY_THRESHOLD = 1_000   # 2 × the 500-char evidence excerpt limit in signal/llm.py
_MIN_BM25_SCORE = 0.01       # ~42% of long linked docs score below this when using first-line market question (live data)


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
    *,
    llm_client: LLMClient | None = None,
    query_text: str = "",
    market_question: str = "",
    summary_model: str | None = None,
) -> tuple[Document, UpsertStatus]:
    """Embed and upsert a document, skipping re-embedding if content unchanged.

    When llm_client, query_text, and market_question are provided, long bodies
    that pass a BM25 relevance gate are summarized before embedding. The summary
    is stored on the document and used as the embedding source.

    Args:
        session:          An open async SQLAlchemy session (caller manages commit).
        embedder:         Local embedder instance.
        raw_doc:          The raw fetched document.
        llm_client:       Optional LLM client for body summarization.
        query_text:       Catalyst query that retrieved this document (for prompt context).
        market_question:  Full market question (for BM25 gate + summarizer prompt).
        summary_model:    Optional model override for the body summarizer.

    Returns:
        A (Document, UpsertStatus) tuple. Status is INSERTED for new docs,
        UPDATED for existing URLs with changed content, DEDUPED for unchanged.

    Raises:
        DocumentSkipped: if the body is empty after HTML stripping.
    """
    body_clean = _sanitize(_strip_html(raw_doc.body))

    if not body_clean.strip():
        log.debug("store.upsert_document.skip", source_url=raw_doc.source_url, reason="empty_body")
        raise DocumentSkipped(raw_doc.source_url)

    content_hash = _sha256(body_clean)

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
        return _row_to_domain(existing), UpsertStatus.DEDUPED

    # If no row matches the URL, check for a different URL with identical content.
    # TV Archive and other sources can return the same snippet under multiple URLs
    # across separate fetch runs — we don't want N copies in the evidence pool.
    if existing is None:
        hash_result = await session.execute(
            select(DocumentRow).where(DocumentRow.content_hash == content_hash)
        )
        existing_by_hash = hash_result.scalars().first()
        if existing_by_hash is not None:
            log.debug(
                "store.upsert_document.skip",
                source_url=raw_doc.source_url,
                reason="duplicate_content_hash",
                existing_url=existing_by_hash.source_url,
            )
            return _row_to_domain(existing_by_hash), UpsertStatus.DEDUPED

    is_update = existing is not None

    # New doc or content changed — optionally summarize long bodies before embedding.
    # Summarization is gated on: body length > threshold AND BM25 score against the
    # market question's first line meets the minimum. The dedup check above ensures
    # we never call the LLM for content we've already processed.
    if llm_client is not None and len(body_clean) > _SUMMARY_THRESHOLD and market_question:
        question_first_line = market_question.split("\n")[0]
        bm25_result = await session.execute(
            sa_text(
                "SELECT ts_rank("
                "  to_tsvector('english', :title || ' ' || :body),"
                "  plainto_tsquery('english', :query)"
                ") AS score"
            ),
            {
                "title": raw_doc.title,
                "body": body_clean[:_SUMMARY_THRESHOLD],
                "query": question_first_line,
            },
        )
        bm25_score: float = bm25_result.scalar() or 0.0

        if bm25_score >= _MIN_BM25_SCORE:
            from freqpred.ingestion.body_summarizer import summarize_body
            summary = await summarize_body(
                raw_doc,
                query_text,
                market_question,
                llm_client,
                model=summary_model or "claude-haiku-4-5-20251001",
            )
            if summary is not None:
                raw_doc.summary = summary
                log.debug(
                    "store.upsert_document.summarized",
                    source_url=raw_doc.source_url,
                    bm25_score=round(bm25_score, 4),
                )
        else:
            log.debug(
                "store.upsert_document.summarize_skip",
                source_url=raw_doc.source_url,
                reason="low_bm25",
                bm25_score=round(bm25_score, 4),
            )

    # Use summary for embedding when present — aligns the embedding vector with
    # the summarized content rather than an arbitrary body truncation.
    summary_clean = _sanitize(raw_doc.summary) if raw_doc.summary else None
    embed_text = summary_clean[:_MAX_EMBED_CHARS] if summary_clean else body_clean[:_MAX_EMBED_CHARS]

    log.debug(
        "store.upsert_document.embed",
        source_url=raw_doc.source_url,
        is_update=is_update,
        embed_source="summary" if summary_clean else "body",
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

    status = UpsertStatus.UPDATED if is_update else UpsertStatus.INSERTED
    return _row_to_domain(row), status


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
