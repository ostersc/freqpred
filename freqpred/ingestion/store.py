"""Document store: dedup, embed (Voyage AI), and upsert into Postgres.

Flow for upsert_document:
1. Strip HTML from body.
2. Compute SHA-256 content_hash.
3. SELECT existing row by source_url.
4. If existing row has same content_hash → return as-is (no re-embed).
5. Otherwise embed the cleaned body, truncated to embedder.max_embed_chars
   (new doc or content changed). The summary is never embedded — see
   derive_embed_text.
6. INSERT … ON CONFLICT (source_url) DO UPDATE — atomic upsert.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from html.parser import HTMLParser
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.rag.models import Document, DocumentMarketLinkRow, DocumentRow
from freqpred.rag.retriever import Embedder

if TYPE_CHECKING:
    from freqpred.llm.client import LLMClient

log = structlog.get_logger()

# Embedding model name is read from the embedder at upsert time — not hardcoded here.


class DocumentSkipped(Exception):
    """Raised when a document is intentionally skipped (e.g. empty body after cleaning)."""


class UpsertStatus(StrEnum):
    INSERTED = "inserted"   # brand-new document
    UPDATED = "updated"     # existing URL, content changed
    DEDUPED = "deduped"     # existing URL, content unchanged — no DB write

# Default truncation limit; overridden per-call by embedder.max_embed_chars.
_MAX_EMBED_CHARS = 2_000

# LLM summarization thresholds.
# Bodies longer than _SUMMARY_THRESHOLD are candidates for LLM summarization.
# _MIN_BM25_SCORE is the ts_rank floor against the market question's first line;
# documents scoring below this are off-topic and not worth summarizing.
_SUMMARY_THRESHOLD = 2_000   # 4 × the 500-char evidence excerpt limit in signal/llm.py
# ~42% of long linked docs score below this when using first-line market question (live data)
_MIN_BM25_SCORE = 0.01
_MAX_BODY_CHARS = 50_000     # skip docs still larger than this after HTML stripping — likely markup/boilerplate soup

# Process-level cache of URLs permanently rejected as body_too_large.
# Avoids re-processing the same oversized live-blog pages on every ingestion cycle.
# Populated lazily; survives for the process lifetime (one re-attempt per restart).
_rejected_urls: set[str] = set()


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


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags from *text* and return the plain-text content."""
    stripper = _HTMLStripper()
    try:
        stripper.feed(text)
        return stripper.get_text()
    except Exception:
        # Fallback for malformed markup (e.g. invalid marked sections like
        # <![Image ...]> from presidency.ucsb.edu) that HTMLParser rejects.
        return " ".join(_TAG_RE.sub(" ", text).split())


def _sanitize(text: str) -> str:
    """Strip null bytes that PostgreSQL's UTF-8 encoding rejects."""
    return text.replace("\x00", "")


# ---------------------------------------------------------------------------
# Embed text derivation
# ---------------------------------------------------------------------------


def derive_embed_text(body: str, max_chars: int) -> str:
    """Return the text sent to the embedder for a document.

    Always the body, never ``summary``. Summaries are written against whichever
    market question triggered the fetch, so embedding one makes a document's
    single vector representation topic-skewed by an accident of ingestion order
    — a doc summarised for "will Trump say Melania" would then represent itself
    to every future retrieval, including for unrelated markets (T100).

    The ``max_chars`` cap remains because one vector over a very long document
    averages unrelated topics into a point near none of them. Chunking with
    max-pooling is the principled fix for that tail and is out of scope here.

    This is the single source of truth for the derivation: ``upsert_document``
    and ``scripts/reindex_embeddings.py`` both call it so the live index and a
    reindex never disagree about what a document's vector represents.
    """
    return _sanitize(body)[:max_chars]


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
    embedder: Embedder,
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
        embedder:         Embedder instance (LocalEmbedder or OllamaEmbedder).
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
    if raw_doc.source_url in _rejected_urls:
        raise DocumentSkipped(raw_doc.source_url)

    body_clean = _sanitize(_strip_html(raw_doc.body))

    if not body_clean.strip():
        log.debug("store.upsert_document.skip", source_url=raw_doc.source_url, reason="empty_body")
        raise DocumentSkipped(raw_doc.source_url)

    if len(body_clean) > _MAX_BODY_CHARS:
        _rejected_urls.add(raw_doc.source_url)
        log.info(
            "store.upsert_document.skip",
            source_url=raw_doc.source_url,
            reason="body_too_large",
            body_len=len(body_clean),
        )
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

    # New doc or content changed — optionally summarize long bodies. The summary is
    # stored for display/evidence use only; it is NOT embedded (see derive_embed_text).
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

    embed_text = derive_embed_text(body_clean, embedder.max_embed_chars)

    log.debug(
        "store.upsert_document.embed",
        source_url=raw_doc.source_url,
        is_update=is_update,
        embed_chars=len(embed_text),
    )
    embedding = await embedder.embed_text(embed_text)
    embed_col = embedder.embedding_column  # "embedding" or "embedding_768"

    doc_id = uuid.uuid4() if existing is None else existing.id

    # For sentence_transformers (embed_col="embedding"): embedding_768 gets None.
    # For Ollama (embed_col="embedding_768"): embedding gets a zero placeholder so
    # the NOT NULL constraint is satisfied; the retriever filters this column out.
    insert_values: dict = {
        "id": doc_id,
        "source_url": raw_doc.source_url,
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
        "embedding_model": embedder.model_name,
        embed_col: embedding,
    }
    if embed_col != "embedding":
        insert_values["embedding"] = [0.0] * 384

    update_values: dict = {
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
        "embedding_model": embedder.model_name,
        embed_col: embedding,
    }

    stmt = (
        pg_insert(DocumentRow)
        .values(**insert_values)
        .on_conflict_do_update(
            index_elements=["source_url"],
            set_=update_values,
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
            linked_at=datetime.now(tz=UTC),
        )
        .on_conflict_do_nothing()
    )
    await session.execute(stmt)
