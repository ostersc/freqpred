"""Hybrid vector + full-text search against the Document store (pgvector).

Retrieval is scoped to documents pre-linked to a specific market via
document_market_links (written at ingestion time). This ensures the retriever
only considers documents that were fetched because of that market's catalyst
queries, not the entire category.

For each candidate document two scores are computed:
  - cosine_sim  : 1 - cosine_distance (pgvector)
  - bm25        : ts_rank from Postgres full-text search (0–1 range)

Final score = vector_weight * norm(cosine_sim) + (1 - vector_weight) * norm(bm25)

Both scores are min-max normalised over the candidate set before blending so
neither dominates purely due to scale differences.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Protocol

import structlog
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.rag.models import Document, DocumentMarketLinkRow, DocumentRow

log = structlog.get_logger()

_VECTOR_WEIGHT = 0.7  # weight for cosine similarity; (1 - this) goes to BM25


class Embedder(Protocol):
    async def embed_text(self, text: str) -> list[float]: ...


async def retrieve(
    session: AsyncSession,
    embedder: Embedder,
    question: str,
    market_id: str,
    top_k: int = 10,
    max_age_days: int = 30,
    vector_weight: float = _VECTOR_WEIGHT,
) -> list[tuple[Document, float]]:
    """Return the top-K most relevant documents for *market_id* using hybrid scoring.

    Scope: only documents linked to *market_id* in document_market_links
    (written at ingestion time). Documents outside that set are never considered.

    Returns a list of (Document, blended_score) tuples sorted best-first.
    blended_score is in [0.0, 1.0] after normalisation.
    """
    query_vector = await embedder.embed_text(question)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    # Subquery: distinct document IDs linked to this market.
    linked_ids_sq = (
        select(DocumentMarketLinkRow.document_id)
        .where(DocumentMarketLinkRow.market_id == market_id)
        .distinct()
        .subquery()
    )

    distance_col = DocumentRow.embedding.cosine_distance(query_vector).label("cosine_distance")
    # Use summary for BM25 when present — summaries are generated with market-question
    # vocabulary so they score better against the market question than the raw body.
    # Use only the first line of the question to avoid boilerplate resolution criteria
    # inflating the tsquery with irrelevant terms (e.g. "market resolv accord rule").
    question_first_line = func.split_part(question, "\n", 1)
    bm25_col = func.ts_rank(
        func.to_tsvector(
            text("'english'"),
            DocumentRow.title + " " + func.coalesce(DocumentRow.summary, DocumentRow.body),
        ),
        func.plainto_tsquery(text("'english'"), question_first_line),
    ).label("bm25_score")

    stmt = (
        select(DocumentRow, distance_col, bm25_col)
        .join(linked_ids_sq, DocumentRow.id == linked_ids_sq.c.document_id)
        .where(DocumentRow.published_at >= cutoff)
    )

    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        log.debug("rag.retrieve.empty", market_id=market_id)
        return []

    # Compute cosine similarities and collect raw scores.
    candidates = [
        (row, 1.0 - float(distance), float(bm25))
        for row, distance, bm25 in rows
    ]

    # Min-max normalise each score over the candidate set so they're on the
    # same [0, 1] scale before blending.
    cosine_vals = [c for _, c, _ in candidates]
    bm25_vals = [b for _, _, b in candidates]

    def _normalise(vals: list[float]) -> list[float]:
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [1.0] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    norm_cosine = _normalise(cosine_vals)
    norm_bm25 = _normalise(bm25_vals)

    scored = [
        (row, vector_weight * nc + (1.0 - vector_weight) * nb)
        for (row, _, _), nc, nb in zip(candidates, norm_cosine, norm_bm25)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    log.debug(
        "rag.retrieve",
        market_id=market_id,
        candidates=len(rows),
        returned=len(top),
        max_age_days=max_age_days,
    )
    return [(_row_to_document(row), score) for row, score in top]


def compute_retrieval_hash(doc_ids: list[str]) -> str:
    """Deterministic SHA-256 hash of a set of document IDs.

    The input list is sorted before hashing so the result is independent of
    input order — same IDs always produce the same hash.
    """
    payload = ",".join(sorted(doc_ids))
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_document(row: DocumentRow) -> Document:
    return Document(
        id=str(row.id),
        source_url=row.source_url,
        content_hash=row.content_hash,
        title=row.title,
        body=row.body,
        source_type=row.source_type,
        source_name=row.source_name,
        category=row.category,
        tags=list(row.tags),
        published_at=row.published_at,
        fetched_at=row.fetched_at,
        embedding=list(row.embedding),
        embedding_model=row.embedding_model,
        summary=row.summary,
    )
