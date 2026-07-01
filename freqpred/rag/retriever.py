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

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Protocol

import structlog
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.rag.models import Document, DocumentMarketLinkRow, DocumentRow

log = structlog.get_logger()

_VECTOR_WEIGHT = 0.7  # weight for cosine similarity; (1 - this) goes to BM25


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product ≈ cosine similarity for unit-norm embeddings."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class Embedder(Protocol):
    model_name: str
    max_embed_chars: int
    embedding_column: str  # "embedding" (384-dim) or "embedding_768" (768-dim)

    async def embed_text(self, text: str) -> list[float]: ...


async def retrieve(
    session: AsyncSession,
    embedder: Embedder,
    question: str,
    market_id: str,
    top_k: int = 10,
    max_age_days: int = 30,
    vector_weight: float = _VECTOR_WEIGHT,
    now: datetime | None = None,
    catalyst_queries: list[str] | None = None,
) -> list[tuple[Document, float]]:
    """Return the top-K most relevant documents for *market_id* using hybrid scoring.

    Scope: only documents linked to *market_id* in document_market_links
    (written at ingestion time). Documents outside that set are never considered.

    *now* is the reference time used to compute the max_age_days cutoff. Defaults
    to the real wall-clock in UTC; tests can pin it for deterministic cutoffs.

    When *catalyst_queries* is provided, at least ``top_k // 2`` slots are
    guaranteed to market-question-ranked docs. The remaining slots are filled by
    taking the top-1 doc per catalyst query (not already in the core set), ranked
    by cosine similarity, up to ``top_k - top_k // 2`` docs. Any unused
    catalyst slots are back-filled with the next-best market-question docs so the
    returned list always reaches ``top_k`` when enough candidates exist.

    Returns a list of (Document, blended_score) tuples sorted best-first.
    blended_score is in [0.0, 1.0] after normalisation.
    """
    query_vector = await embedder.embed_text(question)
    reference = now if now is not None else datetime.now(UTC)
    cutoff = reference - timedelta(days=max_age_days)

    embed_col = embedder.embedding_column  # "embedding" or "embedding_768"
    embed_attr = getattr(DocumentRow, embed_col)

    # Subquery: distinct document IDs linked to this market.
    linked_ids_sq = (
        select(DocumentMarketLinkRow.document_id)
        .where(DocumentMarketLinkRow.market_id == market_id)
        .distinct()
        .subquery()
    )

    distance_col = embed_attr.cosine_distance(query_vector).label("cosine_distance")
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
        .where(embed_attr.is_not(None))  # exclude docs not yet reindexed for this backend
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
        for (row, _, _), nc, nb in zip(candidates, norm_cosine, norm_bm25, strict=True)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Slot budget: at least half reserved for market-question-ranked docs.
    core_min = top_k // 2
    supplemental_max = top_k - core_min

    if catalyst_queries and rows:
        # Guaranteed core: top core_min by market-question blended score.
        core = scored[:core_min]
        core_ids: set[str] = {str(row.id) for row, _ in core}
        remaining = [row for row, _, _ in candidates if str(row.id) not in core_ids]

        # Embed all catalyst query texts in parallel.
        cat_vecs = await asyncio.gather(
            *[embedder.embed_text(q) for q in catalyst_queries]
        )

        # Top-1 per catalyst query, deduped (keep highest sim when multiple queries match same doc).
        supplemental: dict[str, tuple[DocumentRow, float]] = {}
        for cat_vec in cat_vecs:
            best_row: DocumentRow | None = None
            best_sim = -1.0
            for row in remaining:
                sim = _dot(list(getattr(row, embed_col)), cat_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_row = row
            if best_row is not None:
                rid = str(best_row.id)
                if rid not in supplemental or best_sim > supplemental[rid][1]:
                    supplemental[rid] = (best_row, best_sim)

        # Rank-select: take at most supplemental_max catalyst docs by cosine sim.
        extra: list[tuple[DocumentRow, float]] = sorted(
            supplemental.values(), key=lambda x: x[1], reverse=True
        )[:supplemental_max]

        # Back-fill unused catalyst slots with next-best market-question docs.
        leftover_slots = supplemental_max - len(extra)
        if leftover_slots > 0:
            extra_ids = {str(r.id) for r, _ in extra}
            extra_core = [
                (row, sc) for row, sc in scored[core_min:]
                if str(row.id) not in extra_ids
            ][:leftover_slots]
            extra = extra + extra_core

        top = core + extra
    else:
        top = scored[:top_k]

    log.debug(
        "rag.retrieve",
        market_id=market_id,
        candidates=len(rows),
        returned=len(top),
        max_age_days=max_age_days,
        catalyst_count=len(catalyst_queries) if catalyst_queries else 0,
    )
    return [(_row_to_document(row, embed_col), score) for row, score in top]


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


def _row_to_document(row: DocumentRow, embed_col: str = "embedding") -> Document:
    raw_vec = getattr(row, embed_col)
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
        embedding=list(raw_vec) if raw_vec is not None else [],
        embedding_model=row.embedding_model,
        summary=row.summary,
    )
