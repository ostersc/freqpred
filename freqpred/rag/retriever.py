"""Vector search against the Document store (pgvector)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Protocol

import structlog
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.rag.models import Document, DocumentRow

log = structlog.get_logger()


class Embedder(Protocol):
    async def embed_text(self, text: str) -> list[float]: ...


async def retrieve(
    session: AsyncSession,
    embedder: Embedder,
    question: str,
    category: str,
    top_k: int = 10,
    max_age_days: int = 30,
) -> list[tuple[Document, float]]:
    """Embed *question* and return the top-K most relevant documents with cosine similarity scores.

    Filters by *category* and *published_at* recency, then ranks by cosine
    similarity to the question embedding.  Results are sorted most-similar first.

    Returns a list of (Document, similarity_score) tuples where similarity_score
    is in [0.0, 1.0]: 1.0 = identical, 0.0 = maximally dissimilar.
    """
    query_vector = await embedder.embed_text(question)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    distance_col = DocumentRow.embedding.cosine_distance(query_vector).label(
        "cosine_distance"
    )
    stmt = (
        select(DocumentRow, distance_col)
        .where(
            and_(
                DocumentRow.category == category,
                DocumentRow.published_at >= cutoff,
            )
        )
        .order_by(distance_col)
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    log.debug(
        "rag.retrieve",
        category=category,
        top_k=top_k,
        max_age_days=max_age_days,
        returned=len(rows),
    )
    return [(_row_to_document(row), 1.0 - float(distance)) for row, distance in rows]


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
