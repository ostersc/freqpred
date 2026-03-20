"""Unit tests for freqpred/rag/retriever.py.

All database and embedder calls are mocked — no external dependencies.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.rag.retriever import compute_retrieval_hash, retrieve
from freqpred.rag.models import DocumentRow

# Import all ORM models so SQLAlchemy can resolve cross-module relationships
# (e.g. DocumentMarketLinkRow → SignalRow) before any mapper is instantiated.
import freqpred.markets.models  # noqa: F401
import freqpred.signal.models   # noqa: F401
import freqpred.llm.models      # noqa: F401


NOW = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
FAKE_EMBEDDING = [0.1] * 384
MARKET_ID = "KXTEST-26MAR20-FOO"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    category: str = "politics",
    published_at: datetime = NOW,
) -> DocumentRow:
    return DocumentRow(
        id=uuid.uuid4(),
        source_url=f"https://example.com/{uuid.uuid4()}",
        content_hash="abc123",
        title="Test Article",
        body="Body text.",
        source_type="news",
        source_name="Reuters",
        category=category,
        tags=["test"],
        published_at=published_at,
        fetched_at=NOW,
        embedding=FAKE_EMBEDDING,
        embedding_model="all-MiniLM-L6-v2",
        summary=None,
    )


def _make_session(
    rows: list[DocumentRow],
    distances: list[float] | None = None,
    bm25_scores: list[float] | None = None,
) -> AsyncMock:
    """Mock AsyncSession whose execute() returns (DocumentRow, cosine_distance, bm25_score) tuples."""
    if distances is None:
        distances = [0.0] * len(rows)
    if bm25_scores is None:
        bm25_scores = [0.5] * len(rows)
    session = AsyncMock()
    execute_result = MagicMock()
    execute_result.all.return_value = list(zip(rows, distances, bm25_scores))
    session.execute = AsyncMock(return_value=execute_result)
    return session


def _make_embedder(embedding: list[float] = FAKE_EMBEDDING) -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=embedding)
    return embedder


# ---------------------------------------------------------------------------
# compute_retrieval_hash
# ---------------------------------------------------------------------------


def test_hash_deterministic():
    ids = ["a", "b", "c"]
    assert compute_retrieval_hash(ids) == compute_retrieval_hash(ids)


def test_hash_order_independent():
    ids = ["c", "a", "b"]
    assert compute_retrieval_hash(ids) == compute_retrieval_hash(["a", "b", "c"])


def test_hash_different_ids_differ():
    assert compute_retrieval_hash(["x"]) != compute_retrieval_hash(["y"])


def test_hash_empty_list():
    h = compute_retrieval_hash([])
    assert isinstance(h, str) and len(h) == 64


def test_hash_length():
    assert len(compute_retrieval_hash(["a", "b"])) == 64


# ---------------------------------------------------------------------------
# retrieve — embedder interaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_calls_embedder_once():
    session = _make_session([])
    embedder = _make_embedder()

    await retrieve(session, embedder, "Will X happen?", market_id=MARKET_ID)

    embedder.embed_text.assert_awaited_once_with("Will X happen?")


@pytest.mark.asyncio
async def test_retrieve_returns_documents():
    rows = [_make_row(), _make_row()]
    session = _make_session(rows)
    embedder = _make_embedder()

    pairs = await retrieve(session, embedder, "question", market_id=MARKET_ID)

    assert len(pairs) == 2


@pytest.mark.asyncio
async def test_retrieve_empty_result():
    session = _make_session([])
    embedder = _make_embedder()

    pairs = await retrieve(session, embedder, "question", market_id=MARKET_ID)

    assert pairs == []


@pytest.mark.asyncio
async def test_retrieve_respects_top_k():
    """top_k slices the sorted results — more candidates than top_k yields top_k results."""
    rows = [_make_row() for _ in range(8)]
    distances = [float(i) / 10 for i in range(8)]
    session = _make_session(rows, distances=distances)
    embedder = _make_embedder()

    pairs = await retrieve(session, embedder, "question", market_id=MARKET_ID, top_k=3)

    assert len(pairs) == 3


@pytest.mark.asyncio
async def test_retrieve_document_fields_mapped():
    row = _make_row()
    session = _make_session([row])
    embedder = _make_embedder()

    pairs = await retrieve(session, embedder, "question", market_id=MARKET_ID)
    doc, _ = pairs[0]

    assert doc.id == str(row.id)
    assert doc.source_url == row.source_url
    assert doc.title == row.title
    assert doc.body == row.body
    assert doc.category == row.category
    assert doc.embedding == FAKE_EMBEDDING
    assert doc.embedding_model == "all-MiniLM-L6-v2"
    assert doc.summary is None


# ---------------------------------------------------------------------------
# retrieve — hybrid scoring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relevance_scores_are_in_unit_interval():
    """All blended scores must be in [0.0, 1.0]."""
    rows = [_make_row() for _ in range(4)]
    distances = [0.0, 0.5, 0.99, 1.0]
    bm25_scores = [0.8, 0.4, 0.1, 0.0]
    session = _make_session(rows, distances=distances, bm25_scores=bm25_scores)
    embedder = _make_embedder()

    pairs = await retrieve(session, embedder, "question", market_id=MARKET_ID)

    for _, score in pairs:
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_higher_bm25_improves_rank():
    """A doc with identical cosine but higher BM25 should rank first."""
    row_a = _make_row()
    row_b = _make_row()
    # Same cosine distance, but row_b has a higher BM25 score.
    session = _make_session([row_a, row_b], distances=[0.3, 0.3], bm25_scores=[0.1, 0.9])
    embedder = _make_embedder()

    pairs = await retrieve(session, embedder, "question", market_id=MARKET_ID)

    assert pairs[0][0].id == str(row_b.id)


@pytest.mark.asyncio
async def test_results_sorted_best_first():
    """Results are returned in descending blended score order."""
    rows = [_make_row() for _ in range(3)]
    # Intentionally varying distances and BM25 so scores differ.
    session = _make_session(rows, distances=[0.9, 0.1, 0.5], bm25_scores=[0.0, 1.0, 0.5])
    embedder = _make_embedder()

    pairs = await retrieve(session, embedder, "question", market_id=MARKET_ID)

    scores = [s for _, s in pairs]
    assert scores == sorted(scores, reverse=True)
