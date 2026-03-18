"""Unit tests for freqpred/rag/retriever.py.

All database and embedder calls are mocked — no external dependencies.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
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


def _make_session(rows: list[DocumentRow]) -> AsyncMock:
    """Mock AsyncSession whose execute() returns *rows* as scalars."""
    session = AsyncMock()
    scalars_result = MagicMock()
    scalars_result.all.return_value = rows
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_result
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

    await retrieve(session, embedder, "Will X happen?", category="politics")

    embedder.embed_text.assert_awaited_once_with("Will X happen?")


@pytest.mark.asyncio
async def test_retrieve_returns_documents():
    rows = [_make_row(), _make_row()]
    session = _make_session(rows)
    embedder = _make_embedder()

    docs = await retrieve(session, embedder, "question", category="politics")

    assert len(docs) == 2


@pytest.mark.asyncio
async def test_retrieve_maps_category_correctly():
    row = _make_row(category="economics")
    session = _make_session([row])
    embedder = _make_embedder()

    docs = await retrieve(session, embedder, "question", category="economics")

    assert docs[0].category == "economics"


@pytest.mark.asyncio
async def test_retrieve_empty_result():
    session = _make_session([])
    embedder = _make_embedder()

    docs = await retrieve(session, embedder, "question", category="politics")

    assert docs == []


@pytest.mark.asyncio
async def test_retrieve_respects_top_k():
    """top_k is passed to the query — verify the LIMIT clause fires by checking
    the SQL statement constructed (via the session.execute call argument)."""
    session = _make_session([])
    embedder = _make_embedder()

    await retrieve(session, embedder, "question", category="politics", top_k=5)

    # The stmt passed to session.execute should compile to include LIMIT 5.
    stmt = session.execute.call_args[0][0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "5" in compiled  # LIMIT 5 appears in the SQL


@pytest.mark.asyncio
async def test_retrieve_document_fields_mapped():
    row = _make_row()
    session = _make_session([row])
    embedder = _make_embedder()

    docs = await retrieve(session, embedder, "question", category="politics")
    doc = docs[0]

    assert doc.id == str(row.id)
    assert doc.source_url == row.source_url
    assert doc.title == row.title
    assert doc.body == row.body
    assert doc.category == row.category
    assert doc.embedding == FAKE_EMBEDDING
    assert doc.embedding_model == "all-MiniLM-L6-v2"
    assert doc.summary is None
