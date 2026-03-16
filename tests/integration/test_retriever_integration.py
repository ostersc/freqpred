"""Integration tests for freqpred/rag/retriever.py.

Requires a running Postgres + pgvector (docker-compose up -d db).
Uses the freqpred_test database (never production).

Run with:
    DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" \
        uv run pytest tests/integration/test_retriever_integration.py
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    "freqpred_test" not in os.environ.get("DATABASE_URL", ""),
    reason="Integration tests require DATABASE_URL pointing to freqpred_test",
)

from freqpred.db import Base, make_engine, make_session_factory
from freqpred.rag.models import DocumentRow
from freqpred.rag.retriever import compute_retrieval_hash, retrieve
from freqpred.rag.embedder import VoyageEmbedder

# Import all models so Base.metadata is fully populated before create_all.
import freqpred.markets.models  # noqa: F401
import freqpred.signal.models   # noqa: F401
import freqpred.llm.models      # noqa: F401

NOW = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(DATABASE_URL)
    async with eng.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    factory = make_session_factory(engine)
    async with factory() as sess:
        yield sess


def _make_embedder_for_question(question_embedding: list[float]) -> MagicMock:
    """Embedder whose embed_text returns the given vector for the query."""
    embedder = MagicMock(spec=VoyageEmbedder)
    embedder.embed_text = AsyncMock(return_value=question_embedding)
    return embedder


def _unit_vector(dim: int, hot_index: int) -> list[float]:
    """Return a unit vector with 1.0 at *hot_index*, 0 elsewhere.

    Cosine similarity between two such vectors is 1 if same index, 0 if different.
    """
    v = [0.0] * dim
    v[hot_index] = 1.0
    return v


async def _insert_doc(
    session,
    category: str,
    embedding: list[float],
    published_at: datetime = NOW,
    title: str = "Test Article",
) -> DocumentRow:
    row = DocumentRow(
        id=uuid.uuid4(),
        source_url=f"https://example.com/{uuid.uuid4()}",
        content_hash="abc123",
        title=title,
        body="Article body text.",
        source_type="news",
        source_name="Reuters",
        category=category,
        tags=["test"],
        published_at=published_at,
        fetched_at=NOW,
        embedding=embedding,
        embedding_model="voyage-3",
        summary=None,
    )
    session.add(row)
    await session.flush()
    return row


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_returns_correct_category_only(session):
    """Inserting 20 docs across 2 categories — retrieval returns only the
    requested category and excludes the other."""
    dim = 1024

    # Insert 10 politics docs and 10 economics docs, all with same embedding
    shared_embedding = [0.1] * dim
    for i in range(10):
        await _insert_doc(session, category="politics", embedding=shared_embedding)
    for i in range(10):
        await _insert_doc(session, category="economics", embedding=shared_embedding)
    await session.flush()

    embedder = _make_embedder_for_question(shared_embedding)
    docs = await retrieve(session, embedder, "political event?", category="politics", top_k=20)

    assert len(docs) == 10
    assert all(d.category == "politics" for d in docs)


@pytest.mark.asyncio
async def test_retrieval_ranked_by_relevance(session):
    """The most similar document (cosine distance ≈ 0) must rank first."""
    dim = 1024
    # Two distinct embeddings: index 0 hot vs index 1 hot
    emb_a = _unit_vector(dim, 0)
    emb_b = _unit_vector(dim, 1)

    doc_a = await _insert_doc(session, category="politics", embedding=emb_a, title="Doc A")
    doc_b = await _insert_doc(session, category="politics", embedding=emb_b, title="Doc B")
    await session.flush()

    # Query vector identical to emb_a → doc_a must rank first
    embedder = _make_embedder_for_question(emb_a)
    docs = await retrieve(session, embedder, "question", category="politics", top_k=10)

    assert len(docs) == 2
    assert docs[0].id == str(doc_a.id)
    assert docs[1].id == str(doc_b.id)


@pytest.mark.asyncio
async def test_retrieval_excludes_old_documents(session):
    """Documents older than max_age_days must not appear in results."""
    dim = 1024
    shared_embedding = [0.1] * dim
    old_date = NOW - timedelta(days=60)
    recent_date = NOW - timedelta(days=5)

    await _insert_doc(session, category="politics", embedding=shared_embedding,
                      published_at=old_date, title="Old Doc")
    recent_row = await _insert_doc(session, category="politics", embedding=shared_embedding,
                                   published_at=recent_date, title="Recent Doc")
    await session.flush()

    embedder = _make_embedder_for_question(shared_embedding)
    docs = await retrieve(session, embedder, "question", category="politics",
                          top_k=10, max_age_days=30)

    assert len(docs) == 1
    assert docs[0].id == str(recent_row.id)


@pytest.mark.asyncio
async def test_retrieval_top_k_limits_results(session):
    """top_k must cap the result set even when more matching docs exist."""
    dim = 1024
    shared_embedding = [0.1] * dim
    for _ in range(15):
        await _insert_doc(session, category="politics", embedding=shared_embedding)
    await session.flush()

    embedder = _make_embedder_for_question(shared_embedding)
    docs = await retrieve(session, embedder, "question", category="politics", top_k=5)

    assert len(docs) == 5


@pytest.mark.asyncio
async def test_compute_retrieval_hash_deterministic_across_calls():
    ids = [str(uuid.uuid4()) for _ in range(5)]
    h1 = compute_retrieval_hash(ids)
    h2 = compute_retrieval_hash(list(reversed(ids)))
    assert h1 == h2


@pytest.mark.asyncio
async def test_retrieval_empty_category_returns_empty(session):
    """Querying a category with no documents returns an empty list."""
    dim = 1024
    await _insert_doc(session, category="politics", embedding=[0.1] * dim)
    await session.flush()

    embedder = _make_embedder_for_question([0.1] * dim)
    docs = await retrieve(session, embedder, "question", category="sports", top_k=10)

    assert docs == []
