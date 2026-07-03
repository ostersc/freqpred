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
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text

import freqpred.llm.models  # noqa: F401

# Import all models so Base.metadata is fully populated before create_all.
import freqpred.markets.models  # noqa: F401
import freqpred.metrics.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.db import Base, make_engine, make_session_factory
from freqpred.markets.models import MarketRow
from freqpred.rag.embedder import LocalEmbedder
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.rag.retriever import compute_retrieval_hash, retrieve

NOW = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
FUTURE = NOW + timedelta(days=30)
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
    embedder = MagicMock(spec=LocalEmbedder)
    embedder.embed_text = AsyncMock(return_value=question_embedding)
    embedder.embedding_column = "embedding"
    return embedder


def _unit_vector(dim: int, hot_index: int) -> list[float]:
    """Return a unit vector with 1.0 at *hot_index*, 0 elsewhere.

    Cosine similarity between two such vectors is 1 if same index, 0 if different.
    """
    v = [0.0] * dim
    v[hot_index] = 1.0
    return v


async def _seed_market(session, market_id: str, category: str = "politics") -> MarketRow:
    row = MarketRow(
        id=market_id,
        platform="kalshi",
        question=f"Test market {market_id}?",
        category=category,
        close_time=FUTURE,
        yes_bid=0.40,
        yes_ask=0.44,
        mid_price=0.42,
        volume_24h=1000.0,
        open_interest=500.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
        metadata_={},
    )
    session.add(row)
    await session.flush()
    return row


async def _insert_doc(
    session,
    market_id: str,
    embedding: list[float],
    category: str = "politics",
    published_at: datetime = NOW,
    title: str = "Test Article",
    embedding_768: list[float] | None = None,
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
        embedding_768=embedding_768,
        embedding_model="all-MiniLM-L6-v2",
        summary=None,
    )
    session.add(row)
    await session.flush()
    link = DocumentMarketLinkRow(
        id=uuid.uuid4(),
        document_id=row.id,
        market_id=market_id,
        signal_id=None,
        relevance_score=0.0,
        linked_at=NOW,
    )
    session.add(link)
    await session.flush()
    return row


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_returns_correct_market_only(session):
    """Docs linked to market_A must not appear when querying market_B."""
    dim = 384
    shared_embedding = [0.1] * dim

    await _seed_market(session, "RETV-MKT-A", category="politics")
    await _seed_market(session, "RETV-MKT-B", category="economics")

    for _ in range(10):
        await _insert_doc(session, market_id="RETV-MKT-A", embedding=shared_embedding, category="politics")
    for _ in range(10):
        await _insert_doc(session, market_id="RETV-MKT-B", embedding=shared_embedding, category="economics")
    await session.flush()

    embedder = _make_embedder_for_question(shared_embedding)
    docs = await retrieve(session, embedder, "political event?", market_id="RETV-MKT-A",
                          top_k=20, now=NOW)

    assert len(docs) == 10
    assert all(d.category == "politics" for d, _ in docs)


@pytest.mark.asyncio
async def test_retrieval_ranked_by_relevance(session):
    """The most similar document (cosine distance ≈ 0) must rank first."""
    dim = 384
    emb_a = _unit_vector(dim, 0)
    emb_b = _unit_vector(dim, 1)

    await _seed_market(session, "RETV-RANK-MKT")
    doc_a = await _insert_doc(session, market_id="RETV-RANK-MKT", embedding=emb_a, title="Doc A")
    doc_b = await _insert_doc(session, market_id="RETV-RANK-MKT", embedding=emb_b, title="Doc B")
    await session.flush()

    embedder = _make_embedder_for_question(emb_a)
    docs = await retrieve(session, embedder, "question", market_id="RETV-RANK-MKT",
                          top_k=10, now=NOW)

    assert len(docs) == 2
    assert docs[0][0].id == str(doc_a.id)
    assert docs[1][0].id == str(doc_b.id)


@pytest.mark.asyncio
async def test_retrieval_excludes_old_documents(session):
    """Documents older than max_age_days must not appear in results."""
    dim = 384
    shared_embedding = [0.1] * dim
    old_date = NOW - timedelta(days=60)
    recent_date = NOW - timedelta(days=5)

    await _seed_market(session, "RETV-AGE-MKT")
    await _insert_doc(session, market_id="RETV-AGE-MKT", embedding=shared_embedding,
                      published_at=old_date, title="Old Doc")
    recent_row = await _insert_doc(session, market_id="RETV-AGE-MKT", embedding=shared_embedding,
                                   published_at=recent_date, title="Recent Doc")
    await session.flush()

    embedder = _make_embedder_for_question(shared_embedding)
    # Pin *now* so the max_age_days cutoff is deterministic regardless of when
    # the test runs — otherwise wall-clock drift makes both docs age out.
    docs = await retrieve(session, embedder, "question", market_id="RETV-AGE-MKT",
                          top_k=10, max_age_days=30, now=NOW)

    assert len(docs) == 1
    assert docs[0][0].id == str(recent_row.id)


@pytest.mark.asyncio
async def test_retrieval_top_k_limits_results(session):
    """top_k must cap the result set even when more matching docs exist."""
    dim = 384
    shared_embedding = [0.1] * dim

    await _seed_market(session, "RETV-TOPK-MKT")
    for _ in range(15):
        await _insert_doc(session, market_id="RETV-TOPK-MKT", embedding=shared_embedding)
    await session.flush()

    embedder = _make_embedder_for_question(shared_embedding)
    docs = await retrieve(session, embedder, "question", market_id="RETV-TOPK-MKT",
                          top_k=5, now=NOW)

    assert len(docs) == 5


@pytest.mark.asyncio
async def test_compute_retrieval_hash_deterministic_across_calls():
    ids = [str(uuid.uuid4()) for _ in range(5)]
    h1 = compute_retrieval_hash(ids)
    h2 = compute_retrieval_hash(list(reversed(ids)))
    assert h1 == h2


@pytest.mark.asyncio
async def test_retrieval_unlinked_market_returns_empty(session):
    """Querying a market with no linked documents returns an empty list."""
    dim = 384

    await _seed_market(session, "RETV-LINKED-MKT")
    await _seed_market(session, "RETV-EMPTY-MKT")
    await _insert_doc(session, market_id="RETV-LINKED-MKT", embedding=[0.1] * dim)
    await session.flush()

    embedder = _make_embedder_for_question([0.1] * dim)
    docs = await retrieve(session, embedder, "question", market_id="RETV-EMPTY-MKT",
                          top_k=10, now=NOW)

    assert docs == []


@pytest.mark.asyncio
async def test_retrieval_uses_configured_embedding_768_column(session):
    """With a 768-dim embedder backend, retrieve() must score against
    embedding_768 and exclude docs not yet reindexed for that backend.

    Guards the dual-column setup (embedding = 384 sentence-transformers,
    embedding_768 = Ollama): a regression that reads the wrong column would
    silently return nothing (or mis-rank) in production while every 384-based
    test stays green.
    """
    await _seed_market(session, "RETV-768-MKT")

    # All docs share an identical 384 vector — if retrieve() wrongly scores
    # against `embedding`, ranking collapses and the stale doc leaks in.
    same_384 = _unit_vector(384, 5)
    doc_match = await _insert_doc(
        session, market_id="RETV-768-MKT", embedding=same_384,
        embedding_768=_unit_vector(768, 0), title="match",
    )
    doc_other = await _insert_doc(
        session, market_id="RETV-768-MKT", embedding=same_384,
        embedding_768=_unit_vector(768, 3), title="other",
    )
    doc_stale = await _insert_doc(  # not yet reindexed for the 768 backend
        session, market_id="RETV-768-MKT", embedding=same_384, title="stale",
    )
    await session.flush()

    embedder = MagicMock(spec=LocalEmbedder)
    embedder.embed_text = AsyncMock(return_value=_unit_vector(768, 0))
    embedder.embedding_column = "embedding_768"

    docs = await retrieve(session, embedder, "question", market_id="RETV-768-MKT",
                          top_k=10, now=NOW)

    returned_ids = [d.id for d, _ in docs]
    assert str(doc_stale.id) not in returned_ids
    assert set(returned_ids) == {str(doc_match.id), str(doc_other.id)}
    assert returned_ids[0] == str(doc_match.id)  # ranked by 768 cosine similarity
