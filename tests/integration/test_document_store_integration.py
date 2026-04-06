"""Integration tests for freqpred/ingestion/store.py.

Requires a running Postgres + pgvector (docker-compose up -d db).
Uses the freqpred_test database (never production).

Run with:
    DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" \
        uv run pytest tests/integration/
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import select, text

# Skip entire module if DATABASE_URL not pointing at freqpred_test.
pytestmark = pytest.mark.skipif(
    "freqpred_test" not in os.environ.get("DATABASE_URL", ""),
    reason="Integration tests require DATABASE_URL pointing to freqpred_test",
)

from freqpred.db import Base, make_engine, make_session_factory
from freqpred.ingestion.store import RawDocument, upsert_document
from freqpred.rag.embedder import LocalEmbedder
from freqpred.rag.models import DocumentRow

# Import all models so Base.metadata is fully populated before create_all.
import freqpred.markets.models  # noqa: F401
import freqpred.signal.models   # noqa: F401
import freqpred.llm.models      # noqa: F401

NOW = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
FAKE_EMBEDDING = [0.1] * 384
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test",
)


# ---------------------------------------------------------------------------
# Fixtures — all function-scoped to stay on the same event loop as each test
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    """Create a fresh set of tables for each test, then drop them."""
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


@pytest.fixture
def mock_embedder():
    """Voyage AI embedder that returns a fixed vector without hitting the API."""
    embedder = MagicMock(spec=LocalEmbedder)
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)
    return embedder


def _make_raw_doc(url: str, body: str = "Article body.") -> RawDocument:
    return RawDocument(
        source_url=url,
        title="Test Article",
        body=body,
        source_type="news",
        source_name="Reuters",
        category="politics",
        tags=["test"],
        published_at=NOW,
        fetched_at=NOW,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_three_unique_urls_three_embed_calls(session, mock_embedder):
    """3 distinct URLs → 3 embedding calls, 3 rows inserted."""
    docs = [_make_raw_doc(f"https://example.com/article-{i}", body=f"Article body {i}.") for i in range(3)]
    for doc in docs:
        await upsert_document(session, mock_embedder, doc)
    await session.flush()

    assert mock_embedder.embed_text.await_count == 3

    result = await session.execute(
        select(DocumentRow).where(
            DocumentRow.source_url.in_([d.source_url for d in docs])
        )
    )
    rows = result.scalars().all()
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_same_url_twice_one_embed_call(session, mock_embedder):
    """Inserting the same URL twice with identical content → only 1 embed call."""
    url = "https://example.com/dedup-test"
    raw_doc = _make_raw_doc(url, body="Identical content.")

    await upsert_document(session, mock_embedder, raw_doc)
    await session.flush()
    first_call_count = mock_embedder.embed_text.await_count

    await upsert_document(session, mock_embedder, raw_doc)

    assert mock_embedder.embed_text.await_count == first_call_count  # no new call


@pytest.mark.asyncio
async def test_content_change_triggers_re_embed(session, mock_embedder):
    """Same URL but different body → re-embedding must occur."""
    url = "https://example.com/update-test"

    await upsert_document(session, mock_embedder, _make_raw_doc(url, body="Original."))
    await session.flush()
    assert mock_embedder.embed_text.await_count == 1

    await upsert_document(session, mock_embedder, _make_raw_doc(url, body="Updated content."))
    await session.flush()
    assert mock_embedder.embed_text.await_count == 2

    result = await session.execute(
        select(DocumentRow).where(DocumentRow.source_url == url)
    )
    rows = result.scalars().all()
    assert len(rows) == 1  # upserted, not duplicated


@pytest.mark.asyncio
async def test_embedding_dimension_matches_pgvector(session, mock_embedder):
    """Stored embedding must be 384-dimensional."""
    url = "https://example.com/dim-test"
    await upsert_document(session, mock_embedder, _make_raw_doc(url))
    await session.flush()

    result = await session.execute(
        select(DocumentRow).where(DocumentRow.source_url == url)
    )
    row = result.scalar_one()
    assert len(row.embedding) == 384
