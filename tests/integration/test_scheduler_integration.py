"""Integration test for freqpred/ingestion/scheduler.py.

Requires a running Postgres + pgvector (docker-compose up -d db).
Uses the freqpred_test database (never production).

Scenario: seed one market with one active CatalystRun + two CatalystQuery
rows, run one scheduler cycle with mocked fetchers and embedder, then verify
that the document count in the DB increases.

Run with:
    DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" \
        uv run pytest tests/integration/test_scheduler_integration.py
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text

pytestmark = pytest.mark.skipif(
    "freqpred_test" not in os.environ.get("DATABASE_URL", ""),
    reason="Integration tests require DATABASE_URL pointing to freqpred_test",
)

from freqpred.db import Base, make_engine, make_session_factory
from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow
from freqpred.ingestion.scheduler import run_cycle
from freqpred.ingestion.store import RawDocument
from freqpred.markets.models import MarketRow
from freqpred.rag.models import DocumentRow

# Register all models with Base.metadata before create_all.
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models        # noqa: F401
import freqpred.signal.models     # noqa: F401

NOW = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
FUTURE = NOW + timedelta(days=30)
FAKE_EMBEDDING = [0.1] * 384
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


@pytest.fixture
def session_factory(engine):
    return make_session_factory(engine)


@pytest_asyncio.fixture
async def session(engine):
    factory = make_session_factory(engine)
    async with factory() as sess:
        yield sess


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)
    return embedder




def _make_raw_docs(urls: list[str]) -> list[RawDocument]:
    return [
        RawDocument(
            source_url=url,
            title=f"Article {i}",
            body=f"Body content for article {i}.",
            source_type="news",
            source_name="Reuters",
            category="economics",
            tags=[],
            published_at=NOW,
            fetched_at=NOW,
        )
        for i, url in enumerate(urls)
    ]


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


async def _seed_market(session, market_id: str = "SCHED-TEST-MKT") -> MarketRow:
    row = MarketRow(
        id=market_id,
        platform="kalshi",
        question="Will the Fed raise rates in March 2026?",
        category="economics",
        close_time=FUTURE,
        yes_bid=0.30,
        yes_ask=0.34,
        mid_price=0.32,
        volume_24h=5000.0,
        open_interest=1200.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
        metadata_={},
    )
    session.add(row)
    await session.flush()
    return row


async def _seed_catalyst_run(
    session, market_id: str, is_active: bool = True
) -> CatalystRunRow:
    run = CatalystRunRow(
        id=uuid.uuid4(),
        market_id=market_id,
        generation=1,
        llm_query_id=None,
        is_active=is_active,
    )
    session.add(run)
    await session.flush()
    return run


async def _seed_catalyst_queries(
    session, run_id: uuid.UUID, query_texts: list[str]
) -> list[CatalystQueryRow]:
    rows = []
    for text in query_texts:
        row = CatalystQueryRow(
            id=uuid.uuid4(),
            run_id=run_id,
            query_text=text,
        )
        session.add(row)
        rows.append(row)
    await session.flush()
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_cycle_increases_document_count(session, session_factory, mock_embedder):
    """Seeding a market with 1 active run + 2 queries then running one cycle
    must increase the documents table count."""
    market_id = "SCHED-TEST-MKT"
    await _seed_market(session, market_id)
    run = await _seed_catalyst_run(session, market_id, is_active=True)
    await _seed_catalyst_queries(
        session,
        run.id,
        ["Fed rate decision March 2026", "CPI inflation February 2026"],
    )
    await session.commit()

    # Count documents before the cycle.
    before = (await session.execute(select(func.count()).select_from(DocumentRow))).scalar()

    # Mock fetchers to return 2 docs for every call.
    doc_urls_q1 = ["https://reuters.com/fed-1", "https://bloomberg.com/fed-2"]
    doc_urls_q2 = ["https://reuters.com/cpi-1", "https://bloomberg.com/cpi-2"]

    call_count = 0

    async def fake_tavily_fetch(api_key: str, query: str, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_raw_docs(doc_urls_q1)
        return _make_raw_docs(doc_urls_q2)

    with (
        patch(
            "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
            side_effect=fake_tavily_fetch,
        ),
        patch(
            "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.gdelt_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        stats = await run_cycle(
            session_factory,
            mock_embedder,
            tavily_api_key="test-key",
        )

    after = (await session.execute(select(func.count()).select_from(DocumentRow))).scalar()

    assert after > before
    assert stats["docs_stored"] == 4  # 2 queries × 2 docs each
    assert stats["markets_processed"] == 1


@pytest.mark.asyncio
async def test_inactive_catalyst_run_market_excluded(session, session_factory, mock_embedder):
    """Markets whose latest CatalystRun has is_active=False must not be fetched."""
    market_id = "SCHED-INACTIVE-MKT"
    await _seed_market(session, market_id)
    run = await _seed_catalyst_run(session, market_id, is_active=False)
    await _seed_catalyst_queries(session, run.id, ["some query"])
    await session.commit()

    with (
        patch(
            "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_tavily,
        patch(
            "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.gdelt_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        stats = await run_cycle(session_factory, mock_embedder, tavily_api_key="key")

    # Fetcher must not have been called — market excluded
    mock_tavily.assert_not_called()
    assert stats["markets_processed"] == 0


@pytest.mark.asyncio
async def test_cycle_completes_without_error(session, session_factory, mock_embedder):
    """A successful cycle with active queries must complete and report stats."""
    market_id = "SCHED-BASIC-TEST"
    await _seed_market(session, market_id)
    run = await _seed_catalyst_run(session, market_id, is_active=True)
    await _seed_catalyst_queries(session, run.id, ["query one"])
    await session.commit()

    with (
        patch(
            "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.gdelt_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        stats = await run_cycle(session_factory, mock_embedder)

    assert stats["markets_processed"] == 1


@pytest.mark.asyncio
async def test_duplicate_docs_not_reembedded(session, session_factory, mock_embedder):
    """Running the scheduler twice with the same documents must not
    call embed_text a second time for identical content."""
    market_id = "SCHED-DEDUP-TEST"
    await _seed_market(session, market_id)
    run = await _seed_catalyst_run(session, market_id, is_active=True)
    await _seed_catalyst_queries(session, run.id, ["dedup query"])
    await session.commit()

    doc_urls = ["https://reuters.com/dedup-1"]

    with (
        patch(
            "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=_make_raw_docs(doc_urls),
        ),
        patch(
            "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "freqpred.ingestion.scheduler.gdelt_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        await run_cycle(session_factory, mock_embedder, tavily_api_key="key")
        first_embed_count = mock_embedder.embed_text.await_count

        await run_cycle(session_factory, mock_embedder, tavily_api_key="key")
        second_embed_count = mock_embedder.embed_text.await_count

    # Second run with same content must not trigger additional embeddings
    assert second_embed_count == first_embed_count


@pytest.mark.asyncio
async def test_tv_archive_fetcher_stores_transcript_docs(session, session_factory, mock_embedder):
    """When a catalyst query has a tv_query set, the scheduler calls the TV archive
    fetcher and stores the returned transcript clips as documents."""
    market_id = "SCHED-TV-TEST"
    await _seed_market(session, market_id)
    run = await _seed_catalyst_run(session, market_id, is_active=True)

    # Seed one query with a tv_query set (word-mention market pattern).
    tv_query_row = CatalystQueryRow(
        id=uuid.uuid4(),
        run_id=run.id,
        query_text='how often does Trump say "communist" on TV',
        tv_query='trump AND ("communist" OR "communism")',
    )
    session.add(tv_query_row)
    await session.flush()
    await session.commit()

    transcript_urls = [
        "https://archive.org/details/CNN_20260310/start/120/end/180?q=trump+communist",
        "https://archive.org/details/MSNBC_20260312/start/300/end/360?q=trump+communist",
    ]
    transcript_docs = [
        RawDocument(
            source_url=url,
            title=f"CNN NewsRoom : March {10 + i}, 2026",
            body=f"trump mentioned communism on air at this timestamp {i}.",
            source_type="tv_transcript",
            source_name="TVArchive",
            category="",
            tags=["trump", "communism"],
            published_at=NOW,
            fetched_at=NOW,
        )
        for i, url in enumerate(transcript_urls)
    ]

    before = (await session.execute(select(func.count()).select_from(DocumentRow))).scalar()

    with (
        patch("freqpred.ingestion.scheduler.tavily_fetcher.fetch", new_callable=AsyncMock, return_value=[]),
        patch("freqpred.ingestion.scheduler.newsapi_fetcher.fetch", new_callable=AsyncMock, return_value=[]),
        patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch", new_callable=AsyncMock, return_value=[]),
        patch("freqpred.ingestion.scheduler.gdelt_fetcher.fetch", new_callable=AsyncMock, return_value=[]),
        patch(
            "freqpred.ingestion.scheduler.tv_archive_fetcher.fetch",
            new_callable=AsyncMock,
            return_value=transcript_docs,
        ) as mock_tv,
    ):
        stats = await run_cycle(session_factory, mock_embedder)

    after = (await session.execute(select(func.count()).select_from(DocumentRow))).scalar()

    # TV archive fetcher was called with the Solr tv_query string.
    mock_tv.assert_called_once()
    assert mock_tv.call_args.kwargs["query"] == 'trump AND ("communist" OR "communism")'

    # Transcript clips were stored.
    assert after == before + 2
    assert stats["docs_stored"] == 2
