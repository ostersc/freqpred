"""Integration tests for freqpred/markets/repository.py.

Requires a running Postgres (docker-compose up -d db).
Uses the freqpred_test database (never production).

Run with:
    DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" \
        uv run pytest tests/integration/test_market_repository_integration.py
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select, text

pytestmark = pytest.mark.skipif(
    "freqpred_test" not in os.environ.get("DATABASE_URL", ""),
    reason="Integration tests require DATABASE_URL pointing to freqpred_test",
)

from freqpred.db import Base, make_engine, make_session_factory
from freqpred.markets.models import Market, MarketRow
from freqpred.markets.repository import upsert_market, upsert_markets

# Import all models so Base.metadata is fully populated before create_all.
import freqpred.markets.models  # noqa: F401
import freqpred.metrics.models  # noqa: F401
import freqpred.signal.models   # noqa: F401
import freqpred.rag.models      # noqa: F401
import freqpred.llm.models      # noqa: F401

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test",
)
NOW = datetime(2026, 3, 16, 12, 0, 0, tzinfo=UTC)


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


def _make_market(**overrides) -> Market:
    defaults = dict(
        id="KXPRES-25-DEM",
        platform="kalshi",
        question="Will the Dem candidate win?",
        category="politics",
        close_time=NOW + timedelta(days=30),
        yes_bid=0.45,
        yes_ask=0.47,
        mid_price=0.46,
        volume_24h=1000.0,
        open_interest=5000.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
        metadata={"event_ticker": "KXPRES-25", "subtitle": "2026 election"},
    )
    return Market(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_new_market_persists_all_fields(session) -> None:
    """upsert_market inserts a new row with all fields including metadata."""
    market = _make_market()
    await upsert_market(session, market)

    result = await session.execute(select(MarketRow).where(MarketRow.id == market.id))
    row = result.scalar_one()

    assert row.id == market.id
    assert row.platform == market.platform
    assert row.question == market.question
    assert row.category == market.category
    assert row.yes_bid == market.yes_bid
    assert row.yes_ask == market.yes_ask
    assert row.mid_price == market.mid_price
    assert row.volume_24h == market.volume_24h
    assert row.open_interest == market.open_interest
    assert row.metadata_ == market.metadata


@pytest.mark.asyncio
async def test_metadata_column_roundtrip(session) -> None:
    """metadata JSON survives a write/read cycle without corruption.

    This is the regression test for the metadata_ / metadata column name
    collision bug (SQLAlchemy Base.metadata vs the markets.metadata column).
    """
    meta = {"event_ticker": "KXTECH-AI", "subtitle": "AI rank", "status": "active"}
    market = _make_market(id="KXTECH-AI-1", metadata=meta)
    await upsert_market(session, market)

    result = await session.execute(select(MarketRow).where(MarketRow.id == market.id))
    row = result.scalar_one()
    assert row.metadata_ == meta


@pytest.mark.asyncio
async def test_upsert_updates_existing_row(session) -> None:
    """Second upsert with same id updates question and prices."""
    market = _make_market()
    await upsert_market(session, market)

    updated = _make_market(question="Updated question?", yes_bid=0.50, yes_ask=0.52, mid_price=0.51)
    await upsert_market(session, updated)

    result = await session.execute(select(MarketRow).where(MarketRow.id == market.id))
    row = result.scalar_one()
    assert row.question == "Updated question?"
    assert row.yes_bid == 0.50


@pytest.mark.asyncio
async def test_price_updated_at_advances_on_price_change(session) -> None:
    """price_updated_at is bumped when yes_bid/yes_ask/mid_price changes."""
    market = _make_market()
    await upsert_market(session, market)

    result = await session.execute(select(MarketRow).where(MarketRow.id == market.id))
    original_price_updated_at = result.scalar_one().price_updated_at

    changed = _make_market(yes_bid=0.60, yes_ask=0.62, mid_price=0.61)
    await upsert_market(session, changed)

    result = await session.execute(select(MarketRow).where(MarketRow.id == market.id))
    row = result.scalar_one()
    assert row.price_updated_at > original_price_updated_at


@pytest.mark.asyncio
async def test_price_updated_at_unchanged_when_price_stable(session) -> None:
    """price_updated_at is NOT bumped when prices are identical."""
    market = _make_market()
    await upsert_market(session, market)

    result = await session.execute(select(MarketRow).where(MarketRow.id == market.id))
    original_price_updated_at = result.scalar_one().price_updated_at

    # Same prices, different metadata
    same_prices = _make_market(metadata={"event_ticker": "KXPRES-25", "subtitle": "updated"})
    await upsert_market(session, same_prices)

    result = await session.execute(select(MarketRow).where(MarketRow.id == market.id))
    row = result.scalar_one()
    assert row.price_updated_at == original_price_updated_at


@pytest.mark.asyncio
async def test_upsert_markets_batch(session) -> None:
    """upsert_markets writes all markets and returns the correct count."""
    markets = [_make_market(id=f"KXPRES-25-{i}") for i in range(5)]
    written = await upsert_markets(session, markets)
    assert written == 5

    result = await session.execute(
        select(MarketRow).where(MarketRow.id.in_([m.id for m in markets]))
    )
    rows = result.scalars().all()
    assert len(rows) == 5
