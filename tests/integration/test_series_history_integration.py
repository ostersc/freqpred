"""Integration tests for series option history refresh and lookup.

Requires a running Postgres (docker-compose up -d db).
Uses the freqpred_test database (never production).

These tests exercise the real DB path end-to-end, catching bugs that unit tests
with mocked sessions cannot: ORM identity-map issues, asyncpg type binding,
upsert conflict semantics, and the full read path in get_series_history_for_market.

Run with:
    DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" \
        uv run pytest tests/integration/test_series_history_integration.py
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text

pytestmark = pytest.mark.skipif(
    "freqpred_test" not in os.environ.get("DATABASE_URL", ""),
    reason="Integration tests require DATABASE_URL pointing to freqpred_test",
)

from freqpred.db import Base, make_engine, make_session_factory
from freqpred.markets.models import MarketRow
from freqpred.metrics.models import SeriesOptionHistoryRow
from freqpred.metrics.series_history import (
    _SERIES_AGGREGATE_CODE,
    get_series_history_for_market,
    refresh_series_history,
)
from freqpred.signal.models import SignalRow

# Register all ORM models before create_all.
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models        # noqa: F401
import freqpred.metrics.models    # noqa: F401
import freqpred.runtime.models    # noqa: F401
import freqpred.signal.models     # noqa: F401

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test",
)

NOW = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
FUTURE = NOW + timedelta(days=30)


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


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


async def _seed_market(
    session,
    market_id: str,
    series_ticker: str | None = None,
) -> MarketRow:
    row = MarketRow(
        id=market_id,
        platform="kalshi",
        question="Will Trump say Rigged on May 18?",
        category="politics",
        series_ticker=series_ticker,
        close_time=FUTURE,
        yes_bid=0.65,
        yes_ask=0.70,
        mid_price=0.67,
        volume_24h=500.0,
        open_interest=200.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
        metadata_={},
    )
    session.add(row)
    await session.flush()
    return row


async def _seed_signal(
    session,
    market_id: str,
    created_at: datetime | None = None,
) -> SignalRow:
    row = SignalRow(
        id=uuid.uuid4(),
        market_id=market_id,
        estimated_probability=0.65,
        confidence=0.70,
        edge=0.05,
        market_mid_at_signal=0.60,
        direction="yes",
        reasoning="Test reasoning.",
        sources=[],
        retrieval_hash="abc123",
        model_used="claude-sonnet-4-6",
        prompt_version="signal-v7",
        trigger="test",
        raw_context="",
        created_at=created_at or NOW,
    )
    session.add(row)
    await session.flush()
    return row


def _make_settled_market(ticker: str, result: str, yes_sub_title: str = "") -> dict:
    return {"ticker": ticker, "result": result, "yes_sub_title": yes_sub_title}


# ---------------------------------------------------------------------------
# Tests: refresh_series_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_writes_rows_to_db(session, session_factory):
    """refresh_series_history upserts per-option and aggregate rows into the DB."""
    market_id = "KXTRUMPSAY-26MAY18-RIGG"
    await _seed_market(session, market_id, series_ticker="KXTRUMPSAY")
    await _seed_signal(session, market_id, created_at=NOW - timedelta(days=1))
    await session.commit()

    settled = [
        _make_settled_market("KXTRUMPSAY-26MAY11-RIGG", "yes", "Rigged"),
        _make_settled_market("KXTRUMPSAY-26MAY11-RIGG", "yes", "Rigged"),
        _make_settled_market("KXTRUMPSAY-26MAY11-BARA", "no", "Obama"),
        _make_settled_market("KXTRUMPSAY-26MAY11-URAN", "no", "Uranium"),
    ]
    kalshi = AsyncMock()
    kalshi.get_series_settled_history = AsyncMock(return_value=settled)

    async with session_factory() as s:
        rows_upserted = await refresh_series_history(s, kalshi, lookback_days=7, now=NOW)
        await s.commit()

    assert rows_upserted > 0

    async with session_factory() as s:
        result = await s.execute(
            select(SeriesOptionHistoryRow).where(
                SeriesOptionHistoryRow.series_ticker == "KXTRUMPSAY"
            )
        )
        rows = {r.option_code: r for r in result.scalars()}

    # Per-option rows: RIGG, BARA, URAN + aggregate
    assert _SERIES_AGGREGATE_CODE in rows
    assert "RIGG" in rows
    assert "BARA" in rows
    assert "URAN" in rows

    agg = rows[_SERIES_AGGREGATE_CODE]
    assert agg.yes_count == 2  # 2 RIGG yes
    assert agg.no_count == 2   # BARA no + URAN no

    rigg = rows["RIGG"]
    assert rigg.yes_count == 2
    assert rigg.no_count == 0


@pytest.mark.asyncio
async def test_refresh_skips_fresh_series(session, session_factory):
    """refresh_series_history skips a series that was fetched within min_fetch_interval_hours."""
    market_id = "KXTRUMPSAY-26MAY18-RIGG"
    await _seed_market(session, market_id, series_ticker="KXTRUMPSAY")
    await _seed_signal(session, market_id, created_at=NOW - timedelta(days=1))
    await session.commit()

    # Seed a fresh aggregate row (1 hour old, within 6-hour window)
    async with session_factory() as s:
        s.add(SeriesOptionHistoryRow(
            series_ticker="KXTRUMPSAY",
            option_code=_SERIES_AGGREGATE_CODE,
            option_label="KXTRUMPSAY",
            yes_count=5,
            no_count=3,
            last_fetched_at=NOW - timedelta(hours=1),
        ))
        await s.commit()

    kalshi = AsyncMock()
    kalshi.get_series_settled_history = AsyncMock(return_value=[])

    async with session_factory() as s:
        rows_upserted = await refresh_series_history(
            s, kalshi, lookback_days=7, min_fetch_interval_hours=6, now=NOW
        )
        await s.commit()

    assert rows_upserted == 0
    kalshi.get_series_settled_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_refreshes_stale_series(session, session_factory):
    """refresh_series_history re-fetches a series whose last_fetched_at is older than min_fetch_interval."""
    market_id = "KXTRUMPSAY-26MAY18-RIGG"
    await _seed_market(session, market_id, series_ticker="KXTRUMPSAY")
    await _seed_signal(session, market_id, created_at=NOW - timedelta(days=1))
    await session.commit()

    # Seed an aggregate row that is 10 hours old (beyond 6-hour threshold)
    async with session_factory() as s:
        s.add(SeriesOptionHistoryRow(
            series_ticker="KXTRUMPSAY",
            option_code=_SERIES_AGGREGATE_CODE,
            option_label="KXTRUMPSAY",
            yes_count=5,
            no_count=3,
            last_fetched_at=NOW - timedelta(hours=10),
        ))
        await s.commit()

    settled = [_make_settled_market("KXTRUMPSAY-26MAY18-RIGG", "yes", "Rigged")]
    kalshi = AsyncMock()
    kalshi.get_series_settled_history = AsyncMock(return_value=settled)

    async with session_factory() as s:
        rows_upserted = await refresh_series_history(
            s, kalshi, lookback_days=7, min_fetch_interval_hours=6, now=NOW
        )
        await s.commit()

    assert rows_upserted > 0
    kalshi.get_series_settled_history.assert_awaited_once_with("KXTRUMPSAY")


@pytest.mark.asyncio
async def test_refresh_skips_market_without_series_ticker(session, session_factory):
    """Markets with no series_ticker are not included in refresh."""
    market_id = "STANDALONE-26MAY18-YES"
    await _seed_market(session, market_id, series_ticker=None)
    await _seed_signal(session, market_id, created_at=NOW - timedelta(days=1))
    await session.commit()

    kalshi = AsyncMock()
    kalshi.get_series_settled_history = AsyncMock(return_value=[])

    async with session_factory() as s:
        rows_upserted = await refresh_series_history(s, kalshi, lookback_days=7, now=NOW)
        await s.commit()

    assert rows_upserted == 0
    kalshi.get_series_settled_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_upsert_overwrites_existing_counts(session, session_factory):
    """A second refresh replaces stale counts rather than accumulating."""
    market_id = "KXTRUMPSAY-26MAY18-RIGG"
    await _seed_market(session, market_id, series_ticker="KXTRUMPSAY")
    await _seed_signal(session, market_id, created_at=NOW - timedelta(days=1))
    await session.commit()

    first_batch = [
        _make_settled_market("KXTRUMPSAY-26MAY11-RIGG", "yes", "Rigged"),
    ]
    second_batch = [
        _make_settled_market("KXTRUMPSAY-26MAY11-RIGG", "yes", "Rigged"),
        _make_settled_market("KXTRUMPSAY-26MAY11-RIGG", "yes", "Rigged"),
        _make_settled_market("KXTRUMPSAY-26MAY11-RIGG", "no", "Rigged"),
    ]

    kalshi = AsyncMock()
    kalshi.get_series_settled_history = AsyncMock(return_value=first_batch)

    async with session_factory() as s:
        await refresh_series_history(s, kalshi, lookback_days=7, now=NOW - timedelta(hours=10))
        await s.commit()

    # Now re-run with stale data (10 hours later) with updated counts
    kalshi.get_series_settled_history = AsyncMock(return_value=second_batch)
    later = NOW

    async with session_factory() as s:
        await refresh_series_history(s, kalshi, lookback_days=7, min_fetch_interval_hours=6, now=later)
        await s.commit()

    async with session_factory() as s:
        result = await s.execute(
            select(SeriesOptionHistoryRow).where(
                SeriesOptionHistoryRow.series_ticker == "KXTRUMPSAY",
                SeriesOptionHistoryRow.option_code == "RIGG",
            )
        )
        row = result.scalar_one()

    # Second batch: 2 yes + 1 no — previous 1 yes is overwritten
    assert row.yes_count == 2
    assert row.no_count == 1


# ---------------------------------------------------------------------------
# Tests: get_series_history_for_market
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_series_history_returns_correct_rows(session, session_factory):
    """get_series_history_for_market returns series_row and option_row from DB."""
    async with session_factory() as s:
        s.add(SeriesOptionHistoryRow(
            series_ticker="KXTRUMPSAY",
            option_code=_SERIES_AGGREGATE_CODE,
            option_label="KXTRUMPSAY",
            yes_count=165,
            no_count=118,
            last_fetched_at=NOW,
        ))
        s.add(SeriesOptionHistoryRow(
            series_ticker="KXTRUMPSAY",
            option_code="RIGG",
            option_label="Rigged Election",
            yes_count=7,
            no_count=3,
            last_fetched_at=NOW,
        ))
        await s.commit()

    async with session_factory() as s:
        result = await get_series_history_for_market(s, "KXTRUMPSAY", "RIGG")

    assert result is not None
    assert result["series_ticker"] == "KXTRUMPSAY"
    assert result["option_code"] == "RIGG"

    series_row = result["series_row"]
    assert series_row is not None
    assert series_row.yes_count == 165
    assert series_row.no_count == 118

    option_row = result["option_row"]
    assert option_row is not None
    assert option_row.yes_count == 7
    assert option_row.no_count == 3
    assert option_row.option_label == "Rigged Election"


@pytest.mark.asyncio
async def test_get_series_history_returns_none_when_no_data(session, session_factory):
    """get_series_history_for_market returns None when series has no rows."""
    async with session_factory() as s:
        result = await get_series_history_for_market(s, "KXMISSING", "RIGG")

    assert result is None


@pytest.mark.asyncio
async def test_get_series_history_option_row_none_when_code_absent(session, session_factory):
    """option_row is None when the specific option_code has no row, but series_row is returned."""
    async with session_factory() as s:
        s.add(SeriesOptionHistoryRow(
            series_ticker="KXTRUMPPHOTO",
            option_code=_SERIES_AGGREGATE_CODE,
            option_label="KXTRUMPPHOTO",
            yes_count=3,
            no_count=5,
            last_fetched_at=NOW,
        ))
        await s.commit()

    async with session_factory() as s:
        result = await get_series_history_for_market(s, "KXTRUMPPHOTO", "26MAY18")

    assert result is not None
    assert result["series_row"] is not None
    assert result["series_row"].yes_count == 3
    assert result["option_row"] is None


@pytest.mark.asyncio
async def test_get_series_history_multiple_series_isolated(session, session_factory):
    """Rows from different series tickers do not bleed into each other."""
    async with session_factory() as s:
        for ticker in ("KXTRUMPSAY", "KXTRUMPPHOTO"):
            s.add(SeriesOptionHistoryRow(
                series_ticker=ticker,
                option_code=_SERIES_AGGREGATE_CODE,
                option_label=ticker,
                yes_count=10 if ticker == "KXTRUMPSAY" else 20,
                no_count=5,
                last_fetched_at=NOW,
            ))
        await s.commit()

    async with session_factory() as s:
        result = await get_series_history_for_market(s, "KXTRUMPSAY", "RIGG")

    assert result is not None
    assert result["series_row"].yes_count == 10  # KXTRUMPSAY, not KXTRUMPPHOTO


# ---------------------------------------------------------------------------
# Tests: end-to-end refresh → lookup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_then_lookup_returns_data(session, session_factory):
    """Full round-trip: refresh writes data then get_series_history_for_market reads it."""
    market_id = "KXTRUMPSAY-26MAY18-BARA"
    await _seed_market(session, market_id, series_ticker="KXTRUMPSAY")
    await _seed_signal(session, market_id, created_at=NOW - timedelta(days=1))
    await session.commit()

    settled = [
        _make_settled_market("KXTRUMPSAY-26MAY11-BARA", "yes", "Barack Hussein Obama"),
        _make_settled_market("KXTRUMPSAY-26MAY11-BARA", "yes", "Barack Hussein Obama"),
        _make_settled_market("KXTRUMPSAY-26MAY11-BARA", "no", "Barack Hussein Obama"),
        _make_settled_market("KXTRUMPSAY-26MAY04-BARA", "yes", "Barack Hussein Obama"),
    ]
    kalshi = AsyncMock()
    kalshi.get_series_settled_history = AsyncMock(return_value=settled)

    async with session_factory() as s:
        await refresh_series_history(s, kalshi, lookback_days=7, now=NOW)
        await s.commit()

    async with session_factory() as s:
        result = await get_series_history_for_market(s, "KXTRUMPSAY", "BARA")

    assert result is not None

    series_row = result["series_row"]
    assert series_row.yes_count == 3
    assert series_row.no_count == 1

    option_row = result["option_row"]
    assert option_row is not None
    assert option_row.option_label == "Barack Hussein Obama"
    assert option_row.yes_count == 3
    assert option_row.no_count == 1


@pytest.mark.asyncio
async def test_refresh_signal_outside_lookback_not_included(session, session_factory):
    """Markets whose signals are older than lookback_days are not refreshed."""
    market_id = "KXTRUMPSAY-26MAY18-RIGG"
    await _seed_market(session, market_id, series_ticker="KXTRUMPSAY")
    # Signal is 30 days old — beyond 7-day lookback
    await _seed_signal(session, market_id, created_at=NOW - timedelta(days=30))
    await session.commit()

    kalshi = AsyncMock()
    kalshi.get_series_settled_history = AsyncMock(return_value=[])

    async with session_factory() as s:
        rows_upserted = await refresh_series_history(s, kalshi, lookback_days=7, now=NOW)
        await s.commit()

    assert rows_upserted == 0
    kalshi.get_series_settled_history.assert_not_awaited()
