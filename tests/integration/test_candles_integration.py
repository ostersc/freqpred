"""Integration tests for candle storage, backfill, and path reconstruction.

Unit tests cannot catch the two failure modes that matter most here:

  * `load_candle_paths` translates a stored YES-space book into the position's
    own traded side — yes_bid for YES, `1 - yes_ask` for NO. Getting that
    backwards produces a plausible-looking path that is wrong for half the book,
    and only a real query over real rows proves the SQL and the Python agree.
  * the backfill's upsert must be idempotent on the natural key and its cursor
    must only ever widen coverage, or repeat runs either duplicate rows or
    re-request windows already paid for against a rate-limited API.

Requires a running Postgres (docker-compose up -d db) and DATABASE_URL
pointing at freqpred_test.
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select, text

pytestmark = pytest.mark.skipif(
    "freqpred_test" not in os.environ.get("DATABASE_URL", ""),
    reason="Integration tests require DATABASE_URL pointing to freqpred_test",
)

import freqpred.alerts.models  # noqa: F401
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.metrics.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.runtime.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
import freqpred.strategy.models  # noqa: F401
from freqpred.db import Base, make_engine, make_session_factory
from freqpred.markets.candles import backfill_candles, refresh_recent_candles
from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import MarketRow, PositionRow
from freqpred.metrics.models import CandleFetchCursorRow, MarketCandleRow
from freqpred.metrics.weekly_review import load_candle_paths

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test"
)
NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)
SERIES = "KXCANDLE"


class FakeKalshi:
    """Minimal stand-in returning canned candles, or raising a chosen error."""

    def __init__(self, candles=None, error: KalshiAPIError | None = None) -> None:
        self._candles = candles if candles is not None else []
        self._error = error
        self.calls: list[dict] = []

    async def get_market_candlesticks(
        self, series_ticker, market_id, *, start_ts, end_ts, period_interval
    ):
        self.calls.append(
            {
                "market_id": market_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            }
        )
        if self._error is not None:
            raise self._error
        return self._candles


def _candle(offset_hours: int, *, yes_bid_low: str, yes_ask_high: str) -> dict:
    ts = int((NOW - timedelta(days=2) + timedelta(hours=offset_hours)).timestamp())
    return {
        "end_period_ts": ts,
        "price": {"close_dollars": "0.5000"},
        "yes_bid": {"low_dollars": yes_bid_low, "close_dollars": yes_bid_low},
        "yes_ask": {"high_dollars": yes_ask_high, "close_dollars": yes_ask_high},
        "volume_fp": "10",
        "open_interest_fp": "100",
    }


@pytest_asyncio.fixture
async def session():
    engine = make_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = make_session_factory(engine)
    async with factory() as s:
        for table in ("market_candles", "candle_fetch_cursors"):
            await s.execute(text(f"DELETE FROM {table}"))
        # FK order: positions -> signals -> markets. markets.current_signal_id
        # also points back at signals, so it must be nulled before signals go.
        await s.execute(
            text("DELETE FROM positions WHERE market_id LIKE :p"), {"p": f"{SERIES}%"}
        )
        await s.execute(
            text("UPDATE markets SET current_signal_id = NULL WHERE id LIKE :p"),
            {"p": f"{SERIES}%"},
        )
        await s.execute(
            text("DELETE FROM signals WHERE market_id LIKE :p"), {"p": f"{SERIES}%"}
        )
        await s.execute(
            text("DELETE FROM markets WHERE id LIKE :p"), {"p": f"{SERIES}%"}
        )
        await s.commit()
        yield s
    await engine.dispose()


async def _make_market(session, market_id: str) -> None:
    session.add(
        MarketRow(
            id=market_id,
            platform="kalshi",
            question="q",
            category="Politics",
            series_ticker=SERIES,
            open_time=NOW - timedelta(days=3),
            close_time=NOW - timedelta(days=1),
            status="finalized",
            result="yes",
            yes_bid=0.5,
            yes_ask=0.5,
            mid_price=0.5,
            volume_24h=100.0,
            open_interest=100.0,
            last_fetched_at=NOW,
            price_updated_at=NOW,
            metadata_fetched_at=NOW,
        )
    )
    await session.commit()


async def _make_position(session, market_id: str, direction: str, entry: float) -> str:
    pid = uuid.uuid4()
    sid = uuid.uuid4()
    await session.execute(
        text(
            "INSERT INTO signals (id, market_id, estimated_probability, confidence, edge,"
            " market_mid_at_signal, direction, reasoning, sources, retrieval_hash,"
            " model_used, prompt_version, trigger, created_at, raw_context,"
            " market_ask_at_signal)"
            " VALUES (:id, :m, 0.6, 0.7, 0.2, 0.5, :d, '', '{}', 'h', 'm', 'signal-v11',"
            " 'scheduled', :ts, '', 0.4)"
        ),
        {"id": sid, "m": market_id, "d": direction, "ts": NOW - timedelta(days=3)},
    )
    session.add(
        PositionRow(
            id=pid,
            market_id=market_id,
            signal_id=sid,
            strategy_name="T",
            strategy_version="1",
            signal_confidence=0.7,
            signal_edge=0.2,
            signal_estimated_prob=0.6,
            direction=direction,
            contracts=10,
            entry_price=entry,
            entry_time=NOW - timedelta(days=3),
            mode="live",
            status="closed",
            exit_price=0.5,
            exit_time=NOW - timedelta(days=2),
            exit_reason="stoploss",
            pnl=-1.0,
        )
    )
    await session.commit()
    return str(pid)


@pytest.mark.asyncio
async def test_backfill_writes_candles_and_a_cursor(session) -> None:
    await _make_market(session, f"{SERIES}-A")
    client = FakeKalshi([_candle(h, yes_bid_low="0.30", yes_ask_high="0.70") for h in range(3)])

    result = await backfill_candles(
        session, client, series=SERIES, period_interval=60, _now=NOW
    )

    assert result.candles_written == 3
    assert result.markets_fetched == 1
    rows = (await session.execute(select(MarketCandleRow))).scalars().all()
    assert len(rows) == 3
    assert all(r.series_ticker == SERIES for r in rows)
    cursor = (await session.execute(select(CandleFetchCursorRow))).scalar_one()
    assert cursor.expired is False
    assert cursor.candle_count == 3


@pytest.mark.asyncio
async def test_backfill_is_idempotent_and_skips_covered_markets(session) -> None:
    await _make_market(session, f"{SERIES}-A")
    candles = [_candle(h, yes_bid_low="0.30", yes_ask_high="0.70") for h in range(3)]

    first = await backfill_candles(session, FakeKalshi(candles), series=SERIES, _now=NOW)
    second_client = FakeKalshi(candles)
    second = await backfill_candles(session, second_client, series=SERIES, _now=NOW)

    assert first.candles_written == 3
    # Coverage already recorded: no request should be issued at all.
    assert second_client.calls == []
    assert second.markets_skipped_fresh == 1
    assert len((await session.execute(select(MarketCandleRow))).scalars().all()) == 3


@pytest.mark.asyncio
async def test_force_refetches_and_upserts_without_duplicating(session) -> None:
    await _make_market(session, f"{SERIES}-A")
    candles = [_candle(h, yes_bid_low="0.30", yes_ask_high="0.70") for h in range(3)]
    await backfill_candles(session, FakeKalshi(candles), series=SERIES, _now=NOW)

    updated = [_candle(h, yes_bid_low="0.25", yes_ask_high="0.75") for h in range(3)]
    await backfill_candles(
        session, FakeKalshi(updated), series=SERIES, force=True, _now=NOW
    )

    rows = (await session.execute(select(MarketCandleRow))).scalars().all()
    assert len(rows) == 3  # upsert, not insert
    assert all(r.yes_bid_low == pytest.approx(0.25) for r in rows)


@pytest.mark.asyncio
async def test_404_marks_the_market_permanently_expired(session) -> None:
    await _make_market(session, f"{SERIES}-A")
    client = FakeKalshi(error=KalshiAPIError(404, "not found"))

    result = await backfill_candles(session, client, series=SERIES, _now=NOW)

    assert result.markets_expired == 1
    cursor = (await session.execute(select(CandleFetchCursorRow))).scalar_one()
    assert cursor.expired is True

    # A later run must not spend another request learning the same thing.
    retry = FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")])
    await backfill_candles(session, retry, series=SERIES, _now=NOW)
    assert retry.calls == []


@pytest.mark.asyncio
async def test_non_404_errors_do_not_mark_expired(session) -> None:
    """A transient 500 must not be mistaken for permanent expiry."""
    await _make_market(session, f"{SERIES}-A")
    client = FakeKalshi(error=KalshiAPIError(500, "boom"))

    result = await backfill_candles(session, client, series=SERIES, _now=NOW)

    assert result.markets_expired == 0
    assert result.errors
    cursors = (await session.execute(select(CandleFetchCursorRow))).scalars().all()
    assert all(not c.expired for c in cursors)


@pytest.mark.asyncio
async def test_max_requests_budget_stops_cleanly(session) -> None:
    for suffix in ("A", "B", "C"):
        await _make_market(session, f"{SERIES}-{suffix}")
    client = FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")])

    result = await backfill_candles(
        session, client, series=SERIES, max_requests=2, _now=NOW
    )

    assert result.budget_exhausted is True
    assert result.requests_made <= 2
    # What it did fetch is still durable — the next run resumes.
    assert (await session.execute(select(MarketCandleRow))).scalars().all()


@pytest.mark.asyncio
async def test_markets_past_retention_are_skipped_without_a_request(session) -> None:
    old = f"{SERIES}-OLD"
    await _make_market(session, old)
    await session.execute(
        text("UPDATE markets SET close_time = :t, open_time = :o WHERE id = :i"),
        {"t": NOW - timedelta(days=120), "o": NOW - timedelta(days=125), "i": old},
    )
    await session.commit()
    client = FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")])

    result = await backfill_candles(session, client, series=SERIES, _now=NOW)

    assert client.calls == []
    assert result.markets_expired == 1


@pytest.mark.asyncio
async def test_traded_only_excludes_untraded_markets(session) -> None:
    await _make_market(session, f"{SERIES}-TRADED")
    await _make_market(session, f"{SERIES}-UNTRADED")
    await _make_position(session, f"{SERIES}-TRADED", "YES", 0.40)
    client = FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")])

    await backfill_candles(
        session, client, series=SERIES, traded_only=True, _now=NOW
    )

    assert {c["market_id"] for c in client.calls} == {f"{SERIES}-TRADED"}


@pytest.mark.asyncio
async def test_path_uses_yes_bid_for_yes_and_one_minus_ask_for_no(session) -> None:
    """The core direction mapping, proven end-to-end against stored rows."""
    await _make_market(session, f"{SERIES}-A")
    # Book: yes_bid low 0.30, yes_ask high 0.70 => NO exit price = 1 - 0.70 = 0.30.
    # Made asymmetric so a swapped mapping cannot pass: bid 0.30 vs 1-ask 0.20.
    await backfill_candles(
        session,
        FakeKalshi([_candle(1, yes_bid_low="0.30", yes_ask_high="0.80")]),
        series=SERIES,
        _now=NOW,
    )
    yes_id = await _make_position(session, f"{SERIES}-A", "YES", 0.40)
    no_id = await _make_position(session, f"{SERIES}-A", "NO", 0.40)

    paths = await load_candle_paths(
        session, start=NOW - timedelta(days=10), end=NOW, mode="live"
    )

    assert paths[yes_id].lows == pytest.approx((0.30,))
    assert paths[no_id].lows == pytest.approx((0.20,))  # 1 - 0.80


@pytest.mark.asyncio
async def test_path_excludes_periods_with_no_bid(session) -> None:
    """A 0.0 bid is an empty book. Counting it would stop out on illiquidity."""
    await _make_market(session, f"{SERIES}-A")
    await backfill_candles(
        session,
        FakeKalshi(
            [
                _candle(1, yes_bid_low="0.0000", yes_ask_high="1.0000"),
                _candle(2, yes_bid_low="0.3500", yes_ask_high="0.7000"),
            ]
        ),
        series=SERIES,
        _now=NOW,
    )
    yes_id = await _make_position(session, f"{SERIES}-A", "YES", 0.40)
    no_id = await _make_position(session, f"{SERIES}-A", "NO", 0.40)

    paths = await load_candle_paths(
        session, start=NOW - timedelta(days=10), end=NOW, mode="live"
    )

    # YES: the 0.0 bid period is dropped, the 0.35 one kept.
    assert paths[yes_id].lows == pytest.approx((0.35,))
    assert paths[yes_id].n_no_bid == 1
    assert paths[yes_id].n_periods == 2
    # NO: 1 - 1.0 = 0.0 is likewise "no bid" and must be dropped.
    assert paths[no_id].lows == pytest.approx((0.30,))
    assert paths[no_id].n_no_bid == 1


@pytest.mark.asyncio
async def test_refresh_skips_markets_that_are_still_open(session) -> None:
    """An open market's candles are still being written; recording coverage lies."""
    open_id = f"{SERIES}-OPEN"
    await _make_market(session, open_id)
    await session.execute(
        text("UPDATE markets SET close_time = :t, status = 'active' WHERE id = :i"),
        {"t": NOW + timedelta(days=2), "i": open_id},
    )
    await session.commit()
    client = FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")])

    await refresh_recent_candles(session, client, _now=NOW)

    assert open_id not in {c["market_id"] for c in client.calls}


# ---------------------------------------------------------------------------
# Cursor coverage must never claim the future.
#
# Regression cover for the 2026-07-27 backfill that swept 32 still-open markets
# and stamped each cursor with its close_time. Because refresh_recent_candles
# skips any market whose cursor already reaches close_time, those markets were
# excluded permanently — including after they closed and their candles existed.
# ---------------------------------------------------------------------------

async def _make_open_market(session, market_id: str, *, closes_in_days: float) -> None:
    await _make_market(session, market_id)
    await session.execute(
        text("UPDATE markets SET close_time = :t, status = 'active' WHERE id = :i"),
        {"t": NOW + timedelta(days=closes_in_days), "i": market_id},
    )
    await session.commit()


@pytest.mark.asyncio
async def test_backfill_of_an_open_market_does_not_record_future_coverage(session) -> None:
    """Coverage ends where the fetch ended, not at a close_time still days away."""
    open_id = f"{SERIES}-OPEN2"
    await _make_open_market(session, open_id, closes_in_days=6)

    await backfill_candles(
        session,
        FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")]),
        market_id=open_id,
        _now=NOW,
    )

    cursor = (
        await session.execute(
            select(CandleFetchCursorRow).where(CandleFetchCursorRow.market_id == open_id)
        )
    ).scalar_one()
    assert cursor.covered_to <= NOW, (
        "cursor claims coverage of candles that do not exist yet"
    )
    assert cursor.covered_to >= cursor.covered_from, "window must not invert"


@pytest.mark.asyncio
async def test_market_backfilled_while_open_is_refetched_after_it_closes(session) -> None:
    """The end-to-end failure: TIKT was frozen at ten candles for its whole life."""
    mkt = f"{SERIES}-TIKT"
    await _make_open_market(session, mkt, closes_in_days=6)
    await _make_position(session, mkt, "YES", 0.65)  # gives it a recent signal

    # Day 1: backfill runs while the market is still trading.
    await backfill_candles(
        session,
        FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")]),
        market_id=mkt,
        _now=NOW,
    )

    # Day 8: the market has closed. The daily refresh must now pick it up.
    later = NOW + timedelta(days=8)
    client = FakeKalshi([_candle(1, yes_bid_low="0.35", yes_ask_high="0.65")])
    await refresh_recent_candles(session, client, _now=later)

    assert mkt in {c["market_id"] for c in client.calls}, (
        "closed market must be refetched to complete its history"
    )

    cursor = (
        await session.execute(
            select(CandleFetchCursorRow).where(CandleFetchCursorRow.market_id == mkt)
        )
    ).scalar_one()
    assert cursor.covered_to <= later


@pytest.mark.asyncio
async def test_closed_market_coverage_is_unchanged_by_the_clamp(session) -> None:
    """The clamp is a no-op for closed markets — no extra requests are burned."""
    closed_id = f"{SERIES}-CLOSED"
    await _make_market(session, closed_id)  # close_time = NOW - 1 day

    await backfill_candles(
        session,
        FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")]),
        market_id=closed_id,
        _now=NOW,
    )
    cursor = (
        await session.execute(
            select(CandleFetchCursorRow).where(CandleFetchCursorRow.market_id == closed_id)
        )
    ).scalar_one()
    close_time = (
        await session.execute(
            text("SELECT close_time FROM markets WHERE id = :i"), {"i": closed_id}
        )
    ).scalar_one()
    assert cursor.covered_to == close_time, "closed-market coverage must be untouched"

    # And a second backfill still skips it as already covered.
    second = FakeKalshi([_candle(0, yes_bid_low="0.3", yes_ask_high="0.7")])
    await backfill_candles(session, second, market_id=closed_id, _now=NOW)
    assert second.calls == []
