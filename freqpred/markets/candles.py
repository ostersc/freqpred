"""Fetch and store Kalshi market candlesticks.

Why this exists
---------------
Nothing else in the system persists a price *path*. `signals.market_*_at_signal`
gives point observations at roughly the signal cadence — measured at a median of
22 observations across a median 72h live hold, about one every 3.3 hours, with
some positions having none at all after entry. `positions.mae` records only the
worst excursion, and stops updating the moment the position exits, so it cannot
say whether a trade that really stopped at -0.15 would also have breached -0.30.

Candlesticks close both gaps, and they carry `yes_bid` and `yes_ask` OHLC —
both sides of the book, which is what modelling a stop *fill* requires rather
than just a trigger.

Retention
---------
**The endpoint's history is a rolling window, ~67 days when measured on
2026-07-25**: markets closing 2026-05-20 or later returned data; 2026-05-18 and
earlier returned 404. Expired candles never come back. That is the entire reason
this is a store-to-DB tool rather than an on-demand fetch, and why the scheduler
prioritises markets nearest the cliff.

Rate limiting
-------------
Three independent brakes, because a family-wide backfill can otherwise issue
thousands of requests:

1. `KalshiClient._get` throttles to `read_rps` and honours 429 `Retry-After`.
   Requests here are strictly sequential, so that throttle is the real ceiling.
2. `max_requests` caps any single run. Reaching it is a clean partial result —
   cursors record what was covered, so the next run resumes rather than restarts.
3. Every request increments `api_daily_counters` under `CANDLE_SERVICE`, giving
   a persistent cross-process view of spend against the Kalshi read budget.

The cursor table is what makes repeat runs cheap: without it there is no way to
distinguish "fetched, and the market was quiet" from "never fetched", and the
scheduler would re-request empty ranges forever.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.quota import current_window, increment_window_count
from freqpred.markets.kalshi import VALID_PERIOD_INTERVALS, KalshiAPIError
from freqpred.metrics.models import CandleFetchCursorRow, MarketCandleRow

if TYPE_CHECKING:
    from freqpred.markets.kalshi import KalshiClient

log = structlog.get_logger(__name__)

#: Service name for `api_daily_counters` request accounting.
CANDLE_SERVICE = "kalshi_candles"

#: Max candles to request in one call. Measured 2026-07-25: a 1-minute request
#: spanning 48h (2880 candles) succeeded and 168h (10080) returned HTTP 400, so
#: the server cap sits between. 2500 stays clear of it with margin.
MAX_CANDLES_PER_REQUEST = 2500

#: Observed retention. Used only to skip requests that would certainly 404 and
#: to prioritise what is about to expire — never to *infer* that data exists.
RETENTION_DAYS = 67


@dataclass
class BackfillResult:
    markets_attempted: int = 0
    markets_fetched: int = 0
    markets_expired: int = 0
    markets_skipped_fresh: int = 0
    candles_written: int = 0
    requests_made: int = 0
    budget_exhausted: bool = False
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"markets: {self.markets_fetched} fetched, {self.markets_expired} expired, "
            f"{self.markets_skipped_fresh} already covered, "
            f"{self.markets_attempted} attempted | candles: {self.candles_written} | "
            f"requests: {self.requests_made}"
            + (" | BUDGET EXHAUSTED" if self.budget_exhausted else "")
        )


def _dollars(block: dict[str, Any] | None, key: str) -> float | None:
    """Pull one OHLC leg out of a Kalshi candle block.

    Kalshi returns these as decimal *strings* ("0.8400"). A missing block means
    the period had no trades, which is information, not an error.
    """
    if not block:
        return None
    raw = block.get(f"{key}_dollars")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def candle_to_row(
    candle: dict[str, Any],
    *,
    market_id: str,
    series_ticker: str,
    period_interval: int,
    fetched_at: datetime,
) -> dict[str, Any] | None:
    """Map one raw API candle to a `market_candles` row dict, or None if unusable."""
    end_ts = candle.get("end_period_ts")
    if end_ts is None:
        return None
    price = candle.get("price") or {}
    bid = candle.get("yes_bid") or {}
    ask = candle.get("yes_ask") or {}

    def _num(raw: Any) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    return {
        "market_id": market_id,
        "period_interval": period_interval,
        "end_period_ts": datetime.fromtimestamp(int(end_ts), tz=UTC),
        "series_ticker": series_ticker,
        "price_open": _dollars(price, "open"),
        "price_high": _dollars(price, "high"),
        "price_low": _dollars(price, "low"),
        "price_close": _dollars(price, "close"),
        "price_mean": _dollars(price, "mean"),
        "yes_bid_open": _dollars(bid, "open"),
        "yes_bid_high": _dollars(bid, "high"),
        "yes_bid_low": _dollars(bid, "low"),
        "yes_bid_close": _dollars(bid, "close"),
        "yes_ask_open": _dollars(ask, "open"),
        "yes_ask_high": _dollars(ask, "high"),
        "yes_ask_low": _dollars(ask, "low"),
        "yes_ask_close": _dollars(ask, "close"),
        "volume": _num(candle.get("volume_fp")),
        "open_interest": _num(candle.get("open_interest_fp")),
        "fetched_at": fetched_at,
    }


def chunk_ranges(
    start_ts: int, end_ts: int, period_interval: int
) -> list[tuple[int, int]]:
    """Split [start_ts, end_ts] into spans of at most MAX_CANDLES_PER_REQUEST.

    A 1-minute backfill over a week is ~10k candles and the server rejects it
    outright, so long ranges must be split rather than truncated — truncation
    would silently produce a partial price path that later analysis would treat
    as complete.
    """
    if end_ts <= start_ts:
        return []
    span = period_interval * 60 * MAX_CANDLES_PER_REQUEST
    return [
        (chunk, min(chunk + span, end_ts))
        for chunk in range(start_ts, end_ts, span)
    ]


#: asyncpg refuses a statement with more than this many bind parameters.
_MAX_BIND_PARAMS = 32767


def _batch_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split rows so no single INSERT exceeds the bind-parameter limit.

    A candle row is ~20 columns, so the ceiling is ~1600 rows per statement.
    Weekly markets stay well under it (~178 hourly candles), which is why a
    KXTRUMPSAY-only backfill never hit this — but the scheduler also picks up
    monthly series, and KXTRUMPSAYNICKNAME markets run ~2,200 hours, which
    overflowed and failed the whole refresh cycle.

    The batch size is derived from the actual row width rather than hardcoded,
    so adding a column cannot silently reintroduce the overflow.
    """
    if not rows:
        return []
    per_row = max(len(rows[0]), 1)
    size = max(1, (_MAX_BIND_PARAMS // per_row) - 1)
    return [rows[i : i + size] for i in range(0, len(rows), size)]


async def _upsert_candles(session: AsyncSession, rows: list[dict[str, Any]]) -> int:
    """Idempotent upsert on the natural key. Re-fetching a window costs nothing."""
    if not rows:
        return 0
    for batch in _batch_rows(rows):
        stmt = insert(MarketCandleRow).values(batch)
        update_cols = {
            c: getattr(stmt.excluded, c)
            for c in batch[0]
            if c not in ("market_id", "period_interval", "end_period_ts")
        }
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=["market_id", "period_interval", "end_period_ts"],
                set_=update_cols,
            )
        )
    return len(rows)


async def _record_cursor(
    session: AsyncSession,
    *,
    market_id: str,
    period_interval: int,
    covered_from: datetime,
    covered_to: datetime,
    candle_count: int,
    expired: bool,
    now: datetime,
) -> None:
    """Widen the covered window for this (market, interval).

    Coverage only ever grows: a later run over a narrower range must not shrink
    what an earlier wide run already established, or the scheduler would re-fetch
    ground it has already paid for.
    """
    stmt = insert(CandleFetchCursorRow).values(
        market_id=market_id,
        period_interval=period_interval,
        covered_from=covered_from,
        covered_to=covered_to,
        candle_count=candle_count,
        expired=expired,
        last_fetched_at=now,
    )
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=["market_id", "period_interval"],
            set_={
                "covered_from": func_least(
                    CandleFetchCursorRow.covered_from, stmt.excluded.covered_from
                ),
                "covered_to": func_greatest(
                    CandleFetchCursorRow.covered_to, stmt.excluded.covered_to
                ),
                "candle_count": CandleFetchCursorRow.candle_count + candle_count,
                "expired": stmt.excluded.expired,
                "last_fetched_at": stmt.excluded.last_fetched_at,
            },
        )
    )


def func_least(a: Any, b: Any) -> Any:
    from sqlalchemy import func

    return func.least(a, b)


def func_greatest(a: Any, b: Any) -> Any:
    from sqlalchemy import func

    return func.greatest(a, b)


async def fetch_market_candles(
    session: AsyncSession,
    kalshi_client: KalshiClient,
    *,
    market_id: str,
    series_ticker: str,
    start: datetime,
    end: datetime,
    period_interval: int,
    result: BackfillResult,
    max_requests: int,
    now: datetime,
) -> None:
    """Fetch one market's candles over [start, end], chunked and budget-aware.

    Mutates `result`. A 404 marks the market permanently expired so no later run
    retries it; any other API error is recorded and the market is left alone so a
    transient failure does not get mistaken for expiry.
    """
    result.markets_attempted += 1
    rows: list[dict[str, Any]] = []
    chunks = chunk_ranges(int(start.timestamp()), int(end.timestamp()), period_interval)

    for chunk_start, chunk_end in chunks:
        if result.requests_made >= max_requests:
            result.budget_exhausted = True
            break
        result.requests_made += 1
        d, slot = current_window(now)
        await increment_window_count(session, CANDLE_SERVICE, d, slot)
        try:
            candles = await kalshi_client.get_market_candlesticks(
                series_ticker,
                market_id,
                start_ts=chunk_start,
                end_ts=chunk_end,
                period_interval=period_interval,
            )
        except KalshiAPIError as exc:
            if exc.status_code == 404:
                # Settled before the retention cutoff. Permanent — record it so
                # this market is never requested again.
                result.markets_expired += 1
                await _record_cursor(
                    session,
                    market_id=market_id,
                    period_interval=period_interval,
                    covered_from=start,
                    covered_to=end,
                    candle_count=0,
                    expired=True,
                    now=now,
                )
                log.info("candles.expired", market_id=market_id)
                return
            result.errors.append(f"{market_id}: HTTP {exc.status_code}")
            log.warning(
                "candles.fetch_error", market_id=market_id, status=exc.status_code
            )
            return

        for candle in candles:
            row = candle_to_row(
                candle,
                market_id=market_id,
                series_ticker=series_ticker,
                period_interval=period_interval,
                fetched_at=now,
            )
            if row is not None:
                rows.append(row)

    written = await _upsert_candles(session, rows)
    result.candles_written += written
    if written or not result.budget_exhausted:
        result.markets_fetched += 1
    await _record_cursor(
        session,
        market_id=market_id,
        period_interval=period_interval,
        covered_from=start,
        covered_to=end,
        candle_count=written,
        expired=False,
        now=now,
    )
    log.info(
        "candles.fetched",
        market_id=market_id,
        period_interval=period_interval,
        candles=written,
    )


_TARGETS_SQL = """
    SELECT m.id, m.series_ticker,
           COALESCE(m.open_time, m.created_at) AS starts,
           m.close_time
    FROM markets m
    WHERE m.series_ticker IS NOT NULL
      AND (CAST(:market_id AS text) IS NULL OR m.id = CAST(:market_id AS text))
      AND (CAST(:series AS text) IS NULL OR m.series_ticker = CAST(:series AS text))
      AND (CAST(:start AS timestamptz) IS NULL OR m.close_time >= CAST(:start AS timestamptz))
      AND (CAST(:end AS timestamptz) IS NULL OR m.close_time <= CAST(:end AS timestamptz))
      {traded_clause}
    ORDER BY m.close_time DESC
"""


async def select_target_markets(
    session: AsyncSession,
    *,
    market_id: str | None = None,
    series: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    traded_only: bool = False,
) -> list[dict[str, Any]]:
    """Resolve the market/family/window selector into concrete markets.

    Ordered newest-first deliberately: under a request budget, the newest markets
    are both the most analytically useful and the least likely to expire before
    the next run.
    """
    clause = (
        "AND EXISTS (SELECT 1 FROM positions p WHERE p.market_id = m.id)"
        if traded_only
        else ""
    )
    rows = (
        await session.execute(
            text(_TARGETS_SQL.format(traded_clause=clause)),
            {"market_id": market_id, "series": series, "start": start, "end": end},
        )
    ).mappings().all()
    return [dict(r) for r in rows]


async def backfill_candles(
    session: AsyncSession,
    kalshi_client: KalshiClient,
    *,
    market_id: str | None = None,
    series: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    period_interval: int = 60,
    traded_only: bool = False,
    max_requests: int = 500,
    force: bool = False,
    skip_expired: bool = True,
    _now: datetime | None = None,
) -> BackfillResult:
    """Backfill candles for one market or a whole family over a window.

    `start`/`end` filter which markets are in scope by their close time; each
    market is then fetched across its own lifetime, since a candle range outside
    a market's active period returns nothing useful.

    Markets already covered are skipped unless `force`. Markets previously marked
    expired are skipped unless `skip_expired=False` — that data is gone, and
    re-requesting it only burns budget.
    """
    if period_interval not in VALID_PERIOD_INTERVALS:
        raise ValueError(
            f"period_interval must be one of {sorted(VALID_PERIOD_INTERVALS)}"
        )
    now = _now or datetime.now(UTC)
    result = BackfillResult()

    targets = await select_target_markets(
        session,
        market_id=market_id,
        series=series,
        start=start,
        end=end,
        traded_only=traded_only,
    )
    if not targets:
        log.info("candles.no_targets", market_id=market_id, series=series)
        return result

    cursors = {
        (c.market_id, c.period_interval): c
        for c in (
            await session.execute(
                select(CandleFetchCursorRow).where(
                    CandleFetchCursorRow.period_interval == period_interval
                )
            )
        ).scalars()
    }
    retention_floor = now - timedelta(days=RETENTION_DAYS)

    for target in targets:
        if result.requests_made >= max_requests:
            result.budget_exhausted = True
            break

        mid, series_ticker = target["id"], target["series_ticker"]
        cursor = cursors.get((mid, period_interval))
        if cursor is not None and cursor.expired and skip_expired:
            result.markets_expired += 1
            continue

        m_start = target["starts"] or (target["close_time"] - timedelta(days=7))
        m_end = target["close_time"]

        # Certain-404: the market settled before the retention window opened.
        # Recorded as expired so the next run does not spend a request learning
        # the same thing.
        if m_end < retention_floor and not force:
            result.markets_expired += 1
            await _record_cursor(
                session,
                market_id=mid,
                period_interval=period_interval,
                covered_from=m_start,
                covered_to=m_end,
                candle_count=0,
                expired=True,
                now=now,
            )
            continue

        if (
            not force
            and cursor is not None
            and cursor.covered_from <= m_start
            and cursor.covered_to >= m_end
        ):
            result.markets_skipped_fresh += 1
            continue

        await fetch_market_candles(
            session,
            kalshi_client,
            market_id=mid,
            series_ticker=series_ticker,
            start=m_start,
            end=m_end,
            period_interval=period_interval,
            result=result,
            max_requests=max_requests,
            now=now,
        )
        await session.commit()

    await session.commit()
    log.info("candles.backfill_complete", summary=result.summary())
    return result


_REFRESH_TARGETS_SQL = text(
    """
    SELECT m.id, m.series_ticker,
           COALESCE(m.open_time, m.created_at) AS starts,
           m.close_time
    FROM markets m
    WHERE m.series_ticker IS NOT NULL
      AND m.close_time < :now
      AND m.close_time >= :retention_floor
      AND EXISTS (
            SELECT 1 FROM signals s
            WHERE s.market_id = m.id AND s.created_at >= :signal_cutoff
      )
      AND NOT EXISTS (
            SELECT 1 FROM candle_fetch_cursors c
            WHERE c.market_id = m.id
              AND c.period_interval = :interval
              AND (c.expired OR (c.covered_from <= COALESCE(m.open_time, m.created_at)
                                 AND c.covered_to >= m.close_time))
      )
    -- Oldest first: those are the closest to falling out of the retention window,
    -- and unlike the newest they will not still be there tomorrow.
    ORDER BY m.close_time ASC
    LIMIT :limit
    """
)


async def refresh_recent_candles(
    session: AsyncSession,
    kalshi_client: KalshiClient,
    *,
    period_interval: int = 60,
    signal_lookback_days: int = 90,
    max_markets: int = 200,
    max_requests: int = 250,
    _now: datetime | None = None,
) -> BackfillResult:
    """Daily top-up: fetch candles for recently-signalled markets that have closed.

    Only closed markets are fetched — an open market's candles are still being
    written, so any coverage recorded for it would be a lie the cursor then
    prevents correcting.

    Ordering is **oldest-first**, the opposite of the CLI's. Under a per-cycle
    budget the right thing to prioritise is what is about to expire: a market
    closing today will still be fetchable for two months, while one from nine
    weeks ago disappears within days.
    """
    now = _now or datetime.now(UTC)
    result = BackfillResult()
    targets = (
        await session.execute(
            _REFRESH_TARGETS_SQL,
            {
                "now": now,
                "retention_floor": now - timedelta(days=RETENTION_DAYS),
                "signal_cutoff": now - timedelta(days=signal_lookback_days),
                "interval": period_interval,
                "limit": max_markets,
            },
        )
    ).mappings().all()

    for target in targets:
        if result.requests_made >= max_requests:
            result.budget_exhausted = True
            break
        await fetch_market_candles(
            session,
            kalshi_client,
            market_id=target["id"],
            series_ticker=target["series_ticker"],
            start=target["starts"] or (target["close_time"] - timedelta(days=7)),
            end=target["close_time"],
            period_interval=period_interval,
            result=result,
            max_requests=max_requests,
            now=now,
        )
        await session.commit()

    if targets:
        log.info("candles.refresh_complete", summary=result.summary())
    return result
