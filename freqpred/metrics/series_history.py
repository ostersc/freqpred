"""Background refresh for series option settlement history."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.metrics.models import SeriesOptionHistoryRow

if TYPE_CHECKING:
    from freqpred.markets.kalshi import KalshiClient

log = structlog.get_logger(__name__)

_SERIES_AGGREGATE_CODE = "__series__"
MIN_SAMPLE = 3


async def refresh_series_history(
    session: AsyncSession,
    kalshi_client: KalshiClient,
    *,
    lookback_days: int = 7,
    min_fetch_interval_hours: int = 6,
    now: datetime | None = None,
) -> int:
    """Upsert per-option and aggregate settlement history for recently-signalled series.

    Queries markets that have had a signal in the last ``lookback_days`` and have
    a ``series_ticker``. Skips any series whose rows were fetched within
    ``min_fetch_interval_hours``. Returns the number of rows upserted.
    """
    series_result = await session.execute(
        text(
            """
            SELECT DISTINCT m.series_ticker
            FROM markets m
            JOIN signals s ON s.market_id = m.id
            WHERE s.created_at >= :cutoff
              AND m.series_ticker IS NOT NULL
            """
        ),
        {"cutoff": (now or datetime.now(UTC)) - timedelta(days=lookback_days)},
    )
    series_tickers: list[str] = [row[0] for row in series_result]

    if not series_tickers:
        log.debug("series_history.no_series_to_refresh")
        return 0

    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=min_fetch_interval_hours)
    rows_upserted = 0

    for series_ticker in series_tickers:
        freshness_result = await session.execute(
            select(SeriesOptionHistoryRow.last_fetched_at).where(
                SeriesOptionHistoryRow.series_ticker == series_ticker,
                SeriesOptionHistoryRow.option_code == _SERIES_AGGREGATE_CODE,
            )
        )
        last_fetched = freshness_result.scalar_one_or_none()
        if last_fetched is not None and last_fetched >= cutoff:
            log.debug(
                "series_history.skip_fresh",
                series_ticker=series_ticker,
                last_fetched_at=last_fetched.isoformat(),
            )
            continue

        markets = await kalshi_client.get_series_settled_history(series_ticker)
        if not markets:
            log.debug("series_history.no_settled_markets", series_ticker=series_ticker)
            continue

        rows_upserted += await _upsert_series(session, series_ticker, markets, now)

    return rows_upserted


async def _upsert_series(
    session: AsyncSession,
    series_ticker: str,
    markets: list[dict[str, Any]],
    now: datetime,
) -> int:
    """Accumulate counts and upsert all rows (per-option + aggregate) for one series."""
    option_yes: dict[str, int] = {}
    option_no: dict[str, int] = {}
    option_label: dict[str, str] = {}
    series_yes = 0
    series_no = 0

    for m in markets:
        ticker: str = m.get("ticker", "")
        option_code = ticker.rsplit("-", 1)[-1] if "-" in ticker else ticker
        result = m.get("result", "")
        label: str = m.get("yes_sub_title", "") or option_code

        if result == "yes":
            option_yes[option_code] = option_yes.get(option_code, 0) + 1
            series_yes += 1
        elif result == "no":
            option_no[option_code] = option_no.get(option_code, 0) + 1
            series_no += 1
        else:
            # Unknown result — skip this market from counts
            continue

        option_label.setdefault(option_code, label)

    all_codes = set(option_yes) | set(option_no)
    rows: list[dict[str, Any]] = []

    for code in all_codes:
        rows.append({
            "series_ticker": series_ticker,
            "option_code": code,
            "option_label": option_label.get(code, code),
            "yes_count": option_yes.get(code, 0),
            "no_count": option_no.get(code, 0),
            "last_fetched_at": now,
        })

    # Always upsert the aggregate row
    rows.append({
        "series_ticker": series_ticker,
        "option_code": _SERIES_AGGREGATE_CODE,
        "option_label": series_ticker,
        "yes_count": series_yes,
        "no_count": series_no,
        "last_fetched_at": now,
    })

    stmt = insert(SeriesOptionHistoryRow).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["series_ticker", "option_code"],
        set_={
            "option_label": stmt.excluded.option_label,
            "yes_count": stmt.excluded.yes_count,
            "no_count": stmt.excluded.no_count,
            "last_fetched_at": stmt.excluded.last_fetched_at,
        },
    )
    await session.execute(stmt)

    log.info(
        "series_history.upserted",
        series_ticker=series_ticker,
        option_rows=len(all_codes),
        series_yes=series_yes,
        series_no=series_no,
    )
    return len(rows)


async def get_series_history_for_market(
    session: AsyncSession,
    series_ticker: str,
    option_code: str,
) -> dict[str, Any] | None:
    """Return a dict with series_row and option_row for use in prompt building.

    Returns None if no data is available.
    """
    result = await session.execute(
        select(SeriesOptionHistoryRow).where(
            SeriesOptionHistoryRow.series_ticker == series_ticker,
        )
    )
    rows = {r.option_code: r for r in result.scalars()}

    log.debug(
        "series_history.get_for_market",
        series_ticker=series_ticker,
        option_code=option_code,
        row_count=len(rows),
        codes=list(rows.keys()),
    )

    if not rows:
        return None

    series_row = rows.get(_SERIES_AGGREGATE_CODE)
    option_row = rows.get(option_code)

    return {
        "series_ticker": series_ticker,
        "option_code": option_code,
        "series_row": series_row,
        "option_row": option_row,
    }
