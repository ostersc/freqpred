"""Market repository: upsert markets into the database."""
from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import case, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.markets.models import Market, MarketRow

log = structlog.get_logger(__name__)

# Max rows per INSERT statement (keeps parameter count well under Postgres's 65535 limit)
_BATCH_SIZE = 500


def _build_row(market: Market, now: datetime) -> dict:
    return {
        "id": market.id,
        "platform": market.platform,
        "question": market.question,
        "category": market.category,
        "status": market.status,
        "result": market.result,
        "close_time": market.close_time,
        "yes_bid": market.yes_bid,
        "yes_ask": market.yes_ask,
        "mid_price": market.mid_price,
        "last_price": market.last_price,
        "volume_24h": market.volume_24h,
        "open_interest": market.open_interest,
        "liquidity": market.liquidity,
        "last_fetched_at": now,
        "price_updated_at": now,  # overridden by CASE on conflict
        "metadata_fetched_at": market.metadata_fetched_at,
        "metadata": market.metadata,
        "current_signal_id": None,
        "open_time": market.open_time,
    }


async def _upsert_batch(session: AsyncSession, rows: list[dict], now: datetime) -> None:
    """Execute a single batched INSERT ... ON CONFLICT DO UPDATE for a list of rows.

    price_updated_at is only advanced when yes_bid/yes_ask/mid_price changes,
    using a CASE expression against the EXCLUDED pseudo-table — no pre-SELECT needed.
    current_signal_id is never overwritten on conflict.
    """
    tbl = MarketRow.__table__
    stmt = pg_insert(tbl).values(rows)

    price_updated_at_expr = case(
        (
            (stmt.excluded.yes_bid != tbl.c.yes_bid)
            | (stmt.excluded.yes_ask != tbl.c.yes_ask)
            | (stmt.excluded.mid_price != tbl.c.mid_price),
            stmt.excluded.last_fetched_at,
        ),
        else_=tbl.c.price_updated_at,
    )

    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "question": stmt.excluded.question,
            "category": stmt.excluded.category,
            "status": stmt.excluded.status,
            "result": func.coalesce(func.nullif(stmt.excluded.result, ""), tbl.c.result),
            "close_time": stmt.excluded.close_time,
            "yes_bid": stmt.excluded.yes_bid,
            "yes_ask": stmt.excluded.yes_ask,
            "mid_price": stmt.excluded.mid_price,
            "last_price": stmt.excluded.last_price,
            "volume_24h": stmt.excluded.volume_24h,
            "open_interest": stmt.excluded.open_interest,
            "liquidity": stmt.excluded.liquidity,
            "last_fetched_at": stmt.excluded.last_fetched_at,
            "price_updated_at": price_updated_at_expr,
            "metadata_fetched_at": stmt.excluded.metadata_fetched_at,
            "metadata": stmt.excluded.metadata,
            "open_time": func.coalesce(tbl.c.open_time, stmt.excluded.open_time),
        },
    )
    await session.execute(stmt)


async def upsert_market(session: AsyncSession, market: Market) -> None:
    """Insert or update a single market row.

    Rules:
    - ``last_fetched_at`` is always updated to now.
    - ``price_updated_at`` is updated to now only when yes_bid, yes_ask, or
      mid_price has changed since the last fetch.
    - All other fields (question, close_time, volume_24h, etc.) are always
      refreshed to the latest values from the exchange.
    - ``current_signal_id`` is never overwritten.
    """
    now = datetime.now(UTC)
    await _upsert_batch(session, [_build_row(market, now)], now)


async def upsert_markets(session: AsyncSession, markets: list[Market]) -> int:
    """Upsert a batch of markets in chunks. Returns the count written.

    Uses a single INSERT ... ON CONFLICT DO UPDATE per chunk instead of
    per-row SELECT + INSERT, reducing ~72k round-trips to ~73 for 36k markets.
    """
    if not markets:
        return 0

    now = datetime.now(UTC)
    for i in range(0, len(markets), _BATCH_SIZE):
        chunk = markets[i : i + _BATCH_SIZE]
        await _upsert_batch(session, [_build_row(m, now) for m in chunk], now)

    await session.commit()
    log.info("markets_upserted", count=len(markets))
    return len(markets)
