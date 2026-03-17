"""Market repository: upsert markets into the database."""
from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.markets.models import Market, MarketRow

log = structlog.get_logger(__name__)


async def upsert_market(session: AsyncSession, market: Market) -> None:
    """Insert or update a single market row.

    Rules:
    - ``last_fetched_at`` is always updated to now.
    - ``price_updated_at`` is updated to now only when yes_bid, yes_ask, or
      mid_price has changed since the last fetch.
    - All other fields (question, close_time, volume_24h, etc.) are always
      refreshed to the latest values from the exchange.
    """
    now = datetime.now(UTC)

    # Check existing price snapshot to decide whether price actually changed.
    result = await session.execute(
        select(
            MarketRow.yes_bid,
            MarketRow.yes_ask,
            MarketRow.mid_price,
            MarketRow.price_updated_at,
        ).where(MarketRow.id == market.id)
    )
    existing = result.one_or_none()

    if existing is not None:
        price_changed = (
            existing.yes_bid != market.yes_bid
            or existing.yes_ask != market.yes_ask
            or existing.mid_price != market.mid_price
        )
        price_updated_at = now if price_changed else existing.price_updated_at
    else:
        price_updated_at = now

    stmt = (
        pg_insert(MarketRow.__table__)
        .values(
            id=market.id,
            platform=market.platform,
            question=market.question,
            category=market.category,
            close_time=market.close_time,
            yes_bid=market.yes_bid,
            yes_ask=market.yes_ask,
            mid_price=market.mid_price,
            volume_24h=market.volume_24h,
            open_interest=market.open_interest,
            last_fetched_at=now,
            price_updated_at=price_updated_at,
            metadata_fetched_at=market.metadata_fetched_at,
            metadata=market.metadata,
            current_signal_id=None,
        )
        .on_conflict_do_update(
            index_elements=["id"],
            set_={
                "question": market.question,
                "category": market.category,
                "close_time": market.close_time,
                "yes_bid": market.yes_bid,
                "yes_ask": market.yes_ask,
                "mid_price": market.mid_price,
                "volume_24h": market.volume_24h,
                "open_interest": market.open_interest,
                "last_fetched_at": now,
                "price_updated_at": price_updated_at,
                "metadata_fetched_at": market.metadata_fetched_at,
                "metadata": market.metadata,
            },
        )
    )
    await session.execute(stmt)


async def upsert_markets(session: AsyncSession, markets: list[Market]) -> int:
    """Upsert a batch of markets. Returns the count written."""
    for market in markets:
        await upsert_market(session, market)
    await session.commit()
    log.info("markets_upserted", count=len(markets))
    return len(markets)
