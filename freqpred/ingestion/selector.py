"""Market Selector: determines which markets the ingestion pipeline should monitor.

The selector calls ``strategy.is_market_interesting()`` on every registered
strategy. A market is included if *any* strategy returns True. Markets not
selected by any strategy have their latest CatalystRun deactivated so the
ingestion scheduler stops fetching for them.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.models import CatalystRunRow
from freqpred.markets.models import Market, MarketRow

log = structlog.get_logger(__name__)


class StrategyProtocol(Protocol):
    """Minimal protocol the selector requires from a strategy."""

    def is_market_interesting(self, market: Market) -> bool:
        """Return True if this strategy wants to monitor this market."""
        ...


def select_markets(
    markets: list[Market],
    strategies: list[StrategyProtocol],
) -> list[Market]:
    """Return the subset of markets where at least one strategy is interested.

    Args:
        markets:    Candidate markets (typically all active markets from DB).
        strategies: Registered strategy instances to consult.

    Returns:
        Markets selected for catalyst generation and ingestion.
    """
    if not strategies:
        log.warning("selector.no_strategies", total=len(markets))
        return []

    selected = [
        m for m in markets
        if any(s.is_market_interesting(m) for s in strategies)
    ]

    log.info(
        "selector.market_selection",
        total=len(markets),
        selected=len(selected),
        excluded=len(markets) - len(selected),
    )
    return selected


async def deactivate_stale_catalysts(
    session: AsyncSession,
    strategies: list[StrategyProtocol],
    protected_market_ids: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Deactivate catalyst runs for markets that are closed or no longer selected.

    A run is deactivated when:
    - The market's close_time has passed (market resolved), OR
    - All registered strategies return False from is_market_interesting()

    Markets in ``protected_market_ids`` are never deactivated — use this to
    keep catalysts alive for markets with open positions even if they no longer
    pass the strategy's price/day filters.

    Only the latest run per market is checked and potentially deactivated.
    Earlier runs are left as historical records.

    Args:
        session:              Open async session (caller manages commit).
        strategies:           Registered strategy instances.
        protected_market_ids: Market IDs that must not be deactivated.

    Returns:
        List of market IDs whose catalyst runs were deactivated.
    """
    now = datetime.now(UTC)

    # Find all active runs, joining to the market for close_time + question data.
    # We use a subquery to get only the latest run per market.
    from sqlalchemy import func

    latest_run_subq = (
        select(
            CatalystRunRow.market_id,
            func.max(CatalystRunRow.created_at).label("latest_created_at"),
        )
        .where(CatalystRunRow.is_active.is_(True))
        .group_by(CatalystRunRow.market_id)
        .subquery()
    )

    result = await session.execute(
        select(CatalystRunRow, MarketRow)
        .join(
            latest_run_subq,
            (CatalystRunRow.market_id == latest_run_subq.c.market_id)
            & (CatalystRunRow.created_at == latest_run_subq.c.latest_created_at),
        )
        .join(MarketRow, MarketRow.id == CatalystRunRow.market_id)
        .where(CatalystRunRow.is_active.is_(True))
    )
    rows = result.all()

    deactivated_ids: list[str] = []
    for run_row, market_row in rows:
        market = _market_row_to_domain(market_row)

        closed = market.close_time <= now
        no_interest = not any(s.is_market_interesting(market) for s in strategies)
        protected = market.id in protected_market_ids

        if (closed or no_interest) and not protected:
            run_row.is_active = False
            session.add(run_row)
            deactivated_ids.append(market.id)
            log.info(
                "selector.deactivated_catalyst_run",
                market_id=market.id,
                run_id=str(run_row.id),
                generation=run_row.generation,
                reason="closed" if closed else "no_strategy_interest",
            )

    if deactivated_ids:
        await session.flush()

    log.info(
        "selector.deactivation_complete",
        checked=len(rows),
        deactivated=len(deactivated_ids),
    )
    return deactivated_ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _market_row_to_domain(row: MarketRow) -> Market:
    return Market(
        id=row.id,
        platform=row.platform,
        question=row.question,
        category=row.category,
        status=row.status,
        result=row.result,
        close_time=row.close_time,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
        mid_price=row.mid_price,
        last_price=row.last_price,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        yes_bid_size=row.yes_bid_size,
        yes_ask_size=row.yes_ask_size,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
        metadata=dict(row.metadata_),
        open_time=row.open_time,
        series_ticker=row.series_ticker,
    )
