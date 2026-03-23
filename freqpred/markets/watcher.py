"""Market watcher: polls Kalshi for price updates and staleness detection."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.markets.base import IMarketClient
from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import MarketRow
from freqpred.markets.repository import upsert_markets

# Max markets to re-fetch per sweep cycle (rate-limit safety).
_RESOLVED_SWEEP_BATCH = 200

log = structlog.get_logger(__name__)

# Default price-move threshold in dollars (5 cents)
PRICE_MOVE_THRESHOLD = 0.05


def is_stale(last_fetched_at: datetime, polling_interval: int) -> bool:
    """Return True if the market hasn't been fetched within polling_interval × 3 seconds.

    Args:
        last_fetched_at: Timezone-aware datetime of the last successful fetch.
        polling_interval: Configured polling interval in seconds.
    """
    stale_cutoff = timedelta(seconds=polling_interval * 3)
    return (datetime.now(UTC) - last_fetched_at) > stale_cutoff


def price_moved(
    current_mid: float,
    signal_mid: float,
    threshold: float = PRICE_MOVE_THRESHOLD,
) -> bool:
    """Return True if mid_price shifted more than threshold since the last signal.

    Args:
        current_mid: Current mid_price from the latest poll.
        signal_mid: market_mid_at_signal from the most recent Signal record.
        threshold: Minimum absolute move to trigger re-analysis (default $0.05).
    """
    return abs(current_mid - signal_mid) > threshold


class MarketWatcher:
    """Async polling loop that keeps market prices fresh and detects signal triggers.

    Runs as a background asyncio task. On each cycle it:
    1. Fetches all open markets from Kalshi.
    2. Upserts prices and timestamps into the DB via the repository.
    3. Checks which markets have moved > ``price_move_threshold`` since last signal
       and logs them for visibility.
    4. Logs stale markets (last_fetched_at older than polling_interval × 3).
    5. Emits a structured log summary of the cycle.
    """

    def __init__(
        self,
        client: IMarketClient,
        session_factory: async_sessionmaker[AsyncSession],
        polling_interval: int = 300,
        price_move_threshold: float = PRICE_MOVE_THRESHOLD,
    ) -> None:
        self._client = client
        self._session_factory = session_factory
        self._polling_interval = polling_interval
        self._price_move_threshold = price_move_threshold

    async def run(self) -> None:
        """Run the polling loop indefinitely until the task is cancelled."""
        log.info("market_watcher_started", polling_interval=self._polling_interval)
        while True:
            try:
                await self._poll_cycle()
            except asyncio.CancelledError:
                log.info("market_watcher_stopped")
                raise
            except Exception:
                log.exception("market_watcher_poll_error")
            await asyncio.sleep(self._polling_interval)

    async def _poll_cycle(self) -> None:
        """Execute a single poll-upsert-trigger-check cycle."""
        cycle_start = datetime.now(UTC)
        markets = await self._client.list_markets()

        async with self._session_factory() as session:
            await upsert_markets(session, markets)
            triggered = await self._check_price_move_triggers(session, markets)
            stale_ids = await self._detect_stale_markets(session)

        resolved = await self._sweep_closed_markets()

        log.info(
            "market_watcher_cycle_complete",
            markets_polled=len(markets),
            triggers_logged=triggered,
            stale_markets=len(stale_ids),
            resolved_swept=resolved,
            elapsed_s=round((datetime.now(UTC) - cycle_start).total_seconds(), 2),
        )
        if stale_ids:
            log.warning("market_watcher_stale_markets", count=len(stale_ids))

    async def _sweep_closed_markets(self) -> int:
        """Re-fetch markets that have passed close_time or gone stale but aren't marked resolved.

        Kalshi's list_markets() only returns open markets, so resolved markets
        stop appearing in the feed.  This sweep handles two cases:

        1. close_time has passed — normal resolution path.
        2. Market is stale (not returned by list_markets() recently) AND has an open
           position — Kalshi may have closed/cancelled it early, leaving our stored
           close_time in the future.  Without this sweep those positions would be stuck.

        Returns the number of markets successfully updated.
        """
        from sqlalchemy import and_, or_
        from freqpred.markets.models import PositionRow

        now = datetime.now(UTC)
        stale_cutoff = now - timedelta(seconds=self._polling_interval * 3)

        async with self._session_factory() as session:
            # Subquery: market_ids that have at least one open position.
            open_position_market_ids = (
                select(PositionRow.market_id)
                .where(PositionRow.status == "open")
                .scalar_subquery()
            )
            result = await session.execute(
                select(MarketRow.id).where(
                    MarketRow.status.notin_(["resolved", "finalized"]),
                    or_(
                        # Normal path: close_time has passed — sweep all expired
                        # markets regardless of age (no lookback limit).
                        MarketRow.close_time <= now,
                        # Early-close path: stale AND has an open position.
                        and_(
                            MarketRow.last_fetched_at < stale_cutoff,
                            MarketRow.id.in_(open_position_market_ids),
                        ),
                    ),
                ).limit(_RESOLVED_SWEEP_BATCH)
            )
            market_ids = [row.id for row in result.all()]

        if not market_ids:
            return 0

        updated = 0
        markets_to_upsert = []
        for market_id in market_ids:
            try:
                market = await self._client.get_market(market_id)
                markets_to_upsert.append(market)
            except KalshiAPIError as exc:
                if exc.status_code == 404:
                    # Market no longer exists on Kalshi — mark it finalized so
                    # the resolved sweep never queries it again.
                    log.warning(
                        "market_watcher.resolved_sweep_not_found",
                        market_id=market_id,
                        hint="marking finalized so it is not retried",
                    )
                    async with self._session_factory() as session:
                        await session.execute(
                            update(MarketRow)
                            .where(MarketRow.id == market_id)
                            .values(status="finalized")
                        )
                        await session.commit()
                else:
                    log.error(
                        "market_watcher.resolved_sweep_error",
                        market_id=market_id,
                        status_code=exc.status_code,
                        body=exc.body,
                    )
            except Exception:
                log.exception("market_watcher.resolved_sweep_error", market_id=market_id)

        if markets_to_upsert:
            async with self._session_factory() as session:
                await upsert_markets(session, markets_to_upsert)
            updated = len(markets_to_upsert)
            log.info("market_watcher.resolved_sweep", updated=updated)

        return updated

    async def _check_price_move_triggers(
        self,
        session: AsyncSession,
        markets: list,
    ) -> int:
        """For each polled market with a signal, log if price moved enough.

        Returns the count of markets that have moved past the threshold.
        """
        if not markets:
            return 0

        current_mids: dict[str, float] = {m.id: m.mid_price for m in markets}

        from freqpred.signal.models import SignalRow

        result = await session.execute(
            select(MarketRow.id, MarketRow.mid_price, SignalRow.market_mid_at_signal)
            .join(SignalRow, SignalRow.id == MarketRow.current_signal_id)
            .where(MarketRow.current_signal_id.is_not(None))
        )
        rows = result.all()

        triggered = 0
        for row in rows:
            if row.id not in current_mids:
                continue
            market_id: str = row.id
            new_mid: float = current_mids[market_id]
            signal_mid: float = row.market_mid_at_signal

            if price_moved(new_mid, signal_mid, self._price_move_threshold):
                log.info(
                    "market_watcher.price_moved",
                    market_id=market_id,
                    current_mid=new_mid,
                    signal_mid=signal_mid,
                    delta=round(new_mid - signal_mid, 4),
                )
                triggered += 1

        return triggered

    async def _detect_stale_markets(self, session: AsyncSession) -> list[str]:
        """Return IDs of markets in the DB whose last_fetched_at is stale.

        Stale = last_fetched_at older than now - polling_interval × 3.
        """
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self._polling_interval * 3)
        result = await session.execute(
            select(MarketRow.id).where(
                MarketRow.last_fetched_at < cutoff,
                MarketRow.close_time > now,
            )
        )
        return [row.id for row in result.all()]
