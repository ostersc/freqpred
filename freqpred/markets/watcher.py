"""Market watcher: polls Kalshi for price updates and staleness detection."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.markets.base import IMarketClient
from freqpred.markets.models import MarketRow
from freqpred.markets.repository import upsert_markets

if TYPE_CHECKING:
    from redis.asyncio import Redis

log = structlog.get_logger(__name__)

# Redis list key where market IDs are enqueued for signal re-analysis
SIGNAL_TRIGGER_QUEUE = "signal_triggers"

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
    3. Enqueues market IDs into the ``signal_triggers`` Redis list when
       mid_price has moved > ``price_move_threshold`` since the last signal.
    4. Logs stale markets (last_fetched_at older than polling_interval × 3)
       and skips them from signal triggering.
    5. Emits a structured log summary of the cycle.
    """

    def __init__(
        self,
        client: IMarketClient,
        session_factory: async_sessionmaker[AsyncSession],
        redis: "Redis",
        polling_interval: int = 300,
        price_move_threshold: float = PRICE_MOVE_THRESHOLD,
    ) -> None:
        self._client = client
        self._session_factory = session_factory
        self._redis = redis
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

        log.info(
            "market_watcher_cycle_complete",
            markets_polled=len(markets),
            triggers_enqueued=triggered,
            stale_markets=len(stale_ids),
            elapsed_s=round((datetime.now(UTC) - cycle_start).total_seconds(), 2),
        )
        if stale_ids:
            log.warning("market_watcher_stale_markets", market_ids=stale_ids)

    async def _check_price_move_triggers(
        self,
        session: AsyncSession,
        markets: list,
    ) -> int:
        """For each polled market with a signal, enqueue if price moved enough.

        Returns the count of markets enqueued.
        """
        if not markets:
            return 0

        # Build a map of market_id → current mid_price from the just-fetched data
        current_mids: dict[str, float] = {m.id: m.mid_price for m in markets}
        market_ids = list(current_mids.keys())

        # Import here to avoid circular imports (signal models depend on markets)
        from freqpred.signal.models import SignalRow

        result = await session.execute(
            select(MarketRow.id, MarketRow.mid_price, SignalRow.market_mid_at_signal)
            .join(SignalRow, SignalRow.id == MarketRow.current_signal_id)
            .where(MarketRow.current_signal_id.is_not(None))
        )
        rows = result.all()

        enqueued = 0
        for row in rows:
            # Only trigger for markets present in the current poll batch.
            if row.id not in current_mids:
                continue
            market_id: str = row.id
            new_mid: float = current_mids[market_id]
            signal_mid: float = row.market_mid_at_signal

            if price_moved(new_mid, signal_mid, self._price_move_threshold):
                payload = json.dumps(
                    {
                        "market_id": market_id,
                        "trigger": "price_moved",
                        "current_mid": new_mid,
                        "signal_mid": signal_mid,
                        "delta": round(new_mid - signal_mid, 4),
                    }
                )
                await self._redis.rpush(SIGNAL_TRIGGER_QUEUE, payload)
                log.info(
                    "signal_trigger_enqueued",
                    market_id=market_id,
                    current_mid=new_mid,
                    signal_mid=signal_mid,
                    delta=round(new_mid - signal_mid, 4),
                )
                enqueued += 1

        return enqueued

    async def _detect_stale_markets(self, session: AsyncSession) -> list[str]:
        """Return IDs of markets in the DB whose last_fetched_at is stale.

        Stale = last_fetched_at older than now - polling_interval × 3.
        These markets are logged but not enqueued for signal analysis.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=self._polling_interval * 3)
        result = await session.execute(
            select(MarketRow.id).where(MarketRow.last_fetched_at < cutoff)
        )
        return [row.id for row in result.all()]
