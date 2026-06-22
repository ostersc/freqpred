"""Market watcher: polls Kalshi for price updates and staleness detection."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.markets.base import IMarketClient
from freqpred.markets.kalshi import KalshiAPIError, KalshiClient
from freqpred.markets.models import Market, MarketRow, PositionRow
from freqpred.markets.repository import upsert_markets
from freqpred.trading import ledger

if TYPE_CHECKING:
    from freqpred.alerts.dispatcher import AlertDispatcher
    from freqpred.runtime.telemetry import RuntimeTelemetry

from freqpred.runtime.telemetry import SERVICE_MARKET_WATCHER

# Max markets to re-fetch per sweep cycle. Chunked into ≤200-ticker batched
# calls by get_markets_by_tickers(), so this is now bounded by DB/memory cost
# rather than per-market round-trip cost.
_RESOLVED_SWEEP_BATCH = 1000

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
        alert_dispatcher: "AlertDispatcher | None" = None,
        runtime_telemetry: "RuntimeTelemetry | None" = None,
    ) -> None:
        self._client = client
        self._session_factory = session_factory
        self._polling_interval = polling_interval
        self._price_move_threshold = price_move_threshold
        self._alert_dispatcher = alert_dispatcher
        self._runtime_telemetry = runtime_telemetry

    async def run(self) -> None:
        """Run the polling loop indefinitely until the task is cancelled."""
        log.info("market_watcher_started", polling_interval=self._polling_interval)
        while True:
            try:
                await self._poll_cycle()
            except KalshiAPIError as exc:
                if self._runtime_telemetry is not None:
                    await self._runtime_telemetry.record_kalshi_error(
                        "market_watcher",
                        f"market watcher poll failed: {exc}",
                        details={"status_code": exc.status_code},
                    )
                log.exception("market_watcher_poll_error")
            except asyncio.CancelledError:
                log.info("market_watcher_stopped")
                raise
            except Exception as exc:
                if self._runtime_telemetry is not None:
                    await self._runtime_telemetry.record_kalshi_error(
                        "market_watcher",
                        f"market watcher poll failed: {exc}",
                    )
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
        missing_resolved = await self._sweep_missing_results()
        rest_fallback_resolved = await self._resolve_settled_open_positions()

        log.info(
            "market_watcher_cycle_complete",
            markets_polled=len(markets),
            triggers_logged=triggered,
            stale_markets=len(stale_ids),
            resolved_swept=resolved,
            missing_results_resolved=missing_resolved,
            rest_fallback_resolved=rest_fallback_resolved,
            elapsed_s=round((datetime.now(UTC) - cycle_start).total_seconds(), 2),
        )
        if stale_ids:
            log.warning("market_watcher_stale_markets", count=len(stale_ids))
        if self._runtime_telemetry is not None:
            await self._runtime_telemetry.mark_success(
                SERVICE_MARKET_WATCHER,
                details={"markets_polled": len(markets)},
            )

    async def _sweep_closed_markets(self) -> int:
        """Re-fetch markets that have passed close_time or gone stale but aren't marked resolved.

        Kalshi's list_markets() only returns open markets, so resolved markets
        stop appearing in the feed.  This sweep handles two cases:

        1. close_time has passed — normal resolution path.
        2. Market is stale (not returned by list_markets() recently) — Kalshi may
           have closed/cancelled it early.  Re-fetching gets the true current status
           and drains the stale backlog over successive cycles.

        Returns the number of markets successfully updated.
        """
        from sqlalchemy import or_

        now = datetime.now(UTC)
        stale_cutoff = now - timedelta(seconds=self._polling_interval * 3)

        async with self._session_factory() as session:
            result = await session.execute(
                select(MarketRow.id).where(
                    MarketRow.status.notin_(["resolved", "finalized"]),
                    or_(
                        # Normal path: close_time has passed.
                        MarketRow.close_time <= now,
                        # Stale path: not returned by list_markets() recently.
                        MarketRow.last_fetched_at < stale_cutoff,
                    ),
                ).limit(_RESOLVED_SWEEP_BATCH)
            )
            market_ids = [row.id for row in result.all()]

        if not market_ids:
            return 0

        updated = 0
        markets_to_upsert = []
        fetched_markets: list[Market] = []

        if isinstance(self._client, KalshiClient):
            try:
                fetched_markets = await self._client.get_markets_by_tickers(market_ids)
            except Exception as exc:
                if self._runtime_telemetry is not None:
                    await self._runtime_telemetry.record_kalshi_error(
                        "market_watcher",
                        f"resolved sweep batch fetch failed: {exc}",
                        details={"market_ids": market_ids},
                    )
                log.exception("market_watcher.resolved_sweep_error", market_ids=market_ids)
        else:
            for market_id in market_ids:
                try:
                    fetched_markets.append(await self._client.get_market(market_id))
                except Exception as exc:
                    if self._runtime_telemetry is not None:
                        await self._runtime_telemetry.record_kalshi_error(
                            "market_watcher",
                            f"resolved sweep failed for {market_id}: {exc}",
                            details={"market_id": market_id},
                        )
                    log.exception("market_watcher.resolved_sweep_error", market_id=market_id)

        fetched_by_id = {m.id: m for m in fetched_markets}
        missing_ids = [mid for mid in market_ids if mid not in fetched_by_id]

        for market in fetched_markets:
            if market.status == "finalized" and market.result is None:
                # Kalshi sets status before populating result — skip the upsert
                # so the market stays at its current DB status and is retried
                # next cycle.  The 404 handler will eventually catch it if
                # Kalshi removes the market without ever setting a result.
                log.debug(
                    "market_watcher.resolved_sweep_result_pending",
                    market_id=market.id,
                )
            else:
                markets_to_upsert.append(market)

        for market_id in missing_ids:
            # Market gone from live API — try the settled list as a fallback
            # before giving up, since settled retains results for purged markets.
            settled_market = None
            if isinstance(self._client, KalshiClient):
                settled_market = await self._client.get_market_from_settled(market_id)
            if settled_market is not None:
                log.info(
                    "market_watcher.resolved_sweep_recovered_from_settled",
                    market_id=market_id,
                    result=settled_market.result,
                )
                markets_to_upsert.append(settled_market)
            else:
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

        if markets_to_upsert:
            async with self._session_factory() as session:
                await upsert_markets(session, markets_to_upsert)
            updated = len(markets_to_upsert)
            log.info("market_watcher.resolved_sweep", updated=updated)

        return updated

    async def _sweep_missing_results(self) -> int:
        """Back-fill result for markets stuck at finalized with no result.

        Queries the Kalshi settled list endpoint for any market in our DB that is
        finalized but has result=NULL.  If a result is found, updates the market
        and back-fills positions.resolution for all closed positions in that market.

        Returns the number of markets resolved.
        """
        if not isinstance(self._client, KalshiClient):
            return 0

        async with self._session_factory() as session:
            # Scope to markets we've generated at least one signal for —
            # these are the only markets calibration cares about.  The DB
            # can hold 90k+ finalized bracket markets we've never analyzed;
            # fetching results for those wastes API quota with no benefit.
            from freqpred.signal.models import SignalRow  # noqa: PLC0415
            rows = await session.execute(
                select(MarketRow.id)
                .join(SignalRow, SignalRow.market_id == MarketRow.id)
                .where(
                    MarketRow.status == "finalized",
                    MarketRow.result.is_(None),
                )
                .distinct()
            )
            market_ids = [r.id for r in rows.all()]

        if not market_ids:
            return 0

        _BATCH = 100
        resolved = 0
        for i in range(0, len(market_ids), _BATCH):
            batch = market_ids[i : i + _BATCH]
            markets = await self._client.get_markets_from_settled(batch)

            if markets:
                markets_to_upsert = []
                for market in markets:
                    if market.result is None:
                        continue
                    resolution = 1 if market.result == "yes" else 0
                    markets_to_upsert.append(market)
                    async with self._session_factory() as session:
                        await upsert_markets(session, [market])
                        await session.execute(
                            update(PositionRow)
                            .where(
                                PositionRow.market_id == market.id,
                                PositionRow.status == "closed",
                                PositionRow.resolution.is_(None),
                            )
                            .values(resolution=resolution)
                        )
                        await session.commit()
                    log.info(
                        "market_watcher.missing_result_resolved",
                        market_id=market.id,
                        result=market.result,
                        resolution=resolution,
                    )
                    resolved += 1

            # Brief pause between batches to stay well within rate limits.
            if i + _BATCH < len(market_ids):
                await asyncio.sleep(0.5)

        if resolved:
            log.info("market_watcher.missing_results_sweep", resolved=resolved)
        return resolved

    async def _resolve_settled_open_positions(self) -> int:
        """REST-poll fallback: close open positions (any mode) on already-settled markets.

        Two-pass approach:
        1. Markets already resolved in DB (status settled/finalized, result set) → close immediately.
        2. Markets past close_time but not yet resolved in DB → fetch from Kalshi, upsert, then close.
           This handles the case where the sweep batch hasn't reached these markets yet.

        Handles resolutions missed by the WebSocket during reconnect windows,
        and catches paper positions that the WS path skips.
        Idempotent — positions already closed are excluded by the status filter.

        Returns the number of positions closed.
        """
        now = datetime.now(UTC)

        # Pass 1: markets already resolved in DB.
        async with self._session_factory() as session:
            result = await session.execute(
                select(PositionRow.id, PositionRow.direction, PositionRow.market_id, MarketRow.result, MarketRow.settlement_value)
                .join(MarketRow, PositionRow.market_id == MarketRow.id)
                .where(
                    PositionRow.status.in_(["open", "pending"]),
                    MarketRow.status.in_(["resolved", "settled", "finalized"]),
                    MarketRow.result.is_not(None),
                )
            )
            resolved_rows = result.all()

        # Pass 2: open positions where market is past close_time but not yet resolved in DB.
        async with self._session_factory() as session:
            result = await session.execute(
                select(PositionRow.id, PositionRow.market_id)
                .join(MarketRow, PositionRow.market_id == MarketRow.id)
                .where(
                    PositionRow.status.in_(["open", "pending"]),
                    MarketRow.close_time <= now,
                    MarketRow.result.is_(None),
                )
            )
            unresolved_pos = result.all()

        # For unresolved markets with open positions, fetch result from Kalshi now.
        if unresolved_pos:
            unresolved_market_ids = list({row.market_id for row in unresolved_pos})
            for market_id in unresolved_market_ids:
                try:
                    market = await self._client.get_market(market_id)
                    if market.result is not None:
                        async with self._session_factory() as session:
                            await upsert_markets(session, [market])
                        log.info(
                            "market_watcher.fallback_fetched_result",
                            market_id=market_id,
                            result=market.result,
                        )
                    elif isinstance(self._client, KalshiClient):
                        # Try the settled endpoint as a fallback.
                        settled = await self._client.get_market_from_settled(market_id)
                        if settled is not None and settled.result is not None:
                            async with self._session_factory() as session:
                                await upsert_markets(session, [settled])
                            log.info(
                                "market_watcher.fallback_fetched_result_settled",
                                market_id=market_id,
                                result=settled.result,
                            )
                except Exception:
                    log.warning("market_watcher.fallback_fetch_failed", market_id=market_id)

            # Re-query now that markets may have been updated.
            async with self._session_factory() as session:
                result = await session.execute(
                    select(PositionRow.id, PositionRow.direction, PositionRow.market_id, MarketRow.result, MarketRow.settlement_value)
                    .join(MarketRow, PositionRow.market_id == MarketRow.id)
                    .where(
                        PositionRow.status.in_(["open", "pending"]),
                        MarketRow.close_time <= now,
                        MarketRow.result.is_not(None),
                    )
                )
                newly_resolved = result.all()
            # Merge with pass-1 rows, deduplicating by position id.
            seen = {row.id for row in resolved_rows}
            for row in newly_resolved:
                if row.id not in seen:
                    resolved_rows = list(resolved_rows) + [row]
                    seen.add(row.id)

        if not resolved_rows:
            return 0

        closed = 0
        for row in resolved_rows:
            market_result: str = row.result
            settlement_value: float | None = row.settlement_value
            resolution = 1 if market_result.lower() == "yes" else 0
            wins = row.direction.upper() == market_result.upper()
            if settlement_value is not None:
                exit_price = settlement_value if row.direction.upper() == "YES" else round(1.0 - settlement_value, 4)
            else:
                exit_price = 1.0 if wins else 0.0
            async with self._session_factory() as close_session:
                await ledger.close_position(
                    close_session,
                    str(row.id),
                    exit_price=exit_price,
                    exit_reason="market_resolved",
                    resolution=resolution,
                )
            log.info(
                "market_watcher.rest_fallback_resolved",
                position_id=str(row.id),
                direction=row.direction,
                result=market_result,
                exit_price=exit_price,
            )
            closed += 1

        return closed

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
                log.debug(
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
                MarketRow.status.notin_(["resolved", "finalized"]),
            )
        )
        return [row.id for row in result.all()]
