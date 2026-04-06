"""PositionWatcher: persistent Kalshi WebSocket ticker subscription for open live positions.

Maintains a ``ticker`` + ``market_lifecycle_v2`` channel subscription for every market
where freqpred holds an open or pending live position.  Provides sub-second price
updates and triggers PositionMonitor on each tick.  On ``market_lifecycle_v2`` settled
events, closes all open live positions at the correct payout price.  REST polling
(MarketWatcher) continues unchanged for markets without live positions.

Connection lifecycle:
  On connect  → reconcile DB vs Kalshi positions → reconcile pending orders
               → re-query open markets → subscribe to ticker + market_lifecycle channels
  While alive → handle ticker messages → upsert DB price → call PositionMonitor
              → handle market_lifecycle settled → close positions + unsubscribe
  On disconnect → exponential backoff (1s → 2s → 4s … 60s max) → reconnect
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
import websockets
from sqlalchemy import case, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.markets.models import MarketRow, PositionRow
from freqpred.trading import ledger

if TYPE_CHECKING:
    from freqpred.alerts.dispatcher import AlertDispatcher
    from freqpred.markets.kalshi import KalshiClient
    from freqpred.trading.order_manager import OrderManager
    from freqpred.trading.position_monitor import PositionMonitor

log = structlog.get_logger(__name__)

# Kalshi WS signing path (literal, independent of base REST URL).
_WS_SIGN_PATH = "/trade-api/ws/v2"


class PositionWatcher:
    """Persistent WebSocket ticker subscription for all open live positions.

    Instantiate once and run as an asyncio task via ``run()``.  The watcher
    automatically reconnects with exponential backoff on disconnect and
    re-builds its subscription set from the DB on every reconnect.
    """

    def __init__(
        self,
        kalshi_client: "KalshiClient",
        ws_url: str,
        session_factory: async_sessionmaker[AsyncSession],
        position_monitor: "PositionMonitor",
        order_manager: "OrderManager",
        price_move_threshold: float = 0.05,
        alert_dispatcher: "AlertDispatcher | None" = None,
    ) -> None:
        self._kalshi_client = kalshi_client
        self._ws_url = ws_url
        self._session_factory = session_factory
        self._position_monitor = position_monitor
        self._order_manager = order_manager
        self._price_move_threshold = price_move_threshold
        self._alert_dispatcher = alert_dispatcher

        # Set of market_ids currently in the ticker subscription.
        self._subscribed: set[str] = set()
        # Live websockets connection (set while connected, None otherwise).
        self._ws: websockets.WebSocketClientProtocol | None = None
        # Subscription IDs assigned by Kalshi (one per channel subscribed).
        # Kalshi may return multiple sids when subscribing to multiple channels.
        self._subscription_sids: list[int] = []
        # Sid specifically for the ticker channel — used for update_subscription.
        # market_lifecycle_v2 is a global broadcast and cannot be updated with
        # market_tickers; Kalshi requires exactly one sid per update_subscription.
        self._ticker_sid: int | None = None
        # Sequential command ID (Kalshi requires monotonically increasing ids).
        self._msg_id: int = 0
        # In-memory tracking of last known mid_price per market for Δ logging.
        self._last_mid: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Async background loop. Reconnects with exponential backoff on error."""
        log.info("position_watcher.starting", ws_url=self._ws_url)
        backoff = 1.0
        while True:
            try:
                await self._connect_and_subscribe()
                backoff = 1.0  # reset on clean disconnect
            except asyncio.CancelledError:
                log.info("position_watcher.stopped")
                raise
            except Exception:
                log.exception("position_watcher.disconnected", backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def subscribe(self, market_id: str) -> None:
        """Add a market to the ticker + market_lifecycle subscription.

        Called by cli.py after a new live position is opened.
        Safe to call when disconnected — market_id is added to the in-memory
        set and will be included on the next reconnect.
        """
        if market_id in self._subscribed:
            return
        self._subscribed.add(market_id)
        if self._ws is not None:
            if self._ticker_sid is not None:
                await self._send_add_markets(self._ws, [market_id])
            else:
                await self._send_subscribe(self._ws, [market_id])

    async def unsubscribe(self, market_id: str) -> None:
        """Remove a market from the subscription.

        Called after a live position closes or a market settles.
        Safe to call when disconnected.
        """
        self._subscribed.discard(market_id)
        if self._ws is not None and self._ticker_sid is not None:
            await self._send_remove_markets(self._ws, [market_id])

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_subscribe(self) -> None:
        """Establish WebSocket, reconcile, subscribe, then consume messages."""
        auth_headers = self._kalshi_client._make_auth_headers("GET", _WS_SIGN_PATH)

        # Python websockets auto-handles Kalshi's ping/pong frames.
        # No manual heartbeat needed.
        async with websockets.connect(
            self._ws_url, additional_headers=auth_headers
        ) as ws:
            self._ws = ws
            self._subscription_sids = []
            self._ticker_sid = None
            log.info("position_watcher.connected", ws_url=self._ws_url)

            try:
                # Reconcile DB positions vs Kalshi on every connect.
                async with self._session_factory() as session:
                    await self._reconcile_positions(session)

                # Flip any pending positions that filled while disconnected.
                async with self._session_factory() as session:
                    await self._order_manager.reconcile_pending_orders(session)

                # Re-build subscription set from DB (no stale in-memory state).
                async with self._session_factory() as session:
                    self._subscribed = await self._get_open_market_ids(session)

                if self._subscribed:
                    await self._send_subscribe(ws, list(self._subscribed))
                else:
                    log.info("position_watcher.no_open_positions", hint="no ticker subscriptions sent")

                async for raw in ws:
                    await self._handle_message(json.loads(raw))
            finally:
                self._ws = None
                self._subscription_sids = []
                self._ticker_sid = None

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: dict) -> None:
        """Dispatch incoming WebSocket message by type."""
        msg_type = msg.get("type")

        if msg_type == "subscribed":
            inner = msg.get("msg", {})
            sid = inner.get("sid")
            channel = inner.get("channel")
            if sid is not None and sid not in self._subscription_sids:
                self._subscription_sids.append(sid)
            if channel == "ticker" and sid is not None:
                self._ticker_sid = sid
            log.info(
                "position_watcher.subscribed",
                channel=channel,
                sid=sid,
            )

        elif msg_type == "ticker":
            inner = msg.get("msg", {})
            market_id = inner.get("market_ticker")
            log.debug("position_watcher.ticker_raw", market_id=market_id, raw=str(inner)[:500])
            # Kalshi WebSocket v2 sends prices as dollar strings (yes_bid_dollars,
            # yes_ask_dollars), not integer cents.
            yes_bid_raw = inner.get("yes_bid_dollars")
            yes_ask_raw = inner.get("yes_ask_dollars")
            last_price_raw = inner.get("price_dollars")
            if market_id and yes_bid_raw is not None and yes_ask_raw is not None:
                yes_bid = float(yes_bid_raw)
                yes_ask = float(yes_ask_raw)
                last_price = float(last_price_raw) if last_price_raw is not None else 0.0
                # Only process ticks for markets where we hold positions.
                if market_id in self._subscribed:
                    await self._on_ticker_update(market_id, yes_bid, yes_ask, last_price)
                else:
                    log.warning("position_watcher.tick_not_subscribed", market_id=market_id)
            else:
                log.warning("position_watcher.ticker_missing_fields", raw=str(inner)[:200])

        elif msg_type == "market_lifecycle_v2":
            inner = msg.get("msg", {})
            market_id = inner.get("market_ticker")
            status = inner.get("event_type", "")
            if not market_id:
                log.warning(
                    "position_watcher.lifecycle_missing_market_ticker",
                    raw=str(inner)[:200],
                )
                return
            # settlement_value is present on "determined" events only ("1.0000"=YES, "0.0000"=NO).
            # "settled" events carry only settled_ts — no result field.
            sv = inner.get("settlement_value")
            result = ("yes" if float(sv) >= 0.5 else "no") if sv is not None else None
            # Always update DB status/result for any market we have a row for.
            if status:
                await self._update_market_status(market_id, status, result)
            # Only drive position lifecycle for markets we're subscribed to.
            if market_id not in self._subscribed:
                log.debug(
                    "position_watcher.lifecycle_unsubscribed",
                    market_id=market_id,
                    event_type=status,
                )
                return
            await self._on_market_lifecycle(market_id, status, result)

        elif msg_type == "error":
            code = msg.get("msg", {}).get("code")
            error_msg = msg.get("msg", {}).get("msg")
            log.warning("position_watcher.ws_error", code=code, error=error_msg)

        else:
            log.debug("position_watcher.unhandled_message", type=msg_type)

    async def _on_ticker_update(
        self, market_id: str, yes_bid: float, yes_ask: float, last_price: float = 0.0
    ) -> None:
        """Process a ticker update: upsert DB price, log move, trigger monitor."""
        now = datetime.now(UTC)
        spread_mid = round((yes_bid + yes_ask) / 2, 4)
        # Use last_price as mid when available — (bid+ask)/2 is misleading on
        # illiquid markets where the ask side is momentarily wide.
        new_mid = round(last_price, 4) if last_price > 0 else spread_mid
        log.debug(
            "position_watcher.tick",
            market_id=market_id,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            last_price=last_price,
            spread_mid=spread_mid,
            mid=new_mid,
        )

        values: dict = {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "mid_price": new_mid,
            "last_fetched_at": now,
            "price_updated_at": case(
                (MarketRow.yes_bid != yes_bid, now),
                else_=MarketRow.price_updated_at,
            ),
        }
        if last_price > 0:
            values["last_price"] = last_price

        async with self._session_factory() as session:
            await session.execute(
                update(MarketRow)
                .where(MarketRow.id == market_id)
                .values(**values)
            )
            await session.commit()

        # Log significant price moves (for observability and debugging).
        prev_mid = self._last_mid.get(market_id)
        if prev_mid is not None and abs(new_mid - prev_mid) >= self._price_move_threshold:
            log.info(
                "position_watcher.price_moved",
                market_id=market_id,
                prev_mid=prev_mid,
                new_mid=new_mid,
                delta=round(new_mid - prev_mid, 4),
            )
        self._last_mid[market_id] = new_mid

        # Forward raw tick to algo strategies before evaluating exits.
        self._position_monitor.on_tick(market_id, yes_bid, yes_ask, now)

        # Trigger exit evaluation for all open positions.
        await self._position_monitor.check_all_positions()

    async def _on_market_lifecycle(
        self, market_id: str, status: str, result: str | None
    ) -> None:
        """Handle a market_lifecycle_v2 WebSocket event.

        Kalshi lifecycle: "active" → "determined" → "settled"

        - "determined": carries settlement_value (result) — close positions immediately.
        - "settled":    no settlement_value; positions should already be closed from
                        "determined", but if missed we fall back to REST to get result.
        - other:        log at debug level and continue.
        """
        if status == "determined":
            log.info(
                "position_watcher.market_determined",
                market_id=market_id,
                result=result,
            )
            if result is None:
                # Unexpected — determined should carry settlement_value.
                log.warning(
                    "position_watcher.determined_missing_result",
                    market_id=market_id,
                )
                return
            await self._close_positions_for_resolved_market(market_id, result)
            await self.unsubscribe(market_id)
            return

        if status != "settled":
            log.debug(
                "position_watcher.market_lifecycle",
                market_id=market_id,
                status=status,
            )
            return

        # settled event has no settlement_value — positions should already be closed
        # from the earlier determined event.  Unsubscribe and let
        # MarketWatcher._resolve_settled_live_positions handle any missed determinations.
        log.info("position_watcher.market_settled", market_id=market_id)
        await self.unsubscribe(market_id)

    async def _close_positions_for_resolved_market(
        self, market_id: str, result: str
    ) -> None:
        """Close all open live positions for a market that has resolved.

        Payout logic:
          - direction matches result (e.g. YES holds YES-wins market) → exit_price = 1.00
          - direction opposes result → exit_price = 0.00
        """
        resolution = 1 if result.lower() == "yes" else 0

        async with self._session_factory() as session:
            market_res = await session.execute(
                select(MarketRow).where(MarketRow.id == market_id)
            )
            market_row = market_res.scalar_one_or_none()

            pos_res = await session.execute(
                select(PositionRow).where(
                    PositionRow.market_id == market_id,
                    PositionRow.mode == "live",
                    PositionRow.status.in_(["open", "pending"]),
                )
            )
            rows = pos_res.scalars().all()

        if not rows:
            log.info("position_watcher.no_positions_to_resolve", market_id=market_id)
            return

        for row in rows:
            wins = row.direction.upper() == result.upper()
            exit_price = 1.0 if wins else 0.0
            async with self._session_factory() as close_session:
                closed = await ledger.close_position(
                    close_session,
                    str(row.id),
                    exit_price=exit_price,
                    exit_reason="market_resolved",
                    resolution=resolution,
                )
            log.info(
                "position_watcher.position_resolved",
                market_id=market_id,
                position_id=str(row.id),
                direction=row.direction,
                result=result,
                exit_price=exit_price,
                pnl=closed.pnl,
            )
            if self._alert_dispatcher is not None:
                pnl_val = closed.pnl or 0.0
                prefix = "WIN" if pnl_val >= 0 else "LOSS"
                pnl_str = f"{pnl_val:+.4f}"
                question = market_row.question if market_row is not None else market_id
                msg = (
                    f"{prefix}: {question} resolved {result.upper()}\n"
                    f"P&L: {pnl_str}  |  Direction: {closed.direction}  |  "
                    f"Entry: {closed.entry_price:.4f}  Exit: {closed.exit_price:.4f}"
                )
                await self._alert_dispatcher.send(msg)

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------

    async def _reconcile_positions(self, session: AsyncSession) -> None:
        """Sync DB open/pending live positions against Kalshi get_positions().

        For each open/pending live PositionRow in DB:
          - If Kalshi net contracts differ from DB contracts: update DB, log warning.
          - If Kalshi has no position for this market (net=0): auto-close at
            current market mid_price, log warning.

        For each Kalshi position with no matching DB record:
          - Log info and skip (manual trade placed outside freqpred).
        """
        # Load DB live positions.
        result = await session.execute(
            select(PositionRow).where(
                PositionRow.mode == "live",
                PositionRow.status.in_(["open", "pending"]),
            )
        )
        db_rows = {row.market_id: row for row in result.scalars().all()}

        # Fetch Kalshi positions (always needed — kalshi-only logging runs even when db_rows empty).
        kalshi_positions = await self._kalshi_client.get_positions()
        kalshi_net: dict[str, int] = {p.market_id: p.contracts for p in kalshi_positions}

        if db_rows:
            # Load current market mid_prices for auto-close exits.
            market_result = await session.execute(
                select(MarketRow.id, MarketRow.mid_price).where(
                    MarketRow.id.in_(db_rows.keys())
                )
            )
            market_mids: dict[str, float] = {
                row.id: row.mid_price for row in market_result.all()
            }

            # Sync DB → Kalshi.
            rows_to_auto_close: list[tuple[str, float]] = []  # (position_id, exit_price)
            for market_id, row in db_rows.items():
                net = kalshi_net.get(market_id, 0)
                if net == 0:
                    # Kalshi has no position — auto-close at effective price.
                    mid = market_mids.get(market_id, 0.0)
                    exit_price = 1.0 - mid if row.direction == "NO" else mid
                    rows_to_auto_close.append((str(row.id), exit_price))
                    log.warning(
                        "position_watcher.reconcile_auto_close",
                        market_id=market_id,
                        position_id=str(row.id),
                        exit_price=exit_price,
                    )
                elif net != row.contracts:
                    log.warning(
                        "position_watcher.reconcile_contracts_updated",
                        market_id=market_id,
                        position_id=str(row.id),
                        db_contracts=row.contracts,
                        kalshi_contracts=net,
                    )
                    row.contracts = net

            await session.commit()

            # Auto-close zero-net positions in fresh sessions (ledger.close_position
            # needs its own session since it loads and commits the row internally).
            for position_id, exit_price in rows_to_auto_close:
                async with self._session_factory() as close_session:
                    await ledger.close_position(
                        close_session,
                        position_id,
                        exit_price=exit_price,
                        exit_reason="reconcile_auto_close",
                    )

        # Log Kalshi-only positions (manual trades outside freqpred).
        for market_id in kalshi_net:
            if market_id not in db_rows:
                log.info(
                    "position_watcher.reconcile_kalshi_only",
                    market_id=market_id,
                    hint="manual trade placed outside freqpred, skipping",
                )

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    async def _update_market_status(
        self, market_id: str, status: str, result: str | None
    ) -> None:
        """Persist market status/result from a market_lifecycle_v2 event.

        Silently skips if the market has no DB row (markets we don't monitor).
        """
        values: dict = {"status": status}
        if result is not None:
            values["result"] = result
        async with self._session_factory() as session:
            await session.execute(
                update(MarketRow).where(MarketRow.id == market_id).values(**values)
            )
            await session.commit()

    async def _get_open_market_ids(self, session: AsyncSession) -> set[str]:
        """Return market_ids for all open/pending positions (paper and live).

        Paper positions subscribe to the same ticker feed as live positions so
        that TA/algo strategies receive the same sub-second price updates in
        both modes.  Reconciliation (live-only) is handled separately.
        """
        result = await session.execute(
            select(PositionRow.market_id).where(
                PositionRow.status.in_(["open", "pending"]),
            )
        )
        return {row.market_id for row in result.all()}

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send_subscribe(
        self, ws: websockets.WebSocketClientProtocol, market_ids: list[str]
    ) -> None:
        """Send a fresh subscribe command for the given market_ids.

        Subscribes to both ``ticker`` (sub-second price updates) and
        ``market_lifecycle_v2`` (resolution events) in a single command.
        Kalshi may return one or two ``subscribed`` events (one per channel);
        each sid is appended to ``_subscription_sids``.
        """
        log.info("position_watcher.subscribing", market_ids=sorted(market_ids), count=len(market_ids))
        await ws.send(
            json.dumps(
                {
                    "id": self._next_id(),
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["ticker", "market_lifecycle_v2"],
                        "market_tickers": market_ids,
                    },
                }
            )
        )

    async def _send_add_markets(
        self, ws: websockets.WebSocketClientProtocol, market_ids: list[str]
    ) -> None:
        """Extend existing ticker subscription with additional market_ids.

        Uses only the ticker sid — market_lifecycle_v2 is a global broadcast
        and does not support market_ticker filtering. Kalshi requires exactly
        one sid per update_subscription call.
        """
        log.info("position_watcher.adding_markets", market_ids=sorted(market_ids))
        await ws.send(
            json.dumps(
                {
                    "id": self._next_id(),
                    "cmd": "update_subscription",
                    "params": {
                        "sids": [self._ticker_sid],
                        "market_tickers": market_ids,
                        "action": "add_markets",
                    },
                }
            )
        )

    async def _send_remove_markets(
        self, ws: websockets.WebSocketClientProtocol, market_ids: list[str]
    ) -> None:
        """Remove market_ids from the ticker subscription.

        Uses only the ticker sid for the same reason as _send_add_markets.
        """
        await ws.send(
            json.dumps(
                {
                    "id": self._next_id(),
                    "cmd": "update_subscription",
                    "params": {
                        "sids": [self._ticker_sid],
                        "market_tickers": market_ids,
                        "action": "delete_markets",
                    },
                }
            )
        )
