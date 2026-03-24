"""PositionWatcher: persistent Kalshi WebSocket ticker subscription for open live positions.

Maintains a single ``ticker`` channel subscription for every market where freqpred
holds an open or pending live position.  Provides sub-second price updates and
triggers PositionMonitor on each tick.  REST polling (MarketWatcher) continues
unchanged for markets without live positions.

Connection lifecycle:
  On connect  → reconcile DB vs Kalshi positions → reconcile pending orders
               → re-query open markets → subscribe to ticker channel
  While alive → handle ticker messages → upsert DB price → call PositionMonitor
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
    ) -> None:
        self._kalshi_client = kalshi_client
        self._ws_url = ws_url
        self._session_factory = session_factory
        self._position_monitor = position_monitor
        self._order_manager = order_manager
        self._price_move_threshold = price_move_threshold

        # Set of market_ids currently in the ticker subscription.
        self._subscribed: set[str] = set()
        # Live websockets connection (set while connected, None otherwise).
        self._ws: websockets.WebSocketClientProtocol | None = None
        # Subscription ID assigned by Kalshi for the active ticker subscription.
        self._subscription_sid: int | None = None
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
        """Add a market to the ticker subscription.

        Called by cli.py after a new live position is opened.
        Safe to call when disconnected — market_id is added to the in-memory
        set and will be included on the next reconnect.
        """
        if market_id in self._subscribed:
            return
        self._subscribed.add(market_id)
        if self._ws is not None:
            if self._subscription_sid is not None:
                await self._send_add_markets(self._ws, [market_id])
            else:
                await self._send_subscribe(self._ws, [market_id])

    async def unsubscribe(self, market_id: str) -> None:
        """Remove a market from the ticker subscription.

        Called after a live position closes.  Safe to call when disconnected.
        """
        self._subscribed.discard(market_id)
        if self._ws is not None and self._subscription_sid is not None:
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
            self._subscription_sid = None
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

                async for raw in ws:
                    await self._handle_message(json.loads(raw))
            finally:
                self._ws = None
                self._subscription_sid = None

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: dict) -> None:
        """Dispatch incoming WebSocket message by type."""
        msg_type = msg.get("type")

        if msg_type == "subscribed":
            sid = msg.get("msg", {}).get("sid")
            if sid is not None:
                self._subscription_sid = sid
            log.info(
                "position_watcher.subscribed",
                channel=msg.get("msg", {}).get("channel"),
                sid=sid,
            )

        elif msg_type == "ticker":
            inner = msg.get("msg", {})
            market_id = inner.get("market_ticker")
            # Kalshi WebSocket v2 sends yes_bid/yes_ask as integer cents (0–100),
            # not the yes_bid_dollars dollar-string used by the REST API.
            yes_bid_raw = inner.get("yes_bid")
            yes_ask_raw = inner.get("yes_ask")
            if market_id and yes_bid_raw is not None and yes_ask_raw is not None:
                yes_bid = float(yes_bid_raw) / 100  # cents → dollars
                yes_ask = float(yes_ask_raw) / 100  # cents → dollars
                # Only process ticks for markets where we hold positions.
                if market_id in self._subscribed:
                    await self._on_ticker_update(market_id, yes_bid, yes_ask)

        elif msg_type == "error":
            code = msg.get("msg", {}).get("code")
            error_msg = msg.get("msg", {}).get("msg")
            log.warning("position_watcher.ws_error", code=code, error=error_msg)

        else:
            log.debug("position_watcher.unhandled_message", type=msg_type)

    async def _on_ticker_update(
        self, market_id: str, yes_bid: float, yes_ask: float
    ) -> None:
        """Process a ticker update: upsert DB price, log move, trigger monitor."""
        now = datetime.now(UTC)
        new_mid = round((yes_bid + yes_ask) / 2, 4)
        log.debug(
            "position_watcher.tick",
            market_id=market_id,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            mid=new_mid,
        )

        async with self._session_factory() as session:
            await session.execute(
                update(MarketRow)
                .where(MarketRow.id == market_id)
                .values(
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    mid_price=new_mid,
                    last_fetched_at=now,
                    price_updated_at=case(
                        (MarketRow.yes_bid != yes_bid, now),
                        else_=MarketRow.price_updated_at,
                    ),
                )
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
                    # Kalshi has no position — auto-close.
                    exit_price = market_mids.get(market_id, 0.0)
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
        """Send a fresh subscribe command for the given market_ids."""
        await ws.send(
            json.dumps(
                {
                    "id": self._next_id(),
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["ticker"],
                        "market_tickers": market_ids,
                    },
                }
            )
        )

    async def _send_add_markets(
        self, ws: websockets.WebSocketClientProtocol, market_ids: list[str]
    ) -> None:
        """Extend existing subscription with additional market_ids."""
        await ws.send(
            json.dumps(
                {
                    "id": self._next_id(),
                    "cmd": "update_subscription",
                    "params": {
                        "sids": [self._subscription_sid],
                        "market_tickers": market_ids,
                        "action": "add_markets",
                    },
                }
            )
        )

    async def _send_remove_markets(
        self, ws: websockets.WebSocketClientProtocol, market_ids: list[str]
    ) -> None:
        """Remove market_ids from existing subscription."""
        await ws.send(
            json.dumps(
                {
                    "id": self._next_id(),
                    "cmd": "update_subscription",
                    "params": {
                        "sids": [self._subscription_sid],
                        "market_tickers": market_ids,
                        "action": "delete_markets",
                    },
                }
            )
        )
