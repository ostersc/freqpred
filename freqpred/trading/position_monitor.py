"""Position monitor: background loop evaluating exit conditions for open positions.

Exit priority order (per SPEC §8):
  1. Hard stoploss        — framework-enforced, cannot be overridden by strategy
  2. Trailing stoploss    — trails from best mid-price since entry
  3. Force exit           — strategy.force_exit(), signal-independent, every tick
  4. Custom exit          — strategy.custom_exit(), requires fresh signal
  5. Signal exit          — strategy.should_exit(), only when a fresh signal is passed in
  6. Market resolution    — market close_time passed (paper: simulated at current price)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import Market, MarketRow, Order, Position
from freqpred.trading import ledger

if TYPE_CHECKING:
    from freqpred.alerts.dispatcher import AlertDispatcher
    from freqpred.markets.kalshi import KalshiClient
    from freqpred.runtime.telemetry import RuntimeTelemetry
    from freqpred.signal.models import Signal
    from freqpred.strategy.base import IPredictionStrategy
    from freqpred.strategy.config import StrategyConfig

logger = structlog.get_logger(__name__)

# Sentinel returned by check_position when no exit should fire.
_NO_EXIT: tuple[str, float] | None = None


class PositionMonitor:
    """Evaluates exit conditions for all open positions on each price poll.

    Instantiate once and call ``run()`` as an asyncio task, or call
    ``check_all_positions()`` from the main signal loop.

    ``peak_prices`` tracks the best mid-price seen for each open position
    since entry (required for trailing stop). State is in-memory and resets
    on restart — acceptable for paper trading where precision is not critical.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        strategies: dict[str, IPredictionStrategy],
        poll_interval_seconds: float = 60.0,
        alert_dispatcher: "AlertDispatcher | None" = None,
        mode: str = "paper",
        kalshi_client: "KalshiClient | None" = None,
        runtime_telemetry: "RuntimeTelemetry | None" = None,
    ) -> None:
        self._session_factory = session_factory
        self._strategies = strategies
        self._poll_interval = poll_interval_seconds
        self._alert_dispatcher = alert_dispatcher
        self._mode = mode
        self._kalshi_client = kalshi_client
        self._runtime_telemetry = runtime_telemetry
        # position_id → best mid_price seen since entry (used by trailing stop)
        self._peak_prices: dict[str, float] = {}
        # position_id → best/worst effective P&L delta seen (for MAE/MFE)
        self._peak_deltas: dict[str, float] = {}    # MFE
        self._trough_deltas: dict[str, float] = {}  # MAE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_tick(
        self,
        market_id: str,
        yes_bid: float,
        yes_ask: float,
        ts: "datetime",
    ) -> None:
        """Forward a WebSocket tick to every IAlgoStrategy in the registry.

        Sync (non-async) so it can be called from PositionWatcher's async
        handler without an extra await.  No-op for plain IPredictionStrategy
        instances.
        """
        from freqpred.strategy.algo_base import IAlgoStrategy  # local — avoids circular

        for strategy in self._strategies.values():
            if isinstance(strategy, IAlgoStrategy):
                strategy.ingest_tick(market_id, yes_bid, yes_ask, ts)

    async def run(self) -> None:
        """Async background loop. Runs until cancelled."""
        logger.info("position_monitor.started", poll_interval=self._poll_interval)
        while True:
            try:
                await self.check_all_positions()
            except Exception:
                logger.exception("position_monitor.check_error")
            await asyncio.sleep(self._poll_interval)

    async def check_all_positions(
        self,
        *,
        fresh_signals: dict[str, Signal] | None = None,
    ) -> list[Position]:
        """Evaluate all open positions against current market prices.

        Args:
            fresh_signals: mapping of market_id → new Signal, passed in when
                the signal pipeline has just re-analysed a market with an open
                position.  Used for step 5 (signal exit).

        Returns:
            List of positions that were closed during this call.
        """
        fresh_signals = fresh_signals or {}
        closed: list[Position] = []

        async with self._session_factory() as session:
            positions = await ledger.get_open_positions(session, mode=self._mode)
            if not positions:
                return closed

            # Fetch all relevant markets in one query
            market_ids = {p.market_id for p in positions}
            market_rows = await session.execute(
                select(MarketRow).where(MarketRow.id.in_(market_ids))
            )
            markets: dict[str, Market] = {
                row.id: _market_row_to_domain(row)
                for row in market_rows.scalars().all()
            }

        for position in positions:
            market = markets.get(position.market_id)
            if market is None:
                logger.warning(
                    "position_monitor.market_not_found",
                    position_id=position.id,
                    market_id=position.market_id,
                )
                continue

            strategy = self._strategies.get(position.strategy_name)
            if strategy is None:
                logger.warning(
                    "position_monitor.strategy_not_found",
                    position_id=position.id,
                    strategy_name=position.strategy_name,
                )
                continue

            # Use the bid price for the held side — what you'd actually receive on exit.
            # This prevents phantom stoploss triggers on illiquid markets where the ask
            # side is stale and inflates mid_price far above the last traded price.
            # YES position: receive yes_bid when selling.
            # NO position:  receive no_bid = (1 - yes_ask) when selling.
            if position.direction == "NO":
                current_price = round(1.0 - market.yes_ask, 4)
            else:
                current_price = market.yes_bid
            fresh_signal = fresh_signals.get(position.market_id)

            result = self.evaluate_exit(
                position=position,
                market=market,
                current_price=current_price,
                strategy=strategy,
                fresh_signal=fresh_signal,
            )
            if result is None:
                # No exit — update peak price tracker (trailing stop) + MAE/MFE.
                # current_price is already the side bid — pass directly.
                self._update_peak(position, current_price)
                await self._update_excursions(position, current_price)
                continue

            exit_reason, exit_price = result
            # Derive YES/NO resolution from market.result set by Kalshi.
            # Falls back to price-threshold inference if result not yet populated.
            resolution: int | None = None
            if market.result == "yes":
                resolution = 1 if position.direction == "YES" else 0
            elif market.result == "no":
                resolution = 0 if position.direction == "YES" else 1
            elif exit_price >= 0.99:
                resolution = 1 if position.direction == "YES" else 0
            elif exit_price <= 0.01:
                resolution = 0 if position.direction == "YES" else 1
            async with self._session_factory() as session:
                closed_position = await self._execute_exit(
                    session=session,
                    position=position,
                    market=market,
                    exit_reason=exit_reason,
                    exit_price=exit_price,
                    resolution=resolution,
                )
            if closed_position is None:
                # Live exit failed — position stays open; skip cleanup and alert
                logger.warning(
                    "position_monitor.live_exit_skipped",
                    position_id=position.id,
                    exit_reason=exit_reason,
                )
                continue
            # Clean up trackers
            self._peak_prices.pop(position.id, None)
            self._peak_deltas.pop(position.id, None)
            self._trough_deltas.pop(position.id, None)

            logger.info(
                "position_monitor.exit_triggered",
                position_id=position.id,
                market_id=position.market_id,
                strategy=position.strategy_name,
                exit_reason=exit_reason,
                exit_price=exit_price,
                entry_price=position.entry_price,
                contracts=position.contracts,
                pnl=closed_position.pnl,
                mode=position.mode,
            )
            if self._alert_dispatcher is not None:
                try:
                    await self._alert_dispatcher.exit_alert(closed_position, exit_reason)
                except Exception:
                    logger.exception("position_monitor.alert_failed", position_id=position.id)
            closed.append(closed_position)

        return closed

    def evaluate_exit(
        self,
        *,
        position: Position,
        market: Market,
        current_price: float,
        strategy: IPredictionStrategy,
        fresh_signal: Signal | None = None,
    ) -> tuple[str, float] | None:
        """Evaluate all exit conditions for a single position.

        Returns ``(exit_reason, exit_price)`` or ``None`` to hold.
        Pure function — no I/O. Separated for unit-testability.
        """
        config = strategy.config
        # current_price is already the side-specific bid (yes_bid for YES, no_bid for NO).
        # For exit evaluation it is passed through as-is — it already represents
        # what the holder would receive when selling.
        effective_price = current_price
        peak_price = self._peak_prices.get(position.id, position.entry_price)
        now = datetime.now(tz=timezone.utc)

        # 0. If the market result is already known, settle at the correct payout immediately.
        # Stoploss and trailing stop use live market prices which become invalid once a market
        # stops trading (the order book empties, causing yes_ask to default to 1.0 in the
        # Kalshi orderbook fetch, which makes no_bid appear as 0.0 — a phantom stoploss).
        # Checking result first prevents spurious stoploss exits on resolved markets.
        if market.result is not None:
            if market.settlement_value is not None:
                # Scalar market: use actual settlement price, not a rounded 0 or 1.
                exit_price = (
                    market.settlement_value if position.direction.upper() == "YES"
                    else round(1.0 - market.settlement_value, 4)
                )
            else:
                wins = position.direction.upper() == market.result.upper()
                exit_price = 1.0 if wins else 0.0
            return ("market_resolved", exit_price)

        # 0.5. Empty order book guard: when bid collapses to 0 but market.result is not yet
        # set, the zero is an order book artefact (Kalshi clears the book on close/pause before
        # writing result), not a real traded price. Substitute last_price as the effective price
        # so stoploss and trailing stop evaluate against the most recent actual trade.
        #
        # YES position: effective_price = yes_bid → 0; proxy = last_price (last YES trade).
        # NO position:  effective_price = 1 - yes_ask → 0; proxy = 1 - last_price.
        #
        # A genuine collapse to zero still fires: if last_price is also near 0 for a YES
        # position (or near 1 for a NO position), the proxy is similarly small and the
        # stoploss check below will correctly trigger.
        if effective_price == 0.0 and market.result is None:
            if position.direction == "YES":
                proxy = market.last_price
            else:
                proxy = round(1.0 - market.last_price, 4)
            if proxy > 0.0:
                logger.debug(
                    "position_monitor.empty_book_proxy",
                    position_id=position.id,
                    market_id=position.market_id,
                    direction=position.direction,
                    last_price=market.last_price,
                    proxy=proxy,
                )
                effective_price = proxy

        # 1. Hard stoploss (framework-enforced)
        result = _check_stoploss(position, effective_price, config.stoploss)
        if result:
            return result

        # 2. Trailing stoploss
        if config.trailing_stop:
            result = _check_trailing_stop(
                position,
                effective_price,
                peak_price,
                config.stoploss,
                config.trailing_stop_positive,
                config.trailing_stop_positive_offset,
            )
            if result:
                return result

        # 3. Force exit (signal-independent — strategy's own initiative)
        tag = strategy.force_exit(position, market)
        if tag is not None:
            return (f"force_exit:{tag}", effective_price)

        # 4. Custom exit hook (signal-informed)
        if fresh_signal is not None:
            tag = strategy.custom_exit(position, fresh_signal, market)
            if tag is not None:
                return (f"custom_exit:{tag}", effective_price)

        # 5. Signal exit (only when a fresh signal is provided)
        if fresh_signal is not None:
            if strategy.should_exit(position, fresh_signal, market):
                return ("signal", effective_price)

        # 6. Market resolution — Kalshi has confirmed a terminal status.
        # Note: market.result is not None is already handled at step 0 above.
        # Do NOT close on close_time alone — Kalshi may pause/delay settlement and
        # the result field won't be set yet. Wait for _sweep_closed_markets to
        # populate market.result (caught by step 0) or a terminal status.
        if market.status in ("finalized", "resolved", "settled"):
            return ("market_resolved", effective_price)

        return _NO_EXIT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_exit(
        self,
        *,
        session: AsyncSession,
        position: Position,
        market: Market,
        exit_reason: str,
        exit_price: float,
        resolution: int | None,
    ) -> Position | None:
        """Route to live or paper exit branch."""
        if position.mode == "live":
            return await self._execute_live_exit(
                session=session,
                position=position,
                market=market,
                exit_reason=exit_reason,
                resolution=resolution,
            )
        return await self._execute_paper_exit(
            session=session,
            position=position,
            exit_reason=exit_reason,
            exit_price=exit_price,
            resolution=resolution,
        )

    async def _execute_paper_exit(
        self,
        *,
        session: AsyncSession,
        position: Position,
        exit_reason: str,
        exit_price: float,
        resolution: int | None,
    ) -> Position:
        """Close position in the ledger directly — no exchange interaction."""
        return await ledger.close_position(
            session,
            position.id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            resolution=resolution,
        )

    async def _execute_live_exit(
        self,
        *,
        session: AsyncSession,
        position: Position,
        market: Market,
        exit_reason: str,
        resolution: int | None,
    ) -> Position | None:
        """Submit an IOC sell order to Kalshi, then close the position in the ledger.

        Submits a sell order for the same side as the position (sell YES to exit a YES
        position, sell NO to exit a NO position).  The order is IOC so it either fills
        immediately or is cancelled — no resting exit orders are left.

        On KalshiAPIError:
        - Position stays open in DB (caller skips ledger close).
        - Telegram alert is sent so the operator can intervene manually.
        """
        assert self._kalshi_client is not None, "kalshi_client required for live exits"

        # Sell the same side we hold; price at the bid for best IOC fill probability
        if position.direction == "YES":
            limit_price = round(market.yes_bid or 0.0, 4)
        else:
            limit_price = round(1.0 - (market.yes_ask or 1.0), 4)

        if limit_price <= 0:
            # Market is resolved or closed to trading — Kalshi will settle it.
            # Leave the DB position open; reconciliation will handle the payout.
            logger.warning(
                "position_monitor.exit_skip_zero_bid",
                position_id=position.id,
                market_id=position.market_id,
                exit_reason=exit_reason,
            )
            return None

        exit_order = Order(
            market_id=position.market_id,
            direction=position.direction,
            contracts=position.contracts,
            price=limit_price,
            mode="live",
            time_in_force="fill_or_kill",
            action="sell",
        )

        try:
            filled_order = await self._kalshi_client.place_order(exit_order)
        except KalshiAPIError as exc:
            logger.error(
                "position_monitor.live_exit_failed",
                position_id=position.id,
                market_id=position.market_id,
                exit_reason=exit_reason,
                status_code=exc.status_code,
                body=exc.body,
            )
            if self._runtime_telemetry is not None:
                await self._runtime_telemetry.record_kalshi_error(
                    "position_monitor",
                    f"exit order failed for {position.market_id}: {exc}",
                    details={"market_id": position.market_id, "status_code": exc.status_code},
                )
            if self._alert_dispatcher is not None:
                try:
                    await self._alert_dispatcher.send(
                        f"Failed to submit exit order for {market.question} "
                        f"(position {position.id}, reason: {exit_reason}). "
                        "Manual intervention required."
                    )
                except Exception:
                    logger.exception(
                        "position_monitor.alert_failed", position_id=position.id
                    )
            return None

        # Use the confirmed fill price from the exchange response
        confirmed_exit_price = filled_order.price
        return await ledger.close_position(
            session,
            position.id,
            exit_price=confirmed_exit_price,
            exit_reason=exit_reason,
            resolution=resolution,
        )

    def _update_peak(self, position: Position, current_price: float) -> None:
        """Advance peak price if current_price is better than recorded peak."""
        peak = self._peak_prices.get(position.id, position.entry_price)
        if current_price > peak:
            self._peak_prices[position.id] = current_price

    async def _update_excursions(self, position: Position, current_price: float) -> None:
        """Update MAE/MFE for a position and persist to DB when a new extreme is hit."""
        # current_price is the side bid (yes_bid for YES, no_bid for NO) — already converted.
        delta = current_price - position.entry_price

        is_first_observation = position.id not in self._peak_deltas
        prev_best = self._peak_deltas.get(position.id, position.mfe if position.mfe is not None else delta)
        prev_worst = self._trough_deltas.get(position.id, position.mae if position.mae is not None else delta)

        new_best = max(prev_best, delta)
        new_worst = min(prev_worst, delta)

        # Write if a new extreme is found, or on first observation when DB has no value yet
        changed = new_best > prev_best or new_worst < prev_worst or (is_first_observation and position.mae is None)
        self._peak_deltas[position.id] = new_best
        self._trough_deltas[position.id] = new_worst

        if changed:
            async with self._session_factory() as session:
                await ledger.update_position_excursions(
                    session,
                    position.id,
                    mae=new_worst,
                    mfe=new_best,
                )


# ---------------------------------------------------------------------------
# Pure exit-condition helpers (no I/O — easy to unit test independently)
# ---------------------------------------------------------------------------


def _check_stoploss(
    position: Position,
    current_price: float,
    stoploss: float,
) -> tuple[str, float] | None:
    """Return ('stoploss', current_price) if the hard stoploss has been hit.

    ``stoploss`` is an absolute drop on the 0-1 price scale (e.g. -0.10 means
    exit when the price has fallen 10 cents from entry).
    """
    dollar_loss = current_price - position.entry_price
    if dollar_loss <= stoploss:
        return ("stoploss", current_price)
    return None


def _check_trailing_stop(
    position: Position,
    current_price: float,
    peak_price: float,
    stoploss: float,
    trailing_stop_positive: float | None,
    trailing_stop_positive_offset: float,
) -> tuple[str, float] | None:
    """Return ('trailing_stop', current_price) if the trailing stoploss has been hit.

    All thresholds are absolute dollar distances on the 0-1 price scale.

    Logic:
    - If ``trailing_stop_positive`` is set and the position has gained at least
      that many cents from entry, apply the tighter trail
      (``trailing_stop_positive_offset`` cents below peak).
    - Otherwise, apply the normal stoploss distance below peak.
    """
    entry = position.entry_price
    peak_gain = peak_price - entry  # absolute cents gained from entry to peak

    if (
        trailing_stop_positive is not None
        and peak_gain >= trailing_stop_positive
    ):
        # Tight trail: stop = peak - offset (cents)
        trail_distance = trailing_stop_positive_offset
    else:
        # Normal trail: stop = peak - |stoploss| (cents)
        trail_distance = -stoploss  # convert negative to positive distance

    stop_price = peak_price - trail_distance
    if current_price <= stop_price:
        return ("trailing_stop", current_price)
    return None


def _market_row_to_domain(row: MarketRow) -> Market:
    return Market(
        id=row.id,
        platform=row.platform,
        question=row.question,
        category=row.category,
        status=row.status,
        result=row.result,
        settlement_value=row.settlement_value,
        close_time=row.close_time,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
        mid_price=row.mid_price,
        last_price=row.last_price,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        liquidity=row.liquidity,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
        metadata=row.metadata_ or {},
        open_time=row.open_time,
    )
