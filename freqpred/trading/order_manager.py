"""Order manager: paper and live trade execution with risk enforcement."""
from __future__ import annotations

import inspect
import math
import os
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import Market, MarketRow, Order, Position, PositionRow
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.trading import ledger
from freqpred.trading.risk import RiskEngine, TradingCircuitBreakerError

if TYPE_CHECKING:
    from freqpred.markets.kalshi import KalshiClient
    from freqpred.llm.client import LLMClient
    from freqpred.metrics.models import SignalAssessment
    from freqpred.runtime.telemetry import RuntimeTelemetry

logger = structlog.get_logger(__name__)


class PositionNotFoundError(ValueError):
    """Raised by force_exit when the position does not exist in the current mode."""


class PositionNotOpenError(ValueError):
    """Raised by force_exit when the position is not open (already closed/cancelled)."""


class OrderManager:
    def __init__(
        self,
        risk: RiskEngine,
        session_factory: async_sessionmaker[AsyncSession],
        bankroll: float,
        mode: str,
        strategy_version: str = "1.0",
        kalshi_client: KalshiClient | None = None,
        llm_client: LLMClient | None = None,
        judgment_model: str | None = None,
        runtime_telemetry: "RuntimeTelemetry | None" = None,
    ) -> None:
        self._risk = risk
        self._session_factory = session_factory
        self._bankroll = bankroll
        self._mode = mode
        self._strategy_version = strategy_version
        self._kalshi_client = kalshi_client
        self._llm_client = llm_client
        self._judgment_model = judgment_model
        self._runtime_telemetry = runtime_telemetry

    @staticmethod
    def _call_position_size(
        strategy: IPredictionStrategy,
        signal: Signal,
        bankroll: float,
        existing_market_exposure: float,
        assessment: "SignalAssessment | None",
    ) -> float:
        """Call strategy.position_size() without breaking legacy overrides."""
        method = strategy.position_size
        signature = inspect.signature(method)
        params = signature.parameters
        has_var_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        positional_params = [
            p
            for p in params.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        call_kwargs: dict[str, SignalAssessment | float | None] = {}
        if "assessment" in params or has_var_kwargs:
            call_kwargs["assessment"] = assessment

        if "existing_market_exposure" in params:
            call_kwargs["existing_market_exposure"] = existing_market_exposure
            return method(signal, bankroll, **call_kwargs)

        if len(positional_params) >= 3 or has_var_args:
            return method(signal, bankroll, existing_market_exposure, **call_kwargs)

        return method(signal, bankroll, **call_kwargs)

    async def submit(
        self,
        signal: Signal,
        market: Market,
        strategy: IPredictionStrategy,
    ) -> Position | None:
        """Full trade execution flow (paper or live).

        0. Liquidity gate: spread > max_spread → return None
        1. strategy.should_trade(signal, market) → if False, return None
        2. Optional signal assessment → assessment-aware position_size(...) → raw_size
        3. risk.check_position(session, signal, raw_size, bankroll)
           → if not allowed, log reason and return None
           → use decision.capped_size as final size
        4. Convert size to contracts: floor(capped_size / entry_price)
           → if contracts < 1, return None
        5. Route to _submit_live() or _submit_paper()
        6. Log structured event
        7. Return Position
        """
        # Step 0: liquidity gate — reject if spread > max_spread (default: min_edge / 2).
        # Prevents phantom stoploss triggers caused by stale ask-side quotes on illiquid markets.
        spread = round(market.yes_ask - market.yes_bid, 4)
        effective_max_spread = (
            strategy.config.max_spread
            if strategy.config.max_spread is not None
            else strategy.config.min_edge / 2
        )
        if spread > effective_max_spread:
            logger.info(
                "order_manager.spread_too_wide",
                market_id=market.id,
                spread=spread,
                max_spread=effective_max_spread,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
            )
            return None

        # Step 1: strategy gate
        if not strategy.should_trade(signal, market):
            logger.debug(
                "order_manager.strategy_declined",
                market_id=market.id,
                signal_id=signal.id,
                strategy=strategy.config.name,
            )
            return None

        async with self._session_factory() as session:
            # Compute current net bankroll (initial deposit ± all closed P&L) so
            # all risk caps are denominated against the active balance, not the
            # original deposit.
            net_bankroll = await ledger.get_net_bankroll(session, self._bankroll, mode=self._mode)

            # Query existing open exposure for this market so position_size()
            # computes only the *incremental* amount — doubling down only when
            # the new signal's edge/conviction justifies more total exposure.
            existing_exposure_result = await session.execute(
                select(func.coalesce(
                    func.sum(PositionRow.contracts * PositionRow.entry_price), 0.0
                )).where(
                    PositionRow.status == "open",
                    PositionRow.market_id == market.id,
                    PositionRow.mode == self._mode,
                )
            )
            existing_market_exposure: float = float(existing_exposure_result.scalar_one())

            # Price-improvement gate: if we already hold a position in this market,
            # only add to it when the entry price is strictly better than the weighted
            # average of what we already paid.  Blocks assessment-jitter add-ons and
            # price_moved re-entries at the same or worse price.
            if existing_market_exposure > 0.0:
                entry_price_check = (
                    market.yes_ask
                    if signal.direction == "YES"
                    else round(1.0 - market.yes_bid, 4)
                )
                avg_entry_result = await session.execute(
                    select(
                        func.sum(PositionRow.contracts * PositionRow.entry_price)
                        / func.sum(PositionRow.contracts)
                    ).where(
                        PositionRow.status == "open",
                        PositionRow.market_id == market.id,
                        PositionRow.mode == self._mode,
                        PositionRow.direction == signal.direction,
                    )
                )
                avg_entry = avg_entry_result.scalar_one_or_none()
                if avg_entry is not None and entry_price_check >= float(avg_entry):
                    logger.info(
                        "order_manager.no_price_improvement",
                        market_id=market.id,
                        signal_id=signal.id,
                        new_entry_price=entry_price_check,
                        avg_existing_entry=round(float(avg_entry), 4),
                        direction=signal.direction,
                    )
                    return None

            assessment = None
            if self._llm_client is not None and self._judgment_model:
                from freqpred.metrics.assessment import assess_signal_context  # noqa: PLC0415

                assessment = await assess_signal_context(
                    session,
                    signal,
                    market,
                    strategy,
                    self._llm_client,
                    self._judgment_model,
                )

            # Step 2: raw position size (uses net bankroll so Kelly sizing shrinks with losses)
            raw_size = self._call_position_size(
                strategy,
                signal,
                net_bankroll,
                existing_market_exposure,
                assessment,
            )
            if raw_size <= 0.0 and existing_market_exposure > 0.0:
                logger.info(
                    "order_manager.no_incremental_edge",
                    market_id=market.id,
                    signal_id=signal.id,
                    existing_exposure=existing_market_exposure,
                    edge=signal.edge,
                    confidence=signal.confidence,
                )
                return None

            # Circuit breakers fire before any position check.
            from freqpred.alerts.run_state import (  # noqa: PLC0415
                get_daily_loss_ack_at,
                get_drawdown_window,
            )
            _, drawdown_reset_bankroll = await get_drawdown_window(session)
            daily_loss_ack_at = await get_daily_loss_ack_at(session)
            await self._risk.check_circuit_breakers(
                session, net_bankroll, mode=self._mode,
                drawdown_reset_bankroll=drawdown_reset_bankroll,
                daily_loss_ack_at=daily_loss_ack_at,
            )

            # Step 3: risk enforcement
            decision = await self._risk.check_position(
                session,
                signal,
                raw_size,
                net_bankroll,
                market_id=market.id,
                max_market_exposure=strategy.config.max_exposure_per_market * net_bankroll,
                mode=self._mode,
                stoploss_cooldown_hours=strategy.config.stoploss_cooldown_hours,
                block_reentry_after_stoploss=strategy.config.block_reentry_after_stoploss,
                daily_loss_ack_at=daily_loss_ack_at,
            )
            if not decision.allowed:
                logger.info(
                    "order_manager.risk_blocked",
                    market_id=market.id,
                    signal_id=signal.id,
                    reason=decision.reason,
                )
                return None

            # Step 4: size → contracts
            entry_price = (
                market.yes_ask
                if signal.direction == "YES"
                else 1.0 - market.yes_bid
            )
            contracts = math.floor(decision.capped_size / entry_price)
            if contracts < 1:
                logger.info(
                    "order_manager.contracts_below_minimum",
                    market_id=market.id,
                    capped_size=decision.capped_size,
                    entry_price=entry_price,
                )
                return None

            order = Order(
                market_id=market.id,
                direction=signal.direction,
                contracts=contracts,
                price=entry_price,
                mode=self._mode,
            )

            # Step 5: route to live or paper branch
            if self._mode == "live":
                if os.environ.get("LIVE_TRADING_ENABLED") != "true":
                    logger.error(
                        "order_manager.live_blocked",
                        reason="LIVE_TRADING_ENABLED not set",
                    )
                    return None
                position = await self._submit_live(
                    order, signal, market, session, strategy.config.name
                )
            else:
                position = await self._submit_paper(
                    order, signal, market, session, strategy.config.name
                )

        if position is None:
            return None

        # Step 6: structured audit log
        logger.info(
            "order_manager.order_submitted",
            market_id=market.id,
            signal_id=signal.id,
            direction=signal.direction,
            contracts=contracts,
            entry_price=entry_price,
            edge=signal.edge,
            mode=self._mode,
            position_id=position.id,
            exchange_order_id=position.exchange_order_id,
        )

        return position

    async def _submit_paper(
        self,
        order: Order,
        signal: Signal,
        market: Market,
        session: AsyncSession,
        strategy_name: str,
    ) -> Position:
        """Write position as status='open' immediately — no exchange interaction."""
        return await ledger.open_position(
            session,
            market=market,
            signal=signal,
            strategy_name=strategy_name,
            strategy_version=self._strategy_version,
            direction=order.direction,
            contracts=order.contracts,
            entry_price=order.price,
            mode=self._mode,
            status="open",
        )

    async def _submit_live(
        self,
        order: Order,
        signal: Signal,
        market: Market,
        session: AsyncSession,
        strategy_name: str,
    ) -> Position | None:
        """Submit order to Kalshi REST API.

        Records position as status='pending' immediately (before fill confirmation).
        PositionWatcher (T39) will update status to 'open' once the exchange confirms a fill.
        """
        assert self._kalshi_client is not None, "kalshi_client required for live mode"
        try:
            filled_order = await self._kalshi_client.place_order(order)
        except KalshiAPIError as exc:
            logger.warning(
                "order_manager.live_order_failed",
                market_id=order.market_id,
                direction=order.direction,
                contracts=order.contracts,
                price=order.price,
                status_code=exc.status_code,
                body=exc.body,
            )
            if self._runtime_telemetry is not None:
                await self._runtime_telemetry.record_kalshi_error(
                    "order_manager",
                    f"submit live order failed for {order.market_id}: {exc}",
                    details={"market_id": order.market_id, "status_code": exc.status_code},
                )
            return None
        # "executed" means immediately filled; "resting" means GTC order sitting on the book.
        position_status = "open" if filled_order.status == "executed" else "pending"
        # Effective entry cost per contract including exchange fee.
        effective_entry = order.price + (filled_order.fee_usd / order.contracts if order.contracts else 0)
        logger.info(
            "order_manager.live_order_submitted",
            exchange_order_id=filled_order.exchange_order_id,
            market_id=order.market_id,
            direction=order.direction,
            contracts=order.contracts,
            price=order.price,
            fee_usd=filled_order.fee_usd,
            effective_entry_price=round(effective_entry, 6),
            position_status=position_status,
        )
        return await ledger.open_position(
            session,
            market=market,
            signal=signal,
            strategy_name=strategy_name,
            strategy_version=self._strategy_version,
            direction=order.direction,
            contracts=order.contracts,
            entry_price=order.price,
            mode=self._mode,
            status=position_status,
            exchange_order_id=filled_order.exchange_order_id,
            entry_fee_usd=filled_order.fee_usd,
        )

    async def reconcile_pending_orders(self, session: AsyncSession) -> None:
        """Query Kalshi for status of all pending live orders; update positions.

        For each 'pending' live PositionRow:
          - If Kalshi holds a non-zero net position for that market → flip to 'open'.
          - Otherwise → mark 'cancelled' (order did not fill).

        Called by PositionWatcher on startup and after every reconnect.
        """
        if self._kalshi_client is None:
            return

        result = await session.execute(
            select(PositionRow).where(
                PositionRow.status == "pending",
                PositionRow.mode == "live",
            )
        )
        pending = result.scalars().all()
        if not pending:
            return

        kalshi_positions = await self._kalshi_client.get_positions()
        kalshi_by_market: dict[str, int] = {
            p.market_id: p.contracts for p in kalshi_positions
        }

        for row in pending:
            if kalshi_by_market.get(row.market_id, 0) > 0:
                row.status = "open"
                logger.info(
                    "order_manager.pending_filled",
                    position_id=str(row.id),
                    market_id=row.market_id,
                    kalshi_contracts=kalshi_by_market[row.market_id],
                )
            else:
                row.status = "cancelled"
                logger.warning(
                    "order_manager.pending_not_filled",
                    position_id=str(row.id),
                    market_id=row.market_id,
                )

        await session.commit()

    async def force_exit(self, position_id: str, *, exit_reason: str = "force_exit:manual") -> Position:
        """Close an open position manually.

        Scopes the lookup to self._mode so paper positions are not accessible
        from a live-mode manager and vice versa.

        Paper mode: closes the ledger at current market mid.
        Live mode with kalshi_client:
          - Requires LIVE_TRADING_ENABLED=true; raises ValueError otherwise
            (operator/config guard, not a server crash).
          - Raises ValueError if kalshi_client is None (wiring regression guard).
          - If the market is already resolved (result known), closes at settlement
            payout without submitting an exchange order.
          - Otherwise submits an IOC sell at the executable side bid. Uses the
            reconciled/net Kalshi position size (get_positions()) rather than
            PositionRow.contracts to avoid partial-fill drift. If no matching
            live position exists on the exchange, raises ValueError and leaves
            the DB row open (reconciliation required).
          - If limit_price <= 0 (market closed to trading), raises ValueError —
            settlement reconciliation will handle the payout.
          - Propagates KalshiAPIError; position is left open on exchange failure.
        Raises PositionNotFoundError if position is not found for this mode.
        Raises PositionNotOpenError if position status != 'open'.
        Raises ValueError for other precondition failures.
        """
        import uuid as _uuid  # noqa: PLC0415

        if self._mode == "live" and os.environ.get("LIVE_TRADING_ENABLED") != "true":
            raise ValueError("LIVE_TRADING_ENABLED must be 'true' for live force exits")

        async with self._session_factory() as session:
            result = await session.execute(
                select(PositionRow, MarketRow)
                .join(MarketRow, PositionRow.market_id == MarketRow.id)
                .where(
                    PositionRow.id == _uuid.UUID(position_id),
                    PositionRow.mode == self._mode,
                )
            )
            row = result.one_or_none()
            if row is None:
                raise PositionNotFoundError(
                    f"Position {position_id!r} not found for mode={self._mode!r}"
                )

            pos_row, mkt_row = row
            if pos_row.status != "open":
                raise PositionNotOpenError(
                    f"Position {position_id!r} is not open (status={pos_row.status!r})"
                )

            mid = mkt_row.mid_price if mkt_row.mid_price is not None else 0.5
            exit_price = 1.0 - mid if pos_row.direction == "NO" else mid

            if self._mode == "live":
                if self._kalshi_client is None:
                    raise ValueError(
                        f"Live force exit for {position_id!r} requires a KalshiClient; "
                        "none was wired into this OrderManager"
                    )

                # Reconcile against the exchange before any live close path so resolved
                # manual exits use the actual net size held on Kalshi.
                kalshi_positions = await self._kalshi_client.get_positions()
                exchange_pos = next(
                    (
                        p for p in kalshi_positions
                        if p.market_id == pos_row.market_id and p.direction == pos_row.direction
                    ),
                    None,
                )
                if exchange_pos is None or exchange_pos.contracts <= 0:
                    raise ValueError(
                        f"No live exchange position found for {position_id!r}; "
                        "reconciliation required before force exit"
                    )
                if exchange_pos.contracts != pos_row.contracts:
                    pos_row.contracts = exchange_pos.contracts

                # If settlement result is already known, close at payout price
                if mkt_row.result is not None:
                    result_str = (mkt_row.result or "").lower()
                    settlement_price = 1.0 if (
                        (result_str == "yes" and pos_row.direction == "YES") or
                        (result_str == "no" and pos_row.direction == "NO")
                    ) else 0.0
                    return await ledger.close_position(
                        session,
                        position_id,
                        exit_price=settlement_price,
                        exit_reason=exit_reason,
                    )

                bid = mkt_row.yes_bid or 0.0
                ask = mkt_row.yes_ask or 1.0
                limit_price = round(bid if pos_row.direction == "YES" else 1.0 - ask, 4)
                if limit_price <= 0:
                    raise ValueError(
                        f"No executable exit price for {position_id!r} "
                        "(market closed to trading; awaiting settlement reconciliation)"
                    )

                exit_order = Order(
                    market_id=pos_row.market_id,
                    direction=pos_row.direction,
                    contracts=pos_row.contracts,
                    price=limit_price,
                    mode="live",
                    time_in_force="fill_or_kill",
                    action="sell",
                )
                filled = await self._kalshi_client.place_order(exit_order)
                exit_price = filled.price

            return await ledger.close_position(
                session,
                position_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
            )
