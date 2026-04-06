"""Order manager: paper and live trade execution with risk enforcement."""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import Market, Order, Position, PositionRow
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.trading import ledger
from freqpred.trading.risk import RiskEngine, TradingCircuitBreakerError

if TYPE_CHECKING:
    from freqpred.markets.kalshi import KalshiClient

logger = structlog.get_logger(__name__)


class OrderManager:
    def __init__(
        self,
        risk: RiskEngine,
        session_factory: async_sessionmaker[AsyncSession],
        bankroll: float,
        mode: str,
        strategy_version: str = "1.0",
        kalshi_client: KalshiClient | None = None,
    ) -> None:
        self._risk = risk
        self._session_factory = session_factory
        self._bankroll = bankroll
        self._mode = mode
        self._strategy_version = strategy_version
        self._kalshi_client = kalshi_client

    async def submit(
        self,
        signal: Signal,
        market: Market,
        strategy: IPredictionStrategy,
    ) -> Position | None:
        """Full trade execution flow (paper or live).

        0. Liquidity gate: spread > max_spread → return None
        1. strategy.should_trade(signal, market) → if False, return None
        2. strategy.position_size(signal, bankroll) → raw_size
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

            # Step 2: raw position size (uses net bankroll so Kelly sizing shrinks with losses)
            raw_size = strategy.position_size(signal, net_bankroll)

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
