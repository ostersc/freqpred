"""Order manager: paper trade execution with risk enforcement."""
from __future__ import annotations

import math

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.markets.models import Market, Position
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.trading import ledger
from freqpred.trading.risk import RiskEngine, TradingCircuitBreakerError

logger = structlog.get_logger(__name__)


class OrderManager:
    def __init__(
        self,
        risk: RiskEngine,
        session_factory: async_sessionmaker[AsyncSession],
        bankroll: float,
        mode: str,
        strategy_version: str = "1.0",
    ) -> None:
        self._risk = risk
        self._session_factory = session_factory
        self._bankroll = bankroll
        self._mode = mode
        self._strategy_version = strategy_version

    async def submit(
        self,
        signal: Signal,
        market: Market,
        strategy: IPredictionStrategy,
    ) -> Position | None:
        """Full paper trade execution flow.

        1. strategy.should_trade(signal, market) → if False, return None
        2. strategy.position_size(signal, bankroll) → raw_size
        3. risk.check_position(session, signal, raw_size, bankroll)
           → if not allowed, log reason and return None
           → use decision.capped_size as final size
        4. Convert size to contracts: floor(capped_size / entry_price)
           → if contracts < 1, return None
        5. ledger.open_position(session, ...) → Position
        6. Log structured event
        7. Return Position
        """
        # Step 1: strategy gate
        if not strategy.should_trade(signal, market):
            logger.debug(
                "order_manager.strategy_declined",
                market_id=market.id,
                signal_id=signal.id,
                strategy=strategy.config.name,
            )
            return None

        # Step 2: raw position size
        raw_size = strategy.position_size(signal, self._bankroll)

        async with self._session_factory() as session:
            # Circuit breakers fire before any position check
            await self._risk.check_circuit_breakers(session, self._bankroll)

            # Step 3: risk enforcement
            decision = await self._risk.check_position(
                session,
                signal,
                raw_size,
                self._bankroll,
                market_id=market.id,
                max_market_exposure=strategy.config.max_exposure_per_market * self._bankroll,
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
                logger.debug(
                    "order_manager.contracts_below_minimum",
                    market_id=market.id,
                    capped_size=decision.capped_size,
                    entry_price=entry_price,
                )
                return None

            # Step 5: write position
            position = await ledger.open_position(
                session,
                market=market,
                signal=signal,
                strategy_name=strategy.config.name,
                strategy_version=self._strategy_version,
                direction=signal.direction,
                contracts=contracts,
                entry_price=entry_price,
                mode=self._mode,
            )

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
        )

        return position
