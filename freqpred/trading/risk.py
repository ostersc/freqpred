"""Hard cap enforcement and circuit breakers.

IMPORTANT: Strategy code calls risk.py; risk.py has final say.
Strategy position_size() output is ALWAYS passed through here before any order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.config import RiskConfig
from freqpred.markets.models import PositionRow
from freqpred.signal.models import Signal

logger = structlog.get_logger(__name__)


@dataclass
class RiskDecision:
    allowed: bool
    reason: str        # empty string when allowed
    capped_size: float  # actual size to use (may be less than requested)


class TradingCircuitBreakerError(Exception):
    """Raised when a circuit breaker fires. Trading must be halted."""


class RiskEngine:
    def __init__(self, config: RiskConfig) -> None:
        self._config = config

    async def check_position(
        self,
        session: AsyncSession,
        signal: Signal,
        requested_size: float,
        bankroll: float,
    ) -> RiskDecision:
        """Enforce all hard caps. Returns RiskDecision(allowed=False) if any
        limit is breached. Never raises — callers check .allowed.
        Raises TradingCircuitBreakerError if a circuit breaker fires.
        """
        # 1. Edge floor check
        if signal.edge < self._config.min_edge_floor:
            logger.info(
                "risk.edge_below_floor",
                edge=signal.edge,
                floor=self._config.min_edge_floor,
            )
            return RiskDecision(
                allowed=False,
                reason=f"edge {signal.edge:.4f} below floor {self._config.min_edge_floor:.4f}",
                capped_size=0.0,
            )

        # 2. Cap position size at max_position_pct of bankroll
        max_size = bankroll * self._config.max_position_pct
        capped_size = min(requested_size, max_size)

        # 3. Max open positions check
        open_count_result = await session.execute(
            select(func.count()).select_from(PositionRow).where(
                PositionRow.status == "open"
            )
        )
        open_count: int = open_count_result.scalar_one()
        if open_count >= self._config.max_open_positions:
            logger.info(
                "risk.max_open_positions_reached",
                open_count=open_count,
                max=self._config.max_open_positions,
            )
            return RiskDecision(
                allowed=False,
                reason=f"open positions {open_count} >= max {self._config.max_open_positions}",
                capped_size=0.0,
            )

        # 4. Total exposure check
        exposure_result = await session.execute(
            select(func.sum(PositionRow.contracts * PositionRow.entry_price)).where(
                PositionRow.status == "open"
            )
        )
        total_exposure: float = exposure_result.scalar_one() or 0.0
        max_exposure = bankroll * self._config.max_total_exposure_pct
        if total_exposure > max_exposure:
            logger.info(
                "risk.total_exposure_exceeded",
                exposure=total_exposure,
                max_exposure=max_exposure,
            )
            return RiskDecision(
                allowed=False,
                reason=(
                    f"total exposure {total_exposure:.2f} > max {max_exposure:.2f} "
                    f"({self._config.max_total_exposure_pct:.0%} of bankroll)"
                ),
                capped_size=0.0,
            )

        # 5. Daily loss check
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        daily_pnl_result = await session.execute(
            select(func.sum(PositionRow.pnl)).where(
                PositionRow.status == "closed",
                PositionRow.exit_time >= today_start,
            )
        )
        daily_pnl: float = daily_pnl_result.scalar_one() or 0.0
        max_daily_loss = bankroll * self._config.max_daily_loss_pct
        if daily_pnl < 0 and abs(daily_pnl) > max_daily_loss:
            logger.warning(
                "risk.daily_loss_exceeded",
                daily_pnl=daily_pnl,
                max_daily_loss=max_daily_loss,
            )
            return RiskDecision(
                allowed=False,
                reason=(
                    f"daily loss {abs(daily_pnl):.2f} > max {max_daily_loss:.2f} "
                    f"({self._config.max_daily_loss_pct:.0%} of bankroll)"
                ),
                capped_size=0.0,
            )

        logger.debug(
            "risk.position_allowed",
            requested_size=requested_size,
            capped_size=capped_size,
            edge=signal.edge,
        )
        return RiskDecision(allowed=True, reason="", capped_size=capped_size)

    async def check_circuit_breakers(
        self,
        session: AsyncSession,
        bankroll: float,
    ) -> None:
        """Query current state and raise TradingCircuitBreakerError if:
        - daily loss > config.max_daily_loss_pct * bankroll
        - total drawdown > 30% (all-time high vs current)
        Called at the start of each signal loop cycle.
        """
        # Daily loss circuit breaker
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        daily_pnl_result = await session.execute(
            select(func.sum(PositionRow.pnl)).where(
                PositionRow.status == "closed",
                PositionRow.exit_time >= today_start,
            )
        )
        daily_pnl: float = daily_pnl_result.scalar_one() or 0.0
        max_daily_loss = bankroll * self._config.max_daily_loss_pct
        if daily_pnl < 0 and abs(daily_pnl) > max_daily_loss:
            msg = (
                f"Circuit breaker: daily loss {abs(daily_pnl):.2f} exceeds "
                f"{self._config.max_daily_loss_pct:.0%} of bankroll ({max_daily_loss:.2f})"
            )
            logger.error("risk.circuit_breaker.daily_loss", daily_pnl=daily_pnl)
            raise TradingCircuitBreakerError(msg)

        # Drawdown circuit breaker
        # ATH approximation: current bankroll + absolute value of total losses ever.
        # If cumulative P&L is negative, ATH was higher by that delta.
        all_pnl_result = await session.execute(
            select(func.sum(PositionRow.pnl)).where(PositionRow.status == "closed")
        )
        all_pnl: float = all_pnl_result.scalar_one() or 0.0
        ath_bankroll = bankroll + max(0.0, -all_pnl)
        drawdown = (ath_bankroll - bankroll) / ath_bankroll if ath_bankroll > 0 else 0.0
        _DRAWDOWN_LIMIT = 0.30
        if drawdown > _DRAWDOWN_LIMIT:
            msg = (
                f"Circuit breaker: total drawdown {drawdown:.1%} exceeds "
                f"{_DRAWDOWN_LIMIT:.0%} "
                f"(ATH bankroll: {ath_bankroll:.2f}, current: {bankroll:.2f})"
            )
            logger.error("risk.circuit_breaker.drawdown", drawdown=drawdown)
            raise TradingCircuitBreakerError(msg)
