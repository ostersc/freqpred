"""Hard cap enforcement and circuit breakers.

IMPORTANT: Strategy code calls risk.py; risk.py has final say.
Strategy position_size() output is ALWAYS passed through here before any order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

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
        market_id: str,
        max_market_exposure: float,
        mode: str = "paper",
        stoploss_cooldown_hours: float = 0.0,
        block_reentry_after_stoploss: bool = False,
        daily_loss_ack_at: datetime | None = None,
    ) -> RiskDecision:
        """Enforce all hard caps. Returns RiskDecision(allowed=False) if any
        limit is breached. Never raises — callers check .allowed.
        Raises TradingCircuitBreakerError if a circuit breaker fires.

        Args:
            market_id: The market being traded — used to check cumulative exposure.
            max_market_exposure: Max dollar exposure allowed across all open positions
                for this market (strategy.max_exposure_per_market * bankroll).
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

        # 2. Stoploss re-entry guard
        _stoploss_reasons = ("stoploss", "trailing_stop")
        if block_reentry_after_stoploss or stoploss_cooldown_hours > 0:
            stoploss_where = [
                PositionRow.status == "closed",
                PositionRow.exit_reason.in_(_stoploss_reasons),
                PositionRow.market_id == market_id,
                PositionRow.mode == mode,
            ]
            if not block_reentry_after_stoploss:
                # Cooldown window only
                cutoff = datetime.now(timezone.utc) - timedelta(hours=stoploss_cooldown_hours)
                stoploss_where.append(PositionRow.exit_time >= cutoff)
            stoploss_count_result = await session.execute(
                select(func.count(PositionRow.id)).where(*stoploss_where)
            )
            stoploss_count: int = stoploss_count_result.scalar_one()
            if stoploss_count > 0:
                if block_reentry_after_stoploss:
                    reason = f"market {market_id} blocked: stoploss previously fired (block_reentry_after_stoploss=True)"
                else:
                    reason = (
                        f"market {market_id} in stoploss cooldown: "
                        f"{stoploss_count} stoploss exit(s) within the last {stoploss_cooldown_hours:.1f}h"
                    )
                logger.info("risk.stoploss_reentry_blocked", market_id=market_id, stoploss_count=stoploss_count)
                return RiskDecision(allowed=False, reason=reason, capped_size=0.0)

        # 3. Cap position size at max_position_pct of bankroll
        max_size = bankroll * self._config.max_position_pct
        capped_size = min(requested_size, max_size)

        # 3. Per-market cumulative exposure check.
        # Counts existing open positions on this market so that multiple signals
        # on the same market cannot stack exposure beyond the strategy limit.
        market_exposure_result = await session.execute(
            select(func.sum(PositionRow.contracts * PositionRow.entry_price)).where(
                PositionRow.status == "open",
                PositionRow.market_id == market_id,
                PositionRow.mode == mode,
            )
        )
        existing_market_exposure: float = market_exposure_result.scalar_one() or 0.0
        remaining_market_capacity = max_market_exposure - existing_market_exposure
        if remaining_market_capacity <= 0.0:
            logger.info(
                "risk.market_exposure_exceeded",
                market_id=market_id,
                existing_exposure=existing_market_exposure,
                max_market_exposure=max_market_exposure,
            )
            return RiskDecision(
                allowed=False,
                reason=(
                    f"market {market_id} exposure {existing_market_exposure:.2f} >= "
                    f"max {max_market_exposure:.2f} per market"
                ),
                capped_size=0.0,
            )
        # Also cap capped_size so the new position doesn't push over the limit.
        capped_size = min(capped_size, remaining_market_capacity)

        # 4. Max open positions check
        open_count_result = await session.execute(
            select(func.count()).select_from(PositionRow).where(
                PositionRow.status == "open",
                PositionRow.mode == mode,
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

        # 5. Total exposure check
        exposure_result = await session.execute(
            select(func.sum(PositionRow.contracts * PositionRow.entry_price)).where(
                PositionRow.status == "open",
                PositionRow.mode == mode,
            )
        )
        total_exposure: float = exposure_result.scalar_one() or 0.0
        max_exposure = bankroll * self._config.max_total_exposure_pct
        if total_exposure >= max_exposure:
            logger.info(
                "risk.total_exposure_exceeded",
                exposure=total_exposure,
                max_exposure=max_exposure,
            )
            return RiskDecision(
                allowed=False,
                reason=(
                    f"total exposure {total_exposure:.2f} >= max {max_exposure:.2f} "
                    f"({self._config.max_total_exposure_pct:.0%} of bankroll)"
                ),
                capped_size=0.0,
            )
        # Cap so the new position cannot push total exposure over the limit.
        remaining_total_capacity = max_exposure - total_exposure
        capped_size = min(capped_size, remaining_total_capacity)

        # 6. Daily loss check
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # If the operator acknowledged the circuit breaker via /start, measure
        # losses only since that acknowledgement (not since midnight).  This
        # prevents a previously-tripped breaker from immediately re-blocking
        # after resume while still protecting against *new* losses of the same
        # magnitude.
        loss_window_start = (
            max(today_start, daily_loss_ack_at)
            if daily_loss_ack_at is not None
            else today_start
        )
        daily_pnl_result = await session.execute(
            select(func.sum(PositionRow.pnl)).where(
                PositionRow.status == "closed",
                PositionRow.exit_time >= loss_window_start,
                PositionRow.mode == mode,
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
        mode: str,
        drawdown_reset_bankroll: "float | None" = None,
        daily_loss_ack_at: "datetime | None" = None,
    ) -> None:
        """Query current state and raise TradingCircuitBreakerError if:
        - daily loss > config.max_daily_loss_pct * bankroll
        - drawdown from reset baseline > 30%
        Only positions matching ``mode`` are considered.
        Called at the start of each signal loop cycle.

        ``bankroll`` must be the current net bankroll (initial deposit ± all
        closed P&L).  ``drawdown_reset_bankroll`` is the net bankroll stored
        when /reset_drawdown was last called; if None the drawdown check is
        skipped (no baseline established yet).

        ``daily_loss_ack_at`` is set by /start.  When provided, the daily loss
        window starts at ``max(today_start, daily_loss_ack_at)`` so that losses
        incurred before the operator acknowledged the breaker do not immediately
        re-trip it on the next cycle.
        """
        # Daily loss circuit breaker
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        loss_window_start = (
            max(today_start, daily_loss_ack_at)
            if daily_loss_ack_at is not None
            else today_start
        )
        daily_pnl_result = await session.execute(
            select(func.sum(PositionRow.pnl)).where(
                PositionRow.status == "closed",
                PositionRow.exit_time >= loss_window_start,
                PositionRow.mode == mode,
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

        # Drawdown circuit breaker — compare current net bankroll against the
        # value stored when the drawdown window was last reset.
        if drawdown_reset_bankroll is not None and drawdown_reset_bankroll > 0:
            drawdown = max(0.0, (drawdown_reset_bankroll - bankroll) / drawdown_reset_bankroll)
            _DRAWDOWN_LIMIT = 0.30
            if drawdown > _DRAWDOWN_LIMIT:
                msg = (
                    f"Circuit breaker: drawdown {drawdown:.1%} exceeds "
                    f"{_DRAWDOWN_LIMIT:.0%} "
                    f"(baseline: {drawdown_reset_bankroll:.2f}, current: {bankroll:.2f})"
                )
                logger.error("risk.circuit_breaker.drawdown", drawdown=drawdown)
                raise TradingCircuitBreakerError(msg)
