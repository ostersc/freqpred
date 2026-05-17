"""Hard cap enforcement and circuit breakers.

IMPORTANT: Strategy code calls risk.py; risk.py has final say.
Strategy position_size() output is ALWAYS passed through here before any order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.config import RiskConfig
from freqpred.markets.models import PositionRow
from freqpred.signal.models import Signal

if TYPE_CHECKING:
    from freqpred.markets.models import Market

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

    async def _check_stoploss_reentry(
        self,
        session: AsyncSession,
        market_id: str,
        mode: str,
        block_reentry_after_stoploss: bool,
        stoploss_cooldown_hours: float,
    ) -> tuple[bool, str]:
        """Return (blocked, reason) if stoploss re-entry policy prevents a new entry.

        Returns (False, "") when neither guard is configured or no blocking exit found.
        """
        if not block_reentry_after_stoploss and stoploss_cooldown_hours <= 0:
            return False, ""
        # Block on hard stoploss, trailing stop, or any exit that closed at a loss.
        loss_exit_condition = or_(
            PositionRow.exit_reason.in_(("stoploss", "trailing_stop")),
            (PositionRow.exit_reason == "signal") & (PositionRow.pnl < 0),
            (PositionRow.exit_reason == "force_exit:manual") & (PositionRow.pnl < 0),
        )
        loss_where = [
            PositionRow.status == "closed",
            loss_exit_condition,
            PositionRow.market_id == market_id,
            PositionRow.mode == mode,
        ]
        if not block_reentry_after_stoploss:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=stoploss_cooldown_hours)
            loss_where.append(PositionRow.exit_time >= cutoff)
        loss_count_result = await session.execute(
            select(func.count(PositionRow.id)).where(*loss_where)
        )
        loss_count: int = loss_count_result.scalar_one()
        if loss_count > 0:
            if block_reentry_after_stoploss:
                reason = f"market {market_id} blocked: prior loss exit (block_reentry_after_stoploss=True)"
            else:
                reason = (
                    f"market {market_id} in loss cooldown: "
                    f"{loss_count} loss exit(s) within the last {stoploss_cooldown_hours:.1f}h"
                )
            logger.info("risk.loss_reentry_blocked", market_id=market_id, loss_count=loss_count)
            return True, reason
        return False, ""

    async def _check_global_capacity(
        self,
        session: AsyncSession,
        bankroll: float,
        mode: str,
    ) -> tuple[bool, str, float]:
        """Return (blocked, reason, total_exposure) from global position cap checks.

        Checks max_open_positions then max_total_exposure_pct.
        Returns (False, "", total_exposure) when capacity remains.
        Callers that only need (blocked, reason) should use check_entry_capacity().
        """
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
            return True, f"open positions {open_count} >= max {self._config.max_open_positions}", 0.0

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
            return True, (
                f"total exposure {total_exposure:.2f} >= max {max_exposure:.2f} "
                f"({self._config.max_total_exposure_pct:.0%} of bankroll)"
            ), total_exposure
        return False, "", total_exposure

    async def check_entry_capacity(
        self,
        session: AsyncSession,
        bankroll: float,
        mode: str,
    ) -> tuple[bool, str]:
        """Return (blocked, reason) when global caps prevent any new entry.

        Checks max_open_positions and max_total_exposure_pct.
        Returns (False, "") when at least one new position could be opened.
        """
        blocked, reason, _ = await self._check_global_capacity(session, bankroll, mode)
        return blocked, reason

    async def pre_signal_gate(
        self,
        session: AsyncSession,
        market: "Market",
        mode: str,
        *,
        effective_max_spread: float,
        block_reentry_after_stoploss: bool,
        stoploss_cooldown_hours: float,
    ) -> tuple[bool, str]:
        """Return (blocked, reason) if this market should skip signal generation.

        Checks spread (no DB) then stoploss re-entry policy.
        Returns (False, "") if signal generation should proceed.
        Only applies to new-entry markets — callers must exclude markets with
        open positions, which always need signals for exit decisions.
        """
        spread = round(market.yes_ask - market.yes_bid, 4)
        if spread > effective_max_spread:
            logger.info(
                "risk.pre_signal_gate.spread_too_wide",
                market_id=market.id,
                spread=spread,
                max_spread=effective_max_spread,
            )
            return True, f"spread {spread:.4f} > max {effective_max_spread:.4f}"
        return await self._check_stoploss_reentry(
            session, market.id, mode, block_reentry_after_stoploss, stoploss_cooldown_hours
        )

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
        sl_blocked, sl_reason = await self._check_stoploss_reentry(
            session, market_id, mode, block_reentry_after_stoploss, stoploss_cooldown_hours
        )
        if sl_blocked:
            return RiskDecision(allowed=False, reason=sl_reason, capped_size=0.0)

        # 3. Cap position size at max_position_pct of bankroll
        max_size = bankroll * self._config.max_position_pct
        capped_size = min(requested_size, max_size)

        # 4. Per-market cumulative exposure check.
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

        # 5. Max open positions + total exposure (via shared helper)
        cap_blocked, cap_reason, total_exposure = await self._check_global_capacity(session, bankroll, mode)
        if cap_blocked:
            return RiskDecision(allowed=False, reason=cap_reason, capped_size=0.0)

        max_exposure = bankroll * self._config.max_total_exposure_pct
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
