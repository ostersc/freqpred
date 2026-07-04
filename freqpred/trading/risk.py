"""Hard cap enforcement and circuit breakers.

IMPORTANT: Strategy code calls risk.py; risk.py has final say.
Strategy position_size() output is ALWAYS passed through here before any order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import case, func, select
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


@dataclass
class PortfolioSnapshot:
    """Frozen portfolio state consumed by :meth:`RiskEngine.evaluate_position_caps`.

    Gathered from the DB by :meth:`RiskEngine.check_position`; constructed
    directly (with an empty-portfolio default) by the replay harness so the
    same cap arithmetic can run against a fixture state without a session.
    """

    loss_exit_count: int = 0            # blocking loss exits for this market (reentry guard)
    market_open_exposure: float = 0.0   # open exposure on this market
    market_pending_exposure: float = 0.0
    open_count: int = 0                 # global open position count
    pending_count: int = 0
    open_exposure: float = 0.0          # global open exposure
    pending_exposure: float = 0.0
    daily_pnl: float = 0.0              # closed P&L inside the loss window


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
        """Return (blocked, reason) if the loss re-entry policy prevents a new entry.

        Despite the name (kept for config-flag compatibility), this blocks on
        *any* closed loss for this market, not just stoploss exits — see the
        loss_where condition below.
        Returns (False, "") when neither guard is configured or no blocking exit found.
        """
        if not block_reentry_after_stoploss and stoploss_cooldown_hours <= 0:
            return False, ""
        # Block on any closed exit that lost money, regardless of exit_reason —
        # a loss is a loss whether it came from a hard stoploss, a signal flip,
        # a manual/algo force-exit, a resolved market, or a reconcile close.
        loss_where = [
            PositionRow.status == "closed",
            PositionRow.pnl < 0,
            PositionRow.market_id == market_id,
            PositionRow.mode == mode,
        ]
        if not block_reentry_after_stoploss:
            cutoff = datetime.now(UTC) - timedelta(hours=stoploss_cooldown_hours)
            loss_where.append(PositionRow.exit_time >= cutoff)
        loss_count_result = await session.execute(
            select(func.count(PositionRow.id)).where(*loss_where)
        )
        loss_count: int = loss_count_result.scalar_one()
        return self._stoploss_reentry_decision(
            market_id, loss_count, block_reentry_after_stoploss, stoploss_cooldown_hours
        )

    def _edge_floor_decision(self, signal_edge: float) -> RiskDecision | None:
        """Return a rejection when edge is below the hard floor, else None."""
        if signal_edge < self._config.min_edge_floor:
            logger.info(
                "risk.edge_below_floor",
                edge=signal_edge,
                floor=self._config.min_edge_floor,
            )
            return RiskDecision(
                allowed=False,
                reason=f"edge {signal_edge:.4f} below floor {self._config.min_edge_floor:.4f}",
                capped_size=0.0,
            )
        return None

    @staticmethod
    def _stoploss_reentry_decision(
        market_id: str,
        loss_count: int,
        block_reentry_after_stoploss: bool,
        stoploss_cooldown_hours: float,
    ) -> tuple[bool, str]:
        """Pure decision part of the stoploss re-entry guard."""
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
        Pending live orders count as committed exposure — they reserve capital
        on the exchange until they fill or are cancelled.
        Returns (False, "", total_exposure) when capacity remains.
        Callers that only need (blocked, reason) should use check_entry_capacity().
        """
        active_statuses = ("open", "pending")
        count_result = await session.execute(
            select(
                func.sum(
                    case((PositionRow.status == "open", 1), else_=0)
                ).label("open_count"),
                func.sum(
                    case((PositionRow.status == "pending", 1), else_=0)
                ).label("pending_count"),
            )
            .select_from(PositionRow)
            .where(
                PositionRow.status.in_(active_statuses),
                PositionRow.mode == mode,
            )
        )
        counts = count_result.one()
        open_count = int(counts.open_count or 0)
        pending_count = int(counts.pending_count or 0)
        count_blocked, count_reason = self._max_positions_decision(open_count, pending_count)
        if count_blocked:
            return True, count_reason, 0.0

        # COALESCE(requested_contracts, contracts) so pending rows count their
        # request size (the exchange has reserved that capital) while legacy
        # rows missing requested_contracts fall back to their fill count.
        exposure_expr = (
            func.coalesce(PositionRow.requested_contracts, PositionRow.contracts)
            * PositionRow.entry_price
        )
        exposure_result = await session.execute(
            select(
                func.coalesce(
                    func.sum(
                        case(
                            (PositionRow.status == "open", exposure_expr),
                            else_=0.0,
                        )
                    ),
                    0.0,
                ).label("open_exposure"),
                func.coalesce(
                    func.sum(
                        case(
                            (PositionRow.status == "pending", exposure_expr),
                            else_=0.0,
                        )
                    ),
                    0.0,
                ).label("pending_exposure"),
            ).where(
                PositionRow.status.in_(active_statuses),
                PositionRow.mode == mode,
            )
        )
        exposure_row = exposure_result.one()
        open_exposure = float(exposure_row.open_exposure or 0.0)
        pending_exposure = float(exposure_row.pending_exposure or 0.0)
        exposure_blocked, exposure_reason, total_exposure = self._total_exposure_decision(
            open_exposure, pending_exposure, bankroll
        )
        return exposure_blocked, exposure_reason, total_exposure

    def _max_positions_decision(
        self, open_count: int, pending_count: int
    ) -> tuple[bool, str]:
        """Pure decision part of the max-open-positions cap."""
        total_active = open_count + pending_count
        if total_active >= self._config.max_open_positions:
            logger.info(
                "risk.max_open_positions_reached",
                open_count=open_count,
                pending_count=pending_count,
                total_active=total_active,
                max=self._config.max_open_positions,
            )
            return (
                True,
                (
                    f"active positions {total_active} (open={open_count}, "
                    f"pending={pending_count}) >= max {self._config.max_open_positions}"
                ),
            )
        return False, ""

    def _total_exposure_decision(
        self, open_exposure: float, pending_exposure: float, bankroll: float
    ) -> tuple[bool, str, float]:
        """Pure decision part of the total-exposure cap."""
        total_exposure = open_exposure + pending_exposure
        max_exposure = bankroll * self._config.max_total_exposure_pct
        if total_exposure >= max_exposure:
            logger.info(
                "risk.total_exposure_exceeded",
                exposure=total_exposure,
                open_exposure=open_exposure,
                pending_exposure=pending_exposure,
                max_exposure=max_exposure,
            )
            return True, (
                f"total exposure {total_exposure:.2f} (open={open_exposure:.2f}, "
                f"pending={pending_exposure:.2f}) >= max {max_exposure:.2f} "
                f"({self._config.max_total_exposure_pct:.0%} of bankroll)"
            ), total_exposure
        return False, "", total_exposure

    def _global_capacity_decision(
        self,
        open_count: int,
        pending_count: int,
        open_exposure: float,
        pending_exposure: float,
        bankroll: float,
    ) -> tuple[bool, str, float]:
        """Pure composition of the global capacity checks, in production order."""
        count_blocked, count_reason = self._max_positions_decision(open_count, pending_count)
        if count_blocked:
            return True, count_reason, 0.0
        return self._total_exposure_decision(open_exposure, pending_exposure, bankroll)

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
        market: Market,
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
        # Edge floor is checked before any DB work so a too-weak signal is
        # rejected without touching the session.
        edge_rejection = self._edge_floor_decision(signal.edge)
        if edge_rejection is not None:
            return edge_rejection

        snapshot = await self._gather_portfolio_snapshot(
            session,
            market_id=market_id,
            mode=mode,
            block_reentry_after_stoploss=block_reentry_after_stoploss,
            stoploss_cooldown_hours=stoploss_cooldown_hours,
            daily_loss_ack_at=daily_loss_ack_at,
        )
        return self.evaluate_position_caps(
            signal_edge=signal.edge,
            requested_size=requested_size,
            bankroll=bankroll,
            market_id=market_id,
            max_market_exposure=max_market_exposure,
            snapshot=snapshot,
            block_reentry_after_stoploss=block_reentry_after_stoploss,
            stoploss_cooldown_hours=stoploss_cooldown_hours,
        )

    async def _gather_portfolio_snapshot(
        self,
        session: AsyncSession,
        *,
        market_id: str,
        mode: str,
        block_reentry_after_stoploss: bool,
        stoploss_cooldown_hours: float,
        daily_loss_ack_at: datetime | None,
    ) -> PortfolioSnapshot:
        """Query the DB state consumed by evaluate_position_caps()."""
        # Loss-exit count for the stoploss re-entry guard. Skip the query when
        # neither guard is configured (mirrors _check_stoploss_reentry).
        loss_exit_count = 0
        if block_reentry_after_stoploss or stoploss_cooldown_hours > 0:
            # Block on any closed exit that lost money, regardless of exit_reason
            # (mirrors _check_stoploss_reentry) — a loss is a loss whether it came
            # from a hard stoploss, a signal flip, a manual/algo force-exit, a
            # resolved market, or a reconcile close.
            loss_where = [
                PositionRow.status == "closed",
                PositionRow.pnl < 0,
                PositionRow.market_id == market_id,
                PositionRow.mode == mode,
            ]
            if not block_reentry_after_stoploss:
                cutoff = datetime.now(UTC) - timedelta(hours=stoploss_cooldown_hours)
                loss_where.append(PositionRow.exit_time >= cutoff)
            loss_count_result = await session.execute(
                select(func.count(PositionRow.id)).where(*loss_where)
            )
            loss_exit_count = loss_count_result.scalar_one()

        # Per-market and global exposure. COALESCE(requested_contracts, contracts)
        # so pending rows count their request size (the exchange has reserved that
        # capital) while legacy rows fall back to their fill count.
        exposure_expr = (
            func.coalesce(PositionRow.requested_contracts, PositionRow.contracts)
            * PositionRow.entry_price
        )

        market_exposure_result = await session.execute(
            select(
                func.coalesce(
                    func.sum(case((PositionRow.status == "open", exposure_expr), else_=0.0)),
                    0.0,
                ).label("open_exposure"),
                func.coalesce(
                    func.sum(case((PositionRow.status == "pending", exposure_expr), else_=0.0)),
                    0.0,
                ).label("pending_exposure"),
            ).where(
                PositionRow.status.in_(("open", "pending")),
                PositionRow.market_id == market_id,
                PositionRow.mode == mode,
            )
        )
        market_row = market_exposure_result.one()

        global_count_result = await session.execute(
            select(
                func.sum(case((PositionRow.status == "open", 1), else_=0)).label("open_count"),
                func.sum(case((PositionRow.status == "pending", 1), else_=0)).label("pending_count"),
            )
            .select_from(PositionRow)
            .where(
                PositionRow.status.in_(("open", "pending")),
                PositionRow.mode == mode,
            )
        )
        global_counts = global_count_result.one()

        global_exposure_result = await session.execute(
            select(
                func.coalesce(
                    func.sum(case((PositionRow.status == "open", exposure_expr), else_=0.0)),
                    0.0,
                ).label("open_exposure"),
                func.coalesce(
                    func.sum(case((PositionRow.status == "pending", exposure_expr), else_=0.0)),
                    0.0,
                ).label("pending_exposure"),
            ).where(
                PositionRow.status.in_(("open", "pending")),
                PositionRow.mode == mode,
            )
        )
        global_exposures = global_exposure_result.one()

        # Daily loss window. If the operator acknowledged the circuit breaker
        # via /start, measure losses only since that acknowledgement (not since
        # midnight) so a previously-tripped breaker doesn't immediately
        # re-block after resume.
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
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

        return PortfolioSnapshot(
            loss_exit_count=loss_exit_count,
            market_open_exposure=float(market_row.open_exposure or 0.0),
            market_pending_exposure=float(market_row.pending_exposure or 0.0),
            open_count=int(global_counts.open_count or 0),
            pending_count=int(global_counts.pending_count or 0),
            open_exposure=float(global_exposures.open_exposure or 0.0),
            pending_exposure=float(global_exposures.pending_exposure or 0.0),
            daily_pnl=daily_pnl,
        )

    def evaluate_position_caps(
        self,
        *,
        signal_edge: float,
        requested_size: float,
        bankroll: float,
        market_id: str,
        max_market_exposure: float,
        snapshot: PortfolioSnapshot,
        block_reentry_after_stoploss: bool = False,
        stoploss_cooldown_hours: float = 0.0,
    ) -> RiskDecision:
        """Pure cap arithmetic behind check_position().

        Takes portfolio state as an explicit :class:`PortfolioSnapshot` so the
        replay harness can evaluate the identical hard-cap logic against a
        frozen fixture state. check_position() is the only production caller.
        """
        # 1. Edge floor check (also pre-checked by check_position before any
        # DB work; repeated here so the pure path enforces it standalone)
        edge_rejection = self._edge_floor_decision(signal_edge)
        if edge_rejection is not None:
            return edge_rejection

        # 2. Stoploss re-entry guard
        sl_blocked, sl_reason = self._stoploss_reentry_decision(
            market_id, snapshot.loss_exit_count, block_reentry_after_stoploss, stoploss_cooldown_hours
        )
        if sl_blocked:
            return RiskDecision(allowed=False, reason=sl_reason, capped_size=0.0)

        # 3. Cap position size at max_position_pct of bankroll
        max_size = bankroll * self._config.max_position_pct
        capped_size = min(requested_size, max_size)

        # 4. Per-market cumulative exposure check.
        # Counts open AND pending positions on this market so that multiple
        # signals cannot stack exposure beyond the strategy limit while an
        # order is still resting on the exchange.
        existing_market_exposure = snapshot.market_open_exposure + snapshot.market_pending_exposure
        remaining_market_capacity = max_market_exposure - existing_market_exposure
        if remaining_market_capacity <= 0.0:
            logger.info(
                "risk.market_exposure_exceeded",
                market_id=market_id,
                existing_exposure=existing_market_exposure,
                open_exposure=snapshot.market_open_exposure,
                pending_exposure=snapshot.market_pending_exposure,
                max_market_exposure=max_market_exposure,
            )
            return RiskDecision(
                allowed=False,
                reason=(
                    f"market {market_id} exposure {existing_market_exposure:.2f} "
                    f"(open={snapshot.market_open_exposure:.2f}, "
                    f"pending={snapshot.market_pending_exposure:.2f}) >= "
                    f"max {max_market_exposure:.2f} per market"
                ),
                capped_size=0.0,
            )
        # Also cap capped_size so the new position doesn't push over the limit.
        capped_size = min(capped_size, remaining_market_capacity)

        # 5. Max open positions + total exposure (via shared decision helper)
        cap_blocked, cap_reason, total_exposure = self._global_capacity_decision(
            snapshot.open_count,
            snapshot.pending_count,
            snapshot.open_exposure,
            snapshot.pending_exposure,
            bankroll,
        )
        if cap_blocked:
            return RiskDecision(allowed=False, reason=cap_reason, capped_size=0.0)

        max_exposure = bankroll * self._config.max_total_exposure_pct
        remaining_total_capacity = max_exposure - total_exposure
        capped_size = min(capped_size, remaining_total_capacity)

        # 6. Daily loss check
        max_daily_loss = bankroll * self._config.max_daily_loss_pct
        if snapshot.daily_pnl < 0 and abs(snapshot.daily_pnl) > max_daily_loss:
            logger.warning(
                "risk.daily_loss_exceeded",
                daily_pnl=snapshot.daily_pnl,
                max_daily_loss=max_daily_loss,
            )
            return RiskDecision(
                allowed=False,
                reason=(
                    f"daily loss {abs(snapshot.daily_pnl):.2f} > max {max_daily_loss:.2f} "
                    f"({self._config.max_daily_loss_pct:.0%} of bankroll)"
                ),
                capped_size=0.0,
            )

        logger.debug(
            "risk.position_allowed",
            requested_size=requested_size,
            capped_size=capped_size,
            edge=signal_edge,
        )
        return RiskDecision(allowed=True, reason="", capped_size=capped_size)

    async def check_circuit_breakers(
        self,
        session: AsyncSession,
        bankroll: float,
        mode: str,
        drawdown_reset_bankroll: float | None = None,
        daily_loss_ack_at: datetime | None = None,
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
        today_start = datetime.now(UTC).replace(
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
