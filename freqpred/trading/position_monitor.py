"""Position monitor: background loop evaluating exit conditions for open positions.

Exit priority order (per SPEC §8):
  1. Hard stoploss        — framework-enforced, cannot be overridden by strategy
  2. Trailing stoploss    — trails from best mid-price since entry
  3. Minimal ROI          — time-based profit targets
  4. Custom exit          — strategy.custom_exit() hook
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

from freqpred.markets.models import Market, MarketRow, Position
from freqpred.trading import ledger

if TYPE_CHECKING:
    from freqpred.alerts.dispatcher import AlertDispatcher
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
    ) -> None:
        self._session_factory = session_factory
        self._strategies = strategies
        self._poll_interval = poll_interval_seconds
        self._alert_dispatcher = alert_dispatcher
        # position_id → best mid_price seen since entry (used by trailing stop)
        self._peak_prices: dict[str, float] = {}
        # position_id → best/worst effective P&L delta seen (for MAE/MFE)
        self._peak_deltas: dict[str, float] = {}    # MFE
        self._trough_deltas: dict[str, float] = {}  # MAE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            positions = await ledger.get_open_positions(session)
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

            current_price = market.mid_price
            fresh_signal = fresh_signals.get(position.market_id)

            result = self.evaluate_exit(
                position=position,
                market=market,
                current_price=current_price,
                strategy=strategy,
                fresh_signal=fresh_signal,
            )
            if result is None:
                # No exit — update peak price tracker (trailing stop) + MAE/MFE
                self._update_peak(position, current_price)
                await self._update_excursions(position, current_price)
                continue

            exit_reason, exit_price = result
            # Infer YES/NO resolution from the final contract price.
            # When Kalshi settles a market, contract prices snap to ~1.0 or ~0.0.
            # Account for direction: for NO positions, exit_price is the NO contract
            # price, which is near 1.0 when NO wins and near 0.0 when YES wins.
            resolution: int | None = None
            if exit_price >= 0.99:
                resolution = 1 if position.direction == "YES" else 0
            elif exit_price <= 0.01:
                resolution = 0 if position.direction == "YES" else 1
            async with self._session_factory() as session:
                closed_position = await ledger.close_position(
                    session,
                    position.id,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    resolution=resolution,
                )
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
        peak_price = self._peak_prices.get(position.id, position.entry_price)
        now = datetime.now(tz=timezone.utc)

        # 1. Hard stoploss (framework-enforced)
        result = _check_stoploss(position, current_price, config.stoploss)
        if result:
            return result

        # 2. Trailing stoploss
        if config.trailing_stop:
            result = _check_trailing_stop(
                position,
                current_price,
                peak_price,
                config.stoploss,
                config.trailing_stop_positive,
                config.trailing_stop_positive_offset,
            )
            if result:
                return result

        # 3. Minimal ROI
        result = _check_roi(position, current_price, now, config.minimal_roi)
        if result:
            return result

        # 4. Custom exit hook
        if fresh_signal is not None:
            tag = strategy.custom_exit(position, fresh_signal, market)
            if tag is not None:
                return (f"custom_exit:{tag}", current_price)

        # 5. Signal exit (only when a fresh signal is provided)
        if fresh_signal is not None:
            if strategy.should_exit(position, fresh_signal, market):
                return ("signal", current_price)

        # 6. Market resolution — Kalshi status is "resolved" OR close_time has passed
        kalshi_status = market.metadata.get("status", "")
        if kalshi_status == "resolved" or market.close_time <= now:
            return ("market_resolved", current_price)

        return _NO_EXIT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_peak(self, position: Position, current_price: float) -> None:
        """Advance peak price if current_price is better than recorded peak."""
        peak = self._peak_prices.get(position.id, position.entry_price)
        if current_price > peak:
            self._peak_prices[position.id] = current_price

    async def _update_excursions(self, position: Position, current_price: float) -> None:
        """Update MAE/MFE for a position and persist to DB when a new extreme is hit."""
        if position.direction == "YES":
            delta = current_price - position.entry_price
        else:
            delta = (1.0 - current_price) - position.entry_price

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
    """Return ('stoploss', current_price) if the hard stoploss has been hit."""
    pnl_pct = (current_price - position.entry_price) / position.entry_price
    if pnl_pct <= stoploss:
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

    Logic:
    - If ``trailing_stop_positive`` is set and unrealized P&L has crossed that
      threshold, apply the tighter trail (``trailing_stop_positive_offset``
      below peak).
    - Otherwise, apply the normal stoploss distance below peak.
    """
    entry = position.entry_price
    unrealized_pct = (current_price - entry) / entry
    peak_pct = (peak_price - entry) / entry

    if (
        trailing_stop_positive is not None
        and peak_pct >= trailing_stop_positive
    ):
        # Tight trail: stop = peak - offset
        trail_distance = trailing_stop_positive_offset
    else:
        # Normal trail: stop = peak + stoploss (stoploss is negative)
        trail_distance = -stoploss  # positive distance below peak

    stop_price = peak_price * (1.0 - trail_distance)
    if current_price <= stop_price:
        return ("trailing_stop", current_price)
    return None


def _check_roi(
    position: Position,
    current_price: float,
    now: datetime,
    minimal_roi: dict[str, float],
) -> tuple[str, float] | None:
    """Return ('roi', current_price) if a minimal ROI target has been met.

    minimal_roi keys are minutes-since-entry thresholds (as strings).
    The applicable target is the one with the highest threshold that is
    still <= elapsed minutes.
    """
    if not minimal_roi:
        return None

    elapsed_minutes = (now - position.entry_time).total_seconds() / 60.0
    pnl_pct = (current_price - position.entry_price) / position.entry_price

    # Find the ROI target for the current elapsed time:
    # use the largest threshold key that is <= elapsed_minutes
    applicable_target: float | None = None
    best_threshold = -1.0
    for threshold_str, target in minimal_roi.items():
        threshold = float(threshold_str)
        if threshold <= elapsed_minutes and threshold > best_threshold:
            best_threshold = threshold
            applicable_target = target

    if applicable_target is not None and pnl_pct >= applicable_target:
        return ("roi", current_price)
    return None


def _market_row_to_domain(row: MarketRow) -> Market:
    return Market(
        id=row.id,
        platform=row.platform,
        question=row.question,
        category=row.category,
        close_time=row.close_time,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
        mid_price=row.mid_price,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
        metadata=row.metadata_ or {},
    )
