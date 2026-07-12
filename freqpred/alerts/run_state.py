"""DB-backed run-loop state: running / paused / stopped.

The state is stored in a single-row ``run_state`` table so a restart picks
it up without any in-process state.

Usage::

    async with session_factory() as session:
        state = await get_run_state(session)

    async with session_factory() as session:
        await set_run_state(session, "paused")
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.alerts.models import RunStateRow

_DEFAULT_STATE = "running"

_TRADING_MODES = ("paper", "live")


def _has_risk_window(mode: str | None) -> bool:
    """Whether *mode* has mode-scoped risk-window fields in run_state.

    Paper and live each get their own drawdown baseline and daily-loss
    acknowledgement — their bankrolls evolve on entirely different scales, so
    a baseline captured in one mode is always wrong for the other. Signal-only
    (or None) places no orders, so no risk window exists for it. Any other
    value is a programming error, not a mode without a window.
    """
    if mode in _TRADING_MODES:
        return True
    if mode is None or mode == "signal-only":
        return False
    raise ValueError(
        f"unknown trading mode {mode!r}; expected one of "
        f"{_TRADING_MODES + ('signal-only',)}"
    )


def drawdown_window_from_row(
    row: RunStateRow | None, mode: str | None
) -> tuple[datetime | None, float | None]:
    """(reset_at, reset_bankroll) for *mode* from an already-loaded row.

    Returns (None, None) when no row exists or *mode* has no risk window.
    """
    if row is None or not _has_risk_window(mode):
        return None, None
    if mode == "paper":
        return row.drawdown_reset_at_paper, row.drawdown_reset_bankroll_paper
    return row.drawdown_reset_at_live, row.drawdown_reset_bankroll_live


def daily_loss_ack_from_row(
    row: RunStateRow | None, mode: str | None
) -> datetime | None:
    """Daily-loss acknowledgement timestamp for *mode* from an already-loaded row."""
    if row is None or not _has_risk_window(mode):
        return None
    if mode == "paper":
        return row.daily_loss_ack_at_paper
    return row.daily_loss_ack_at_live


async def get_run_state(session: AsyncSession) -> str:
    """Return the current run state; defaults to 'running' if no row exists."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    return row.state if row is not None else _DEFAULT_STATE


async def get_drawdown_window(
    session: AsyncSession, mode: str | None
) -> tuple[datetime | None, float | None]:
    """Return (reset_at, reset_bankroll) for *mode*'s drawdown window.

    ``reset_bankroll`` is the net bankroll stored at the time of the last
    /reset_drawdown call made while running in *mode*. Both values are None
    if that mode has never been reset (or has no risk window: signal-only).
    """
    result = await session.execute(select(RunStateRow).limit(1))
    return drawdown_window_from_row(result.scalar_one_or_none(), mode)


async def reset_drawdown(
    session: AsyncSession, mode: str, net_bankroll: float
) -> datetime:
    """Set *mode*'s drawdown baseline to now/current value; the other mode's
    window is untouched.

    ``net_bankroll`` is the current net account value in *mode* and becomes
    the baseline against which all future drawdown measurements for that mode
    are made. Returns the new reset timestamp. Raises ValueError for modes
    without a risk window (signal-only) — there is no baseline to reset.
    """
    if not _has_risk_window(mode):
        raise ValueError(f"cannot reset drawdown baseline in mode {mode!r}")
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        row = RunStateRow(id=1, state=_DEFAULT_STATE, updated_at=now)
        session.add(row)
    if mode == "paper":
        row.drawdown_reset_at_paper = now
        row.drawdown_reset_bankroll_paper = net_bankroll
    else:
        row.drawdown_reset_at_live = now
        row.drawdown_reset_bankroll_live = net_bankroll
    row.updated_at = now
    await session.commit()
    return now


async def get_daily_loss_ack_at(
    session: AsyncSession, mode: str | None
) -> datetime | None:
    """Return when the daily loss circuit breaker was last acknowledged in *mode*.

    Set whenever run state transitions to 'running' while *mode* is the active
    trading mode. The daily loss window in risk checks uses
    ``max(today_start, daily_loss_ack_at)`` so that losses incurred *before*
    the acknowledgement don't immediately re-trip the breaker. Returns None if
    never acknowledged in that mode (full day window applies).
    """
    result = await session.execute(select(RunStateRow).limit(1))
    return daily_loss_ack_from_row(result.scalar_one_or_none(), mode)


async def set_run_state(
    session: AsyncSession, state: str, mode: str | None = None
) -> None:
    """Upsert the run state singleton row and commit.

    When transitioning to 'running' with a trading *mode* (paper/live), stamps
    that mode's ``daily_loss_ack_at_<mode> = now`` so the daily loss circuit
    breaker window resets to the current moment — for that mode only. This
    prevents an already-tripped breaker from immediately re-firing on the next
    loop cycle after the user resumes via /start, without acknowledging the
    breaker for the other mode's next run. ``mode=None`` (or signal-only)
    stamps neither column.
    """
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        row = RunStateRow(id=1, state=state, updated_at=now)
        session.add(row)
    else:
        row.state = state
        row.updated_at = now
    if state == "running" and _has_risk_window(mode):
        if mode == "paper":
            row.daily_loss_ack_at_paper = now
        else:
            row.daily_loss_ack_at_live = now
    await session.commit()


async def get_strategy_name(session: AsyncSession) -> str | None:
    """Return the strategy name written by the active run loop, or None."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    return row.strategy_name if row is not None else None


async def set_strategy_name(session: AsyncSession, strategy_name: str) -> None:
    """Write (or update) the active strategy name in the run_state row and commit."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        session.add(
            RunStateRow(id=1, state=_DEFAULT_STATE, updated_at=now, strategy_name=strategy_name)
        )
    else:
        row.strategy_name = strategy_name
        row.updated_at = now
    await session.commit()


async def get_mode(session: AsyncSession) -> str | None:
    """Return the trading mode written by the active run loop, or None."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    return row.mode if row is not None else None


async def set_mode(session: AsyncSession, mode: str) -> None:
    """Write (or update) the trading mode in the run_state row and commit."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        session.add(RunStateRow(id=1, state=_DEFAULT_STATE, updated_at=now, mode=mode))
    else:
        row.mode = mode
        row.updated_at = now
    await session.commit()


async def set_cb_state(
    session: AsyncSession, active: bool, reason: str | None = None
) -> None:
    """Persist circuit breaker state to run_state and commit.

    Call with ``active=True, reason=<message>`` when a CB fires.
    Call with ``active=False, reason=None`` when a CB clears.
    """
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        session.add(
            RunStateRow(
                id=1, state=_DEFAULT_STATE, updated_at=now,
                cb_active=active, cb_reason=reason,
            )
        )
    else:
        row.cb_active = active
        row.cb_reason = reason
        row.updated_at = now
    await session.commit()
