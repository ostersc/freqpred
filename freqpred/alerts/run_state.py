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


async def get_run_state(session: AsyncSession) -> str:
    """Return the current run state; defaults to 'running' if no row exists."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    return row.state if row is not None else _DEFAULT_STATE


async def get_drawdown_reset_at(session: AsyncSession) -> datetime | None:
    """Return the timestamp of the last drawdown reset, or None if never reset."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    return row.drawdown_reset_at if row is not None else None


async def get_drawdown_window(session: AsyncSession) -> tuple[datetime | None, float | None]:
    """Return (reset_at, reset_bankroll) for the current drawdown window.

    ``reset_bankroll`` is the net bankroll stored at the time of the last
    /reset_drawdown call.  Both values are None if no reset has ever been done.
    """
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    if row is None:
        return None, None
    return row.drawdown_reset_at, row.drawdown_reset_bankroll


async def reset_drawdown(session: AsyncSession, net_bankroll: float) -> datetime:
    """Set drawdown_reset_at and drawdown_reset_bankroll to now/current value.

    ``net_bankroll`` is the current net account value and becomes the baseline
    against which all future drawdown measurements are made.
    Returns the new reset timestamp.
    """
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        session.add(RunStateRow(
            id=1, state=_DEFAULT_STATE, updated_at=now,
            drawdown_reset_at=now, drawdown_reset_bankroll=net_bankroll,
        ))
    else:
        row.drawdown_reset_at = now
        row.drawdown_reset_bankroll = net_bankroll
        row.updated_at = now
    await session.commit()
    return now


async def get_daily_loss_ack_at(session: AsyncSession) -> datetime | None:
    """Return the timestamp when the daily loss circuit breaker was last acknowledged.

    Set whenever run state transitions to 'running'. The daily loss window in
    risk checks uses ``max(today_start, daily_loss_ack_at)`` so that losses
    incurred *before* the acknowledgement don't immediately re-trip the breaker.
    Returns None if never acknowledged (full day window applies).
    """
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    return row.daily_loss_ack_at if row is not None else None


async def set_run_state(session: AsyncSession, state: str) -> None:
    """Upsert the run state singleton row and commit.

    When transitioning to 'running', stamps ``daily_loss_ack_at = now`` so
    that the daily loss circuit breaker window resets to the current moment.
    This prevents an already-tripped breaker from immediately re-firing on
    the next loop cycle after the user resumes via /start.
    """
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        session.add(RunStateRow(
            id=1,
            state=state,
            updated_at=now,
            daily_loss_ack_at=now if state == "running" else None,
        ))
    else:
        row.state = state
        row.updated_at = now
        if state == "running":
            row.daily_loss_ack_at = now
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
