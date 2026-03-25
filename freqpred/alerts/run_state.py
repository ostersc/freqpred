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


async def reset_drawdown(session: AsyncSession) -> datetime:
    """Set drawdown_reset_at to now and commit. Returns the new reset timestamp."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        session.add(RunStateRow(id=1, state=_DEFAULT_STATE, updated_at=now, drawdown_reset_at=now))
    else:
        row.drawdown_reset_at = now
        row.updated_at = now
    await session.commit()
    return now


async def set_run_state(session: AsyncSession, state: str) -> None:
    """Upsert the run state singleton row and commit."""
    result = await session.execute(select(RunStateRow).limit(1))
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        session.add(RunStateRow(id=1, state=state, updated_at=now))
    else:
        row.state = state
        row.updated_at = now
    await session.commit()
