"""Persistence helpers for runtime strategy config overrides."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.strategy.models import RuntimeConfigOverrideRow


async def load_overrides(session: AsyncSession, strategy_name: str) -> dict:
    """Return the persisted overrides dict for *strategy_name*, or {} if none."""
    result = await session.execute(
        select(RuntimeConfigOverrideRow).where(
            RuntimeConfigOverrideRow.strategy_name == strategy_name
        )
    )
    row = result.scalar_one_or_none()
    return dict(row.overrides) if row is not None else {}


async def save_overrides(
    session: AsyncSession,
    strategy_name: str,
    overrides: dict,
) -> None:
    """Upsert the overrides dict for *strategy_name* and commit."""
    result = await session.execute(
        select(RuntimeConfigOverrideRow).where(
            RuntimeConfigOverrideRow.strategy_name == strategy_name
        )
    )
    row = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if row is None:
        session.add(
            RuntimeConfigOverrideRow(
                strategy_name=strategy_name,
                overrides=overrides,
                updated_at=now,
            )
        )
    else:
        row.overrides = overrides
        row.updated_at = now
    await session.commit()
