"""Position tracking and P&L calculation."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.markets.models import Market, Position, PositionRow
from freqpred.signal.models import Signal


async def open_position(
    session: AsyncSession,
    *,
    market: Market,
    signal: Signal,
    strategy_name: str,
    strategy_version: str,
    direction: str,
    contracts: int,
    entry_price: float,
    mode: str,
) -> Position:
    """Insert a new Position row with status='open'. Commits the session."""
    row = PositionRow(
        id=uuid.uuid4(),
        market_id=market.id,
        signal_id=uuid.UUID(signal.id) if isinstance(signal.id, str) else signal.id,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        signal_confidence=signal.confidence,
        signal_edge=signal.edge,
        signal_estimated_prob=signal.estimated_probability,
        direction=direction,
        contracts=contracts,
        entry_price=entry_price,
        entry_time=datetime.now(tz=timezone.utc),
        mode=mode,
        status="open",
    )
    session.add(row)
    await session.commit()
    return _row_to_position(row)


async def close_position(
    session: AsyncSession,
    position_id: str,
    *,
    exit_price: float,
    resolution: int,
) -> Position:
    """Update position: set exit_price, exit_time, resolution, status='closed', pnl, pnl_pct."""
    result = await session.execute(
        select(PositionRow).where(PositionRow.id == uuid.UUID(position_id))
    )
    row: PositionRow = result.scalar_one()

    pnl = (exit_price - row.entry_price) * row.contracts
    pnl_pct = pnl / (row.entry_price * row.contracts)

    row.exit_price = exit_price
    row.exit_time = datetime.now(tz=timezone.utc)
    row.resolution = resolution
    row.status = "closed"
    row.pnl = round(pnl, 4)
    row.pnl_pct = round(pnl_pct, 6)

    await session.commit()
    return _row_to_position(row)


async def get_open_positions(session: AsyncSession) -> list[Position]:
    """Return all positions with status='open', ordered by entry_time desc."""
    result = await session.execute(
        select(PositionRow)
        .where(PositionRow.status == "open")
        .order_by(PositionRow.entry_time.desc())
    )
    return [_row_to_position(row) for row in result.scalars().all()]


async def get_daily_pnl(session: AsyncSession) -> float:
    """Sum of pnl for all positions closed today (UTC). Returns 0.0 if none."""
    today_start = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    result = await session.execute(
        select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
            PositionRow.status == "closed",
            PositionRow.exit_time >= today_start,
        )
    )
    return float(result.scalar_one())


async def get_portfolio_summary(session: AsyncSession) -> dict:
    """Return portfolio summary with open count, exposure, and P&L totals."""
    open_count_result = await session.execute(
        select(func.count()).where(PositionRow.status == "open")
    )
    open_count = int(open_count_result.scalar_one())

    exposure_result = await session.execute(
        select(
            func.coalesce(
                func.sum(PositionRow.contracts * PositionRow.entry_price), 0.0
            )
        ).where(PositionRow.status == "open")
    )
    total_exposure = float(exposure_result.scalar_one())

    daily_pnl = await get_daily_pnl(session)

    all_time_result = await session.execute(
        select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
            PositionRow.status == "closed"
        )
    )
    all_time_pnl = float(all_time_result.scalar_one())

    return {
        "open_count": open_count,
        "total_exposure_usd": total_exposure,
        "daily_pnl_usd": daily_pnl,
        "all_time_pnl_usd": all_time_pnl,
    }


def _row_to_position(row: PositionRow) -> Position:
    return Position(
        id=str(row.id),
        market_id=row.market_id,
        signal_id=str(row.signal_id),
        strategy_name=row.strategy_name,
        strategy_version=row.strategy_version,
        signal_confidence=row.signal_confidence,
        signal_edge=row.signal_edge,
        signal_estimated_prob=row.signal_estimated_prob,
        direction=row.direction,
        contracts=row.contracts,
        entry_price=row.entry_price,
        entry_time=row.entry_time,
        mode=row.mode,
        status=row.status,
        exit_price=row.exit_price,
        exit_time=row.exit_time,
        resolution=row.resolution,
        pnl=row.pnl,
        pnl_pct=row.pnl_pct,
    )
