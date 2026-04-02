"""Position tracking and P&L calculation."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.markets.models import Market, MarketRow, Position, PositionRow
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
    status: str = "open",
    exchange_order_id: str | None = None,
    entry_fee_usd: float = 0.0,
) -> Position:
    """Insert a new Position row. Commits the session.

    ``status`` is "open" for paper trades and immediately-filled live orders;
    "pending" for live GTC orders awaiting fill confirmation (T39 will flip these).
    ``exchange_order_id`` and ``entry_fee_usd`` are populated for live orders.
    """
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
        status=status,
        exchange_order_id=exchange_order_id,
        entry_fee_usd=entry_fee_usd,
    )
    session.add(row)
    await session.commit()
    return _row_to_position(row)


async def close_position(
    session: AsyncSession,
    position_id: str,
    *,
    exit_price: float,
    exit_reason: str | None = None,
    resolution: int | None = None,
) -> Position:
    """Update position: set exit_price, exit_time, exit_reason, resolution, status='closed', pnl, pnl_pct."""
    result = await session.execute(
        select(PositionRow).where(PositionRow.id == uuid.UUID(position_id))
    )
    row: PositionRow = result.scalar_one()

    fee = row.entry_fee_usd or 0.0
    gross_pnl = (exit_price - row.entry_price) * row.contracts
    pnl = gross_pnl - fee
    cost_basis = row.entry_price * row.contracts + fee
    pnl_pct = pnl / cost_basis if cost_basis else 0.0

    row.exit_price = exit_price
    row.exit_time = datetime.now(tz=timezone.utc)
    row.exit_reason = exit_reason
    row.resolution = resolution
    row.status = "closed"
    row.pnl = round(pnl, 4)
    row.pnl_pct = round(pnl_pct, 6)

    await session.commit()
    return _row_to_position(row)


async def update_position_excursions(
    session: AsyncSession,
    position_id: str,
    *,
    mae: float,
    mfe: float,
) -> None:
    """Update mae/mfe on an open position. Does not commit — caller decides."""
    await session.execute(
        update(PositionRow)
        .where(PositionRow.id == uuid.UUID(position_id))
        .values(mae=round(mae, 6), mfe=round(mfe, 6))
    )
    await session.commit()


async def get_net_bankroll(session: AsyncSession, initial_bankroll: float, mode: str = "paper") -> float:
    """Current effective bankroll = initial_bankroll + all closed P&L for *mode*.

    Uses realized P&L only (closed positions). Floored at 0.0 so risk
    checks never operate on a negative bankroll.
    """
    result = await session.execute(
        select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
            PositionRow.status == "closed",
            PositionRow.mode == mode,
        )
    )
    all_time_pnl: float = float(result.scalar_one())
    return max(0.0, initial_bankroll + all_time_pnl)


async def get_open_positions(session: AsyncSession, mode: str = "paper") -> list[Position]:
    """Return all positions with status='open' for *mode*, ordered by entry_time desc."""
    result = await session.execute(
        select(PositionRow)
        .where(PositionRow.status == "open", PositionRow.mode == mode)
        .order_by(PositionRow.entry_time.desc())
    )
    return [_row_to_position(row) for row in result.scalars().all()]


async def get_daily_pnl(session: AsyncSession, mode: str = "paper") -> float:
    """Sum of pnl for all positions closed today (UTC) matching *mode*. Returns 0.0 if none."""
    today_start = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    result = await session.execute(
        select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
            PositionRow.status == "closed",
            PositionRow.exit_time >= today_start,
            PositionRow.mode == mode,
        )
    )
    return float(result.scalar_one())


async def get_portfolio_summary(session: AsyncSession, mode: str = "paper") -> dict:
    """Return portfolio summary with open count, exposure, and P&L totals for *mode*."""
    open_count_result = await session.execute(
        select(func.count()).where(
            PositionRow.status == "open",
            PositionRow.mode == mode,
        )
    )
    open_count = int(open_count_result.scalar_one())

    exposure_result = await session.execute(
        select(
            func.coalesce(
                func.sum(PositionRow.contracts * PositionRow.entry_price), 0.0
            )
        ).where(
            PositionRow.status == "open",
            PositionRow.mode == mode,
        )
    )
    total_exposure = float(exposure_result.scalar_one())

    daily_pnl = await get_daily_pnl(session, mode=mode)

    all_time_result = await session.execute(
        select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
            PositionRow.status == "closed",
            PositionRow.mode == mode,
        )
    )
    all_time_pnl = float(all_time_result.scalar_one())

    # Unrealized P&L, exposure breakdown, and MAE/MFE across all open positions.
    # YES: contracts * (mid - entry_price)
    # NO:  contracts * ((1 - mid) - entry_price)
    open_rows_result = await session.execute(
        select(
            PositionRow.contracts,
            PositionRow.entry_price,
            PositionRow.direction,
            PositionRow.mae,
            PositionRow.mfe,
            MarketRow.mid_price,
        )
        .join(MarketRow, PositionRow.market_id == MarketRow.id)
        .where(
            PositionRow.status == "open",
            PositionRow.mode == mode,
        )
    )
    unrealized_pnl = 0.0
    long_exposure = 0.0
    short_exposure = 0.0
    mae_dollar_sum = 0.0
    mfe_dollar_sum = 0.0
    mae_contract_sum = 0
    mfe_contract_sum = 0

    for contracts, entry_price, direction, mae, mfe, mid_price in open_rows_result.all():
        if direction == "YES":
            unrealized_pnl += contracts * (mid_price - entry_price)
            long_exposure += contracts * entry_price
        else:
            unrealized_pnl += contracts * ((1.0 - mid_price) - entry_price)
            short_exposure += contracts * entry_price

        if mae is not None:
            mae_dollar_sum += mae * contracts
            mae_contract_sum += contracts
        if mfe is not None:
            mfe_dollar_sum += mfe * contracts
            mfe_contract_sum += contracts

    net_exposure = long_exposure - short_exposure
    portfolio_mae_pct = mae_dollar_sum / mae_contract_sum if mae_contract_sum > 0 else None
    portfolio_mfe_pct = mfe_dollar_sum / mfe_contract_sum if mfe_contract_sum > 0 else None

    return {
        "open_count": open_count,
        "total_exposure_usd": total_exposure,
        "net_exposure_usd": net_exposure,
        "daily_pnl_usd": daily_pnl,
        "all_time_pnl_usd": all_time_pnl,
        "unrealized_pnl_usd": unrealized_pnl,
        "portfolio_mae_usd": mae_dollar_sum if mae_contract_sum > 0 else None,
        "portfolio_mfe_usd": mfe_dollar_sum if mfe_contract_sum > 0 else None,
        "portfolio_mae_pct": portfolio_mae_pct,
        "portfolio_mfe_pct": portfolio_mfe_pct,
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
        exit_reason=row.exit_reason,
        resolution=row.resolution,
        pnl=row.pnl,
        pnl_pct=row.pnl_pct,
        mae=row.mae,
        mfe=row.mfe,
        exchange_order_id=row.exchange_order_id,
        entry_fee_usd=row.entry_fee_usd or 0.0,
    )
