"""Position tracking and P&L calculation."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import Market, MarketRow, Position, PositionRow
from freqpred.signal.models import Signal, SignalRow


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
    requested_contracts: int | None = None,
    exchange_order_status: str | None = None,
    last_exchange_sync_at: datetime | None = None,
) -> Position:
    """Insert a new Position row. Commits the session.

    ``status`` is "open" for paper trades and immediately-filled live orders;
    "pending" for live GTC orders awaiting fill confirmation. Reconcile flips
    pending → open / cancelled based on Kalshi's get_order response.
    ``exchange_order_id``, ``entry_fee_usd``, ``requested_contracts``,
    ``exchange_order_status``, and ``last_exchange_sync_at`` are populated for
    live orders only.
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
        requested_contracts=requested_contracts,
        exchange_order_status=exchange_order_status,
        last_exchange_sync_at=last_exchange_sync_at,
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


async def partial_close_position(
    session: AsyncSession,
    position: "Position",
    *,
    filled_contracts: int,
    fill_price: float,
    fee_usd: float,
    exit_reason: str,
    exit_order_id: str | None = None,
    exit_requested_contracts: int | None = None,
    resolution: int | None = None,
    _now: datetime | None = None,
) -> "Position":
    """Realize P&L on a subset of contracts; leave the residual open.

    Accumulates gross P&L (without entry fee) into realized_pnl_accumulator
    and exit fees into exit_fee_usd. When the last contract closes (contracts
    drops to 0) the position is transitioned to 'closed' with a weighted-avg
    exit_price and net P&L that includes both entry and all exit fees.
    """
    now = _now or datetime.now(tz=timezone.utc)

    result = await session.execute(
        select(PositionRow).where(PositionRow.id == uuid.UUID(str(position.id)))
    )
    row: PositionRow = result.scalar_one()

    # Accumulate gross P&L contribution for this tranche (no entry-fee deduction;
    # that happens once on final close). This lets us derive weighted-avg exit_price
    # purely from the accumulator + entry_price on final close.
    partial_gross = (fill_price - row.entry_price) * filled_contracts
    row.realized_pnl_accumulator = (row.realized_pnl_accumulator or 0.0) + partial_gross

    # Accumulate exit fees across all IOC orders so final pnl net of all fees.
    row.exit_fee_usd = (row.exit_fee_usd or 0.0) + fee_usd

    # Running total of contracts closed via exit orders (used for weighted-avg
    # on final close and the dashboard's exit fill breakdown).
    prev_filled = row.exit_filled_contracts or 0
    row.exit_filled_contracts = prev_filled + filled_contracts

    # Per-order metadata: latest order ID + requested count for the dashboard.
    if exit_order_id is not None:
        row.exit_order_id = exit_order_id
    if exit_requested_contracts is not None:
        row.exit_requested_contracts = exit_requested_contracts

    row.contracts -= filled_contracts

    if row.contracts <= 0:
        # Final close: derive weighted-avg exit_price and net P&L.
        total_closed = row.exit_filled_contracts  # fully accumulated at this point
        if total_closed and total_closed > 0:
            weighted_avg_exit = (row.realized_pnl_accumulator / total_closed) + row.entry_price
        else:
            weighted_avg_exit = fill_price

        entry_fee = row.entry_fee_usd or 0.0
        total_exit_fee = row.exit_fee_usd or 0.0
        pnl = row.realized_pnl_accumulator - entry_fee - total_exit_fee
        cost_basis = row.entry_price * (total_closed or 0) + entry_fee
        pnl_pct = pnl / cost_basis if cost_basis else 0.0

        row.exit_price = round(weighted_avg_exit, 6)
        row.exit_time = now
        row.exit_reason = exit_reason
        row.resolution = resolution
        row.status = "closed"
        row.pnl = round(pnl, 4)
        row.pnl_pct = round(pnl_pct, 6)
        row.contracts = 0

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


async def get_pending_positions(session: AsyncSession, mode: str = "paper") -> list[Position]:
    """Return all positions with status='pending' for *mode*, ordered by entry_time desc."""
    result = await session.execute(
        select(PositionRow)
        .where(PositionRow.status == "pending", PositionRow.mode == mode)
        .order_by(PositionRow.entry_time.desc())
    )
    return [_row_to_position(row) for row in result.scalars().all()]


async def promote_pending_to_open(
    session: AsyncSession,
    position_id: str,
    *,
    fill_price: float | None = None,
) -> None:
    """Flip a pending position's status to 'open'. Commits the session.

    ``fill_price``, when provided, also updates entry_price — used in paper mode
    to record price improvement when the ask was already below the limit at fill time.
    """
    values: dict = {"status": "open"}
    if fill_price is not None:
        values["entry_price"] = fill_price
    await session.execute(
        update(PositionRow)
        .where(PositionRow.id == uuid.UUID(position_id))
        .values(**values)
    )
    await session.commit()


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


async def get_pnl_time_series(
    session: AsyncSession,
    mode: str = "paper",
    lookback_days: int | None = None,
    strategy_name: str | None = None,
    model_used: str | None = None,
    prompt_version: str | None = None,
    direction: str | None = None,
    category: str | None = None,
    series_ticker: str | None = None,
    market_id: str | None = None,
) -> list[dict]:
    """Return daily realized P&L buckets for closed positions matching all filters.

    Only closed positions with a non-null exit_time are included. Filters are
    AND-combined. Signal-level filters require a JOIN to SignalRow; market-level
    filters require a JOIN to MarketRow — both JOINs are conditional so the
    common case (no such filters) avoids the extra join cost.

    Returns a list sorted by date asc, each entry:
        {"date": "YYYY-MM-DD", "daily_pnl": float, "cumulative_pnl": float, "trade_count": int}
    """
    base_filters = [
        PositionRow.status == "closed",
        PositionRow.mode == mode,
        PositionRow.exit_time.is_not(None),
    ]

    if lookback_days is not None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
        base_filters.append(PositionRow.exit_time >= cutoff)

    if direction:
        base_filters.append(PositionRow.direction == direction.upper())

    if strategy_name:
        base_filters.append(PositionRow.strategy_name == strategy_name)

    if market_id:
        base_filters.append(PositionRow.market_id == market_id)

    stmt = select(
        func.date(PositionRow.exit_time).label("day"),
        func.coalesce(func.sum(PositionRow.pnl), 0.0).label("daily_pnl"),
        func.count(PositionRow.id).label("trade_count"),
    ).where(*base_filters)

    needs_signal_join = model_used is not None or prompt_version is not None
    needs_market_join = category is not None or series_ticker is not None

    if needs_signal_join:
        stmt = stmt.join(SignalRow, SignalRow.id == PositionRow.signal_id)
        if model_used:
            stmt = stmt.where(SignalRow.model_used == model_used)
        if prompt_version:
            stmt = stmt.where(SignalRow.prompt_version == prompt_version)

    if needs_market_join:
        stmt = stmt.join(MarketRow, MarketRow.id == PositionRow.market_id)
        if category:
            stmt = stmt.where(MarketRow.category == category)
        if series_ticker:
            stmt = stmt.where(MarketRow.series_ticker == series_ticker)

    stmt = stmt.group_by(func.date(PositionRow.exit_time)).order_by(
        func.date(PositionRow.exit_time).asc()
    )

    rows = (await session.execute(stmt)).all()

    cumulative = 0.0
    result = []
    for day, daily_pnl, trade_count in rows:
        daily_pnl_f = float(daily_pnl)
        cumulative += daily_pnl_f
        result.append(
            {
                "date": str(day),
                "daily_pnl": round(daily_pnl_f, 4),
                "cumulative_pnl": round(cumulative, 4),
                "trade_count": int(trade_count),
            }
        )
    return result


async def get_llm_spend_time_series(
    session: AsyncSession,
    lookback_days: int | None = None,
) -> list[dict]:
    """Return daily LLM spend buckets, unfiltered by position dimensions.

    LLM spend is global — it is not attributed to individual positions or
    strategies and therefore ignores any position-level filter the caller may
    hold. Only the time window is respected.

    Returns a list sorted by date asc, each entry:
        {"date": "YYYY-MM-DD", "daily_spend": float, "cumulative_spend": float}
    """
    filters = []
    if lookback_days is not None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
        filters.append(LLMQueryRow.timestamp >= cutoff)

    stmt = (
        select(
            func.date(LLMQueryRow.timestamp).label("day"),
            func.coalesce(func.sum(LLMQueryRow.cost_usd), 0.0).label("daily_spend"),
        )
        .where(*filters)
        .group_by(func.date(LLMQueryRow.timestamp))
        .order_by(func.date(LLMQueryRow.timestamp).asc())
    )

    rows = (await session.execute(stmt)).all()

    cumulative = 0.0
    result = []
    for day, daily_spend in rows:
        daily_spend_f = float(daily_spend)
        cumulative += daily_spend_f
        result.append(
            {
                "date": str(day),
                "daily_spend": round(daily_spend_f, 6),
                "cumulative_spend": round(cumulative, 6),
            }
        )
    return result


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
        exit_order_id=row.exit_order_id,
        exit_fee_usd=row.exit_fee_usd or 0.0,
        exit_requested_contracts=row.exit_requested_contracts,
        exit_filled_contracts=row.exit_filled_contracts,
        realized_pnl_accumulator=row.realized_pnl_accumulator or 0.0,
        exchange_stoploss_order_id=row.exchange_stoploss_order_id,
    )
