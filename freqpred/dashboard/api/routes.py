"""FastAPI router — all /api/* endpoints."""
from __future__ import annotations

import uuid as _uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.llm.audit import get_daily_spend_usd
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import PositionRow
from freqpred.metrics.calibration import compute_calibration
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.signal.models import SignalRow
from freqpred.trading.ledger import get_portfolio_summary

from .schemas import (
    ApiErrorStateOut,
    CalibrationBucketOut,
    CalibrationResponse,
    CircuitBreakerStateOut,
    DocumentLinkOut,
    HealthResponse,
    LedgerResponse,
    LLMCostResponse,
    PositionListResponse,
    PositionOut,
    SignalDetailOut,
    SignalListResponse,
    SignalOut,
    StrategyConfigOut,
    StrategyConfigUpdateRequest,
    SystemHealthResponse,
    WebSocketStateOut,
)

# Fields that cannot be changed at runtime (require a process restart).
_IMMUTABLE_FIELDS: frozenset[str] = frozenset({"name", "categories"})

router = APIRouter()
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return request.app.state.session_factory


async def get_db(
    sf: Annotated[async_sessionmaker[AsyncSession], Depends(_session_factory)],
) -> AsyncSession:  # type: ignore[override]
    async with sf() as session:
        yield session


def _daily_cap(request: Request) -> float:
    return float(request.app.state.daily_cap_usd)


async def _get_mode(session: AsyncSession) -> str:
    """Read the active trading mode from run_state; default to 'paper' if not set."""
    import freqpred.alerts.models  # noqa: F401 — register RunStateRow  # noqa: PLC0415
    from freqpred.alerts.run_state import get_mode  # noqa: PLC0415

    return (await get_mode(session)) or "paper"


def _risk_config(request: Request) -> object | None:
    return getattr(request.app.state, "risk_config", None)


def _bankroll_usd(request: Request) -> float:
    return float(getattr(request.app.state, "bankroll_usd", 0.0))


def _started_at(request: Request) -> datetime:
    return request.app.state.started_at


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal_row_to_out(row: SignalRow) -> SignalOut:
    return SignalOut(
        id=str(row.id),
        market_id=row.market_id,
        estimated_probability=row.estimated_probability,
        confidence=row.confidence,
        edge=row.edge,
        market_mid_at_signal=row.market_mid_at_signal,
        direction=row.direction,
        reasoning=row.reasoning,
        sources=list(row.sources or []),
        retrieval_hash=row.retrieval_hash,
        model_used=row.model_used,
        prompt_version=row.prompt_version,
        trigger=row.trigger,
        created_at=row.created_at,
        social_sentiment_summary=row.social_sentiment_summary,
    )


def _position_row_to_out(row: PositionRow) -> PositionOut:
    return PositionOut(
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
        created_at=row.created_at,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _load_active_strategy_config(session: AsyncSession) -> object | None:
    """Load the active strategy config from the DB.

    1. Read the strategy name written to ``run_state`` by ``freqpred run``.
    2. Load the strategy's default config via the strategy loader.
    3. Apply any runtime overrides persisted in ``runtime_config_overrides``.

    Returns ``None`` if no strategy is active (run loop not started).
    """
    import freqpred.alerts.models  # noqa: F401 — register RunStateRow  # noqa: PLC0415
    from freqpred.alerts.run_state import get_strategy_name  # noqa: PLC0415
    from freqpred.strategy.config_store import load_overrides  # noqa: PLC0415
    from freqpred.strategy.loader import load_strategy  # noqa: PLC0415

    strategy_name = await get_strategy_name(session)
    if strategy_name is None:
        return None

    try:
        cfg = load_strategy(strategy_name).config
    except Exception:
        log.warning("dashboard.strategy_load_failed", strategy_name=strategy_name)
        return None

    overrides = await load_overrides(session, strategy_name)
    for key, value in overrides.items():
        setattr(cfg, key, value)

    return cfg


def _strategy_config_to_out(cfg: object) -> StrategyConfigOut:
    return StrategyConfigOut(
        name=cfg.name,
        min_edge=cfg.min_edge,
        min_confidence=cfg.min_confidence,
        kelly_fraction=cfg.kelly_fraction,
        max_exposure_per_market=cfg.max_exposure_per_market,
        categories=list(cfg.categories),
        min_volume_24h=cfg.min_volume_24h,
        max_days_to_close=cfg.max_days_to_close,
        min_days_to_close=cfg.min_days_to_close,
        stoploss=cfg.stoploss,
        trailing_stop=cfg.trailing_stop,
        trailing_stop_positive=cfg.trailing_stop_positive,
        trailing_stop_positive_offset=cfg.trailing_stop_positive_offset,
        min_mid_price=cfg.min_mid_price,
        max_mid_price=cfg.max_mid_price,
        max_spread=cfg.max_spread,
        block_reentry_after_stoploss=cfg.block_reentry_after_stoploss,
        stoploss_cooldown_hours=cfg.stoploss_cooldown_hours,
    )


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


@router.get("/signals", response_model=SignalListResponse)
async def list_signals(
    session: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    market_id: str | None = Query(default=None),
    direction: str | None = Query(default=None),
) -> SignalListResponse:
    stmt = select(SignalRow).order_by(SignalRow.created_at.desc())
    count_stmt = select(func.count()).select_from(SignalRow)

    if market_id:
        stmt = stmt.where(SignalRow.market_id == market_id)
        count_stmt = count_stmt.where(SignalRow.market_id == market_id)
    if direction:
        stmt = stmt.where(SignalRow.direction == direction.upper())
        count_stmt = count_stmt.where(SignalRow.direction == direction.upper())

    total = int((await session.execute(count_stmt)).scalar_one())
    rows = (await session.execute(stmt.offset(offset).limit(limit))).scalars().all()

    return SignalListResponse(
        items=[_signal_row_to_out(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/signals/{signal_id}", response_model=SignalDetailOut)
async def get_signal(
    signal_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> SignalDetailOut:
    try:
        uid = _uuid.UUID(signal_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Signal not found")

    row = (
        await session.execute(select(SignalRow).where(SignalRow.id == uid))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Signal not found")

    # Fetch document links with source URL and title
    link_rows = (
        await session.execute(
            select(
                DocumentMarketLinkRow.document_id,
                DocumentMarketLinkRow.relevance_score,
                DocumentRow.source_url,
                DocumentRow.title,
            )
            .join(DocumentRow, DocumentRow.id == DocumentMarketLinkRow.document_id)
            .where(DocumentMarketLinkRow.signal_id == uid)
            .order_by(DocumentMarketLinkRow.relevance_score.desc())
        )
    ).all()

    doc_links = [
        DocumentLinkOut(
            document_id=str(doc_id),
            source_url=source_url,
            title=title or "",
            relevance_score=relevance_score,
        )
        for doc_id, relevance_score, source_url, title in link_rows
    ]

    base = _signal_row_to_out(row)
    return SignalDetailOut(**base.model_dump(), document_links=doc_links)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@router.get("/positions", response_model=PositionListResponse)
async def list_positions(
    session: Annotated[AsyncSession, Depends(get_db)],
    status: str = Query(default="all", pattern="^(open|closed|all)$"),
) -> PositionListResponse:
    app_mode = await _get_mode(session)
    stmt = select(PositionRow).where(PositionRow.mode == app_mode).order_by(PositionRow.entry_time.desc())
    count_stmt = select(func.count()).select_from(PositionRow).where(PositionRow.mode == app_mode)

    if status != "all":
        stmt = stmt.where(PositionRow.status == status)
        count_stmt = count_stmt.where(PositionRow.status == status)

    total = int((await session.execute(count_stmt)).scalar_one())
    rows = (await session.execute(stmt)).scalars().all()

    return PositionListResponse(
        items=[_position_row_to_out(r) for r in rows],
        total=total,
    )


@router.get("/positions/{position_id}", response_model=PositionOut)
async def get_position(
    position_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> PositionOut:
    try:
        uid = _uuid.UUID(position_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Position not found")

    row = (
        await session.execute(select(PositionRow).where(PositionRow.id == uid))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Position not found")

    return _position_row_to_out(row)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@router.get("/ledger", response_model=LedgerResponse)
async def get_ledger(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> LedgerResponse:
    app_mode = await _get_mode(session)
    summary = await get_portfolio_summary(session, mode=app_mode)
    return LedgerResponse(
        open_count=summary["open_count"],
        total_exposure_usd=summary["total_exposure_usd"],
        daily_pnl_usd=summary["daily_pnl_usd"],
        all_time_pnl_usd=summary["all_time_pnl_usd"],
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@router.get("/calibration", response_model=CalibrationResponse)
async def get_calibration(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> CalibrationResponse:
    app_mode = await _get_mode(session)
    report = await compute_calibration(session, mode=app_mode)
    return CalibrationResponse(
        brier_score=report.brier_score,
        market_brier_score=report.market_brier_score,
        n_samples=report.n_samples,
        buckets=[
            CalibrationBucketOut(
                lower=b.lower,
                upper=b.upper,
                count=b.count,
                mean_estimated_prob=b.mean_estimated_prob,
                actual_resolution_rate=b.actual_resolution_rate,
            )
            for b in report.buckets
        ],
    )


# ---------------------------------------------------------------------------
# LLM cost
# ---------------------------------------------------------------------------


@router.get("/llm/cost", response_model=LLMCostResponse)
async def get_llm_cost(
    session: Annotated[AsyncSession, Depends(get_db)],
    daily_cap: Annotated[float, Depends(_daily_cap)],
) -> LLMCostResponse:
    today_usd = await get_daily_spend_usd(session)

    # Weekly spend
    week_start = datetime.now(UTC) - timedelta(days=7)
    weekly_result = await session.execute(
        select(func.coalesce(func.sum(LLMQueryRow.cost_usd), 0.0)).where(
            LLMQueryRow.timestamp >= week_start
        )
    )
    weekly_usd = float(weekly_result.scalar_one())

    # Breakdown by query_type for today
    today = datetime.now(UTC).date()
    by_type_result = await session.execute(
        select(LLMQueryRow.query_type, func.sum(LLMQueryRow.cost_usd))
        .where(func.date(LLMQueryRow.timestamp) == today)
        .group_by(LLMQueryRow.query_type)
    )
    by_query_type = {row[0]: round(float(row[1]), 6) for row in by_type_result.all()}

    pct_used = (today_usd / daily_cap * 100.0) if daily_cap > 0 else 0.0

    return LLMCostResponse(
        today_usd=today_usd,
        weekly_usd=weekly_usd,
        daily_cap_usd=daily_cap,
        pct_used=round(pct_used, 2),
        by_query_type=by_query_type,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(
    session: Annotated[AsyncSession, Depends(get_db)],
    daily_cap: Annotated[float, Depends(_daily_cap)],
) -> HealthResponse:
    db_status = "connected"
    open_positions = 0
    llm_remaining = daily_cap
    try:
        app_mode = await _get_mode(session)
        open_result = await session.execute(
            select(func.count()).where(PositionRow.status == "open", PositionRow.mode == app_mode)
        )
        open_positions = int(open_result.scalar_one())
        daily_spend = await get_daily_spend_usd(session)
        llm_remaining = max(0.0, daily_cap - daily_spend)
    except Exception:
        log.exception("health.db_check_failed")
        db_status = "error"

    overall = "ok" if db_status == "connected" else "degraded"

    return HealthResponse(
        status=overall,
        db=db_status,
        open_positions=open_positions,
        llm_daily_budget_remaining_usd=round(llm_remaining, 4),
    )


# ---------------------------------------------------------------------------
# Strategy config
# ---------------------------------------------------------------------------


@router.get("/strategy/config", response_model=StrategyConfigOut)
async def get_strategy_config(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> StrategyConfigOut:
    cfg = await _load_active_strategy_config(session)
    if cfg is None:
        raise HTTPException(
            status_code=503,
            detail="No active strategy — freqpred run is not running.",
        )
    return _strategy_config_to_out(cfg)


@router.put("/strategy/config", response_model=StrategyConfigOut)
async def update_strategy_config(
    update: StrategyConfigUpdateRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> StrategyConfigOut:
    cfg = await _load_active_strategy_config(session)
    if cfg is None:
        raise HTTPException(
            status_code=503,
            detail="No active strategy — freqpred run is not running.",
        )

    # Detect immutable fields sent by the caller.
    sent_fields = set(update.model_dump(exclude_unset=True).keys())
    immutable_sent = sorted(sent_fields & _IMMUTABLE_FIELDS)
    if immutable_sent:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Fields {immutable_sent} are immutable and require a process "
                "restart to change."
            ),
        )

    # Merge new updates on top of existing persisted overrides and save.
    mutable_updates = {
        k: v for k, v in update.model_dump(exclude_unset=True).items()
        if k not in _IMMUTABLE_FIELDS and v is not None
    }

    from freqpred.strategy.config_store import load_overrides, save_overrides  # noqa: PLC0415

    existing = await load_overrides(session, cfg.name)
    existing.update(mutable_updates)
    await save_overrides(session, cfg.name, existing)

    # Return the fully merged config as confirmation.
    for key, value in existing.items():
        setattr(cfg, key, value)

    return _strategy_config_to_out(cfg)


# ---------------------------------------------------------------------------
# System health
# ---------------------------------------------------------------------------


@router.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health(
    session: Annotated[AsyncSession, Depends(get_db)],
    daily_cap: Annotated[float, Depends(_daily_cap)],
    initial_bankroll: Annotated[float, Depends(_bankroll_usd)],
    risk_cfg: Annotated[object | None, Depends(_risk_config)],
    started_at: Annotated[datetime, Depends(_started_at)],
) -> SystemHealthResponse:
    import freqpred.alerts.models  # noqa: F401 — ensure RunStateRow is registered  # noqa: PLC0415
    from freqpred.alerts.models import RunStateRow as _RunStateRow  # noqa: PLC0415
    from freqpred.trading.ledger import get_net_bankroll  # noqa: PLC0415

    db_ok = True
    app_mode = "paper"
    run_state = "running"
    cb_halted = False
    cb_reason: str | None = None
    daily_pnl: float = 0.0
    net_bankroll: float = initial_bankroll
    llm_budget_used: float = 0.0
    pending_orders: int = 0
    open_positions: int = 0
    llm_errors_last_hour: int = 0

    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    hour_ago = datetime.now(UTC) - timedelta(hours=1)

    try:
        app_mode = await _get_mode(session)

        # Read run_state row directly to get CB state alongside run_state + drawdown.
        rs_result = await session.execute(select(_RunStateRow).limit(1))
        rs_row = rs_result.scalar_one_or_none()
        if rs_row is not None:
            run_state = rs_row.state
            cb_halted = bool(rs_row.cb_active)
            cb_reason = rs_row.cb_reason

        net_bankroll = await get_net_bankroll(session, initial_bankroll, mode=app_mode)

        daily_pnl_result = await session.execute(
            select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
                PositionRow.status == "closed",
                PositionRow.exit_time >= today_start,
                PositionRow.mode == app_mode,
            )
        )
        daily_pnl = float(daily_pnl_result.scalar_one())

        llm_budget_used = await get_daily_spend_usd(session)

        pending_result = await session.execute(
            select(func.count(PositionRow.id)).where(
                PositionRow.status == "pending",
                PositionRow.mode == app_mode,
            )
        )
        pending_orders = int(pending_result.scalar_one())

        open_result = await session.execute(
            select(func.count(PositionRow.id)).where(
                PositionRow.status == "open",
                PositionRow.mode == app_mode,
            )
        )
        open_positions = int(open_result.scalar_one())

        llm_errors_result = await session.execute(
            select(func.count(LLMQueryRow.id)).where(
                LLMQueryRow.success.is_(False),
                LLMQueryRow.timestamp >= hour_ago,
            )
        )
        llm_errors_last_hour = int(llm_errors_result.scalar_one())

    except Exception:
        log.exception("system_health.db_query_failed")
        db_ok = False

    max_daily_loss_pct: float = (
        risk_cfg.max_daily_loss_pct if risk_cfg is not None else 0.15
    )
    daily_loss_pct: float = (
        abs(daily_pnl) / net_bankroll if (net_bankroll > 0 and daily_pnl < 0) else 0.0
    )

    uptime_seconds = int((datetime.now(UTC) - started_at).total_seconds())

    return SystemHealthResponse(
        run_state=run_state,
        mode=app_mode,
        circuit_breakers=CircuitBreakerStateOut(
            trading_halted=cb_halted,
            reason=cb_reason,
            daily_loss_pct=round(daily_loss_pct, 4),
            daily_loss_limit_pct=max_daily_loss_pct,
            llm_budget_used_usd=round(llm_budget_used, 4),
            llm_budget_cap_usd=daily_cap,
        ),
        websocket=WebSocketStateOut(
            connected=None,
            subscribed_markets=None,
            last_message_at=None,
        ),
        api_errors=ApiErrorStateOut(
            kalshi_errors_last_hour=0,
            llm_errors_last_hour=llm_errors_last_hour,
            consecutive_llm_errors=None,
        ),
        pending_orders=pending_orders,
        open_positions=open_positions,
        db_ok=db_ok,
        uptime_seconds=uptime_seconds,
    )
