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
from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import MarketRow, PositionRow
from freqpred.metrics.calibration import compute_calibration, compute_source_brier_scores
from freqpred.metrics.models import SignalAssessmentRow, SourceQualityScoreRow
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.signal.models import SignalRow
from freqpred.trading.ledger import get_portfolio_summary
from freqpred.trading.order_manager import PositionNotFoundError, PositionNotOpenError

from .schemas import (
    AnalyzeResponse,
    ApiErrorStateOut,
    CalibrationBucketOut,
    CalibrationResponse,
    CircuitBreakerStateOut,
    DocumentLinkOut,
    HealthResponse,
    LedgerResponse,
    LLMCostResponse,
    LLMQueryDetailOut,
    LLMQueryListResponse,
    LLMQueryOut,
    MarketDetailOut,
    MarketListResponse,
    MarketOut,
    PositionDetailOut,
    PositionListResponse,
    PositionOut,
    SignalAssessmentOut,
    SignalDetailOut,
    SignalListResponse,
    SignalOut,
    SourceQualityListResponse,
    SourceQualityScoreOut,
    StrategyConfigOut,
    StrategyConfigUpdateRequest,
    StrategyDecisionListResponse,
    StrategyDecisionOut,
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


def _signal_pipeline(request: Request) -> object | None:
    return getattr(request.app.state, "signal_pipeline", None)


def _get_order_manager(request: Request) -> object | None:
    return getattr(request.app.state, "order_manager", None)


def _started_at(request: Request) -> datetime:
    return request.app.state.started_at


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal_row_to_out(row: SignalRow, market_question: str | None = None) -> SignalOut:
    return SignalOut(
        id=str(row.id),
        market_id=row.market_id,
        market_question=market_question,
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


def _assessment_row_to_out(row: SignalAssessmentRow) -> SignalAssessmentOut:
    return SignalAssessmentOut(
        trust_score=row.trust_score,
        size_multiplier=row.size_multiplier,
        verdict=row.verdict,
        reasoning=row.reasoning,
        key_factors=list(row.key_factors or []),
        warnings=list(row.warnings or []),
        source_breakdown=list(row.source_breakdown or []),
        similar_market_summary=dict(row.similar_market_summary or {}),
        llm_query_id=row.llm_query_id,
        created_at=row.created_at,
    )


def _effective_mid(mid_price: float, yes_bid: float, yes_ask: float, last_price: float) -> float:
    """Return the best available mid price for unrealized P&L calculation.

    When a market stops trading, Kalshi clears the order book (yes_bid→0, yes_ask→1.0)
    before writing the result field.  mid_price computed from (0+1)/2=0.5 is misleading;
    last_price (the last actual trade) is a far better proxy for the true settlement value.
    """
    if yes_bid == 0.0 and yes_ask >= 0.95 and last_price > 0.0:
        return last_price
    return mid_price


def _position_row_to_out(row: PositionRow, current_mid: float | None = None) -> PositionOut:
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None
    if row.status == "open" and current_mid is not None:
        if row.direction == "YES":
            unrealized_pnl = row.contracts * (current_mid - row.entry_price)
        else:
            unrealized_pnl = row.contracts * ((1.0 - current_mid) - row.entry_price)
        cost_basis = row.entry_price * row.contracts
        unrealized_pnl_pct = unrealized_pnl / cost_basis if cost_basis else 0.0

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
        exit_reason=row.exit_reason,
        resolution=row.resolution,
        pnl=row.pnl,
        pnl_pct=row.pnl_pct,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        created_at=row.created_at,
    )


async def _load_signal_assessment(
    session: AsyncSession,
    signal_id: _uuid.UUID,
) -> SignalAssessmentOut | None:
    assessment_row = (
        await session.execute(
            select(SignalAssessmentRow)
            .where(SignalAssessmentRow.signal_id == signal_id)
            .order_by(SignalAssessmentRow.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if assessment_row is None:
        return None
    return _assessment_row_to_out(assessment_row)


async def _load_document_links(
    session: AsyncSession,
    signal_id: _uuid.UUID,
) -> list[DocumentLinkOut]:
    link_rows = (
        await session.execute(
            select(
                DocumentMarketLinkRow.document_id,
                DocumentMarketLinkRow.relevance_score,
                DocumentRow.source_url,
                DocumentRow.title,
                DocumentRow.source_type,
                DocumentRow.source_name,
                DocumentRow.published_at,
                DocumentRow.fetched_at,
                DocumentRow.summary,
                DocumentRow.body,
            )
            .join(DocumentRow, DocumentRow.id == DocumentMarketLinkRow.document_id)
            .where(DocumentMarketLinkRow.signal_id == signal_id)
            .order_by(DocumentMarketLinkRow.relevance_score.desc())
        )
    ).all()

    return [
        DocumentLinkOut(
            document_id=str(doc_id),
            source_url=source_url,
            title=title or "",
            relevance_score=relevance_score,
            source_type=source_type,
            source_name=source_name,
            published_at=published_at,
            fetched_at=fetched_at,
            summary=summary,
            body_excerpt=(body or "")[:400],
        )
        for doc_id, relevance_score, source_url, title,
            source_type, source_name, published_at, fetched_at, summary, body
        in link_rows
    ]


async def _build_signal_detail(
    session: AsyncSession,
    signal_row: SignalRow,
    market_question: str | None,
) -> SignalDetailOut:
    base = _signal_row_to_out(signal_row, market_question)
    return SignalDetailOut(
        **base.model_dump(),
        document_links=await _load_document_links(session, signal_row.id),
        assessment=await _load_signal_assessment(session, signal_row.id),
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
        assessment_scale_min=cfg.assessment_scale_min,
        assessment_scale_max=cfg.assessment_scale_max,
        similar_market_min_signals=cfg.similar_market_min_signals,
        similar_market_min_trades=cfg.similar_market_min_trades,
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
    stmt = (
        select(SignalRow, MarketRow.question)
        .outerjoin(MarketRow, MarketRow.id == SignalRow.market_id)
        .order_by(SignalRow.created_at.desc())
    )
    count_stmt = select(func.count()).select_from(SignalRow)

    if market_id:
        stmt = stmt.where(SignalRow.market_id == market_id)
        count_stmt = count_stmt.where(SignalRow.market_id == market_id)
    if direction:
        stmt = stmt.where(SignalRow.direction == direction.upper())
        count_stmt = count_stmt.where(SignalRow.direction == direction.upper())

    total = int((await session.execute(count_stmt)).scalar_one())
    result_rows = (await session.execute(stmt.offset(offset).limit(limit))).all()

    return SignalListResponse(
        items=[_signal_row_to_out(r, q) for r, q in result_rows],
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

    result = (
        await session.execute(
            select(SignalRow, MarketRow.question)
            .outerjoin(MarketRow, MarketRow.id == SignalRow.market_id)
            .where(SignalRow.id == uid)
        )
    ).one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail="Signal not found")
    row, market_question = result

    return await _build_signal_detail(session, row, market_question)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@router.get("/positions", response_model=PositionListResponse)
async def list_positions(
    session: Annotated[AsyncSession, Depends(get_db)],
    status: str = Query(default="all", pattern="^(open|closed|all)$"),
) -> PositionListResponse:
    app_mode = await _get_mode(session)
    stmt = (
        select(PositionRow, MarketRow.mid_price, MarketRow.yes_bid, MarketRow.yes_ask, MarketRow.last_price)
        .outerjoin(MarketRow, MarketRow.id == PositionRow.market_id)
        .where(PositionRow.mode == app_mode)
        .order_by(PositionRow.entry_time.desc())
    )
    count_stmt = select(func.count()).select_from(PositionRow).where(PositionRow.mode == app_mode)

    if status != "all":
        stmt = stmt.where(PositionRow.status == status)
        count_stmt = count_stmt.where(PositionRow.status == status)

    total = int((await session.execute(count_stmt)).scalar_one())
    rows = (await session.execute(stmt)).all()

    return PositionListResponse(
        items=[
            _position_row_to_out(
                r,
                current_mid=_effective_mid(mid, yes_bid or 0.0, yes_ask or 0.0, last_price or 0.0)
                if mid is not None else None,
            )
            for r, mid, yes_bid, yes_ask, last_price in rows
        ],
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


@router.get("/positions/{position_id}/detail", response_model=PositionDetailOut)
async def get_position_detail(
    position_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> PositionDetailOut:
    try:
        uid = _uuid.UUID(position_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Position not found")

    pos_row = (
        await session.execute(select(PositionRow).where(PositionRow.id == uid))
    ).scalar_one_or_none()
    if pos_row is None:
        raise HTTPException(status_code=404, detail="Position not found")

    # Fetch market for question + current mid
    market_row = (
        await session.execute(select(MarketRow).where(MarketRow.id == pos_row.market_id))
    ).scalar_one_or_none()
    market_question = market_row.question if market_row else None
    current_mid = (
        _effective_mid(market_row.mid_price, market_row.yes_bid, market_row.yes_ask, market_row.last_price)
        if market_row else None
    )

    # Fetch entry signal + document links
    entry_signal_uid = pos_row.signal_id
    sig_result = (
        await session.execute(
            select(SignalRow, MarketRow.question)
            .outerjoin(MarketRow, MarketRow.id == SignalRow.market_id)
            .where(SignalRow.id == entry_signal_uid)
        )
    ).one_or_none()
    if sig_result is None:
        raise HTTPException(status_code=404, detail="Entry signal not found")
    sig_row, _ = sig_result

    entry_signal = await _build_signal_detail(session, sig_row, market_question)

    # Fetch all signals for this market (chronological, capped at 100)
    market_sig_rows = (
        await session.execute(
            select(SignalRow)
            .where(SignalRow.market_id == pos_row.market_id)
            .order_by(SignalRow.created_at.asc())
            .limit(100)
        )
    ).scalars().all()
    market_signals = [_signal_row_to_out(r, market_question) for r in market_sig_rows]

    return PositionDetailOut(
        **_position_row_to_out(pos_row, current_mid=current_mid).model_dump(),
        market_question=market_question,
        current_mid=current_mid,
        entry_signal=entry_signal,
        market_signals=market_signals,
    )


@router.post("/positions/{position_id}/force-exit", response_model=PositionOut)
async def force_exit_position(
    position_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
    order_manager: Annotated[object | None, Depends(_get_order_manager)],
) -> PositionOut:
    if order_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Force exit requires freqpred run (order manager not available)",
        )
    try:
        uid = _uuid.UUID(position_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Position not found")

    # Delegate all checks to the order manager — no preflight query here.
    try:
        await order_manager.force_exit(position_id)  # type: ignore[union-attr]
    except PositionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PositionNotOpenError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KalshiAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Exchange error: {exc}") from exc

    # Fresh SELECT — order_manager.force_exit() commits in its own session.
    updated = (
        await session.execute(select(PositionRow).where(PositionRow.id == uid))
    ).scalar_one()
    return _position_row_to_out(updated)


# ---------------------------------------------------------------------------
# Strategy decisions (exited-position post-mortem)
# ---------------------------------------------------------------------------


def _our_side_win_value(direction: str, market_result: str | None) -> float | None:
    """Return 1.0 if our side won, 0.0 if it lost, None if the market is unresolved.

    Markets store ``result`` as ``"yes" | "no" | None``. A YES position wins when
    ``result='yes'``, a NO position wins when ``result='no'``.
    """
    if market_result not in ("yes", "no"):
        return None
    if direction == "YES":
        return 1.0 if market_result == "yes" else 0.0
    if direction == "NO":
        return 1.0 if market_result == "no" else 0.0
    return None


def _decision_row_to_out(
    row: PositionRow,
    market_result: str | None,
    market_question: str | None,
    best_prior_ask: float | None,
) -> StrategyDecisionOut:
    base = _position_row_to_out(row).model_dump()

    # --- Exit decision counterfactual --------------------------------------
    # If the market is unresolved we cannot compute the counterfactual — leave
    # both counterfactual and delta fields as None.
    win_value = _our_side_win_value(row.direction, market_result)
    counterfactual_pc: float | None = None
    counterfactual_usd: float | None = None
    exit_delta_pc: float | None = None
    exit_delta_usd: float | None = None
    if win_value is not None and row.exit_price is not None:
        counterfactual_pc = win_value - row.entry_price
        counterfactual_usd = counterfactual_pc * row.contracts
        exit_delta_pc = row.exit_price - win_value
        exit_delta_usd = exit_delta_pc * row.contracts

    # --- Entry efficiency vs best prior signal -----------------------------
    entry_eff_pc: float | None = None
    entry_eff_usd: float | None = None
    if best_prior_ask is not None:
        entry_eff_pc = best_prior_ask - row.entry_price
        entry_eff_usd = entry_eff_pc * row.contracts

    return StrategyDecisionOut(
        **base,
        market_question=market_question,
        market_result=market_result,
        counterfactual_pnl_per_contract=counterfactual_pc,
        counterfactual_pnl_usd=counterfactual_usd,
        exit_delta_per_contract=exit_delta_pc,
        exit_delta_usd=exit_delta_usd,
        best_prior_ask=best_prior_ask,
        entry_efficiency_per_contract=entry_eff_pc,
        entry_efficiency_usd=entry_eff_usd,
    )


@router.get("/strategy-decisions", response_model=StrategyDecisionListResponse)
async def list_strategy_decisions(
    session: Annotated[AsyncSession, Depends(get_db)],
    strategy: str | None = Query(default=None),
    exit_reason: str | None = Query(default=None),
    ticker_prefix: str | None = Query(default=None),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> StrategyDecisionListResponse:
    app_mode = await _get_mode(session)

    # Correlated subquery: lowest market_ask_at_signal across signals prior to
    # entry_time for the same (market, direction) with positive edge.
    best_prior_ask_subq = (
        select(func.min(SignalRow.market_ask_at_signal))
        .where(SignalRow.market_id == PositionRow.market_id)
        .where(SignalRow.direction == PositionRow.direction)
        .where(SignalRow.edge > 0)
        .where(SignalRow.created_at < PositionRow.entry_time)
        .where(SignalRow.market_ask_at_signal.isnot(None))
        .correlate(PositionRow)
        .scalar_subquery()
    )

    base_filters = [
        PositionRow.mode == app_mode,
        PositionRow.status == "closed",
    ]
    filters = list(base_filters)
    if strategy:
        filters.append(PositionRow.strategy_name == strategy)
    if exit_reason:
        # Prefix ILIKE match so `exit_reason=force_exit` captures
        # `force_exit:time_based`, `force_exit:risk`, etc.
        filters.append(PositionRow.exit_reason.ilike(f"{exit_reason}%"))
    if ticker_prefix:
        filters.append(PositionRow.market_id.ilike(f"{ticker_prefix}%"))
    if date_from:
        filters.append(PositionRow.exit_time >= date_from)
    if date_to:
        filters.append(PositionRow.exit_time <= date_to)

    stmt = (
        select(
            PositionRow,
            MarketRow.result,
            MarketRow.question,
            best_prior_ask_subq.label("best_prior_ask"),
        )
        .outerjoin(MarketRow, MarketRow.id == PositionRow.market_id)
        .where(*filters)
        .order_by(PositionRow.exit_time.desc())
        .offset(offset)
        .limit(limit)
    )

    count_stmt = (
        select(func.count()).select_from(PositionRow).where(*filters)
    )

    # distinct_strategies / distinct_exit_reasons are computed over the full
    # closed-position set (only `base_filters`, not the current selection) so
    # the filter dropdowns are stable regardless of what is currently selected.
    distinct_strategies_stmt = (
        select(PositionRow.strategy_name)
        .where(*base_filters)
        .distinct()
        .order_by(PositionRow.strategy_name)
    )
    distinct_exit_reasons_stmt = (
        select(PositionRow.exit_reason)
        .where(*base_filters)
        .where(PositionRow.exit_reason.isnot(None))
        .distinct()
        .order_by(PositionRow.exit_reason)
    )

    total = int((await session.execute(count_stmt)).scalar_one())
    rows = (await session.execute(stmt)).all()
    distinct_strategies = [
        s for s in (await session.execute(distinct_strategies_stmt)).scalars().all()
    ]
    distinct_exit_reasons = [
        r for r in (await session.execute(distinct_exit_reasons_stmt)).scalars().all()
    ]

    items = [
        _decision_row_to_out(
            pos_row,
            market_result=market_result,
            market_question=market_question,
            best_prior_ask=best_prior_ask,
        )
        for pos_row, market_result, market_question, best_prior_ask in rows
    ]

    return StrategyDecisionListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        distinct_strategies=distinct_strategies,
        distinct_exit_reasons=distinct_exit_reasons,
    )


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------

_ANALYZE_COOLDOWN_SECS = 60


def _market_row_to_out(row: MarketRow) -> MarketOut:
    return MarketOut(
        id=row.id,
        question=row.question,
        status=row.status,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
        mid_price=row.mid_price,
        volume_24h=row.volume_24h,
        close_time=row.close_time,
        last_fetched_at=row.last_fetched_at,
        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
    )


@router.get("/markets", response_model=MarketListResponse)
async def list_markets(
    session: Annotated[AsyncSession, Depends(get_db)],
    search: str | None = Query(default=None),
    status: str = Query(default="open", pattern="^(open|closed|all)$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MarketListResponse:
    from sqlalchemy import or_  # noqa: PLC0415

    stmt = select(MarketRow).order_by(MarketRow.last_fetched_at.desc())
    count_stmt = select(func.count()).select_from(MarketRow)

    if status != "all":
        db_status = "active" if status == "open" else status
        stmt = stmt.where(MarketRow.status == db_status)
        count_stmt = count_stmt.where(MarketRow.status == db_status)

    if search:
        like = f"%{search}%"
        condition = or_(MarketRow.question.ilike(like), MarketRow.id.ilike(like))
        stmt = stmt.where(condition)
        count_stmt = count_stmt.where(condition)

    total = int((await session.execute(count_stmt)).scalar_one())
    rows = (await session.execute(stmt.offset(offset).limit(limit))).scalars().all()

    return MarketListResponse(
        items=[_market_row_to_out(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/markets/{market_id}", response_model=MarketDetailOut)
async def get_market(
    market_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> MarketDetailOut:
    row = (
        await session.execute(select(MarketRow).where(MarketRow.id == market_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Market not found")

    current_signal: SignalOut | None = None
    if row.current_signal_id:
        sig_result = (
            await session.execute(
                select(SignalRow, MarketRow.question)
                .outerjoin(MarketRow, MarketRow.id == SignalRow.market_id)
                .where(SignalRow.id == row.current_signal_id)
            )
        ).one_or_none()
        if sig_result is not None:
            sig_row, market_question = sig_result
            current_signal = _signal_row_to_out(sig_row, market_question)

    base = _market_row_to_out(row)
    return MarketDetailOut(**base.model_dump(), current_signal=current_signal)


@router.post("/markets/{market_id}/analyze", response_model=AnalyzeResponse)
async def analyze_market(
    market_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
    pipeline: Annotated[object | None, Depends(_signal_pipeline)],
) -> AnalyzeResponse:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Signal pipeline not available — start freqpred with API credentials.",
        )

    market_row = (
        await session.execute(select(MarketRow).where(MarketRow.id == market_id))
    ).scalar_one_or_none()
    if market_row is None:
        raise HTTPException(status_code=404, detail="Market not found")

    # 429 cooldown: if current signal was created within the last 60 s, return it cached.
    if market_row.current_signal_id:
        sig_row = (
            await session.execute(
                select(SignalRow).where(SignalRow.id == market_row.current_signal_id)
            )
        ).scalar_one_or_none()
        if sig_row is not None:
            age_secs = (datetime.now(UTC) - sig_row.created_at).total_seconds()
            if age_secs < _ANALYZE_COOLDOWN_SECS:
                return AnalyzeResponse(
                    signal=_signal_row_to_out(sig_row, market_row.question),
                    cached=True,
                )

    from freqpred.markets.models import Market  # noqa: PLC0415

    market = Market(
        id=market_row.id,
        platform=market_row.platform,
        question=market_row.question,
        category=market_row.category,
        status=market_row.status,
        result=market_row.result,
        close_time=market_row.close_time,
        yes_bid=market_row.yes_bid,
        yes_ask=market_row.yes_ask,
        mid_price=market_row.mid_price,
        last_price=market_row.last_price,
        volume_24h=market_row.volume_24h,
        open_interest=market_row.open_interest,
        liquidity=market_row.liquidity,
        last_fetched_at=market_row.last_fetched_at,
        price_updated_at=market_row.price_updated_at,
        metadata_fetched_at=market_row.metadata_fetched_at,
        current_signal_id=(
            str(market_row.current_signal_id) if market_row.current_signal_id else None
        ),
    )

    new_signal = await pipeline.analyze(market, trigger="manual", force=True)  # type: ignore[union-attr]
    if new_signal is None:
        raise HTTPException(
            status_code=503,
            detail="No documents have been ingested for this market yet. Run the ingestion pipeline first.",
        )

    # Reload the saved SignalRow to build the response (new_signal is a domain Signal dataclass).
    sig_row = (
        await session.execute(
            select(SignalRow).where(SignalRow.id == _uuid.UUID(new_signal.id))
        )
    ).scalar_one_or_none()
    if sig_row is None:
        raise HTTPException(status_code=500, detail="Signal was generated but could not be retrieved.")

    return AnalyzeResponse(
        signal=_signal_row_to_out(sig_row, market_row.question),
        cached=False,
    )


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
    lookback_days: Annotated[int | None, Query(ge=1)] = None,
    category: str | None = Query(default=None),
) -> CalibrationResponse:
    app_mode = await _get_mode(session)
    report = await compute_calibration(
        session,
        mode=app_mode,
        lookback_days=lookback_days,
        market_category=category,
    )
    categories_result = await session.execute(
        select(MarketRow.category)
        .join(SignalRow, SignalRow.market_id == MarketRow.id)
        .where(
            MarketRow.status == "finalized",
            MarketRow.result.is_not(None),
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
        .distinct()
        .order_by(MarketRow.category)
    )
    available_categories = [row[0] for row in categories_result.all()]

    def _map_buckets(buckets: list) -> list[CalibrationBucketOut]:
        return [
            CalibrationBucketOut(
                lower=b.lower,
                upper=b.upper,
                count=b.count,
                mean_estimated_prob=b.mean_estimated_prob,
                actual_resolution_rate=b.actual_resolution_rate,
            )
            for b in buckets
        ]

    return CalibrationResponse(
        brier_score=report.brier_score,
        market_brier_score=report.market_brier_score,
        n_samples=report.n_samples,
        buckets=_map_buckets(report.buckets),
        market_buckets=_map_buckets(report.market_buckets),
        available_categories=available_categories,
    )


async def _compute_source_quality_rows_live(
    session: AsyncSession,
    *,
    lookback_days: int | None,
    category: str | None,
) -> list[SourceQualityScoreOut]:
    computed_at = datetime.now(UTC)
    categories: list[str | None]
    if category is not None:
        categories = [category]
    else:
        distinct_categories = await session.execute(
            select(MarketRow.category)
            .join(SignalRow, SignalRow.market_id == MarketRow.id)
            .where(
                MarketRow.status == "finalized",
                MarketRow.result.is_not(None),
                SignalRow.model_used != "demo_harness",
                SignalRow.prompt_version != "demo",
            )
            .distinct()
            .order_by(MarketRow.category)
        )
        categories = [None, *[row[0] for row in distinct_categories.all()]]

    items: list[SourceQualityScoreOut] = []
    for market_category in categories:
        calibration = await compute_calibration(
            session,
            lookback_days=lookback_days,
            market_category=market_category,
        )
        if calibration.n_samples == 0:
            continue

        scores = await compute_source_brier_scores(
            session,
            lookback_days=lookback_days,
            market_category=market_category,
        )
        items.extend(
            SourceQualityScoreOut(
                source_name=score.source_name,
                market_category=market_category,
                weighted_brier=score.weighted_brier_score,
                overall_brier=calibration.brier_score,
                n_signals=score.n_signals,
                total_doc_uses=score.total_doc_appearances,
                computed_at=computed_at,
            )
            for score in scores
        )

    items.sort(
        key=lambda item: (
            item.weighted_brier,
            "" if item.market_category is None else item.market_category,
            item.source_name,
        )
    )
    return items


@router.get("/metrics/source-quality", response_model=SourceQualityListResponse)
async def get_source_quality(
    session: Annotated[AsyncSession, Depends(get_db)],
    category: str | None = Query(default=None),
    lookback_days: Annotated[int | None, Query(ge=1)] = None,
) -> SourceQualityListResponse:
    filters = []
    if category is not None:
        filters.append(SourceQualityScoreRow.market_category == category)
    if lookback_days is not None:
        filters.append(SourceQualityScoreRow.lookback_days == lookback_days)

    latest_snapshot = (
        await session.execute(
            select(func.max(SourceQualityScoreRow.computed_at)).where(*filters)
        )
    ).scalar_one()
    if latest_snapshot is None:
        return SourceQualityListResponse(
            items=await _compute_source_quality_rows_live(
                session,
                lookback_days=lookback_days,
                category=category,
            )
        )

    rows = (
        await session.execute(
            select(SourceQualityScoreRow)
            .where(SourceQualityScoreRow.computed_at == latest_snapshot, *filters)
            .order_by(
                SourceQualityScoreRow.weighted_brier.asc(),
                SourceQualityScoreRow.market_category.asc().nullsfirst(),
                SourceQualityScoreRow.source_name.asc(),
            )
        )
    ).scalars().all()

    return SourceQualityListResponse(
        items=[
            SourceQualityScoreOut(
                source_name=row.source_name,
                market_category=row.market_category,
                weighted_brier=row.weighted_brier,
                overall_brier=row.overall_brier,
                n_signals=row.n_signals,
                total_doc_uses=row.total_doc_uses,
                computed_at=row.computed_at,
            )
            for row in rows
        ]
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


@router.get("/llm/queries", response_model=LLMQueryListResponse)
async def list_llm_queries(
    session: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> LLMQueryListResponse:
    count_result = await session.execute(select(func.count()).select_from(LLMQueryRow))
    total = int(count_result.scalar_one())

    rows = (
        await session.execute(
            select(LLMQueryRow)
            .order_by(LLMQueryRow.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
    ).scalars().all()

    return LLMQueryListResponse(
        items=[
            LLMQueryOut(
                id=r.id,
                timestamp=r.timestamp,
                query_type=r.query_type,
                market_id=r.market_id,
                model_used=r.model_used,
                tokens_total=r.tokens_total,
                cost_usd=r.cost_usd,
                latency_ms=r.latency_ms,
                success=r.success,
            )
            for r in rows
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/llm/queries/{query_id}", response_model=LLMQueryDetailOut)
async def get_llm_query(
    query_id: int,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> LLMQueryDetailOut:
    row = (
        await session.execute(select(LLMQueryRow).where(LLMQueryRow.id == query_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="LLM query not found")

    return LLMQueryDetailOut(
        id=row.id,
        timestamp=row.timestamp,
        query_type=row.query_type,
        market_id=row.market_id,
        model_used=row.model_used,
        tokens_total=row.tokens_total,
        cost_usd=row.cost_usd,
        latency_ms=row.latency_ms,
        success=row.success,
        prompt=row.prompt,
        response=row.response,
        error_message=row.error_message,
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
    daily_loss_ack_at: datetime | None = None
    daily_pnl: float = 0.0
    net_bankroll: float = initial_bankroll
    llm_budget_used: float = 0.0
    pending_orders: int = 0
    open_positions: int = 0
    llm_errors_last_hour: int = 0

    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    hour_ago = datetime.now(UTC) - timedelta(hours=1)
    loss_window_start: datetime = today_start

    try:
        app_mode = await _get_mode(session)

        # Read run_state row directly to get CB state alongside run_state + drawdown.
        rs_result = await session.execute(select(_RunStateRow).limit(1))
        rs_row = rs_result.scalar_one_or_none()
        if rs_row is not None:
            run_state = rs_row.state
            cb_halted = bool(rs_row.cb_active)
            cb_reason = rs_row.cb_reason
            daily_loss_ack_at = rs_row.daily_loss_ack_at

        net_bankroll = await get_net_bankroll(session, initial_bankroll, mode=app_mode)

        # Mirror the risk engine's daily loss window: max(today_start, daily_loss_ack_at).
        # Displaying the raw midnight-to-now total would diverge from what risk.py actually
        # enforces whenever the operator has /start-ed (which stamps daily_loss_ack_at).
        if daily_loss_ack_at is not None:
            loss_window_start = max(today_start, daily_loss_ack_at)
        daily_pnl_result = await session.execute(
            select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
                PositionRow.status == "closed",
                PositionRow.exit_time >= loss_window_start,
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
            daily_loss_window_start=loss_window_start,
            daily_loss_ack_at=daily_loss_ack_at,
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
