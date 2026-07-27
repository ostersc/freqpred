"""FastAPI router — all /api/* endpoints."""
from __future__ import annotations

import importlib.metadata
import os
import signal
import subprocess
import uuid as _uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.ingestion.models import FactbasePhraseRow
from freqpred.llm.audit import get_daily_spend_usd
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.kalshi import KalshiAPIError
from freqpred.markets.models import MarketRow, PositionRow
from freqpred.metrics.calibration import (
    compute_calibration,
    compute_calibration_heatmap,
    compute_calibration_time_series,
    compute_source_brier_scores,
    prompt_version_sort_key,
)
from freqpred.metrics.models import SignalAssessmentRow, SourceQualityScoreRow
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.runtime.models import RuntimeEventRow
from freqpred.runtime.telemetry import (
    EVENT_CATEGORY_KALSHI_API,
    RuntimeTelemetry,
    list_service_heartbeats,
)
from freqpred.signal.models import SignalRow
from freqpred.trading.ledger import (
    get_llm_spend_time_series,
    get_net_bankroll,
    get_pnl_time_series,
    get_portfolio_summary,
)
from freqpred.trading.order_manager import PositionNotFoundError, PositionNotOpenError

from .schemas import (
    AnalyzeResponse,
    ApiErrorStateOut,
    CalibrationBucketOut,
    CalibrationHeatmapCellOut,
    CalibrationHeatmapResponse,
    CalibrationHeatmapRowOut,
    CalibrationResponse,
    CalibrationTimeSeriesPointOut,
    CalibrationTimeSeriesResponse,
    ChangelogStatusOut,
    CircuitBreakerStateOut,
    DocumentLinkOut,
    ExchangeStatusOut,
    HealthResponse,
    KalshiApiTierOut,
    LedgerResponse,
    LLMCostResponse,
    LLMQueryDetailOut,
    LLMQueryListResponse,
    LLMQueryOut,
    LLMSpendDayOut,
    MarketDetailOut,
    MarketListResponse,
    MarketOut,
    PendingOrderSummary,
    PnLDayOut,
    PnLTimeSeriesResponse,
    PositionDetailOut,
    PositionListResponse,
    PositionOut,
    PromptVersionStart,
    ServiceFreshnessOut,
    SettlementSourceSummaryOut,
    SettlementSourceSummaryResponse,
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

if TYPE_CHECKING:
    from freqpred.markets.kalshi import KalshiClient

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


def _kalshi_base_url(request: Request) -> str:
    return getattr(request.app.state, "kalshi_base_url", "https://api.elections.kalshi.com/trade-api/v2")


def _runtime_telemetry(request: Request) -> RuntimeTelemetry | None:
    telemetry = getattr(request.app.state, "runtime_telemetry", None)
    return telemetry if isinstance(telemetry, RuntimeTelemetry) else None


def _kalshi_client(request: Request) -> KalshiClient | None:
    from freqpred.markets.kalshi import KalshiClient as _KC  # noqa: PLC0415
    client = getattr(request.app.state, "kalshi_client", None)
    return client if isinstance(client, _KC) else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal_row_to_out(
    row: SignalRow,
    market_question: str | None = None,
    rag_hit_count: int = 0,
    has_factbase: bool = False,
    series_ticker: str | None = None,
    has_assessment: bool = False,
    has_open_position: bool = False,
) -> SignalOut:
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
        llm_query_id=row.llm_query_id,
        rag_hit_count=rag_hit_count,
        has_factbase=has_factbase,
        series_ticker=series_ticker,
        has_assessment=has_assessment,
        has_open_position=has_open_position,
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


def _position_row_to_out(
    row: PositionRow,
    current_mid: float | None = None,
    has_factbase: bool = False,
    series_ticker: str | None = None,
) -> PositionOut:
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None
    if row.status == "open" and current_mid is not None:
        # Mirrors ledger.close_position's realized pnl/cost_basis formula —
        # gross P&L net of entry fee, cost basis including it — so an open
        # position's displayed P&L doesn't jump the moment it actually
        # closes purely because fee accounting kicks in.
        fee = row.entry_fee_usd or 0.0
        if row.direction == "YES":
            gross_pnl = row.contracts * (current_mid - row.entry_price)
        else:
            gross_pnl = row.contracts * ((1.0 - current_mid) - row.entry_price)
        unrealized_pnl = gross_pnl - fee
        cost_basis = row.entry_price * row.contracts + fee
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
        entry_fee_usd=row.entry_fee_usd or 0.0,
        effective_entry_price=(
            row.entry_price + (row.entry_fee_usd or 0.0) / row.contracts
            if row.contracts
            else row.entry_price
        ),
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
        current_mid=current_mid if row.status == "open" else None,
        created_at=row.created_at,
        has_factbase=has_factbase,
        series_ticker=series_ticker,
        exchange_order_id=row.exchange_order_id,
        requested_contracts=row.requested_contracts,
        exchange_order_status=row.exchange_order_status,
        last_exchange_sync_at=row.last_exchange_sync_at,
        exit_order_id=row.exit_order_id,
        exit_fee_usd=row.exit_fee_usd or 0.0,
        exit_requested_contracts=row.exit_requested_contracts,
        exit_filled_contracts=row.exit_filled_contracts,
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
    series_ticker: str | None = None,
) -> SignalDetailOut:
    document_links = await _load_document_links(session, signal_row.id)
    assessment = await _load_signal_assessment(session, signal_row.id)
    has_factbase = bool(
        (await session.execute(
            select(func.count())
            .select_from(FactbasePhraseRow)
            .where(FactbasePhraseRow.market_id == signal_row.market_id)
        )).scalar_one()
    )
    base = _signal_row_to_out(
        signal_row,
        market_question,
        rag_hit_count=len(document_links),
        has_factbase=has_factbase,
        series_ticker=series_ticker,
        has_assessment=assessment is not None,
    )
    return SignalDetailOut(
        **base.model_dump(),
        document_links=document_links,
        assessment=assessment,
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
        max_edge=cfg.max_edge,
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
    min_edge: float | None = Query(default=None),
    max_edge: float | None = Query(default=None),
    min_confidence: float | None = Query(default=None),
    max_confidence: float | None = Query(default=None),
    has_factbase: bool | None = Query(default=None),
    has_docs: bool | None = Query(default=None),
    trigger: str | None = Query(default=None),
    series_ticker: str | None = Query(default=None),
) -> SignalListResponse:
    app_mode = await _get_mode(session)

    rag_count_subq = (
        select(func.count())
        .select_from(DocumentMarketLinkRow)
        .where(DocumentMarketLinkRow.signal_id == SignalRow.id)
        .correlate(SignalRow)
        .scalar_subquery()
    )
    has_factbase_subq = (
        select(func.count())
        .select_from(FactbasePhraseRow)
        .where(FactbasePhraseRow.market_id == SignalRow.market_id)
        .correlate(SignalRow)
        .scalar_subquery()
    )
    has_assessment_subq = (
        select(func.count())
        .select_from(SignalAssessmentRow)
        .where(SignalAssessmentRow.signal_id == SignalRow.id)
        .correlate(SignalRow)
        .scalar_subquery()
    )
    has_open_position_subq = (
        select(func.count())
        .select_from(PositionRow)
        .where(PositionRow.market_id == SignalRow.market_id)
        .where(PositionRow.status == "open")
        .where(PositionRow.mode == app_mode)
        .correlate(SignalRow)
        .scalar_subquery()
    )
    stmt = (
        select(
            SignalRow,
            MarketRow.question,
            MarketRow.series_ticker,
            rag_count_subq.label("rag_hit_count"),
            has_factbase_subq.label("has_factbase"),
            has_assessment_subq.label("has_assessment"),
            has_open_position_subq.label("has_open_position"),
        )
        .outerjoin(MarketRow, MarketRow.id == SignalRow.market_id)
        .order_by(SignalRow.created_at.desc())
    )
    count_stmt = (
        select(func.count())
        .select_from(SignalRow)
        .outerjoin(MarketRow, MarketRow.id == SignalRow.market_id)
    )

    if market_id:
        stmt = stmt.where(SignalRow.market_id == market_id)
        count_stmt = count_stmt.where(SignalRow.market_id == market_id)
    if direction:
        stmt = stmt.where(SignalRow.direction == direction.upper())
        count_stmt = count_stmt.where(SignalRow.direction == direction.upper())
    if min_edge is not None:
        stmt = stmt.where(SignalRow.edge >= min_edge)
        count_stmt = count_stmt.where(SignalRow.edge >= min_edge)
    if max_edge is not None:
        stmt = stmt.where(SignalRow.edge <= max_edge)
        count_stmt = count_stmt.where(SignalRow.edge <= max_edge)
    if min_confidence is not None:
        stmt = stmt.where(SignalRow.confidence >= min_confidence)
        count_stmt = count_stmt.where(SignalRow.confidence >= min_confidence)
    if max_confidence is not None:
        stmt = stmt.where(SignalRow.confidence <= max_confidence)
        count_stmt = count_stmt.where(SignalRow.confidence <= max_confidence)
    if has_factbase is not None:
        cond = has_factbase_subq > 0 if has_factbase else has_factbase_subq == 0
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)
    if has_docs is not None:
        cond = rag_count_subq > 0 if has_docs else rag_count_subq == 0
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)
    if trigger:
        stmt = stmt.where(SignalRow.trigger == trigger)
        count_stmt = count_stmt.where(SignalRow.trigger == trigger)
    if series_ticker:
        stmt = stmt.where(MarketRow.series_ticker == series_ticker)
        count_stmt = count_stmt.where(MarketRow.series_ticker == series_ticker)

    total = int((await session.execute(count_stmt)).scalar_one())
    result_rows = (await session.execute(stmt.offset(offset).limit(limit))).all()

    distinct_triggers = (
        await session.execute(select(SignalRow.trigger).distinct().order_by(SignalRow.trigger))
    ).scalars().all()
    distinct_series_tickers = (
        await session.execute(
            select(MarketRow.series_ticker)
            .join(SignalRow, SignalRow.market_id == MarketRow.id)
            .where(MarketRow.series_ticker.isnot(None))
            .distinct()
            .order_by(MarketRow.series_ticker)
        )
    ).scalars().all()

    return SignalListResponse(
        items=[
            _signal_row_to_out(r, q, int(rag), bool(fb), st, bool(asmnt), bool(hop))
            for r, q, st, rag, fb, asmnt, hop in result_rows
        ],
        total=total,
        limit=limit,
        offset=offset,
        distinct_triggers=list(distinct_triggers),
        distinct_series_tickers=list(distinct_series_tickers),
    )


@router.get("/signals/{signal_id}", response_model=SignalDetailOut)
async def get_signal(
    signal_id: str,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> SignalDetailOut:
    try:
        uid = _uuid.UUID(signal_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Signal not found") from exc

    result = (
        await session.execute(
            select(SignalRow, MarketRow.question, MarketRow.series_ticker)
            .outerjoin(MarketRow, MarketRow.id == SignalRow.market_id)
            .where(SignalRow.id == uid)
        )
    ).one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail="Signal not found")
    row, market_question, series_ticker = result

    return await _build_signal_detail(session, row, market_question, series_ticker)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@router.get("/positions", response_model=PositionListResponse)
async def list_positions(
    session: Annotated[AsyncSession, Depends(get_db)],
    status: str = Query(default="all", pattern="^(open|closed|all)$"),
) -> PositionListResponse:
    app_mode = await _get_mode(session)
    pos_has_factbase_subq = (
        select(func.count())
        .select_from(FactbasePhraseRow)
        .where(FactbasePhraseRow.market_id == PositionRow.market_id)
        .correlate(PositionRow)
        .scalar_subquery()
    )
    stmt = (
        select(
            PositionRow,
            MarketRow.mid_price,
            MarketRow.yes_bid,
            MarketRow.yes_ask,
            MarketRow.last_price,
            MarketRow.series_ticker,
            pos_has_factbase_subq.label("has_factbase"),
        )
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
                has_factbase=bool(fb),
                series_ticker=st,
            )
            for r, mid, yes_bid, yes_ask, last_price, st, fb in rows
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
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Position not found") from exc

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
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Position not found") from exc

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

    entry_signal = await _build_signal_detail(
        session, sig_row, market_question,
        series_ticker=market_row.series_ticker if market_row else None,
    )

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
        **_position_row_to_out(
            pos_row,
            current_mid=current_mid,
            has_factbase=entry_signal.has_factbase,
            series_ticker=market_row.series_ticker if market_row else None,
        ).model_dump(),
        market_question=market_question,
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
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Position not found") from exc

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
    distinct_strategies = list((await session.execute(distinct_strategies_stmt)).scalars().all())
    distinct_exit_reasons = list((await session.execute(distinct_exit_reasons_stmt)).scalars().all())

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


@router.get(
    "/markets/settlement-sources/summary",
    response_model=SettlementSourceSummaryResponse,
)
async def get_settlement_sources_summary(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> SettlementSourceSummaryResponse:
    """Group active markets by official settlement source, ranked by market count.

    Surfaces candidates for a dedicated fetcher the way FactBase (T73) was
    found manually for KXTRUMPSAY markets — a source appearing on many active
    markets is worth scraping directly.
    """
    rows = (
        await session.execute(
            select(MarketRow.metadata_).where(MarketRow.status == "active")
        )
    ).scalars().all()

    # Kalshi's own settlement_sources values are inconsistent about trailing
    # slashes on otherwise-identical URLs (e.g. "espn.com" vs "espn.com/"),
    # which fragments counts for what is really one source. Normalize on the
    # grouping key only; display the first-seen (name, url) for that key.
    counts: dict[tuple[str, str], int] = {}
    display: dict[tuple[str, str], tuple[str, str]] = {}
    for metadata in rows:
        sources = (metadata or {}).get("settlement_sources") or []
        for source in sources:
            name = source.get("name", "")
            url = source.get("url", "")
            if not name and not url:
                continue
            key = (name.strip().lower(), url.strip().lower().rstrip("/"))
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, (name, url))

    items = [
        SettlementSourceSummaryOut(name=display[key][0], url=display[key][1], market_count=count)
        for key, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return SettlementSourceSummaryResponse(items=items)


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
        yes_bid_size=market_row.yes_bid_size,
        yes_ask_size=market_row.yes_ask_size,
        last_fetched_at=market_row.last_fetched_at,
        price_updated_at=market_row.price_updated_at,
        metadata_fetched_at=market_row.metadata_fetched_at,
        current_signal_id=(
            str(market_row.current_signal_id) if market_row.current_signal_id else None
        ),
        open_time=market_row.open_time,
        series_ticker=market_row.series_ticker,
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


@router.get("/calibration/time-series", response_model=CalibrationTimeSeriesResponse)
async def get_calibration_time_series(
    session: Annotated[AsyncSession, Depends(get_db)],
    lookback_days: Annotated[int | None, Query(ge=1)] = None,
    category: str | None = Query(default=None),
    ticker_prefix: str | None = Query(default=None),
    direction: str | None = Query(default=None),
    model_used: str | None = Query(default=None),
    prompt_version: str | None = Query(default=None),
    series_ticker: str | None = Query(default=None),
    min_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    max_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> CalibrationTimeSeriesResponse:
    app_mode = await _get_mode(session)
    ts = await compute_calibration_time_series(
        session,
        mode=app_mode,
        lookback_days=lookback_days,
        market_category=category,
        ticker_prefix=ticker_prefix,
        direction=direction,
        model_used=model_used,
        prompt_version=prompt_version,
        series_ticker=series_ticker,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    # prompt_version_starts: first signal date per prompt version
    pv_starts_result = await session.execute(
        select(
            SignalRow.prompt_version,
            func.min(func.date(SignalRow.created_at)).label("first_date"),
        )
        .where(
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
        .group_by(SignalRow.prompt_version)
        .order_by(func.min(func.date(SignalRow.created_at)))
    )
    prompt_version_starts = [
        PromptVersionStart(version=row[0], date=str(row[1]))
        for row in pv_starts_result.all()
        if row[0] is not None
    ]

    _base_where = [
        MarketRow.status == "finalized",
        MarketRow.result.is_not(None),
        SignalRow.model_used != "demo_harness",
        SignalRow.prompt_version != "demo",
    ]

    async def _distinct(col):  # type: ignore[no-untyped-def]
        res = await session.execute(
            select(col)
            .select_from(MarketRow)
            .join(SignalRow, SignalRow.market_id == MarketRow.id)
            .where(*_base_where)
            .distinct()
            .order_by(col)
        )
        return [row[0] for row in res.all() if row[0] is not None]

    available_categories = await _distinct(MarketRow.category)
    available_models = await _distinct(SignalRow.model_used)
    available_prompt_versions = sorted(
        await _distinct(SignalRow.prompt_version), key=prompt_version_sort_key
    )
    available_directions = await _distinct(SignalRow.direction)
    available_series_tickers = await _distinct(MarketRow.series_ticker)

    return CalibrationTimeSeriesResponse(
        series=[
            CalibrationTimeSeriesPointOut(
                date=pt.date,
                brier_score=pt.brier_score,
                market_brier_score=pt.market_brier_score,
                n_samples=pt.n_samples,
            )
            for pt in ts.points
        ],
        prompt_version_starts=prompt_version_starts,
        available_categories=available_categories,
        available_models=available_models,
        available_prompt_versions=available_prompt_versions,
        available_directions=available_directions,
        available_series_tickers=available_series_tickers,
    )


@router.get("/calibration/by-option", response_model=CalibrationHeatmapResponse)
async def get_calibration_by_option(
    session: Annotated[AsyncSession, Depends(get_db)],
    lookback_days: Annotated[int | None, Query(ge=1)] = None,
    category: str | None = Query(default=None),
    ticker_prefix: str | None = Query(default=None),
    direction: str | None = Query(default=None),
    model_used: str | None = Query(default=None),
    series_ticker: str | None = Query(default=None),
    min_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    max_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> CalibrationHeatmapResponse:
    app_mode = await _get_mode(session)
    report = await compute_calibration_heatmap(
        session,
        mode=app_mode,
        lookback_days=lookback_days,
        market_category=category,
        ticker_prefix=ticker_prefix,
        direction=direction,
        model_used=model_used,
        series_ticker=series_ticker,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    _base_where = [
        MarketRow.status == "finalized",
        MarketRow.result.is_not(None),
        SignalRow.model_used != "demo_harness",
        SignalRow.prompt_version != "demo",
    ]

    async def _distinct(col):  # type: ignore[no-untyped-def]
        res = await session.execute(
            select(col)
            .select_from(MarketRow)
            .join(SignalRow, SignalRow.market_id == MarketRow.id)
            .where(*_base_where)
            .distinct()
            .order_by(col)
        )
        return [row[0] for row in res.all() if row[0] is not None]

    available_categories = await _distinct(MarketRow.category)
    available_models = await _distinct(SignalRow.model_used)
    available_directions = await _distinct(SignalRow.direction)
    available_series_tickers = await _distinct(MarketRow.series_ticker)

    def _map_cells(
        cells: dict,
    ) -> dict[str, CalibrationHeatmapCellOut]:
        return {
            k: CalibrationHeatmapCellOut(
                brier_score=c.brier_score,
                market_brier_score=c.market_brier_score,
                n_samples=c.n_samples,
                delta=c.delta,
            )
            for k, c in cells.items()
        }

    return CalibrationHeatmapResponse(
        rows=[
            CalibrationHeatmapRowOut(
                series_ticker=r.series_ticker,
                option_code=r.option_code,
                option_label=r.option_label,
                cells=_map_cells(r.cells),
            )
            for r in report.rows
        ],
        prompt_versions=report.prompt_versions,
        available_categories=available_categories,
        available_models=available_models,
        available_directions=available_directions,
        available_series_tickers=available_series_tickers,
    )


@router.get("/calibration", response_model=CalibrationResponse)
async def get_calibration(
    session: Annotated[AsyncSession, Depends(get_db)],
    lookback_days: Annotated[int | None, Query(ge=1)] = None,
    category: str | None = Query(default=None),
    ticker_prefix: str | None = Query(default=None),
    direction: str | None = Query(default=None),
    model_used: str | None = Query(default=None),
    prompt_version: str | None = Query(default=None),
    series_ticker: str | None = Query(default=None),
    min_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    max_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> CalibrationResponse:
    app_mode = await _get_mode(session)
    report = await compute_calibration(
        session,
        mode=app_mode,
        lookback_days=lookback_days,
        market_category=category,
        ticker_prefix=ticker_prefix,
        direction=direction,
        model_used=model_used,
        prompt_version=prompt_version,
        series_ticker=series_ticker,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    # Fetch all available filter option values from finalized resolved signals
    _base_where = [
        MarketRow.status == "finalized",
        MarketRow.result.is_not(None),
        SignalRow.model_used != "demo_harness",
        SignalRow.prompt_version != "demo",
    ]

    async def _distinct(col):  # type: ignore[no-untyped-def]
        res = await session.execute(
            select(col)
            .select_from(MarketRow)
            .join(SignalRow, SignalRow.market_id == MarketRow.id)
            .where(*_base_where)
            .distinct()
            .order_by(col)
        )
        return [row[0] for row in res.all() if row[0] is not None]

    available_categories = await _distinct(MarketRow.category)
    available_models = await _distinct(SignalRow.model_used)
    available_prompt_versions = sorted(
        await _distinct(SignalRow.prompt_version), key=prompt_version_sort_key
    )
    available_directions = await _distinct(SignalRow.direction)
    available_series_tickers = await _distinct(MarketRow.series_ticker)

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
        available_models=available_models,
        available_prompt_versions=available_prompt_versions,
        available_directions=available_directions,
        available_series_tickers=available_series_tickers,
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
# P&L over time
# ---------------------------------------------------------------------------


@router.get("/pnl/time-series", response_model=PnLTimeSeriesResponse)
async def get_pnl_time_series_endpoint(
    session: Annotated[AsyncSession, Depends(get_db)],
    initial_bankroll: Annotated[float, Depends(_bankroll_usd)],
    lookback_days: Annotated[int | None, Query(ge=1)] = None,
    strategy: str | None = Query(default=None),
    model_used: str | None = Query(default=None),
    prompt_version: str | None = Query(default=None),
    direction: str | None = Query(default=None),
    category: str | None = Query(default=None),
    series_ticker: str | None = Query(default=None),
    market_id: str | None = Query(default=None),
) -> PnLTimeSeriesResponse:
    app_mode = await _get_mode(session)

    pnl_series_raw = await get_pnl_time_series(
        session,
        mode=app_mode,
        lookback_days=lookback_days,
        strategy_name=strategy,
        model_used=model_used,
        prompt_version=prompt_version,
        direction=direction,
        category=category,
        series_ticker=series_ticker,
        market_id=market_id,
    )

    llm_series_raw = await get_llm_spend_time_series(
        session,
        lookback_days=lookback_days,
    )

    # Stable filter options — use the full closed-position set for this mode,
    # not the narrowed set from the active filters.  Same pattern as /calibration.
    _base = [
        PositionRow.status == "closed",
        PositionRow.mode == app_mode,
        PositionRow.exit_time.is_not(None),
    ]

    async def _distinct_pos(col):  # type: ignore[no-untyped-def]
        res = await session.execute(
            select(col).where(*_base).distinct().order_by(col)
        )
        return [r[0] for r in res.all() if r[0] is not None]

    async def _distinct_via(col, join_model, join_condition):  # type: ignore[no-untyped-def]
        res = await session.execute(
            select(col)
            .select_from(PositionRow)
            .join(join_model, join_condition)
            .where(*_base)
            .distinct()
            .order_by(col)
        )
        return [r[0] for r in res.all() if r[0] is not None]

    available_strategies = await _distinct_pos(PositionRow.strategy_name)
    available_directions = await _distinct_pos(PositionRow.direction)
    available_market_ids = await _distinct_pos(PositionRow.market_id)
    available_models = await _distinct_via(
        SignalRow.model_used, SignalRow, SignalRow.id == PositionRow.signal_id
    )
    available_prompt_versions = sorted(
        await _distinct_via(SignalRow.prompt_version, SignalRow, SignalRow.id == PositionRow.signal_id),
        key=prompt_version_sort_key,
    )
    available_categories = await _distinct_via(
        MarketRow.category, MarketRow, MarketRow.id == PositionRow.market_id
    )
    available_series_tickers = await _distinct_via(
        MarketRow.series_ticker, MarketRow, MarketRow.id == PositionRow.market_id
    )

    # First closed-position date per prompt version (for chart milestone flags)
    pv_res = await session.execute(
        select(SignalRow.prompt_version, func.min(func.date(PositionRow.exit_time)))
        .select_from(PositionRow)
        .join(SignalRow, SignalRow.id == PositionRow.signal_id)
        .where(*_base)
        .group_by(SignalRow.prompt_version)
        .order_by(func.min(func.date(PositionRow.exit_time)))
    )
    prompt_version_starts = [
        PromptVersionStart(version=str(row[0]), date=str(row[1]))
        for row in pv_res.all()
        if row[0] is not None and row[1] is not None
    ]

    total_trades = sum(d["trade_count"] for d in pnl_series_raw)
    all_time_pnl = pnl_series_raw[-1]["cumulative_pnl"] if pnl_series_raw else 0.0

    # Preset-independent: the same value the risk checks size against, so the
    # projection anchor does not drift when the lookback toggle changes.
    net_bankroll_now = await get_net_bankroll(session, initial_bankroll, mode=app_mode)

    return PnLTimeSeriesResponse(
        pnl_series=[PnLDayOut(**d) for d in pnl_series_raw],
        llm_series=[LLMSpendDayOut(**d) for d in llm_series_raw],
        prompt_version_starts=prompt_version_starts,
        initial_bankroll=initial_bankroll,
        net_bankroll_now=net_bankroll_now,
        total_trades=total_trades,
        all_time_pnl=all_time_pnl,
        available_strategies=available_strategies,
        available_models=available_models,
        available_prompt_versions=available_prompt_versions,
        available_directions=available_directions,
        available_categories=available_categories,
        available_series_tickers=available_series_tickers,
        available_market_ids=available_market_ids,
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

    # Breakdown by query_type for today (UTC). Half-open range instead of
    # date(timestamp) == today so ix_llm_queries_timestamp is usable.
    now = datetime.now(UTC)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    by_type_result = await session.execute(
        select(LLMQueryRow.query_type, func.sum(LLMQueryRow.cost_usd))
        .where(
            LLMQueryRow.timestamp >= day_start,
            LLMQueryRow.timestamp < day_start + timedelta(days=1),
        )
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
    runtime_telemetry: Annotated[RuntimeTelemetry | None, Depends(_runtime_telemetry)],
    kalshi_base_url: Annotated[str, Depends(_kalshi_base_url)],
    kalshi_client: Annotated["KalshiClient | None", Depends(_kalshi_client)],
) -> SystemHealthResponse:
    import httpx as _httpx  # noqa: PLC0415

    import freqpred.alerts.models  # noqa: F401 — ensure RunStateRow is registered  # noqa: PLC0415
    from freqpred.alerts.models import RunStateRow as _RunStateRow  # noqa: PLC0415
    from freqpred.alerts.run_state import daily_loss_ack_from_row  # noqa: PLC0415

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
    oldest_pending_order_age_seconds: int | None = None
    pending_orders_detail: list[PendingOrderSummary] = []
    open_positions: int = 0
    llm_errors_last_hour: int = 0
    kalshi_errors_last_hour: int = 0
    service_rows: list[ServiceFreshnessOut] = []
    changelog_status = ChangelogStatusOut(
        unreviewed_count=0,
        has_unreviewed_breaking_change=False,
        last_reviewed_at=None,
        last_checked_at=None,
    )
    websocket_state = (
        runtime_telemetry.websocket_state() if runtime_telemetry is not None else {}
    )
    exchange_status = ExchangeStatusOut(
        exchange_active=None, trading_active=None, fetched_at=None
    )

    try:
        async with _httpx.AsyncClient(timeout=5.0) as _hc:
            _ex_resp = await _hc.get(f"{kalshi_base_url}/exchange/status")
            _ex_resp.raise_for_status()
            _ex_data = _ex_resp.json()
        exchange_status = ExchangeStatusOut(
            exchange_active=bool(_ex_data.get("exchange_active")),
            trading_active=bool(_ex_data.get("trading_active")),
            fetched_at=datetime.now(UTC),
        )
    except Exception:
        log.warning("system_health.exchange_status_fetch_failed", base_url=kalshi_base_url)

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
            daily_loss_ack_at = daily_loss_ack_from_row(rs_row, app_mode)

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

        pending_min_result = await session.execute(
            select(func.min(PositionRow.entry_time)).where(
                PositionRow.status == "pending",
                PositionRow.mode == app_mode,
            )
        )
        oldest_pending_entry_time = pending_min_result.scalar_one()
        if isinstance(oldest_pending_entry_time, datetime):
            oldest_pending_order_age_seconds = int(
                (datetime.now(UTC) - oldest_pending_entry_time).total_seconds()
            )

        # Per-pending-order detail, oldest-first so the first row matches
        # oldest_pending_order_age_seconds.
        pending_detail_result = await session.execute(
            select(
                PositionRow.id,
                PositionRow.market_id,
                PositionRow.requested_contracts,
                PositionRow.contracts,
                PositionRow.exchange_order_status,
                PositionRow.entry_time,
                PositionRow.last_exchange_sync_at,
            )
            .where(
                PositionRow.status == "pending",
                PositionRow.mode == app_mode,
            )
            .order_by(PositionRow.entry_time.asc())
        )
        now_ts = datetime.now(UTC)
        for pid, mid_, req_c, filled_c, ex_status, entry_t, sync_at in pending_detail_result.all():
            entry_norm = entry_t if entry_t.tzinfo else entry_t.replace(tzinfo=UTC)
            pending_orders_detail.append(
                PendingOrderSummary(
                    position_id=str(pid),
                    market_id=mid_,
                    requested_contracts=req_c,
                    filled_contracts=filled_c,
                    exchange_order_status=ex_status,
                    age_seconds=int((now_ts - entry_norm).total_seconds()),
                    last_exchange_sync_at=sync_at,
                )
            )

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

        kalshi_errors_result = await session.execute(
            select(func.count(RuntimeEventRow.id)).where(
                RuntimeEventRow.category == EVENT_CATEGORY_KALSHI_API,
                RuntimeEventRow.created_at >= hour_ago,
            )
        )
        kalshi_errors_last_hour = int(kalshi_errors_result.scalar_one())

        from freqpred.runtime.models import (
            KalshiChangelogStateRow as _ChangelogRow,  # noqa: PLC0415
        )
        _cl_result = await session.execute(select(_ChangelogRow).where(_ChangelogRow.id == 1))
        _cl_row = _cl_result.scalar_one_or_none()
        if _cl_row is not None:
            changelog_status = ChangelogStatusOut(
                unreviewed_count=_cl_row.unreviewed_count,
                has_unreviewed_breaking_change=_cl_row.has_unreviewed_breaking_change,
                last_reviewed_at=_cl_row.last_reviewed_at,
                last_checked_at=_cl_row.last_checked_at,
            )

        if runtime_telemetry is not None:
            heartbeats = await list_service_heartbeats(session)
            service_states = runtime_telemetry.evaluate_service_states(
                heartbeats,
                run_state=run_state,
                now=datetime.now(UTC),
            )
            service_rows = [
                ServiceFreshnessOut(
                    service_name=state.service_name,
                    label=state.label,
                    status=state.status,
                    last_success_at=state.last_success_at,
                    last_error_at=state.last_error_at,
                    last_error_message=state.last_error_message,
                    stale_after_seconds=state.stale_after_seconds,
                    age_seconds=state.age_seconds,
                )
                for state in service_states
            ]

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

    api_tier: KalshiApiTierOut | None = None
    if kalshi_client is not None:
        try:
            limits_data = await kalshi_client.get_account_limits()
            raw_level: str | None = (
                limits_data.get("usage_tier")
                or limits_data.get("api_usage_level")
                or limits_data.get("usage_level")
            )
            if raw_level is None:
                for v in limits_data.values():
                    if isinstance(v, dict):
                        raw_level = (
                            v.get("usage_tier")
                            or v.get("api_usage_level")
                            or v.get("usage_level")
                        )
                        if raw_level:
                            break
            if raw_level is None:
                log.info("system_health.api_tier_unknown", limits_keys=list(limits_data.keys()))
            api_tier = KalshiApiTierOut(
                api_usage_level=raw_level,
                # Only offer upgrade when we know the current tier and it isn't advanced.
                can_upgrade=raw_level is not None and raw_level.lower() != "advanced",
                fetched_at=datetime.now(UTC),
            )
        except Exception:
            log.warning("system_health.api_tier_fetch_failed")

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
            status=(
                next(
                    (state.status for state in service_rows if state.service_name == "position_watcher_last_message"),
                    "unknown",
                )
                if runtime_telemetry is not None
                else "unknown"
            ),
            connected=websocket_state.get("connected"),
            subscribed_markets=websocket_state.get("subscribed_markets"),
            last_message_at=websocket_state.get("last_message_at"),
            last_reconcile_at=websocket_state.get("last_reconcile_at"),
        ),
        api_errors=ApiErrorStateOut(
            kalshi_errors_last_hour=kalshi_errors_last_hour,
            llm_errors_last_hour=llm_errors_last_hour,
            consecutive_llm_errors=None,
        ),
        services=service_rows,
        exchange=exchange_status,
        changelog=changelog_status,
        pending_orders=pending_orders,
        oldest_pending_order_age_seconds=oldest_pending_order_age_seconds,
        pending_orders_detail=pending_orders_detail,
        open_positions=open_positions,
        db_ok=db_ok,
        uptime_seconds=uptime_seconds,
        api_tier=api_tier,
    )


# ---------------------------------------------------------------------------
# API tier upgrade
# ---------------------------------------------------------------------------


@router.post("/system/api-tier/upgrade", status_code=200)
async def upgrade_api_tier_endpoint(
    kalshi_client: Annotated["KalshiClient | None", Depends(_kalshi_client)],
) -> dict[str, bool]:
    if kalshi_client is None:
        raise HTTPException(status_code=503, detail="Kalshi client not available")
    try:
        await kalshi_client.upgrade_api_tier()
    except KalshiAPIError as exc:
        if exc.status_code == 404:
            log.warning("api_tier.upgrade_not_available")
            raise HTTPException(status_code=503, detail="Upgrade endpoint not yet available") from exc
        if exc.status_code == 403:
            if "insufficient_scope" in exc.body:
                log.warning("api_tier.upgrade_insufficient_scope")
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Kalshi API key lacks write/trading scope. Generate a key with "
                        "trading permissions in your Kalshi account settings."
                    ),
                ) from exc
            log.warning("api_tier.upgrade_no_api_order")
            raise HTTPException(
                status_code=403,
                detail=(
                    "Kalshi requires at least one order placed via the API in your "
                    "last 100 orders before granting the Advanced tier. Place an "
                    "API order, then retry."
                ),
            ) from exc
        log.exception("api_tier.upgrade_failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("api_tier.upgrade_failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True}


# ---------------------------------------------------------------------------
# Run-state control
# ---------------------------------------------------------------------------

_VALID_STATES = {"running", "paused", "stopped"}


@router.post("/system/run-state", status_code=200)
async def set_run_state_endpoint(
    session: Annotated[AsyncSession, Depends(get_db)],
    body: dict,
) -> dict:
    """Set the run-loop state to running / paused / stopped."""
    from freqpred.alerts.run_state import get_mode, set_run_state  # noqa: PLC0415

    state = body.get("state", "")
    if state not in _VALID_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state '{state}'. Must be one of: {sorted(_VALID_STATES)}")
    # Resuming acknowledges the daily-loss breaker for the running loop's mode
    # only — mirror the Telegram /start handler.
    mode = await get_mode(session)
    await set_run_state(session, state, mode=mode)
    log.info("dashboard.run_state_set", state=state, mode=mode)
    return {"state": state}


@router.post("/system/shutdown", status_code=200)
async def shutdown_endpoint() -> dict:
    """Send SIGTERM to the process — graceful shutdown. Cannot be undone from the dashboard."""
    log.info("dashboard.shutdown_requested")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"ok": True}


@router.get("/system/version")
async def get_version() -> dict:
    try:
        version = importlib.metadata.version("freqpred")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    try:
        git_hash = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_hash = "unknown"
    return {"version": version, "git_hash": git_hash}
