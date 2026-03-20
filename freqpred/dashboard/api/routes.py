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
    CalibrationBucketOut,
    CalibrationResponse,
    DocumentLinkOut,
    HealthResponse,
    LedgerResponse,
    LLMCostResponse,
    PositionListResponse,
    PositionOut,
    SignalDetailOut,
    SignalListResponse,
    SignalOut,
)

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
    stmt = select(PositionRow).order_by(PositionRow.entry_time.desc())
    count_stmt = select(func.count()).select_from(PositionRow)

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
    summary = await get_portfolio_summary(session)
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
    report = await compute_calibration(session)
    return CalibrationResponse(
        brier_score=report.brier_score,
        naive_brier_score=report.naive_brier_score,
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
        open_result = await session.execute(
            select(func.count()).where(PositionRow.status == "open")
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
