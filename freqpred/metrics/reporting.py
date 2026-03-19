"""Daily digest and report generation."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.llm.audit import get_daily_spend_usd
from freqpred.llm.client import LLMClient
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import PositionRow
from freqpred.metrics.calibration import compute_calibration

log = structlog.get_logger(__name__)

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_DIGEST_SYSTEM = (
    "You are a concise trading system reporter. "
    "Write a single paragraph of ≤150 words summarizing the state of a "
    "prediction market trading system. Use plain English, include all key "
    "numbers, and highlight anything worth attention. No bullet points."
)


async def generate_daily_digest(
    session: AsyncSession,
    llm_client: LLMClient,
) -> str:
    """
    Assembles a structured data snapshot (open positions, yesterday P&L,
    LLM spend, calibration score) and passes it to Claude Haiku for a
    concise natural-language summary. Logs the LLM call via audit.
    Returns the formatted digest string.
    """
    now = datetime.now(UTC)
    yesterday_start = (now - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    yesterday_end = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # --- Open positions ---
    open_result = await session.execute(
        select(
            func.count(PositionRow.id),
            func.coalesce(
                func.sum(PositionRow.contracts * PositionRow.entry_price), 0.0
            ),
        ).where(PositionRow.status == "open")
    )
    open_count, total_exposure = open_result.one()
    open_count = int(open_count)
    total_exposure = float(total_exposure)

    # --- Yesterday's closed P&L ---
    pnl_result = await session.execute(
        select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
            PositionRow.status == "closed",
            PositionRow.exit_time >= yesterday_start,
            PositionRow.exit_time < yesterday_end,
        )
    )
    yesterday_pnl = float(pnl_result.scalar_one())

    # --- LLM spend yesterday vs daily cap ---
    yesterday_spend_result = await session.execute(
        select(func.coalesce(func.sum(LLMQueryRow.cost_usd), 0.0)).where(
            LLMQueryRow.timestamp >= yesterday_start,
            LLMQueryRow.timestamp < yesterday_end,
        )
    )
    yesterday_llm_spend = float(yesterday_spend_result.scalar_one())

    today_llm_spend = await get_daily_spend_usd(session)

    # --- Calibration ---
    calibration = await compute_calibration(session)

    # --- Circuit breaker events (LLM errors in last 24h as proxy) ---
    cb_result = await session.execute(
        select(func.count(LLMQueryRow.id)).where(
            LLMQueryRow.success.is_(False),
            LLMQueryRow.timestamp >= now - timedelta(hours=24),
        )
    )
    circuit_breaker_events = int(cb_result.scalar_one())

    # --- Build prompt ---
    calibration_str = (
        f"Brier score {calibration.brier_score:.3f} over {calibration.n_samples} resolved markets"
        if calibration.n_samples > 0
        else "no resolved markets yet (calibration unavailable)"
    )
    prompt = (
        f"Daily digest as of {now.strftime('%Y-%m-%d %H:%M UTC')}:\n"
        f"- Open positions: {open_count} with ${total_exposure:.2f} total exposure\n"
        f"- Yesterday's closed P&L: ${yesterday_pnl:+.2f}\n"
        f"- LLM spend yesterday: ${yesterday_llm_spend:.4f}; today so far: ${today_llm_spend:.4f}\n"
        f"- Signal calibration: {calibration_str}\n"
        f"- LLM errors / circuit-breaker events (last 24h): {circuit_breaker_events}\n\n"
        "Write a single natural-language paragraph (≤150 words) summarizing "
        "system health and anything worth attention."
    )

    response = await llm_client.complete(
        prompt=prompt,
        model=_HAIKU_MODEL,
        query_type="daily_digest",
        system=_DIGEST_SYSTEM,
        max_tokens=300,
    )

    log.info(
        "daily_digest.generated",
        open_positions=open_count,
        yesterday_pnl=round(yesterday_pnl, 4),
        brier_score=round(calibration.brier_score, 4),
        n_samples=calibration.n_samples,
        llm_query_id=response.llm_query_id,
    )

    return response.content
