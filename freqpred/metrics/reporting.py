"""Daily digest and report generation."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.ingestion.models import FetcherRateLimitRow
from freqpred.llm.audit import get_daily_spend_usd
from freqpred.llm.client import LLMClient
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import MarketRow, PositionRow
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
    trading_mode: str = "paper",
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

    # --- Open positions + unrealized P&L + excursion metrics ---
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

    # Unrealized P&L, net exposure, and portfolio MAE/MFE: join to markets for current mid_price
    unreal_rows_result = await session.execute(
        select(
            PositionRow.contracts,
            PositionRow.entry_price,
            PositionRow.direction,
            PositionRow.mae,
            PositionRow.mfe,
            MarketRow.mid_price,
        )
        .join(MarketRow, PositionRow.market_id == MarketRow.id)
        .where(PositionRow.status == "open")
    )
    unrealized_pnl = 0.0
    net_exposure = 0.0
    mae_dollar_sum = 0.0
    mfe_dollar_sum = 0.0
    mae_contract_sum = 0
    mfe_contract_sum = 0
    for contracts, entry_price, direction, mae, mfe, mid_price in unreal_rows_result.all():
        if direction == "YES":
            unrealized_pnl += contracts * (mid_price - entry_price)
            net_exposure += contracts * entry_price
        else:
            unrealized_pnl += contracts * ((1.0 - mid_price) - entry_price)
            net_exposure -= contracts * entry_price
        if mae is not None:
            mae_dollar_sum += mae * contracts
            mae_contract_sum += contracts
        if mfe is not None:
            mfe_dollar_sum += mfe * contracts
            mfe_contract_sum += contracts

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

    # --- LLM errors (last 24h) ---
    llm_error_result = await session.execute(
        select(func.count(LLMQueryRow.id)).where(
            LLMQueryRow.success.is_(False),
            LLMQueryRow.timestamp >= now - timedelta(hours=24),
        )
    )
    llm_errors = int(llm_error_result.scalar_one())

    # --- Fetcher backoff state (from fetcher_rate_limits table) ---
    backoff_result = await session.execute(
        select(
            FetcherRateLimitRow.service,
            FetcherRateLimitRow.skip_cycles_remaining,
            FetcherRateLimitRow.skip_cycles_next,
            FetcherRateLimitRow.tripped_at,
        ).where(FetcherRateLimitRow.tripped_at >= now - timedelta(hours=24))
    )
    backoff_rows = backoff_result.all()

    if backoff_rows:
        parts = []
        for service, remaining, next_skip, tripped_at in backoff_rows:
            tripped_str = tripped_at.strftime("%H:%MZ") if tripped_at else "unknown"
            if remaining > 0:
                parts.append(f"{service} backed off ({remaining} cycles remaining, tripped {tripped_str})")
            else:
                parts.append(f"{service} rate-limited earlier ({tripped_str}, now recovered)")
        fetcher_status = "; ".join(parts)
    else:
        fetcher_status = "all fetchers healthy"

    # --- Build prompt ---
    calibration_str = (
        f"Brier score {calibration.brier_score:.3f} over {calibration.n_samples} resolved markets"
        if calibration.n_samples > 0
        else "no resolved markets yet (calibration unavailable)"
    )
    portfolio_mae_str = (
        f"${mae_dollar_sum:+.2f} ({mae_dollar_sum / mae_contract_sum:+.4f} wtd avg)"
        if mae_contract_sum > 0 else "N/A (no excursion data yet)"
    )
    portfolio_mfe_str = (
        f"${mfe_dollar_sum:+.2f} ({mfe_dollar_sum / mfe_contract_sum:+.4f} wtd avg)"
        if mfe_contract_sum > 0 else "N/A (no excursion data yet)"
    )

    mode_label = trading_mode.upper()
    prompt = (
        f"Daily digest as of {now.strftime('%Y-%m-%d %H:%M UTC')} [mode: {mode_label}]:\n"
        f"- Trading mode: {mode_label} ({'real money' if trading_mode == 'live' else 'simulated / no real money'})\n"
        f"- Open positions: {open_count} with ${total_exposure:.2f} gross exposure, "
        f"${net_exposure:+.2f} net exposure, ${unrealized_pnl:+.2f} unrealized P&L\n"
        f"- Portfolio MAE (worst excursion seen): {portfolio_mae_str}\n"
        f"- Portfolio MFE (best excursion seen): {portfolio_mfe_str}\n"
        f"- Yesterday's closed P&L: ${yesterday_pnl:+.2f}\n"
        f"- LLM spend yesterday: ${yesterday_llm_spend:.4f}; today so far: ${today_llm_spend:.4f}\n"
        f"- LLM errors (last 24h): {llm_errors}\n"
        f"- Signal calibration: {calibration_str}\n"
        f"- Fetcher status: {fetcher_status}\n\n"
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
        trading_mode=trading_mode,
    )

    mode_banner = f"[{mode_label} MODE]\n"
    return mode_banner + response.content


def _seconds_until_next(time_str: str, tz: ZoneInfo) -> float:
    """Return seconds until the next occurrence of HH:MM in the given timezone."""
    hour, minute = (int(p) for p in time_str.split(":"))
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def run_digest_scheduler(
    session_factory: async_sessionmaker[AsyncSession],
    llm_client: "LLMClient",
    alert_dispatcher: object,
    digest_time: str = "07:00",
    digest_timezone: str = "America/New_York",
    trading_mode: str = "paper",
) -> None:
    """Background task: generate and send the daily digest at the configured time.

    Fires once per day at *digest_time* in *digest_timezone*. Designed to run
    as an asyncio task alongside the signal loop.
    """
    tz = ZoneInfo(digest_timezone)
    log.info(
        "digest_scheduler.started",
        digest_time=digest_time,
        digest_timezone=digest_timezone,
    )

    while True:
        wait = _seconds_until_next(digest_time, tz)
        log.debug("digest_scheduler.sleeping", seconds=round(wait))
        await asyncio.sleep(wait)

        try:
            async with session_factory() as session:
                digest = await generate_daily_digest(session, llm_client, trading_mode=trading_mode)
            await alert_dispatcher.digest_alert(digest)
            log.info("digest_scheduler.sent")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("digest_scheduler.error")
