"""Daily digest and report generation.

The digest has two parts:

1. A **deterministic stat header** assembled in code — run state, open
   positions vs cap, exposure, unrealized P&L, session P&L with win/loss
   counts, drawdown, LLM spend vs cap, calibration vs market baseline,
   signal activity, and service health. Numbers never pass through the LLM,
   so they cannot be garbled.
2. An **LLM analyst take** — Claude Haiku receives the header plus detail
   the reader does not see (per-position P&L, top signals, exit breakdown,
   stale-service errors) and writes 3–5 prioritized bullets flagging only
   what deserves attention. It is explicitly told not to restate header
   numbers.

The digest is plain text (no HTML/markdown) because it is dispatched to
both Telegram and Discord via the alert path.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    from freqpred.runtime.telemetry import RuntimeTelemetry

from freqpred.alerts.command_handlers import _fmt_age_secs, _fmt_usd, _truncate
from freqpred.ingestion.models import FetcherRateLimitRow
from freqpred.llm.audit import get_daily_spend_usd
from freqpred.llm.client import LLMClient
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import MarketRow, PositionRow
from freqpred.metrics.calibration import compute_calibration

log = structlog.get_logger(__name__)

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_DIGEST_SYSTEM = (
    "You are the daily-brief analyst for a prediction-market trading system. "
    "The reader already sees a deterministic stat header above your output; "
    "your job is the analyst take underneath it. Write 3-5 bullets, each on "
    "its own line starting with '• ', each at most 18 words. Surface only "
    "what deserves attention or action: risks, anomalies, win/loss streaks, "
    "stale or erroring services, unusual LLM spend, calibration shifts, "
    "positions deep underwater or near resolution, halted run state. Order "
    "by importance, most important first. Never restate a number the header "
    "already shows unless you are flagging it as a problem. If everything is "
    "routine, write one bullet saying so plus at most one observation. "
    "Plain text only - no markdown, no headers, no preamble."
)


async def generate_daily_digest(
    session: AsyncSession,
    llm_client: LLMClient,
    trading_mode: str = "paper",
    bankroll: float = 0.0,
    model: str = _HAIKU_MODEL,
    llm_daily_cap: float = 0.0,
    max_open_positions: int | None = None,
    telemetry: RuntimeTelemetry | None = None,
) -> str:
    """
    Assembles a deterministic stat header (positions, P&L, drawdown, LLM
    spend, calibration, signal activity, service health) and asks Claude
    Haiku for a short prioritized analyst take underneath it. Logs the LLM
    call via audit. Returns header + analyst bullets as plain text.
    """
    now = datetime.now(UTC)
    yesterday_start = (now - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    yesterday_end = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # --- Run state + drawdown ---
    from freqpred.alerts.run_state import get_drawdown_window, get_run_state  # noqa: PLC0415
    run_state = await get_run_state(session)
    drawdown_reset_at, drawdown_reset_bankroll = await get_drawdown_window(session)

    # --- Open positions + unrealized P&L + excursion metrics ---
    open_result = await session.execute(
        select(
            func.count(PositionRow.id),
            func.coalesce(
                func.sum(PositionRow.contracts * PositionRow.entry_price), 0.0
            ),
        ).where(PositionRow.status == "open", PositionRow.mode == trading_mode)
    )
    open_count, total_exposure = open_result.one()
    open_count = int(open_count)
    total_exposure = float(total_exposure)

    # Unrealized P&L, net exposure, and portfolio MAE/MFE: join to markets for current mid_price.
    # unrealized_pnl mirrors close_position's realized formula — gross P&L net
    # of entry fee — so this number doesn't jump the moment positions close.
    unreal_rows_result = await session.execute(
        select(
            PositionRow.contracts,
            PositionRow.entry_price,
            PositionRow.entry_fee_usd,
            PositionRow.direction,
            PositionRow.mae,
            PositionRow.mfe,
            MarketRow.mid_price,
            PositionRow.market_id,
            MarketRow.question,
        )
        .join(MarketRow, PositionRow.market_id == MarketRow.id)
        .where(PositionRow.status == "open", PositionRow.mode == trading_mode)
    )
    unrealized_pnl = 0.0
    net_exposure = 0.0
    mae_dollar_sum = 0.0
    mfe_dollar_sum = 0.0
    mae_contract_sum = 0
    mfe_contract_sum = 0
    position_details: list[tuple[float, str]] = []  # (unrealized, detail line for LLM)
    for (
        contracts, entry_price, entry_fee_usd, direction, mae, mfe,
        mid_price, market_id, question,
    ) in unreal_rows_result.all():
        fee = entry_fee_usd or 0.0
        if direction == "YES":
            pos_unreal = contracts * (mid_price - entry_price) - fee
            net_exposure += contracts * entry_price
            current_price = mid_price
        else:
            pos_unreal = contracts * ((1.0 - mid_price) - entry_price) - fee
            net_exposure -= contracts * entry_price
            current_price = 1.0 - mid_price
        unrealized_pnl += pos_unreal
        position_details.append((
            pos_unreal,
            f"{direction} {contracts}x {market_id} "
            f"(entry {entry_price * 100:.0f}c, now {current_price * 100:.0f}c, "
            f"unrealized ${pos_unreal:+.2f}): {_truncate(question, 70)}",
        ))
        if mae is not None:
            mae_dollar_sum += mae * contracts
            mae_contract_sum += contracts
        if mfe is not None:
            mfe_dollar_sum += mfe * contracts
            mfe_contract_sum += contracts

    # --- Session P&L: yesterday midnight through now (captures full prior day + intraday) ---
    pnl_result = await session.execute(
        select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
            PositionRow.status == "closed",
            PositionRow.exit_time >= yesterday_start,
            PositionRow.mode == trading_mode,
        )
    )
    session_pnl = float(pnl_result.scalar_one())

    # Exit reason breakdown for the same window (surfaces stop losses, circuit breaker exits, etc.)
    exits_result = await session.execute(
        select(
            PositionRow.exit_reason,
            func.count(PositionRow.id),
            func.coalesce(func.sum(PositionRow.pnl), 0.0),
        )
        .where(
            PositionRow.status == "closed",
            PositionRow.exit_time >= yesterday_start,
            PositionRow.mode == trading_mode,
        )
        .group_by(PositionRow.exit_reason)
    )
    exit_rows = exits_result.all()

    if exit_rows:
        exit_parts = [
            f"{reason or 'resolved'}: {count} trade(s) ${pnl:+.2f}"
            for reason, count, pnl in exit_rows
        ]
        session_exit_str = "; ".join(exit_parts)
    else:
        session_exit_str = "no closed trades in session"

    # Win/loss counts for the same window
    winloss_result = await session.execute(
        select(
            func.coalesce(func.sum(case((PositionRow.pnl > 0, 1), else_=0)), 0),
            func.coalesce(func.sum(case((PositionRow.pnl < 0, 1), else_=0)), 0),
        ).where(
            PositionRow.status == "closed",
            PositionRow.exit_time >= yesterday_start,
            PositionRow.mode == trading_mode,
        )
    )
    session_wins, session_losses = (int(v) for v in winloss_result.one())

    # --- Drawdown: current net vs stored baseline at reset time ---
    # net_value derived from bankroll + all-time closed P&L (unrealized excluded to match ledger)
    all_time_pnl_result = await session.execute(
        select(func.coalesce(func.sum(PositionRow.pnl), 0.0)).where(
            PositionRow.status == "closed", PositionRow.mode == trading_mode
        )
    )
    net_value: float = bankroll + float(all_time_pnl_result.scalar_one())

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
    calibration = await compute_calibration(session, mode=trading_mode)

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
        for service, remaining, _next_skip, tripped_at in backoff_rows:
            tripped_str = tripped_at.strftime("%H:%MZ") if tripped_at else "unknown"
            if remaining > 0:
                parts.append(f"{service} backed off ({remaining} cycles remaining, tripped {tripped_str})")
            else:
                parts.append(f"{service} rate-limited earlier ({tripped_str}, now recovered)")
        fetcher_status = "; ".join(parts)
    else:
        fetcher_status = "all fetchers healthy"

    # --- Signal activity (last 24h) ---
    from freqpred.signal.models import SignalRow  # noqa: PLC0415

    signals_result = await session.execute(
        select(
            SignalRow.direction,
            SignalRow.edge,
            SignalRow.estimated_probability,
            SignalRow.market_mid_at_signal,
            SignalRow.confidence,
            SignalRow.market_id,
        ).where(SignalRow.created_at >= now - timedelta(hours=24))
    )
    signal_rows = signals_result.all()
    n_signals = len(signal_rows)
    actionable = [r for r in signal_rows if (r.direction or "").upper() != "SKIP"]
    top_signals = sorted(actionable, key=lambda r: abs(r.edge), reverse=True)[:3]
    top_signal_lines = [
        f"{r.direction.upper()} {r.market_id} edge {r.edge:+.1%} "
        f"(est {r.estimated_probability:.0%} vs mkt {r.market_mid_at_signal:.0%}, "
        f"conf {r.confidence:.2f})"
        for r in top_signals
    ]

    # --- Service health (runtime freshness telemetry) ---
    health_line: str | None = None
    stale_detail_lines: list[str] = []
    if telemetry is not None:
        from freqpred.runtime.telemetry import list_service_heartbeats  # noqa: PLC0415

        heartbeats = await list_service_heartbeats(session)
        states = telemetry.evaluate_service_states(heartbeats, run_state=run_state)
        ok_count = sum(1 for s in states if s.status == "ok")
        stale = [s for s in states if s.status == "stale"]
        health_line = f"Health {ok_count}/{len(states)} services ok"
        if stale:
            shown = ", ".join(
                f"{s.label} ({_fmt_age_secs(s.age_seconds)})" if s.age_seconds is not None
                else f"{s.label} (never)"
                for s in stale[:3]
            )
            more = f" +{len(stale) - 3} more" if len(stale) > 3 else ""
            health_line += f" · stale: {shown}{more}"
            stale_detail_lines = [
                f"{s.label}: stale for {_fmt_age_secs(s.age_seconds) if s.age_seconds is not None else 'ever'}"
                + (f"; last error: {_truncate(s.last_error_message, 140)}" if s.last_error_message else "")
                for s in stale
            ]

    # --- Drawdown line ---
    if drawdown_reset_bankroll is not None and drawdown_reset_bankroll > 0:
        drawdown = max(0.0, (drawdown_reset_bankroll - net_value) / drawdown_reset_bankroll)
        reset_label = (
            drawdown_reset_at.strftime("%m-%d %H:%MZ")
            if drawdown_reset_at is not None
            else "unknown"
        )
        drawdown_str = f"Drawdown {drawdown:.1%} from ${drawdown_reset_bankroll:,.2f} (baseline {reset_label})"
    elif bankroll > 0:
        drawdown_str = "Drawdown: no baseline set (use /reset_drawdown)"
    else:
        drawdown_str = "Drawdown: unknown (bankroll not provided)"

    # ------------------------------------------------------------------
    # Deterministic stat header — shown verbatim to the reader
    # ------------------------------------------------------------------
    mode_label = trading_mode.upper()
    state_str = run_state if run_state == "running" else f"{run_state} (!)"
    open_cap = f"{open_count}/{max_open_positions}" if max_open_positions else str(open_count)

    session_line = (
        f"Session P&L {_fmt_usd(session_pnl)} ({session_wins}W/{session_losses}L)"
    )
    if bankroll > 0:
        session_line += f" · net value ${net_value:,.2f}"

    llm_line = f"LLM ${today_llm_spend:.2f} today"
    if llm_daily_cap > 0:
        llm_line += f" / ${llm_daily_cap:.2f} cap ({today_llm_spend / llm_daily_cap:.0%})"
    llm_line += f" · ${yesterday_llm_spend:.2f} yesterday"
    if llm_errors:
        llm_line += f" · (!) {llm_errors} LLM errors 24h"

    if calibration.n_samples > 0:
        improvement = calibration.market_brier_score - calibration.brier_score
        better = "better" if improvement >= 0 else "worse"
        calibration_line = (
            f"Brier {calibration.brier_score:.3f} vs market "
            f"{calibration.market_brier_score:.3f} ({improvement:+.3f} {better}, "
            f"n={calibration.n_samples})"
        )
    else:
        calibration_line = "Calibration: no resolved markets yet"

    signals_line = (
        f"Signals 24h: {n_signals} ({len(actionable)} actionable)"
        if n_signals else "Signals 24h: none"
    )

    header_lines = [
        f"[{mode_label}] Daily digest — {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"State {state_str} · open {open_cap} · exposure ${total_exposure:,.2f}"
        f" · unrealized {_fmt_usd(unrealized_pnl)}",
        session_line,
        drawdown_str,
        llm_line,
        f"{calibration_line} · {signals_line}",
    ]
    if health_line:
        header_lines.append(health_line)
    header = "\n".join(header_lines)

    # ------------------------------------------------------------------
    # LLM analyst take — gets the header plus detail the reader can't see
    # ------------------------------------------------------------------
    portfolio_mae_str = (
        f"${mae_dollar_sum:+.2f} ({mae_dollar_sum / mae_contract_sum:+.4f} wtd avg)"
        if mae_contract_sum > 0 else "N/A (no excursion data yet)"
    )
    portfolio_mfe_str = (
        f"${mfe_dollar_sum:+.2f} ({mfe_dollar_sum / mfe_contract_sum:+.4f} wtd avg)"
        if mfe_contract_sum > 0 else "N/A (no excursion data yet)"
    )

    top_positions = sorted(position_details, key=lambda t: abs(t[0]), reverse=True)[:8]
    position_block = (
        "\n".join(f"  - {line}" for _, line in top_positions)
        if top_positions else "  (none)"
    )
    top_signal_block = (
        "\n".join(f"  - {line}" for line in top_signal_lines)
        if top_signal_lines else "  (none actionable)"
    )
    stale_block = (
        "\n".join(f"  - {line}" for line in stale_detail_lines)
        if stale_detail_lines
        else ("  (telemetry unavailable)" if telemetry is None else "  (none stale)")
    )

    run_state_note = (
        f"{run_state} — signal analysis HALTED" if run_state != "running" else "running"
    )
    prompt = (
        f"STAT HEADER (already shown to the reader — do not restate these numbers):\n"
        f"{header}\n\n"
        f"DETAIL (not shown to the reader):\n"
        f"- Run state: {run_state_note}; mode {mode_label} "
        f"({'real money' if trading_mode == 'live' else 'simulated, no real money'})\n"
        f"- Open positions by |unrealized P&L| (worst/best first):\n{position_block}\n"
        f"- Portfolio MAE (worst excursion seen): {portfolio_mae_str}; "
        f"MFE (best seen): {portfolio_mfe_str}\n"
        f"- Session closed trades (yesterday through now): {session_exit_str}\n"
        f"- Top signals by |edge| in last 24h:\n{top_signal_block}\n"
        f"- Stale services:\n{stale_block}\n"
        f"- Fetcher rate-limit status: {fetcher_status}\n\n"
        "Write the analyst take now."
    )

    response = await llm_client.complete(
        prompt=prompt,
        model=model,
        query_type="daily_digest",
        system=_DIGEST_SYSTEM,
        max_tokens=300,
    )

    log.info(
        "daily_digest.generated",
        open_positions=open_count,
        session_pnl=round(session_pnl, 4),
        brier_score=round(calibration.brier_score, 4),
        n_samples=calibration.n_samples,
        n_signals_24h=n_signals,
        llm_query_id=response.llm_query_id,
        trading_mode=trading_mode,
    )

    return f"{header}\n\n{response.content.strip()}"


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
    llm_client: LLMClient,
    alert_dispatcher: object,
    digest_time: str = "07:00",
    digest_timezone: str = "America/New_York",
    trading_mode: str = "paper",
    bankroll: float = 0.0,
    model: str = _HAIKU_MODEL,
    llm_daily_cap: float = 0.0,
    max_open_positions: int | None = None,
    telemetry: RuntimeTelemetry | None = None,
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
                digest = await generate_daily_digest(
                    session,
                    llm_client,
                    trading_mode=trading_mode,
                    bankroll=bankroll,
                    model=model,
                    llm_daily_cap=llm_daily_cap,
                    max_open_positions=max_open_positions,
                    telemetry=telemetry,
                )
            await alert_dispatcher.digest_alert(digest)
            log.info("digest_scheduler.sent")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("digest_scheduler.error")
