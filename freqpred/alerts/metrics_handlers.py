"""Telegram bot command handlers for T29: metrics and performance commands.

Registers /profit /daily /weekly /monthly /stats /balance /budget /calibration
/source_calibration onto a TelegramCommandHandler.

Usage::

    register_metrics_commands(
        cmd_handler=telegram_cmd_handler,
        session_factory=session_factory,
        config=config,
        mode=mode,
    )
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.alerts.command_handlers import _clip, _esc, _fmt_usd
from freqpred.alerts.run_state import get_drawdown_window
from freqpred.metrics.calibration import compute_calibration, compute_source_brier_scores
from freqpred.trading.ledger import get_portfolio_summary

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from freqpred.alerts.telegram_commands import TelegramCommandHandler
    from freqpred.config import Settings
    from freqpred.llm.client import LLMClient
    from freqpred.runtime.telemetry import RuntimeTelemetry

log = structlog.get_logger(__name__)

_TELEGRAM_MAX_LEN = 4096
_TABLE_LIMIT = _TELEGRAM_MAX_LEN - 150  # headroom for <pre> wrapper and headers


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _table_rows_clip(header: str, divider: str, rows: list[str]) -> str:
    """Format a monospace <pre> table, appending '... and N more' if too long.

    Rendered with parse_mode=HTML — <pre> keeps column alignment in Telegram.
    Header and rows are HTML-escaped here; callers pass raw text.
    """
    lines: list[str] = ["<pre>", _esc(header), _esc(divider)]
    char_count = len(header) + len(divider) + 13  # wrapper tags + newlines
    included = 0

    for row in rows:
        tentative = char_count + len(row) + 1
        remaining = len(rows) - included
        if tentative + len(f"... and {remaining} more") + 6 > _TABLE_LIMIT:
            lines.append(f"... and {remaining} more")
            break
        lines.append(_esc(row))
        char_count = tentative
        included += 1

    lines.append("</pre>")
    return "\n".join(lines)


def _parse_int_arg(args: list[str], default: int, cmd: str) -> int | str:
    """Return parsed int from args[0], or an error string, or default."""
    if not args:
        return default
    try:
        return int(args[0])
    except ValueError:
        return f"Usage: /{cmd} [n] — n must be a number, got {args[0]!r}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_metrics_commands(
    cmd_handler: TelegramCommandHandler,
    session_factory: async_sessionmaker[AsyncSession],
    config: Settings,
    mode: str,
    llm_client: LLMClient | None = None,
    telemetry: RuntimeTelemetry | None = None,
) -> None:
    """Register all T29 metrics commands onto *cmd_handler*."""

    # ------------------------------------------------------------------ #
    # /profit [n]  — summary over last n days (default: all time)         #
    # ------------------------------------------------------------------ #

    async def handle_profit(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import PositionRow
        from freqpred.signal.models import SignalRow

        n_days: int | None = None
        if args:
            try:
                n_days = int(args[0])
            except ValueError:
                return f"Usage: /profit [n] — n must be a number of days, got {args[0]!r}"

        cutoff: datetime | None = None
        if n_days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=n_days)

        async with session_factory() as session:
            stmt = select(PositionRow).where(
                PositionRow.status == "closed",
                PositionRow.mode == mode,
            )
            if cutoff:
                stmt = stmt.where(PositionRow.exit_time >= cutoff)
            result = await session.execute(stmt)
            rows = result.scalars().all()

            # Brier score for the same window
            brier_stmt = (
                select(SignalRow.estimated_probability, PositionRow.resolution)
                .join(SignalRow, SignalRow.id == PositionRow.signal_id)
                .where(
                    PositionRow.status == "closed",
                    PositionRow.resolution.is_not(None),
                    PositionRow.mode == mode,
                )
            )
            if cutoff:
                brier_stmt = brier_stmt.where(PositionRow.exit_time >= cutoff)
            brier_result = await session.execute(brier_stmt)
            brier_rows = brier_result.all()

        if not rows:
            period = f"last {n_days} day(s)" if n_days else "all time"
            return f"No closed trades ({period})."

        pnls = [float(r.pnl) for r in rows if r.pnl is not None]
        total_pnl = sum(pnls)
        total_invested = sum(
            float(r.contracts) * float(r.entry_price)
            for r in rows
            if r.contracts is not None and r.entry_price is not None
        )
        pnl_pct = total_pnl / total_invested if total_invested > 0 else 0.0
        win_count = sum(1 for p in pnls if p > 0)
        trade_count = len(rows)
        win_rate = win_count / trade_count if trade_count else 0.0

        # Avg hold duration
        durations = []
        for r in rows:
            if r.entry_time and r.exit_time:
                et = r.entry_time if r.entry_time.tzinfo else r.entry_time.replace(tzinfo=UTC)
                xt = r.exit_time if r.exit_time.tzinfo else r.exit_time.replace(tzinfo=UTC)
                durations.append((xt - et).total_seconds())
        avg_hold = "N/A"
        if durations:
            avg_secs = int(sum(durations) / len(durations))
            avg_hold = f"{avg_secs // 3600}h {(avg_secs % 3600) // 60}m"

        best = max(pnls) if pnls else None
        worst = min(pnls) if pnls else None

        if brier_rows:
            brier_score = sum(
                (float(p) - float(y)) ** 2 for p, y in brier_rows
            ) / len(brier_rows)
            brier_str = f"{brier_score:.3f} ({len(brier_rows)} sample(s))"
        else:
            brier_str = "N/A"

        period_label = f"last {n_days} day(s)" if n_days else "all time"
        lines = [
            f"<b>P&amp;L summary</b> ({period_label})",
            f"Trades: {trade_count} ({win_count} wins, {win_rate:.0%} win rate)",
            f"Total P&L: {_fmt_usd(total_pnl)} ({pnl_pct:+.1%} on ${total_invested:,.2f} invested)",
            f"Best trade: {_fmt_usd(best)}" if best is not None else "Best trade: N/A",
            f"Worst trade: {_fmt_usd(worst)}" if worst is not None else "Worst trade: N/A",
            f"Avg hold time: {avg_hold}",
            f"Brier score: {brier_str}",
        ]
        return _clip("\n".join(lines))

    # ------------------------------------------------------------------ #
    # /daily [n]  — table: date | trades | P&L $ | P&L %  (default 7)    #
    # ------------------------------------------------------------------ #

    async def handle_daily(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import PositionRow

        n = _parse_int_arg(args, 7, "daily")
        if isinstance(n, str):
            return n

        cutoff = datetime.now(UTC) - timedelta(days=n)

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow).where(
                    PositionRow.status == "closed",
                    PositionRow.exit_time >= cutoff,
                    PositionRow.mode == mode,
                )
            )
            rows = result.scalars().all()

        # Group by UTC date
        by_date: dict[str, list] = {}
        for r in rows:
            if r.exit_time is None:
                continue
            xt = r.exit_time if r.exit_time.tzinfo else r.exit_time.replace(tzinfo=UTC)
            day = xt.strftime("%Y-%m-%d")
            by_date.setdefault(day, []).append(r)

        if not by_date:
            return f"No closed trades in the last {n} day(s)."

        # Build sorted table rows (newest first)
        header = f"{'Date':<12} {'Trades':>6} {'P&L $':>9} {'P&L %':>7}"
        divider = "-" * len(header)
        table_rows: list[str] = []
        for day in sorted(by_date.keys(), reverse=True):
            day_rows = by_date[day]
            pnl = sum(float(r.pnl) for r in day_rows if r.pnl is not None)
            invested = sum(
                float(r.contracts) * float(r.entry_price)
                for r in day_rows
                if r.contracts and r.entry_price
            )
            pct = pnl / invested if invested > 0 else 0.0
            table_rows.append(
                f"{day:<12} {len(day_rows):>6} {pnl:>+9.2f} {pct:>+7.1%}"
            )
        return _table_rows_clip(header, divider, table_rows)

    # ------------------------------------------------------------------ #
    # /weekly [n]  — table: week | trades | P&L $ | P&L %  (default 8)   #
    # ------------------------------------------------------------------ #

    async def handle_weekly(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import PositionRow

        n = _parse_int_arg(args, 8, "weekly")
        if isinstance(n, str):
            return n

        cutoff = datetime.now(UTC) - timedelta(weeks=n)

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow).where(
                    PositionRow.status == "closed",
                    PositionRow.exit_time >= cutoff,
                    PositionRow.mode == mode,
                )
            )
            rows = result.scalars().all()

        by_week: dict[str, list] = {}
        for r in rows:
            if r.exit_time is None:
                continue
            xt = r.exit_time if r.exit_time.tzinfo else r.exit_time.replace(tzinfo=UTC)
            # ISO week start (Monday)
            week_start = xt - timedelta(days=xt.weekday())
            week_key = week_start.strftime("%Y-%m-%d")
            by_week.setdefault(week_key, []).append(r)

        if not by_week:
            return f"No closed trades in the last {n} week(s)."

        header = f"{'Week start':<12} {'Trades':>6} {'P&L $':>9} {'P&L %':>7}"
        divider = "-" * len(header)
        table_rows: list[str] = []
        for week in sorted(by_week.keys(), reverse=True):
            week_rows = by_week[week]
            pnl = sum(float(r.pnl) for r in week_rows if r.pnl is not None)
            invested = sum(
                float(r.contracts) * float(r.entry_price)
                for r in week_rows
                if r.contracts and r.entry_price
            )
            pct = pnl / invested if invested > 0 else 0.0
            table_rows.append(
                f"{week:<12} {len(week_rows):>6} {pnl:>+9.2f} {pct:>+7.1%}"
            )
        return _table_rows_clip(header, divider, table_rows)

    # ------------------------------------------------------------------ #
    # /monthly [n]  — table: month | trades | P&L $ | P&L %  (default 6) #
    # ------------------------------------------------------------------ #

    async def handle_monthly(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import PositionRow

        n = _parse_int_arg(args, 6, "monthly")
        if isinstance(n, str):
            return n

        cutoff = datetime.now(UTC) - timedelta(days=n * 30)

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow).where(
                    PositionRow.status == "closed",
                    PositionRow.exit_time >= cutoff,
                    PositionRow.mode == mode,
                )
            )
            rows = result.scalars().all()

        by_month: dict[str, list] = {}
        for r in rows:
            if r.exit_time is None:
                continue
            xt = r.exit_time if r.exit_time.tzinfo else r.exit_time.replace(tzinfo=UTC)
            month_key = xt.strftime("%Y-%m")
            by_month.setdefault(month_key, []).append(r)

        if not by_month:
            return f"No closed trades in the last {n} month(s)."

        header = f"{'Month':<8} {'Trades':>6} {'P&L $':>9} {'P&L %':>7}"
        divider = "-" * len(header)
        table_rows: list[str] = []
        for month in sorted(by_month.keys(), reverse=True):
            month_rows = by_month[month]
            pnl = sum(float(r.pnl) for r in month_rows if r.pnl is not None)
            invested = sum(
                float(r.contracts) * float(r.entry_price)
                for r in month_rows
                if r.contracts and r.entry_price
            )
            pct = pnl / invested if invested > 0 else 0.0
            table_rows.append(
                f"{month:<8} {len(month_rows):>6} {pnl:>+9.2f} {pct:>+7.1%}"
            )
        return _table_rows_clip(header, divider, table_rows)

    # ------------------------------------------------------------------ #
    # /stats  — all-time aggregate stats grouped by exit reason           #
    # ------------------------------------------------------------------ #

    async def handle_stats(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import PositionRow

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow).where(
                    PositionRow.status == "closed",
                    PositionRow.mode == mode,
                )
            )
            rows = result.scalars().all()

        if not rows:
            return "No closed trades yet."

        pnls = [float(r.pnl) for r in rows if r.pnl is not None]
        win_count = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        total_invested = sum(
            float(r.contracts) * float(r.entry_price)
            for r in rows
            if r.contracts and r.entry_price
        )
        pnl_pct = total_pnl / total_invested if total_invested > 0 else 0.0
        win_rate = win_count / len(rows)

        best = max(pnls) if pnls else None
        worst = min(pnls) if pnls else None

        durations = []
        for r in rows:
            if r.entry_time and r.exit_time:
                et = r.entry_time if r.entry_time.tzinfo else r.entry_time.replace(tzinfo=UTC)
                xt = r.exit_time if r.exit_time.tzinfo else r.exit_time.replace(tzinfo=UTC)
                durations.append((xt - et).total_seconds())
        avg_hold = "N/A"
        if durations:
            avg_secs = int(sum(durations) / len(durations))
            avg_hold = f"{avg_secs // 3600}h {(avg_secs % 3600) // 60}m"

        # Breakdown by exit reason
        reason_counts: dict[str, tuple[int, int, float]] = {}  # reason -> (total, wins, pnl)
        for r in rows:
            reason = r.exit_reason or "resolved"
            pnl_val = float(r.pnl) if r.pnl is not None else 0.0
            is_win = 1 if pnl_val > 0 else 0
            existing = reason_counts.get(reason, (0, 0, 0.0))
            reason_counts[reason] = (
                existing[0] + 1,
                existing[1] + is_win,
                existing[2] + pnl_val,
            )

        lines = [
            "<b>All-time stats</b>",
            f"Total trades: {len(rows)} ({win_count} wins, {win_rate:.0%} win rate)",
            f"Total P&L: {_fmt_usd(total_pnl)} ({pnl_pct:+.1%} on ${total_invested:,.2f} invested)",
            f"Best trade: {_fmt_usd(best)}" if best is not None else "Best trade: N/A",
            f"Worst trade: {_fmt_usd(worst)}" if worst is not None else "Worst trade: N/A",
            f"Avg hold time: {avg_hold}",
            "",
            "<b>By exit reason</b>",
        ]
        reason_header = f"{'Reason':<22} {'N':>3} {'Wins':>4} {'P&L $':>9}"
        reason_rows = [
            f"{reason[:22]:<22} {count:>3} {wins:>4} {reason_pnl:>+9.2f}"
            for reason, (count, wins, reason_pnl) in sorted(reason_counts.items())
        ]
        table_block = _table_rows_clip(reason_header, "-" * len(reason_header), reason_rows)
        return _clip("\n".join(lines) + "\n" + table_block)

    # ------------------------------------------------------------------ #
    # /balance  — portfolio snapshot                                       #
    # ------------------------------------------------------------------ #

    async def handle_balance(chat_id: int, args: list[str]) -> str:
        bankroll = config.trading.bankroll_usd

        async with session_factory() as session:
            summary = await get_portfolio_summary(session, mode=mode)
            drawdown_reset_at, drawdown_reset_bankroll = await get_drawdown_window(
                session, mode
            )

        all_time_pnl = summary["all_time_pnl_usd"]
        net_value = bankroll + all_time_pnl
        exposure = summary["total_exposure_usd"]
        exposure_pct = exposure / bankroll if bankroll > 0 else 0.0
        daily_pnl = summary["daily_pnl_usd"]
        open_count = summary["open_count"]
        unrealized_pnl = summary["unrealized_pnl_usd"]
        net_exposure = summary["net_exposure_usd"]
        portfolio_mae_usd = summary["portfolio_mae_usd"]
        portfolio_mfe_usd = summary["portfolio_mfe_usd"]
        portfolio_mae_pct = summary["portfolio_mae_pct"]
        portfolio_mfe_pct = summary["portfolio_mfe_pct"]

        def _excursion_line(label: str, usd: float | None, pct: float | None) -> str:
            if usd is None:
                return f"{label}: —"
            return f"{label}: {_fmt_usd(usd)} ({pct:+.4f} wtd avg)"

        if drawdown_reset_bankroll is not None and drawdown_reset_bankroll > 0:
            drawdown = max(0.0, (drawdown_reset_bankroll - net_value) / drawdown_reset_bankroll)
            reset_ts = (
                drawdown_reset_at.strftime("%Y-%m-%d %H:%MZ")
                if drawdown_reset_at is not None
                else "unknown"
            )
            drawdown_line = (
                f"Drawdown: {drawdown:.1%} from ${drawdown_reset_bankroll:,.2f}"
                f" (reset {reset_ts})"
            )
        else:
            drawdown_line = "Drawdown: no baseline set (use /reset_drawdown)"
        lines = [
            f"<b>Balance snapshot</b> ({mode} mode)",
            f"Bankroll: ${bankroll:,.2f}",
            f"All-time P&L: {_fmt_usd(all_time_pnl)}",
            f"Net value: ${net_value:,.2f}",
            f"Gross exposure: ${exposure:,.2f} ({exposure_pct:.1%} of bankroll)",
            f"Net exposure: {_fmt_usd(net_exposure)}",
            f"Unrealized P&L: {_fmt_usd(unrealized_pnl)}",
            f"Today's P&L: {_fmt_usd(daily_pnl)}",
            f"Open positions: {open_count}",
            _excursion_line("Portfolio MAE", portfolio_mae_usd, portfolio_mae_pct),
            _excursion_line("Portfolio MFE", portfolio_mfe_usd, portfolio_mfe_pct),
            drawdown_line,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # /budget  — LLM cost breakdown                                        #
    # ------------------------------------------------------------------ #

    async def handle_budget(chat_id: int, args: list[str]) -> str:
        from freqpred.llm.models import LLMQueryRow

        cap = config.risk.max_daily_llm_spend_usd
        now = datetime.now(UTC)
        today = now.date()
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        async with session_factory() as session:
            # Today's spend by query_type
            today_result = await session.execute(
                select(
                    LLMQueryRow.query_type,
                    func.coalesce(func.sum(LLMQueryRow.cost_usd), 0.0),
                )
                .where(func.date(LLMQueryRow.timestamp) == today)
                .group_by(LLMQueryRow.query_type)
            )
            today_by_type = {qt: float(cost) for qt, cost in today_result.all()}

            week_result = await session.execute(
                select(func.coalesce(func.sum(LLMQueryRow.cost_usd), 0.0)).where(
                    LLMQueryRow.timestamp >= week_start
                )
            )
            week_spend = float(week_result.scalar_one())

            month_result = await session.execute(
                select(func.coalesce(func.sum(LLMQueryRow.cost_usd), 0.0)).where(
                    LLMQueryRow.timestamp >= month_start
                )
            )
            month_spend = float(month_result.scalar_one())

            alltime_result = await session.execute(
                select(func.coalesce(func.sum(LLMQueryRow.cost_usd), 0.0))
            )
            alltime_spend = float(alltime_result.scalar_one())

        today_total = sum(today_by_type.values())
        cap_pct = today_total / cap if cap > 0 else 0.0
        resets_at = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        resets_in = resets_at - now
        resets_in_h = int(resets_in.total_seconds()) // 3600
        resets_in_m = (int(resets_in.total_seconds()) % 3600) // 60

        lines = [
            "<b>LLM budget</b>",
            f"Today: ${today_total:.2f} / ${cap:.2f} cap ({cap_pct:.0%})"
            f" · resets in {resets_in_h}h {resets_in_m}m",
        ]

        if today_by_type:
            type_header = f"{'Query type':<25} {'Today $':>8}"
            type_rows = [
                f"{qt[:25]:<25} {cost:>8.4f}" for qt, cost in sorted(today_by_type.items())
            ]
            lines.append(_table_rows_clip(type_header, "-" * len(type_header), type_rows))

        lines += [
            f"This week: ${week_spend:.2f}",
            f"This month: ${month_spend:.2f}",
            f"All-time: ${alltime_spend:.2f}",
        ]
        return _clip("\n".join(lines))

    # ------------------------------------------------------------------ #
    # /calibration  — Brier score + per-bucket breakdown                   #
    # ------------------------------------------------------------------ #

    async def handle_calibration(chat_id: int, args: list[str]) -> str:
        lookback_days: int | None = None
        if args:
            try:
                lookback_days = int(args[0])
            except ValueError:
                return f"Invalid days argument: {args[0]!r} — usage: /calibration [days]"

        async with session_factory() as session:
            report = await compute_calibration(session, mode=mode, lookback_days=lookback_days)

        if report.n_samples == 0:
            return "No resolved positions yet — calibration unavailable."

        improvement = report.market_brier_score - report.brier_score
        direction = "better" if improvement > 0 else "worse"

        period = f"last {lookback_days}d" if lookback_days is not None else "all-time"
        header_lines = [
            f"<b>Calibration</b> ({period}, {report.n_samples} samples)",
            f"Brier score: {report.brier_score:.3f} vs market baseline {report.market_brier_score:.3f}",
            f"Improvement: {improvement:+.3f} ({direction} than market)",
        ]

        non_empty = [b for b in report.buckets if b.count > 0]
        if not non_empty:
            return "\n".join(header_lines)

        tbl_header = f"{'Bucket':<11} {'Count':>5} {'MeanEst':>8} {'ActualRate':>11}"
        divider = "-" * len(tbl_header)
        tbl_rows = [
            f"{b.lower:.2f}–{b.upper:.2f}   {b.count:>5} "
            f"{b.mean_estimated_prob:>8.3f} {b.actual_resolution_rate:>11.3f}"
            for b in non_empty
        ]

        table_block = _table_rows_clip(tbl_header, divider, tbl_rows)
        return _clip("\n".join(header_lines) + "\n" + table_block)

    # ------------------------------------------------------------------ #
    # /source_calibration [days] [min_docs]                               #
    # ------------------------------------------------------------------ #

    async def handle_source_calibration(chat_id: int, args: list[str]) -> str:
        lookback_days: int | None = None
        min_docs = 50
        if args:
            try:
                lookback_days = int(args[0])
            except ValueError:
                return f"Invalid days argument: {args[0]!r} — usage: /source_calibration [days] [min_docs]"
        if len(args) >= 2:
            try:
                min_docs = int(args[1])
            except ValueError:
                return f"Invalid min_docs argument: {args[1]!r} — usage: /source_calibration [days] [min_docs]"

        async with session_factory() as session:
            scores = await compute_source_brier_scores(
                session, lookback_days=lookback_days, min_docs=min_docs
            )
            calibration = await compute_calibration(session, lookback_days=lookback_days)

        period = f"last {lookback_days}d" if lookback_days is not None else "all-time"
        min_docs_label = f", min {min_docs} uses" if min_docs > 0 else ""
        if not scores:
            return f"No qualifying sources ({period}{min_docs_label}). Try /source_calibration {lookback_days or ''} 0"

        overall = calibration.brier_score
        header_line = f"<b>Source Brier</b> ({period}{min_docs_label}) — overall: {overall:.4f}"
        tbl_header = f"{'Source':<22} {'Brier':>6} {'Delta':>7} {'Uses':>6}"
        divider = "-" * len(tbl_header)
        tbl_rows = [
            f"{s.source_name:<22} {s.weighted_brier_score:>6.4f} "
            f"{s.weighted_brier_score - overall:>+7.4f} {s.total_doc_appearances:>6}"
            for s in scores
        ]

        table_block = _table_rows_clip(tbl_header, divider, tbl_rows)
        return _clip(header_line + "\n" + table_block)

    # ------------------------------------------------------------------ #
    # /digest  — on-demand daily digest via Claude Haiku                  #
    # ------------------------------------------------------------------ #

    async def handle_digest(chat_id: int, args: list[str]) -> str:
        from freqpred.metrics.reporting import generate_daily_digest

        if llm_client is None:
            return "Digest unavailable: LLM client not configured."

        async with session_factory() as session:
            digest = await generate_daily_digest(
                session,
                llm_client,
                trading_mode=mode,
                bankroll=config.trading.bankroll_usd,
                model=config.anthropic.cheap_model,
                llm_daily_cap=config.risk.max_daily_llm_spend_usd,
                max_open_positions=config.risk.max_open_positions,
                telemetry=telemetry,
            )
        # LLM output is free text — escape so it can't break HTML parsing.
        return _clip(_esc(digest))

    # ------------------------------------------------------------------ #
    # Register all handlers
    # ------------------------------------------------------------------ #

    cmd_handler.register(
        "profit", handle_profit,
        description="[days] — P&L summary (default all time)", category="Performance")
    cmd_handler.register(
        "daily", handle_daily,
        description="[n] — per-day P&L table", category="Performance")
    cmd_handler.register(
        "weekly", handle_weekly,
        description="[n] — per-week P&L table", category="Performance")
    cmd_handler.register(
        "monthly", handle_monthly,
        description="[n] — per-month P&L table", category="Performance")
    cmd_handler.register(
        "stats", handle_stats,
        description="All-time stats by exit reason", category="Performance")
    cmd_handler.register(
        "balance", handle_balance,
        description="Bankroll, exposure, and P&L snapshot", category="Performance")
    cmd_handler.register(
        "budget", handle_budget,
        description="LLM spend vs daily cap", category="Diagnostics")
    cmd_handler.register(
        "calibration", handle_calibration,
        description="[days] — Brier score vs market", category="Performance")
    cmd_handler.register(
        "source_calibration", handle_source_calibration,
        description="[days] [min_docs] — Brier score per source", category="Performance")
    cmd_handler.register(
        "digest", handle_digest,
        description="LLM-written summary of the day", category="Performance")
