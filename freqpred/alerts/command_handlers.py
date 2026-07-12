"""Telegram bot command handlers for T28: system + status commands.

Registers all /start /pause /stop /show_config /logs /version
/status /trades /signals /health handlers onto a TelegramCommandHandler.

Replies use Telegram HTML markup (see TelegramCommandHandler._send_reply):
dynamic strings must be escaped with _esc() before interpolation.

Usage::

    log_buffer = LogBuffer()
    register_system_commands(
        cmd_handler=telegram_cmd_handler,
        session_factory=session_factory,
        config=config,
        mode=mode,
        strategy_name=strategy_name,
        log_buffer=log_buffer,
    )
"""
from __future__ import annotations

import asyncio
import collections
import html
import importlib.metadata
import logging
import os
import signal
import subprocess
import uuid as _uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.alerts.run_state import get_run_state, reset_drawdown, set_cb_state, set_run_state

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from freqpred.alerts.telegram_commands import TelegramCommandHandler
    from freqpred.config import Settings
    from freqpred.runtime.telemetry import RuntimeTelemetry

log = structlog.get_logger(__name__)

_TELEGRAM_MAX_LEN = 4096


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------


class LogBuffer:
    """Bounded in-memory ring buffer that captures stdlib log lines."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._buf: collections.deque[tuple[str, str]] = collections.deque(maxlen=maxlen)

    def append(self, logger_name: str, line: str) -> None:
        self._buf.append((logger_name, line))

    def last(self, n: int, filter: str | None = None) -> list[str]:
        """Return the last *n* lines, optionally filtered by logger prefix.

        *filter* matches if the logger name equals it or starts with ``filter.``.
        E.g. filter="scheduler" matches "freqpred.ingestion.scheduler" and
        "freqpred.ingestion.scheduler.market_cycle_complete".
        """
        entries = list(self._buf)
        if filter is not None:
            entries = [
                (name, line) for name, line in entries
                if name == filter
                or name.endswith(f".{filter}")
                or f".{filter}." in name
                or name.startswith(f"{filter}.")
            ]
        lines = [line for _, line in entries]
        return lines[-n:] if n < len(lines) else lines


class _LogBufferHandler(logging.Handler):
    """Logging handler that writes formatted records to a LogBuffer."""

    def __init__(self, buf: LogBuffer) -> None:
        super().__init__()
        self._buf = buf

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buf.append(record.name, self.format(record))
        except Exception:
            pass


def install_log_buffer(buf: LogBuffer) -> None:
    """Attach a _LogBufferHandler to the root logger."""
    handler = _LogBufferHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)


# ---------------------------------------------------------------------------
# Formatting helpers (shared by metrics_handlers)
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape a dynamic string for a parse_mode=HTML reply."""
    return html.escape(str(text), quote=False)


def _truncate(text: str, n: int) -> str:
    """Truncate to at most *n* chars, cutting at a word boundary."""
    if len(text) <= n:
        return text
    cut = text[:n]
    space = cut.rfind(" ")
    # Only back up to the word boundary when it doesn't eat most of the text.
    if space > n * 2 // 3:
        cut = cut[:space]
    return cut.rstrip() + "…"


def _clip(text: str) -> str:
    """Truncate to Telegram's 4096-char message limit at a line boundary."""
    if len(text) <= _TELEGRAM_MAX_LEN:
        return text
    suffix = "\n…[truncated]"
    cut = text[: _TELEGRAM_MAX_LEN - len(suffix)]
    nl = cut.rfind("\n")
    if nl > _TELEGRAM_MAX_LEN // 2:
        cut = cut[:nl]
    return cut + suffix


def _fmt_usd(value: float) -> str:
    """Format a signed dollar amount: +$1.20 / -$0.35."""
    sign = "-" if value < 0 else "+"
    return f"{sign}${abs(value):,.2f}"


def _fmt_price(price: float) -> str:
    """Format a 0–1 contract price in cents: 43¢, or 43.5¢ if fractional."""
    cents = price * 100
    if abs(cents - round(cents)) < 0.05:
        return f"{round(cents):d}¢"
    return f"{cents:.1f}¢"


def _fmt_age_secs(secs: int) -> str:
    """Format an age in seconds: 12s / 5m / 2h 15m / 3d 4h."""
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m"
    hours, mins = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h {mins}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def _ago(dt: datetime, now: datetime | None = None) -> str:
    """Format a past datetime as an age string ('2h 15m'). Naive dt = UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return _fmt_age_secs(max(0, int((current - dt).total_seconds())))


def _unrealized_pnl(direction: str, contracts: int, entry_price: float, mid: float) -> float:
    """Unrealized P&L in dollars. NO positions are valued at (1 - mid)."""
    current = mid if direction.upper() == "YES" else 1.0 - mid
    return contracts * (current - entry_price)


_STATE_ICONS = {"running": "▶️", "paused": "⏸", "stopped": "⛔"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_system_commands(
    cmd_handler: TelegramCommandHandler,
    session_factory: async_sessionmaker[AsyncSession],
    config: Settings,
    mode: str,
    strategy_name: str,
    log_buffer: LogBuffer | None = None,
    telemetry: RuntimeTelemetry | None = None,
) -> None:
    """Register all T28 commands onto *cmd_handler*."""

    # ------------------------------------------------------------------ #
    # /start — set state to running                                        #
    # ------------------------------------------------------------------ #

    async def handle_start(chat_id: int, args: list[str]) -> str:
        async with session_factory() as session:
            # mode-scoped: /start only acknowledges the daily-loss breaker for
            # the mode this process is running in, never the other mode's.
            await set_run_state(session, "running", mode=mode)
        log.info("telegram.start", chat_id=chat_id)
        return "Run loop state set to running."

    # ------------------------------------------------------------------ #
    # /pause — no new positions; existing positions still managed          #
    # ------------------------------------------------------------------ #

    async def handle_pause(chat_id: int, args: list[str]) -> str:
        async with session_factory() as session:
            await set_run_state(session, "paused")
        log.info("telegram.pause", chat_id=chat_id)
        return "Run loop paused. No new positions will be opened. Existing positions still managed."

    # ------------------------------------------------------------------ #
    # /stop — halt signal loop entirely                                    #
    # ------------------------------------------------------------------ #

    async def handle_stop(chat_id: int, args: list[str]) -> str:
        async with session_factory() as session:
            await set_run_state(session, "stopped")
        log.info("telegram.stop", chat_id=chat_id)
        return "Run loop stopped. Signal analysis halted. Use /start to resume."

    # ------------------------------------------------------------------ #
    # /shutdown — send SIGTERM to the process (graceful shutdown)          #
    # ------------------------------------------------------------------ #

    _SHUTDOWN_TIMEOUT_SECS = 30

    async def handle_shutdown(chat_id: int, args: list[str]) -> str | None:
        nonce = _uuid.uuid4().hex[:12]
        confirm_data = f"confirm_shutdown:{nonce}"
        cancel_data = f"cancel_shutdown:{nonce}"

        async def on_confirm(cb_chat_id: int, data: str, cb_query_id: str) -> str | None:
            cmd_handler.unregister_callback(confirm_data)
            cmd_handler.unregister_callback(cancel_data)
            log.info("telegram.shutdown", chat_id=cb_chat_id)
            os.kill(os.getpid(), signal.SIGTERM)
            return "Shutting down. Goodbye."

        async def on_cancel(cb_chat_id: int, data: str, cb_query_id: str) -> str | None:
            cmd_handler.unregister_callback(confirm_data)
            cmd_handler.unregister_callback(cancel_data)
            return "Shutdown cancelled."

        cmd_handler.register_callback(confirm_data, on_confirm)
        cmd_handler.register_callback(cancel_data, on_cancel)

        await cmd_handler.send_inline_keyboard(
            chat_id,
            "Are you sure you want to shut down freqpred?",
            [
                [
                    {"text": "Confirm Shutdown", "callback_data": confirm_data},
                    {"text": "Cancel", "callback_data": cancel_data},
                ]
            ],
        )

        async def _timeout_notify() -> None:
            await asyncio.sleep(_SHUTDOWN_TIMEOUT_SECS)
            if confirm_data in cmd_handler._callback_handlers:
                cmd_handler.unregister_callback(confirm_data)
                cmd_handler.unregister_callback(cancel_data)
                await cmd_handler._send_reply(
                    chat_id,
                    "Confirmation timed out after 30 s. Shutdown cancelled.",
                )

        asyncio.create_task(_timeout_notify())
        return None

    # ------------------------------------------------------------------ #
    # /reset_drawdown — reset drawdown circuit breaker start date          #
    # ------------------------------------------------------------------ #

    async def handle_reset_drawdown(chat_id: int, args: list[str]) -> str:
        if mode not in ("paper", "live"):
            return f"No drawdown baseline in {mode} mode — nothing to reset."
        from freqpred.trading import ledger as _ledger  # noqa: PLC0415
        async with session_factory() as session:
            net_bankroll = await _ledger.get_net_bankroll(
                session, config.trading.bankroll_usd, mode=mode
            )
            reset_at = await reset_drawdown(session, mode, net_bankroll)
            await set_cb_state(session, active=False, reason=None)
        log.info(
            "telegram.reset_drawdown",
            chat_id=chat_id, mode=mode,
            reset_at=reset_at.isoformat(), net_bankroll=net_bankroll,
        )
        return (
            f"Drawdown reset ({mode}). Baseline set to ${net_bankroll:,.2f} "
            f"at {reset_at.strftime('%Y-%m-%d %H:%M UTC')}."
        )

    # ------------------------------------------------------------------ #
    # /show_config                                                          #
    # ------------------------------------------------------------------ #

    async def handle_show_config(chat_id: int, args: list[str]) -> str:
        lines = [
            "<b>Current configuration</b>",
            f"Strategy: {_esc(strategy_name)}",
            f"Mode: {mode}",
            f"Min edge: {config.risk.min_edge_floor:.2%}",
            f"Max position: {config.risk.max_position_pct:.2%} of bankroll",
            f"Max open positions: {config.risk.max_open_positions}",
            f"LLM budget: ${config.risk.max_daily_llm_spend_usd:.2f}/day",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # /logs [n] [filter]                                                   #
    # ------------------------------------------------------------------ #

    async def handle_logs(chat_id: int, args: list[str]) -> str:
        n = 20
        log_filter: str | None = None
        remaining = list(args)
        # Parse optional leading integer for count
        if remaining:
            try:
                n = int(remaining[0])
                remaining = remaining[1:]
            except ValueError:
                pass
        # Remaining arg (if any) is the logger filter
        if remaining:
            log_filter = remaining[0]
        if log_buffer is None:
            return "Log capture not available."
        lines = log_buffer.last(n, filter=log_filter)
        if not lines:
            filter_str = f" matching {log_filter!r}" if log_filter else ""
            return f"No log lines captured yet{filter_str}."
        filter_label = f" [{_esc(log_filter)}]" if log_filter else ""
        header = f"Last {len(lines)} log line(s){filter_label}:"
        # Budget the <pre> body so the wrapped message stays under the limit.
        budget = _TELEGRAM_MAX_LEN - len(header) - len("\n<pre></pre>") - 20
        body = _esc("\n".join(lines))
        if len(body) > budget:
            body = "…" + body[-budget:]
        return f"{header}\n<pre>{body}</pre>"

    # ------------------------------------------------------------------ #
    # /version                                                             #
    # ------------------------------------------------------------------ #

    async def handle_version(chat_id: int, args: list[str]) -> str:
        try:
            version = importlib.metadata.version("freqpred")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"

        try:
            result = subprocess.run(  # noqa: S603
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            git_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            git_hash = "unknown"

        return f"freqpred {version} (git {git_hash})"

    # ------------------------------------------------------------------ #
    # /status [position_id_or_market_ticker]                              #
    # ------------------------------------------------------------------ #

    async def _status_detail(target: str) -> str:
        """Detailed single-position view. Accepts a UUID or market ticker."""
        from freqpred.markets.models import MarketRow, PositionRow  # noqa: PLC0415

        stmt = (
            select(PositionRow, MarketRow.question, MarketRow.mid_price)
            .join(MarketRow, PositionRow.market_id == MarketRow.id)
        )
        try:
            stmt = stmt.where(PositionRow.id == _uuid.UUID(target))
        except ValueError:
            # Not a UUID — treat as a market ticker; prefer the open position.
            stmt = (
                stmt.where(PositionRow.market_id == target)
                .order_by(PositionRow.status != "open", PositionRow.entry_time.desc())
                .limit(1)
            )

        async with session_factory() as session:
            result = await session.execute(stmt)
            row = result.first()

        if row is None:
            return f"No position found for: {_esc(target)}"

        pos, question, mid = row
        direction = pos.direction.upper()
        cost = pos.contracts * pos.entry_price
        current_price = mid if direction == "YES" else 1.0 - mid

        lines = [
            f"<b>{direction} {pos.contracts}×</b> {_esc(pos.market_id)}",
            _esc(question),
            "",
            f"Status: {pos.status}"
            + (f" · open {_ago(pos.entry_time)}" if pos.status == "open" and pos.entry_time else ""),
        ]

        if pos.status == "open":
            unreal = _unrealized_pnl(direction, pos.contracts, pos.entry_price, mid)
            unreal_pct = unreal / cost if cost > 0 else 0.0
            lines.append(
                f"Price: {_fmt_price(pos.entry_price)} → {_fmt_price(current_price)}"
                f" · P&L {_fmt_usd(unreal)} ({unreal_pct:+.1%})"
            )
        else:
            exit_bits = []
            if pos.exit_price is not None:
                exit_bits.append(f"Price: {_fmt_price(pos.entry_price)} → {_fmt_price(pos.exit_price)}")
            if pos.pnl is not None:
                pct = f" ({pos.pnl_pct:+.1%})" if pos.pnl_pct is not None else ""
                exit_bits.append(f"P&L {_fmt_usd(pos.pnl)}{pct}")
            if exit_bits:
                lines.append(" · ".join(exit_bits))
            if pos.exit_reason:
                lines.append(f"Exit reason: {_esc(pos.exit_reason)}")

        lines.append(f"Cost basis: ${cost:,.2f}")

        # Signal snapshot at entry
        sig_bits = []
        if pos.signal_estimated_prob is not None:
            sig_bits.append(f"est {pos.signal_estimated_prob:.0%}")
        if pos.signal_edge is not None:
            sig_bits.append(f"edge {pos.signal_edge:+.1%}")
        if pos.signal_confidence is not None:
            sig_bits.append(f"conf {pos.signal_confidence:.2f}")
        if sig_bits:
            lines.append("Signal at entry: " + " · ".join(sig_bits))

        def _excursion(delta: float | None) -> str:
            if delta is None:
                return "—"
            return f"{_fmt_usd(delta * pos.contracts)} ({delta:+.3f}/contract)"

        lines += [
            f"MAE (worst seen): {_excursion(pos.mae)}",
            f"MFE (best seen): {_excursion(pos.mfe)}",
            f"ID: <code>{pos.id}</code>",
        ]
        return "\n".join(lines)

    async def handle_status(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import MarketRow, PositionRow  # noqa: PLC0415

        if args:
            return await _status_detail(args[0])

        # List all open positions
        from freqpred.alerts.run_state import get_drawdown_window  # noqa: PLC0415
        from freqpred.trading import ledger as _ledger  # noqa: PLC0415

        async with session_factory() as session:
            current_state = await get_run_state(session)
            reset_at, reset_bankroll = await get_drawdown_window(session, mode)
            net_bankroll = await _ledger.get_net_bankroll(
                session, config.trading.bankroll_usd, mode=mode
            )
            if reset_bankroll is not None and reset_bankroll > 0:
                drawdown_pct = max(0.0, (reset_bankroll - net_bankroll) / reset_bankroll) * 100
            else:
                drawdown_pct = 0.0

            result = await session.execute(
                select(PositionRow, MarketRow.question, MarketRow.mid_price)
                .join(MarketRow, PositionRow.market_id == MarketRow.id)
                .where(PositionRow.status == "open", PositionRow.mode == mode)
                .order_by(PositionRow.entry_time.desc())
            )
            rows = result.all()

        icon = _STATE_ICONS.get(current_state, "❓")
        header = [
            f"{icon} <b>{current_state.upper()}</b> · {mode} · {_esc(strategy_name)}",
        ]

        if reset_bankroll is not None and reset_at is not None:
            drawdown_line = (
                f"Drawdown {drawdown_pct:.1f}% from ${reset_bankroll:,.0f}"
                f" (baseline {reset_at.strftime('%m-%d %H:%M')})"
            )
        else:
            drawdown_line = "Drawdown: no baseline (use /reset_drawdown)"
        if drawdown_pct >= 30.0:
            drawdown_line = f"🚨 {drawdown_line} — CIRCUIT BREAKER ACTIVE"

        if current_state != "running":
            header.append(f"⚠️ Signal loop is {current_state} — /start to resume")

        if not rows:
            header.append(f"Open 0/{config.risk.max_open_positions} · no open positions")
            header.append(drawdown_line)
            return "\n".join(header)

        total_unrealized = 0.0
        blocks: list[str] = []
        for pos, question, mid in rows:
            direction = pos.direction.upper()
            unreal = _unrealized_pnl(direction, pos.contracts, pos.entry_price, mid)
            total_unrealized += unreal
            cost = pos.contracts * pos.entry_price
            unreal_pct = unreal / cost if cost > 0 else 0.0
            current_price = mid if direction == "YES" else 1.0 - mid
            age = f" · open {_ago(pos.entry_time)}" if pos.entry_time else ""
            blocks.append(
                f"<b>{direction} {pos.contracts}×</b> {_esc(pos.market_id)}\n"
                f"{_esc(_truncate(question, 80))}\n"
                f"{_fmt_price(pos.entry_price)} → {_fmt_price(current_price)}"
                f" · {_fmt_usd(unreal)} ({unreal_pct:+.1%}){age}"
            )

        header.append(
            f"Open {len(rows)}/{config.risk.max_open_positions}"
            f" · unrealized {_fmt_usd(total_unrealized)}"
        )
        header.append(drawdown_line)
        body = "\n".join(header) + "\n\n" + "\n\n".join(blocks)
        body += "\n\n/status &lt;ticker&gt; for detail · /fx &lt;ticker&gt; to close"
        return _clip(body)

    # ------------------------------------------------------------------ #
    # /trades [n]                                                          #
    # ------------------------------------------------------------------ #

    async def handle_trades(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import MarketRow, PositionRow  # noqa: PLC0415

        n = 10
        if args:
            try:
                n = int(args[0])
            except ValueError:
                return f"Usage: /trades [n] — n must be a number, got {args[0]!r}"

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow, MarketRow.question)
                .join(MarketRow, PositionRow.market_id == MarketRow.id)
                .where(PositionRow.status == "closed", PositionRow.mode == mode)
                .order_by(PositionRow.exit_time.desc())
                .limit(n)
            )
            rows = result.all()

        if not rows:
            return "No resolved positions yet."

        blocks: list[str] = []
        total_pnl = 0.0
        for pos, question in rows:
            pnl = pos.pnl if pos.pnl is not None else 0.0
            total_pnl += pnl
            icon = "✅" if pnl > 0 else ("❌" if pnl < 0 else "➖")
            pct = f" ({pos.pnl_pct:+.1%})" if pos.pnl_pct is not None else ""
            held = ""
            if pos.entry_time and pos.exit_time:
                entry = pos.entry_time if pos.entry_time.tzinfo else pos.entry_time.replace(tzinfo=UTC)
                exit_ = pos.exit_time if pos.exit_time.tzinfo else pos.exit_time.replace(tzinfo=UTC)
                held = f" · held {_fmt_age_secs(max(0, int((exit_ - entry).total_seconds())))}"
            reason = pos.exit_reason or "resolved"
            blocks.append(
                f"{icon} {_fmt_usd(pnl)}{pct} · {pos.direction.upper()}"
                f" · {_esc(reason)}{held}\n"
                f"{_esc(_truncate(question, 80))}"
            )

        header = f"<b>Last {len(rows)} closed trade(s)</b> · net {_fmt_usd(total_pnl)}"
        return _clip(header + "\n\n" + "\n\n".join(blocks))

    # ------------------------------------------------------------------ #
    # /signals [n]                                                         #
    # ------------------------------------------------------------------ #

    async def handle_signals(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import MarketRow  # noqa: PLC0415
        from freqpred.signal.models import SignalRow  # noqa: PLC0415

        n = 10
        if args:
            try:
                n = int(args[0])
            except ValueError:
                return f"Usage: /signals [n] — n must be a number, got {args[0]!r}"

        async with session_factory() as session:
            result = await session.execute(
                select(SignalRow, MarketRow.question)
                .join(MarketRow, SignalRow.market_id == MarketRow.id)
                .order_by(SignalRow.created_at.desc())
                .limit(n)
            )
            rows = result.all()

        if not rows:
            return "No signals recorded yet."

        blocks: list[str] = []
        for sig, question in rows:
            age = f"{_ago(sig.created_at)} ago" if sig.created_at else "?"
            direction = sig.direction.upper()
            icon = {"YES": "🟢", "NO": "🔴", "SKIP": "⏭"}.get(direction, "•")
            blocks.append(
                f"{icon} <b>{direction}</b> · {_esc(sig.market_id)} · {age}\n"
                f"{_esc(_truncate(question, 80))}\n"
                f"est {sig.estimated_probability:.0%} vs mkt {sig.market_mid_at_signal:.0%}"
                f" → edge {sig.edge:+.1%} · conf {sig.confidence:.2f}"
                f" · {_esc(sig.trigger)}"
            )

        header = f"<b>Last {len(rows)} signal(s)</b>"
        return _clip(header + "\n\n" + "\n\n".join(blocks))

    # ------------------------------------------------------------------ #
    # /health — scheduled-service freshness telemetry                      #
    # ------------------------------------------------------------------ #

    async def handle_health(chat_id: int, args: list[str]) -> str:
        if telemetry is None:
            return "Health telemetry not available in this run mode."
        from freqpred.runtime.telemetry import list_service_heartbeats  # noqa: PLC0415

        async with session_factory() as session:
            current_state = await get_run_state(session)
            heartbeats = await list_service_heartbeats(session)
        states = telemetry.evaluate_service_states(heartbeats, run_state=current_state)

        icons = {"ok": "✅", "stale": "🔴", "idle": "⏸", "unknown": "⚪"}
        rank = {"stale": 0, "unknown": 1, "idle": 2, "ok": 3}
        ok_count = sum(1 for s in states if s.status == "ok")

        lines = [f"<b>Service health</b> — {ok_count}/{len(states)} ok"]
        ws = telemetry.websocket_state()
        if ws["connected"] is not None:
            ws_icon = "✅" if ws["connected"] else "🔴"
            markets = ws["subscribed_markets"]
            markets_str = f" · {markets} markets" if markets is not None else ""
            lines.append(f"{ws_icon} WebSocket {'connected' if ws['connected'] else 'disconnected'}{markets_str}")
        lines.append("")

        for s in sorted(states, key=lambda s: (rank.get(s.status, 1), s.label)):
            icon = icons.get(s.status, "❓")
            age = _fmt_age_secs(s.age_seconds) + " ago" if s.age_seconds is not None else "never"
            line = f"{icon} {_esc(s.label)} · {age}"
            if s.status == "stale" and s.last_error_message:
                line += f"\n    ↳ {_esc(_truncate(s.last_error_message, 120))}"
            lines.append(line)
        return _clip("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Register all handlers                                                #
    # ------------------------------------------------------------------ #

    cmd_handler.register(
        "start", handle_start,
        description="Resume the signal loop", category="System")
    cmd_handler.register(
        "pause", handle_pause,
        description="Pause new entries (exits still run)", category="System")
    cmd_handler.register(
        "stop", handle_stop,
        description="Halt signal analysis entirely", category="System")
    cmd_handler.register(
        "shutdown", handle_shutdown,
        description="Gracefully shut down the process", category="System")
    cmd_handler.register(
        "reset_drawdown", handle_reset_drawdown,
        description="Reset the drawdown circuit-breaker baseline", category="System")
    cmd_handler.register(
        "show_config", handle_show_config,
        description="Show strategy, mode, and risk limits", category="System")
    cmd_handler.register(
        "logs", handle_logs,
        description="[n] [filter] — recent log lines", category="Diagnostics")
    cmd_handler.register(
        "version", handle_version,
        description="Version and git commit", category="Diagnostics")
    cmd_handler.register(
        "health", handle_health,
        description="Freshness of schedulers and fetchers", category="Diagnostics")
    cmd_handler.register(
        "status", handle_status,
        description="[id|ticker] — open positions or one position in detail",
        category="Positions")
    cmd_handler.register(
        "trades", handle_trades,
        description="[n] — recent closed trades", category="Positions")
    cmd_handler.register(
        "signals", handle_signals,
        description="[n] — recent signals", category="Positions")
