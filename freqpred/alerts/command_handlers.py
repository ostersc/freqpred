"""Telegram bot command handlers for T28: system + status commands.

Registers all /start /pause /stop /show_config /logs /version
/status /count /trades /signals handlers onto a TelegramCommandHandler.

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
import importlib.metadata
import logging
import os
import signal
import subprocess
import uuid as _uuid
from datetime import UTC, datetime, timezone
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.alerts.run_state import get_run_state, reset_drawdown, set_run_state

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from freqpred.alerts.telegram_commands import TelegramCommandHandler
    from freqpred.config import Settings

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
# Helper
# ---------------------------------------------------------------------------


def _truncate(text: str, n: int) -> str:
    return text[:n] + "…" if len(text) > n else text


def _clip(text: str) -> str:
    """Truncate to Telegram's 4096-char message limit."""
    if len(text) <= _TELEGRAM_MAX_LEN:
        return text
    suffix = "\n...[truncated]"
    return text[: _TELEGRAM_MAX_LEN - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_system_commands(
    cmd_handler: "TelegramCommandHandler",
    session_factory: "async_sessionmaker[AsyncSession]",
    config: "Settings",
    mode: str,
    strategy_name: str,
    log_buffer: LogBuffer | None = None,
) -> None:
    """Register all T28 commands onto *cmd_handler*."""

    # ------------------------------------------------------------------ #
    # /start — set state to running                                        #
    # ------------------------------------------------------------------ #

    async def handle_start(chat_id: int, args: list[str]) -> str:
        async with session_factory() as session:
            await set_run_state(session, "running")
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
        from freqpred.trading import ledger as _ledger  # noqa: PLC0415
        async with session_factory() as session:
            net_bankroll = await _ledger.get_net_bankroll(
                session, config.trading.bankroll_usd, mode=mode
            )
            reset_at = await reset_drawdown(session, net_bankroll)
        log.info(
            "telegram.reset_drawdown",
            chat_id=chat_id, reset_at=reset_at.isoformat(), net_bankroll=net_bankroll,
        )
        return (
            f"Drawdown reset. Baseline set to ${net_bankroll:,.2f} "
            f"at {reset_at.strftime('%Y-%m-%d %H:%M UTC')}."
        )

    # ------------------------------------------------------------------ #
    # /show_config                                                          #
    # ------------------------------------------------------------------ #

    async def handle_show_config(chat_id: int, args: list[str]) -> str:
        lines = [
            "Current configuration:",
            f"  strategy    : {strategy_name}",
            f"  mode        : {mode}",
            f"  min edge    : {config.risk.min_edge_floor:.2%}",
            f"  max position: {config.risk.max_position_pct:.2%} of bankroll",
            f"  llm budget  : ${config.risk.max_daily_llm_spend_usd:.2f}/day",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # /logs [n]                                                            #
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
        body = "\n".join(lines)
        filter_label = f" [{log_filter}]" if log_filter else ""
        return _clip(f"Last {len(lines)} log line(s){filter_label}:\n```\n{body}\n```")

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
    # /status [position_id]                                               #
    # ------------------------------------------------------------------ #

    async def handle_status(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import MarketRow, PositionRow

        if args:
            # Detailed single-position view
            pos_id_str = args[0]
            try:
                pos_uuid = _uuid.UUID(pos_id_str)
            except ValueError:
                return f"Invalid position ID: {pos_id_str!r}"

            async with session_factory() as session:
                result = await session.execute(
                    select(PositionRow, MarketRow.question)
                    .join(MarketRow, PositionRow.market_id == MarketRow.id)
                    .where(PositionRow.id == pos_uuid)
                )
                row = result.one_or_none()

            if row is None:
                return f"Position {pos_id_str} not found."

            pos, question = row
            time_open = "N/A"
            if pos.entry_time:
                delta = datetime.now(timezone.utc) - pos.entry_time.replace(tzinfo=timezone.utc if pos.entry_time.tzinfo is None else pos.entry_time.tzinfo)
                hours, rem = divmod(int(delta.total_seconds()), 3600)
                mins = rem // 60
                time_open = f"{hours}h {mins}m"

            conf_str = f"{pos.signal_confidence:.2f}" if pos.signal_confidence is not None else "N/A"
            edge_str = f"{pos.signal_edge:+.3f}" if pos.signal_edge is not None else "N/A"
            prob_str = f"{pos.signal_estimated_prob:.3f}" if pos.signal_estimated_prob is not None else "N/A"

            def _excursion_str(delta: float | None, contracts: int) -> str:
                if delta is None:
                    return "N/A"
                return f"{delta:+.4f}  (${delta * contracts:+.2f})"

            lines = [
                f"Position: {pos.id}",
                f"Market  : {_truncate(question, 60)}",
                f"Direction: {pos.direction}  |  Contracts: {pos.contracts}",
                f"Entry price     : {pos.entry_price:.4f}",
                f"Est. probability: {prob_str}",
                f"Edge at entry   : {edge_str}",
                f"Confidence      : {conf_str}",
                f"MAE (worst seen): {_excursion_str(pos.mae, pos.contracts)}",
                f"MFE (best seen) : {_excursion_str(pos.mfe, pos.contracts)}",
                f"Status  : {pos.status}",
                f"Time open: {time_open}",
            ]
            return "\n".join(lines)

        # List all open positions
        from freqpred.alerts.run_state import get_drawdown_window  # noqa: PLC0415
        from freqpred.trading import ledger as _ledger  # noqa: PLC0415

        async with session_factory() as session:
            current_state = await get_run_state(session)
            reset_at, reset_bankroll = await get_drawdown_window(session)
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

        if reset_bankroll is not None and reset_at is not None:
            reset_label = f" from ${reset_bankroll:,.0f} (reset {reset_at.strftime('%m-%d %H:%M')})"
        else:
            reset_label = " (no baseline)"
        drawdown_str = f"drawdown={drawdown_pct:.1f}%{reset_label}"
        if drawdown_pct >= 30.0:
            drawdown_str += " *** CIRCUIT BREAKER ACTIVE ***"

        state_line = f"state={current_state} | strategy={strategy_name} | mode={mode} | {drawdown_str}"
        if current_state != "running":
            state_line += f"\n*** signal loop is {current_state.upper()} — use /start to resume ***"

        if not rows:
            return f"{state_line}\nNo open positions."

        lines = [state_line, "Open positions:"]
        total_unrealized = 0.0
        for pos, question, mid in rows:
            q = _truncate(question, 60)
            # Unrealized P&L estimate
            if pos.direction == "YES":
                unreal_pnl = pos.contracts * (mid - pos.entry_price)
            else:
                unreal_pnl = pos.contracts * ((1.0 - mid) - pos.entry_price)
            total_unrealized += unreal_pnl
            prob_str = f"{pos.signal_estimated_prob:.3f}" if pos.signal_estimated_prob is not None else "N/A"
            mae_str = f"{pos.mae:+.4f}" if pos.mae is not None else "—"
            mfe_str = f"{pos.mfe:+.4f}" if pos.mfe is not None else "—"
            lines.append(
                f"  [{pos.direction}] {pos.market_id}\n"
                f"    {q}\n"
                f"    entry={pos.entry_price:.4f}  prob={prob_str}  unreal_pnl=${unreal_pnl:+.2f}"
                f"  mae={mae_str}  mfe={mfe_str}"
            )
        lines.append(f"\nTotal unrealized P&L: ${total_unrealized:+.2f}")
        return _clip("\n".join(lines))

    # ------------------------------------------------------------------ #
    # /count                                                               #
    # ------------------------------------------------------------------ #

    async def handle_count(chat_id: int, args: list[str]) -> str:
        from sqlalchemy import func
        from freqpred.markets.models import PositionRow

        async with session_factory() as session:
            result = await session.execute(
                select(func.count()).select_from(PositionRow).where(
                    PositionRow.status == "open", PositionRow.mode == mode,
                )
            )
            open_count: int = result.scalar_one()

        max_pos = config.risk.max_open_positions
        return f"Open: {open_count} / Max: {max_pos}"

    # ------------------------------------------------------------------ #
    # /trades [n]                                                          #
    # ------------------------------------------------------------------ #

    async def handle_trades(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import MarketRow, PositionRow

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

        lines = [f"Last {len(rows)} resolved position(s):"]
        for pos, question in rows:
            q = _truncate(question, 80)
            pnl_str = f"{pos.pnl:+.4f}" if pos.pnl is not None else "N/A"
            duration = "N/A"
            if pos.entry_time and pos.exit_time:
                entry = pos.entry_time
                exit_ = pos.exit_time
                if entry.tzinfo is None:
                    entry = entry.replace(tzinfo=timezone.utc)
                if exit_.tzinfo is None:
                    exit_ = exit_.replace(tzinfo=timezone.utc)
                delta = exit_ - entry
                hours, rem = divmod(int(delta.total_seconds()), 3600)
                mins = rem // 60
                duration = f"{hours}h {mins}m"
            reason = pos.exit_reason or "resolved"
            lines.append(f"  {q}\n    exit={reason}  pnl={pnl_str}  held={duration}")
        return _clip("\n".join(lines))

    # ------------------------------------------------------------------ #
    # /signals [n]                                                         #
    # ------------------------------------------------------------------ #

    async def handle_signals(chat_id: int, args: list[str]) -> str:
        from freqpred.markets.models import MarketRow
        from freqpred.signal.models import SignalRow

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

        lines = [f"Last {len(rows)} signal(s):"]
        for sig, question in rows:
            q = _truncate(question, 80)
            lines.append(
                f"  {q}\n"
                f"    prob={sig.estimated_probability:.3f}  mid={sig.market_mid_at_signal:.3f}"
                f"  edge={sig.edge:+.3f}  dir={sig.direction}"
            )
        return _clip("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Register all handlers
    # ------------------------------------------------------------------ #

    cmd_handler.register("start", handle_start)
    cmd_handler.register("pause", handle_pause)
    cmd_handler.register("stop", handle_stop)
    cmd_handler.register("shutdown", handle_shutdown)
    cmd_handler.register("reset_drawdown", handle_reset_drawdown)
    cmd_handler.register("show_config", handle_show_config)
    cmd_handler.register("logs", handle_logs)
    cmd_handler.register("version", handle_version)
    cmd_handler.register("status", handle_status)
    cmd_handler.register("count", handle_count)
    cmd_handler.register("trades", handle_trades)
    cmd_handler.register("signals", handle_signals)
