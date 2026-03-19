"""Telegram bot command handlers for T30: position management commands.

Registers /forceexit (/fx) and /delete handlers onto a TelegramCommandHandler.

Confirmation rules
------------------
- /forceexit <id>  in paper mode  → execute immediately
- /forceexit <id>  in live mode   → inline keyboard confirmation required
- /forceexit all   (any mode)     → inline keyboard confirmation required
- /delete <id>     in paper mode  → inline keyboard confirmation required
- /delete <id>     in live mode   → rejected with an error message

If confirmation is not received within 30 seconds the action is cancelled and
the bot sends a timeout notice. Pending state is stored in memory only — it is
lost on process restart, which is acceptable per the spec.

Usage::

    register_position_commands(
        cmd_handler=telegram_cmd_handler,
        session_factory=session_factory,
        config=config,
        mode=mode,
    )
"""
from __future__ import annotations

import asyncio
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import delete as sa_delete, select

from freqpred.markets.models import MarketRow, PositionRow
from freqpred.trading import ledger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from freqpred.alerts.telegram_commands import TelegramCommandHandler
    from freqpred.config import Settings

log = structlog.get_logger(__name__)

_CONFIRM_TIMEOUT_SECS = 30
_TELEGRAM_MAX_LEN = 4096


def _clip(text: str) -> str:
    if len(text) <= _TELEGRAM_MAX_LEN:
        return text
    suffix = "\n...[truncated]"
    return text[: _TELEGRAM_MAX_LEN - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_position_commands(
    cmd_handler: "TelegramCommandHandler",
    session_factory: "async_sessionmaker[AsyncSession]",
    config: "Settings",
    mode: str,
) -> None:
    """Register all T30 position management commands onto *cmd_handler*."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _close_position_paper(position_id: str) -> str:
        """Close a single position in paper mode at current market mid price."""
        try:
            pos_uuid = _uuid.UUID(position_id)
        except ValueError:
            return f"Invalid position ID: {position_id!r}"

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow, MarketRow.mid_price)
                .join(MarketRow, PositionRow.market_id == MarketRow.id)
                .where(PositionRow.id == pos_uuid)
            )
            row = result.one_or_none()

            if row is None:
                return f"Position {position_id} not found."

            pos, mid_price = row
            if pos.status != "open":
                return f"Position {position_id} is already {pos.status}."

            closed = await ledger.close_position(
                session,
                position_id,
                exit_price=mid_price,
                exit_reason="manual_telegram",
            )

        pnl_str = f"{closed.pnl:+.4f}" if closed.pnl is not None else "N/A"
        log.info("telegram.forceexit", position_id=position_id, pnl=closed.pnl)
        return (
            f"Position {position_id} closed.\n"
            f"  exit_price={mid_price:.4f}  pnl={pnl_str}  reason=manual_telegram"
        )

    async def _close_position_live(position_id: str) -> str:
        """Close a single position in live mode.

        Closes the ledger record and logs that a corresponding Kalshi order
        would need to be submitted (full live order submission is handled by
        the Order Manager which currently only opens positions).
        """
        try:
            pos_uuid = _uuid.UUID(position_id)
        except ValueError:
            return f"Invalid position ID: {position_id!r}"

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow, MarketRow.mid_price)
                .join(MarketRow, PositionRow.market_id == MarketRow.id)
                .where(PositionRow.id == pos_uuid)
            )
            row = result.one_or_none()

            if row is None:
                return f"Position {position_id} not found."

            pos, mid_price = row
            if pos.status != "open":
                return f"Position {position_id} is already {pos.status}."

            closed = await ledger.close_position(
                session,
                position_id,
                exit_price=mid_price,
                exit_reason="manual_telegram",
            )

        pnl_str = f"{closed.pnl:+.4f}" if closed.pnl is not None else "N/A"
        log.info("telegram.forceexit_live", position_id=position_id, pnl=closed.pnl)
        return (
            f"Position {position_id} force-closed (LIVE).\n"
            f"  exit_price={mid_price:.4f}  pnl={pnl_str}  reason=manual_telegram\n"
            "Note: submit a corresponding close order on Kalshi manually if not already done."
        )

    async def _close_all_positions() -> str:
        """Close all open positions at current market mid price."""
        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow, MarketRow.mid_price)
                .join(MarketRow, PositionRow.market_id == MarketRow.id)
                .where(PositionRow.status == "open")
            )
            rows = result.all()

            if not rows:
                return "No open positions to close."

            closed_summaries: list[str] = []
            for pos, mid_price in rows:
                closed = await ledger.close_position(
                    session,
                    str(pos.id),
                    exit_price=mid_price,
                    exit_reason="manual_telegram",
                )
                pnl_str = f"{closed.pnl:+.4f}" if closed.pnl is not None else "N/A"
                closed_summaries.append(f"  {pos.id}  pnl={pnl_str}")

            log.info("telegram.forceexit_all", count=len(closed_summaries))
            header = f"Closed {len(closed_summaries)} position(s):"
            return _clip(header + "\n" + "\n".join(closed_summaries))

    async def _delete_position(position_id: str) -> str:
        """Hard-delete a paper position from the DB."""
        try:
            pos_uuid = _uuid.UUID(position_id)
        except ValueError:
            return f"Invalid position ID: {position_id!r}"

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow).where(PositionRow.id == pos_uuid)
            )
            row = result.scalar_one_or_none()

            if row is None:
                return f"Position {position_id} not found."

            await session.execute(
                sa_delete(PositionRow).where(PositionRow.id == pos_uuid)
            )
            await session.commit()

        log.info("telegram.delete_position", position_id=position_id)
        return f"Paper position {position_id} deleted from the database."

    # ------------------------------------------------------------------
    # Confirmation flow helper
    # ------------------------------------------------------------------

    async def _require_confirmation(
        chat_id: int,
        prompt: str,
        action_fn: "Callable[[], Awaitable[str]]",
    ) -> None:
        """Send an inline keyboard confirmation prompt and wire up callbacks.

        If the user confirms within 30 s the action is executed and the reply
        is sent.  If they cancel or time out, a notice is sent instead.
        The caller should return ``None`` after calling this coroutine so the
        command handler does not send a duplicate plain-text reply.
        """
        nonce = _uuid.uuid4().hex[:12]
        confirm_data = f"confirm:{nonce}"
        cancel_data = f"cancel:{nonce}"

        async def on_confirm(cb_chat_id: int, data: str, cb_query_id: str) -> str | None:
            cmd_handler.unregister_callback(confirm_data)
            cmd_handler.unregister_callback(cancel_data)
            result = await action_fn()
            return result

        async def on_cancel(cb_chat_id: int, data: str, cb_query_id: str) -> str | None:
            cmd_handler.unregister_callback(confirm_data)
            cmd_handler.unregister_callback(cancel_data)
            return "Action cancelled."

        cmd_handler.register_callback(confirm_data, on_confirm)
        cmd_handler.register_callback(cancel_data, on_cancel)

        await cmd_handler.send_inline_keyboard(
            chat_id,
            prompt,
            [
                [
                    {"text": "Confirm", "callback_data": confirm_data},
                    {"text": "Cancel", "callback_data": cancel_data},
                ]
            ],
        )

        # Background task that fires a timeout notice after 30 s.
        async def _timeout_notify() -> None:
            await asyncio.sleep(_CONFIRM_TIMEOUT_SECS)
            if confirm_data in cmd_handler._callback_handlers:
                cmd_handler.unregister_callback(confirm_data)
                cmd_handler.unregister_callback(cancel_data)
                await cmd_handler._send_reply(
                    chat_id,
                    "Confirmation timed out after 30 s. Action cancelled.",
                )

        asyncio.create_task(_timeout_notify())

    # ------------------------------------------------------------------
    # /forceexit [<position_id> | all]
    # ------------------------------------------------------------------

    async def handle_forceexit(chat_id: int, args: list[str]) -> str | None:
        if not args:
            return "Usage: /forceexit <position_id> | /forceexit all"

        target = args[0]

        if target.lower() == "all":
            # Always requires confirmation regardless of mode.
            await _require_confirmation(
                chat_id,
                "Force-close ALL open positions?",
                _close_all_positions,
            )
            return None

        # Single position.
        if mode == "paper":
            return await _close_position_paper(target)

        # Live mode — require confirmation.
        await _require_confirmation(
            chat_id,
            f"Force-close position {target} in LIVE mode?",
            lambda: _close_position_live(target),
        )
        return None

    # ------------------------------------------------------------------
    # /fx — alias for /forceexit
    # ------------------------------------------------------------------

    async def handle_fx(chat_id: int, args: list[str]) -> str | None:
        return await handle_forceexit(chat_id, args)

    # ------------------------------------------------------------------
    # /delete <position_id>
    # ------------------------------------------------------------------

    async def handle_delete(chat_id: int, args: list[str]) -> str | None:
        if not args:
            return "Usage: /delete <position_id>"

        if mode == "live":
            return (
                "/delete is not available in live mode. "
                "Use /forceexit to close a live position via the Order Manager."
            )

        position_id = args[0]

        # Validate UUID before showing confirmation prompt.
        try:
            _uuid.UUID(position_id)
        except ValueError:
            return f"Invalid position ID: {position_id!r}"

        await _require_confirmation(
            chat_id,
            f"Permanently delete paper position {position_id} from the database?",
            lambda: _delete_position(position_id),
        )
        return None

    # ------------------------------------------------------------------
    # Register all handlers
    # ------------------------------------------------------------------

    cmd_handler.register("forceexit", handle_forceexit)
    cmd_handler.register("fx", handle_fx)
    cmd_handler.register("delete", handle_delete)
