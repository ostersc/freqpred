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
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import delete as sa_delete
from sqlalchemy import select

from freqpred.markets.models import PositionRow

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from freqpred.alerts.telegram_commands import TelegramCommandHandler
    from freqpred.config import Settings
    from freqpred.trading.order_manager import OrderManager

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
    cmd_handler: TelegramCommandHandler,
    session_factory: async_sessionmaker[AsyncSession],
    config: Settings,
    mode: str,
    order_manager: OrderManager | None = None,
) -> None:
    """Register all T30 position management commands onto *cmd_handler*."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _force_exit_one(position_id: str) -> str:
        """Close a single position via OrderManager.force_exit()."""
        if order_manager is None:
            return "Force exit not available (signal-only mode)."
        try:
            closed = await order_manager.force_exit(position_id, exit_reason="force_exit:manual")
        except ValueError as exc:
            return str(exc)
        except Exception as exc:
            return f"Force exit failed: {exc}"
        pnl_str = f"{closed.pnl:+.4f}" if closed.pnl is not None else "N/A"
        log.info("telegram.forceexit", position_id=position_id, pnl=closed.pnl)
        return (
            f"Position {position_id} closed.\n"
            f"  exit_price={closed.exit_price:.4f}  pnl={pnl_str}  reason={closed.exit_reason}"
        )

    async def _close_all_positions() -> str:
        """Close all open positions (matching current mode) via OrderManager.force_exit()."""
        if order_manager is None:
            return "Force exit not available (signal-only mode)."

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow.id).where(
                    PositionRow.status == "open",
                    PositionRow.mode == mode,
                )
            )
            pos_ids = [str(r) for r in result.scalars().all()]

        if not pos_ids:
            return "No open positions to close."

        closed_summaries: list[str] = []
        errors: list[str] = []
        for pos_id in pos_ids:
            try:
                closed = await order_manager.force_exit(pos_id, exit_reason="force_exit:manual")
                pnl_str = f"{closed.pnl:+.4f}" if closed.pnl is not None else "N/A"
                closed_summaries.append(f"  {pos_id}  pnl={pnl_str}")
            except Exception as exc:
                errors.append(f"  {pos_id}: {exc}")

        log.info("telegram.forceexit_all", count=len(closed_summaries))
        parts: list[str] = []
        if closed_summaries:
            parts.append(f"Closed {len(closed_summaries)} position(s):\n" + "\n".join(closed_summaries))
        if errors:
            parts.append(f"{len(errors)} error(s):\n" + "\n".join(errors))
        return _clip("\n".join(parts))

    async def _delete_position(position_id: str) -> str:
        """Hard-delete a paper position from the DB."""
        try:
            pos_uuid = _uuid.UUID(position_id)
        except ValueError:
            return f"Invalid position ID: {position_id!r}"

        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow).where(PositionRow.id == pos_uuid, PositionRow.mode == "paper")
            )
            row = result.scalar_one_or_none()

            if row is None:
                return f"Paper position {position_id} not found (live positions cannot be deleted)."

            await session.execute(
                sa_delete(PositionRow).where(PositionRow.id == pos_uuid, PositionRow.mode == "paper")
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
        action_fn: Callable[[], Awaitable[str]],
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

    async def _resolve_to_position_id(target: str) -> str | None:
        """Return a position UUID for *target*.

        Accepts either a UUID string or a market ID (e.g. KXTRUMPSAY-26APR06-AUTO).
        Returns None if no matching open position is found.
        """
        try:
            _uuid.UUID(target)
            return target  # already a valid UUID
        except ValueError:
            pass
        # Treat as market_id — look up the open position
        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow.id).where(
                    PositionRow.market_id == target,
                    PositionRow.status == "open",
                    PositionRow.mode == mode,
                ).limit(1)
            )
            row = result.scalar_one_or_none()
            return str(row) if row is not None else None

    async def handle_forceexit(chat_id: int, args: list[str]) -> str | None:
        if not args:
            return "Usage: /forceexit <position_id_or_market_id> | /forceexit all"

        target = args[0]

        if target.lower() == "all":
            # Always requires confirmation regardless of mode.
            await _require_confirmation(
                chat_id,
                "Force-close ALL open positions?",
                _close_all_positions,
            )
            return None

        # Resolve market ID → position UUID if needed.
        position_id = await _resolve_to_position_id(target)
        if position_id is None:
            return f"No open position found for: {target!r}"

        # Single position.
        if mode == "paper":
            return await _force_exit_one(position_id)

        # Live mode — require confirmation.
        await _require_confirmation(
            chat_id,
            f"Force-close position {target} in LIVE mode?",
            lambda: _force_exit_one(position_id),
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
