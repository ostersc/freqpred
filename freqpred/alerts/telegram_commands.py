"""Telegram bot inbound command handler — polling loop + command router.

Separate from TelegramSender (outbound-only) so alert flow is not affected.

Usage:
    handler = TelegramCommandHandler(
        bot_token="...",
        authorized_users=["alice", "123456789"],
    )
    handler.register("status", my_status_handler)

    # Start alongside other async tasks:
    task = asyncio.create_task(handler.run())
    ...
    task.cancel()
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

_API_BASE = "https://api.telegram.org"

# Callable type: receives (chat_id, args) and returns a reply string or None.
CommandHandler = Callable[[int, list[str]], Awaitable[str | None]]

# Callable type for inline keyboard callbacks.
# Receives (chat_id, callback_data, callback_query_id) and returns a reply or None.
CallbackHandler = Callable[[int, str, str], Awaitable[str | None]]


class TelegramCommandHandler:
    """Inbound Telegram command handler.

    Runs a long-polling ``getUpdates`` loop as an asyncio task. Parses
    ``/command [arg1 arg2...]`` messages, checks the sender against
    ``authorized_users``, then dispatches to a registered async handler.

    Auth rules:
    - ``authorized_users`` is a list of strings that can be either a
      Telegram username (e.g. ``"alice"``) or a numeric user ID as a string
      (e.g. ``"123456789"``).
    - If ``authorized_users`` is empty, no one can send commands.
    - Unauthorized senders are silently dropped.

    Disables itself (becomes a no-op) if ``bot_token`` is empty.
    """

    def __init__(
        self,
        bot_token: str,
        authorized_users: list[str],
    ) -> None:
        self._token = bot_token
        self._authorized_users = set(authorized_users)
        self._enabled = bool(bot_token)
        self._offset: int = 0
        self._handlers: dict[str, CommandHandler] = {}
        self._callback_handlers: dict[str, CallbackHandler] = {}

        if not self._enabled:
            log.info("telegram_commands_disabled", reason="missing bot_token")
        else:
            # Register built-in /help command.
            self.register("help", self._help_handler)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, command: str, handler: CommandHandler) -> None:
        """Register *handler* for */command*.

        ``command`` should be given without the leading ``/``.
        Re-registering the same command replaces the previous handler.
        """
        self._handlers[command.lstrip("/")] = handler
        log.debug("telegram_command_registered", command=command)

    def register_callback(self, data: str, handler: CallbackHandler) -> None:
        """Register *handler* for an inline keyboard callback with the given *data*.

        When a user presses an inline button whose ``callback_data`` equals *data*,
        *handler* is called with ``(chat_id, callback_data, callback_query_id)``.
        Re-registering the same data string replaces the previous handler.
        """
        self._callback_handlers[data] = handler
        log.debug("telegram_callback_registered", data=data)

    def unregister_callback(self, data: str) -> None:
        """Remove a previously registered callback handler (no-op if not found)."""
        self._callback_handlers.pop(data, None)

    async def send_inline_keyboard(
        self,
        chat_id: int,
        text: str,
        buttons: list[list[dict[str, str]]],
    ) -> int | None:
        """Send *text* with an inline keyboard to *chat_id*.

        *buttons* is a list of rows; each row is a list of button dicts with
        ``text`` and ``callback_data`` keys.  Returns the Telegram ``message_id``
        of the sent message, or ``None`` on error.
        """
        if not self._enabled:
            return None
        url = f"{_API_BASE}/bot{self._token}/sendMessage"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "reply_markup": {"inline_keyboard": buttons},
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("result", {}).get("message_id")
        except Exception as exc:
            log.warning("telegram_send_inline_keyboard_error", chat_id=chat_id, error=str(exc))
            return None

    async def run(self) -> None:
        """Long-polling loop. Runs until the task is cancelled."""
        if not self._enabled:
            return

        log.info("telegram_command_handler.started")
        try:
            while True:
                updates = await self._get_updates()
                for update in updates:
                    await self._handle_update(update)
        except asyncio.CancelledError:
            log.info("telegram_command_handler.stopped")
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_updates(self) -> list[dict[str, Any]]:
        """Poll Telegram for new updates (long-poll, 30 s timeout).

        On HTTP / network errors, logs and returns an empty list so the
        loop continues on the next iteration.
        """
        url = f"{_API_BASE}/bot{self._token}/getUpdates"
        params: dict[str, Any] = {"timeout": 30}
        if self._offset:
            params["offset"] = self._offset

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=35.0)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, Exception) as exc:
            log.warning("telegram_get_updates_error", error=str(exc))
            await asyncio.sleep(2)
            return []

        updates: list[dict[str, Any]] = data.get("result", [])
        if updates:
            # Advance offset past the last received update.
            self._offset = updates[-1]["update_id"] + 1
        return updates

    async def _handle_update(self, update: dict[str, Any]) -> None:
        """Parse a single update and dispatch if it's an authorized command."""
        if callback_query := update.get("callback_query"):
            await self._handle_callback_query(callback_query)
            return

        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        text: str = message.get("text", "")
        if not text.startswith("/"):
            return

        from_user: dict[str, Any] = message.get("from") or {}
        chat_id: int = message["chat"]["id"]

        if not self._is_authorized(from_user):
            log.debug(
                "telegram_command_unauthorized",
                user_id=from_user.get("id"),
                username=from_user.get("username"),
            )
            return

        command, args = self._parse_command(text)
        handler = self._handlers.get(command)
        if handler is None:
            await self._send_reply(
                chat_id,
                f"Unknown command: /{command}\nType /help for available commands.",
            )
            return

        log.info("telegram_command_dispatched", command=command, chat_id=chat_id)
        try:
            reply = await handler(chat_id, args)
        except Exception as exc:
            log.exception("telegram_command_handler_error", command=command, error=str(exc))
            await self._send_reply(chat_id, f"Error executing /{command}: {exc}")
            return

        if reply:
            await self._send_reply(chat_id, reply)

    def _is_authorized(self, from_user: dict[str, Any]) -> bool:
        """Return True if the sender is in the authorized_users list."""
        if not self._authorized_users:
            return False
        user_id_str = str(from_user.get("id", ""))
        username = from_user.get("username", "")
        return user_id_str in self._authorized_users or username in self._authorized_users

    @staticmethod
    def _parse_command(text: str) -> tuple[str, list[str]]:
        """Parse ''/cmd arg1 arg2'' → ('cmd', ['arg1', 'arg2']).

        Strips the bot mention suffix (e.g. ``/cmd@BotName``).
        """
        parts = text.split()
        raw_command = parts[0].lstrip("/")
        # Strip @BotName suffix if present.
        if "@" in raw_command:
            raw_command = raw_command.split("@", 1)[0]
        return raw_command, parts[1:]

    async def _handle_callback_query(self, callback_query: dict[str, Any]) -> None:
        """Dispatch an inline keyboard callback query to its registered handler."""
        callback_query_id: str = callback_query.get("id", "")
        from_user: dict[str, Any] = callback_query.get("from") or {}
        data: str = callback_query.get("data", "")
        chat_id: int = (callback_query.get("message") or {}).get("chat", {}).get("id", 0)

        if not self._is_authorized(from_user):
            await self._answer_callback_query(callback_query_id)
            return

        handler = self._callback_handlers.get(data)
        if handler is None:
            await self._answer_callback_query(callback_query_id, "Action expired or unknown.")
            return

        log.info("telegram_callback_dispatched", data=data, chat_id=chat_id)
        try:
            reply = await handler(chat_id, data, callback_query_id)
        except Exception as exc:
            log.exception("telegram_callback_handler_error", data=data, error=str(exc))
            await self._answer_callback_query(callback_query_id, f"Error: {exc}")
            return

        await self._answer_callback_query(callback_query_id)
        if reply and chat_id:
            await self._send_reply(chat_id, reply)

    async def _answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        """Acknowledge a callback query (required by Telegram to dismiss the loading state)."""
        if not self._enabled:
            return
        url = f"{_API_BASE}/bot{self._token}/answerCallbackQuery"
        try:
            async with httpx.AsyncClient() as client:
                payload: dict[str, Any] = {"callback_query_id": callback_query_id}
                if text:
                    payload["text"] = text
                response = await client.post(url, json=payload, timeout=10.0)
                response.raise_for_status()
        except Exception as exc:
            log.warning("telegram_answer_callback_error", error=str(exc))

    async def _send_reply(self, chat_id: int, text: str) -> None:
        """Send a plain-text reply to chat_id."""
        url = f"{_API_BASE}/bot{self._token}/sendMessage"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={"chat_id": chat_id, "text": text},
                    timeout=10.0,
                )
                response.raise_for_status()
        except Exception as exc:
            log.warning("telegram_send_reply_error", chat_id=chat_id, error=str(exc))

    # ------------------------------------------------------------------
    # Built-in /help handler
    # ------------------------------------------------------------------

    async def _help_handler(self, chat_id: int, args: list[str]) -> str:
        commands = sorted(self._handlers.keys())
        lines = ["Available commands:"] + [f"  /{cmd}" for cmd in commands]
        return "\n".join(lines)
