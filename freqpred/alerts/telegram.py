"""Telegram alert sender."""
from __future__ import annotations

import structlog
import httpx

log = structlog.get_logger(__name__)

_API_BASE = "https://api.telegram.org"


class TelegramSender:
    """Sends messages via Telegram Bot API.

    Silently disabled if token or chat_id are empty.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        if not self._enabled:
            log.info("telegram_alerts_disabled", reason="missing bot_token or chat_id")

    async def send(self, message: str) -> None:
        if not self._enabled:
            return
        url = f"{_API_BASE}/bot{self._token}/sendMessage"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"chat_id": self._chat_id, "text": message},
                timeout=10.0,
            )
            response.raise_for_status()
        log.debug("telegram_alert_sent", chat_id=self._chat_id)
