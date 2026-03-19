"""Discord alert sender."""
from __future__ import annotations

import structlog
import httpx

log = structlog.get_logger(__name__)


class DiscordSender:
    """Sends messages via Discord webhook.

    Silently disabled if webhook_url is empty.
    """

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url
        self._enabled = bool(webhook_url)
        if not self._enabled:
            log.info("discord_alerts_disabled", reason="missing webhook_url")

    async def send(self, message: str) -> None:
        if not self._enabled:
            return
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._url,
                json={"content": message},
                timeout=10.0,
            )
            response.raise_for_status()
        log.debug("discord_alert_sent")
