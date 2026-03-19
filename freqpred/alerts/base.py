"""Shared protocol for alert senders."""
from __future__ import annotations

from typing import Protocol


class AlertSender(Protocol):
    async def send(self, message: str) -> None: ...
