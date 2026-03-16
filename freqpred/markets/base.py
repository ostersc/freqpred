"""Abstract market client interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from freqpred.markets.models import Market, Order, Position


class IMarketClient(ABC):
    """Abstract interface over a prediction market exchange (e.g. Kalshi)."""

    @abstractmethod
    async def get_active_markets(self) -> list[Market]:
        """Fetch all currently active markets."""
        ...

    @abstractmethod
    async def get_market(self, market_id: str) -> Market:
        """Fetch a single market by ID."""
        ...

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Submit an order to the exchange."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Fetch open positions from the exchange."""
        ...

    @abstractmethod
    async def get_balance(self) -> float:
        """Return current account balance in USD."""
        ...
