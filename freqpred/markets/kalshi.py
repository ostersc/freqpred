"""Kalshi prediction market API adapter."""
from __future__ import annotations

from freqpred.markets.base import IMarketClient
from freqpred.markets.models import Market, Order, Position


class KalshiClient(IMarketClient):
    """Concrete IMarketClient implementation for the Kalshi exchange."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def get_active_markets(self) -> list[Market]:
        raise NotImplementedError

    async def get_market(self, market_id: str) -> Market:
        raise NotImplementedError

    async def place_order(self, order: Order) -> Order:
        raise NotImplementedError

    async def get_positions(self) -> list[Position]:
        raise NotImplementedError

    async def get_balance(self) -> float:
        raise NotImplementedError
