"""Kalshi prediction market API adapter."""
from __future__ import annotations

import asyncio
import base64
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from freqpred.markets.base import IMarketClient
from freqpred.markets.models import (
    KalshiMarketsResponse,
    KalshiSingleMarketResponse,
    Market,
    Order,
    Position,
)

log = structlog.get_logger(__name__)


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Kalshi API error {status_code}: {body}")

# Known Kalshi event_ticker prefixes → our category label.
# The event_ticker first segment (before the first '-') is the series ticker,
# which encodes the market type.  This avoids the impractical /series approach
# which returns thousands of series and requires an API call per series.
_TICKER_PREFIX_TO_CATEGORY: dict[str, str] = {
    # Sports — NBA props dominate open markets
    "KXNBA": "sports",
    "KXNFL": "sports",
    "KXMLB": "sports",
    "KXNHL": "sports",
    "KXNCAA": "sports",
    "KXSOC": "sports",
    "KXPGA": "sports",
    "KXTENNIS": "sports",
    "KXUFC": "sports",
    "KXMMA": "sports",
    "KXF1": "sports",
    "KXNASCAR": "sports",
    "KXBOXING": "sports",
    # Politics
    "KXPRES": "politics",
    "KXSEN": "politics",
    "KXHOUSE": "politics",
    "KXGOV": "politics",
    "KXELECT": "politics",
    "KXTRUMP": "politics",
    "KXBIDEN": "politics",
    # Economics
    "KXECON": "economics",
    "KXCPI": "economics",
    "KXFED": "economics",
    "KXGDP": "economics",
    "WRECSS": "economics",
    # Technology
    "KXTECH": "technology",
    "OAIAGI": "technology",
    "NYTOAI": "technology",
    "AMAZONFTC": "technology",
    "APPLEUS": "technology",
    "TESLAOPTIMUS": "technology",
    "EVSHARE": "technology",
    # Science / Space
    "KXSPACE": "science",
    "STARSHIPMARS": "science",
    "MOON": "science",
    # Climate
    "KXCLIMATE": "climate",
    "USCLIMATE": "climate",
    # Finance
    "KXSP500": "finance",
    "KXNASDAQ": "finance",
}


def _infer_category(event_ticker: str) -> str:
    """Map a Kalshi event_ticker to our category via known series prefix table.

    The first dash-separated segment of ``event_ticker`` is the series ticker.
    We do a longest-prefix match against ``_TICKER_PREFIX_TO_CATEGORY`` so that
    e.g. ``KXNBA3PT`` (starts with ``KXNBA``) correctly resolves to 'sports'.
    """
    prefix = event_ticker.split("-")[0].upper() if event_ticker else ""
    # Exact match first; then longest matching known prefix
    if prefix in _TICKER_PREFIX_TO_CATEGORY:
        return _TICKER_PREFIX_TO_CATEGORY[prefix]
    best = ""
    for known in _TICKER_PREFIX_TO_CATEGORY:
        if prefix.startswith(known) and len(known) > len(best):
            best = known
    return _TICKER_PREFIX_TO_CATEGORY[best] if best else "other"


_MAX_RETRIES = 3
_PAGE_SIZE = 1000


class KalshiClient(IMarketClient):
    """Concrete IMarketClient implementation for the Kalshi exchange.

    Authentication uses RSA-PSS signing per Kalshi's API spec:
      KALSHI-ACCESS-KEY: your API key ID
      KALSHI-ACCESS-TIMESTAMP: current time in milliseconds
      KALSHI-ACCESS-SIGNATURE: base64(RSA-PSS-SHA256(timestamp + METHOD + path))

    If no private key is configured, auth headers are omitted (useful for
    public/demo endpoints or testing).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        private_key_path: str = "",
        read_rps: int = 15,
    ) -> None:
        # Kalshi read rate limits by tier: Basic=20/s, Advanced=30/s,
        # Premier=100/s, Prime=400/s.  Default 15 is safe for Basic.
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # Extract just the path portion of the base URL for use in RSA signing.
        # Kalshi signs the full path (e.g. /trade-api/v2/markets), not the
        # relative endpoint path (/markets).
        from urllib.parse import urlparse
        self._base_path = urlparse(self._base_url).path.rstrip("/")
        self._private_key = None
        if private_key_path:
            key_bytes = Path(private_key_path).read_bytes()
            self._private_key = load_pem_private_key(key_bytes, password=None)
        self._http = httpx.AsyncClient(timeout=30.0)
        self._min_interval = 1.0 / max(read_rps, 1)
        self._last_request_at: float = 0.0
        log.info(
            "kalshi.client.init",
            base_url=self._base_url,
            api_key_configured=bool(self._api_key),
            private_key_configured=bool(self._private_key),
        )

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _make_auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Build Kalshi auth headers. Returns empty dict if no key configured."""
        if not self._private_key or not self._api_key:
            return {}
        ts_ms = str(int(time.time() * 1000))
        path_no_query = path.split("?")[0]
        message = f"{ts_ms}{method.upper()}{path_no_query}".encode()
        sig_bytes = self._private_key.sign(  # type: ignore[union-attr]
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig_bytes).decode(),
        }

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Authenticated GET with simple rate limiting and 429 backoff."""
        url = self._base_url + path

        for attempt in range(_MAX_RETRIES):
            # Throttle to stay within read_rps
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_at = time.monotonic()

            headers = self._make_auth_headers("GET", self._base_path + path)
            resp = await self._http.get(url, params=params, headers=headers)

            if resp.status_code == 429:
                backoff = float(resp.headers.get("Retry-After", 2 ** attempt))
                log.warning(
                    "kalshi_rate_limited",
                    path=path,
                    retry_after=backoff,
                    attempt=attempt,
                )
                await asyncio.sleep(backoff)
                continue

            if resp.status_code >= 400:
                raise KalshiAPIError(resp.status_code, resp.text)
            return resp.json()

        raise RuntimeError(f"Kalshi GET {path} failed after {_MAX_RETRIES} retries")

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        """Authenticated POST request to Kalshi API.

        Uses the same RSA-PSS signing as _get() — method="POST".
        Raises KalshiAPIError on non-2xx response (no retry — POSTs are not idempotent).
        """
        url = self._base_url + path
        headers = self._make_auth_headers("POST", self._base_path + path)
        headers["Content-Type"] = "application/json"
        resp = await self._http.post(url, json=body, headers=headers)
        if resp.status_code >= 400:
            raise KalshiAPIError(resp.status_code, resp.text)
        return resp.json()

    async def _paginate_markets(self, params: dict[str, Any]) -> list[Any]:
        """Fetch all pages from GET /markets and return validated KalshiMarketSchema list."""
        from freqpred.markets.models import KalshiMarketSchema

        results: list[KalshiMarketSchema] = []
        cursor: str | None = None

        while True:
            page_params: dict[str, Any] = {**params, "limit": _PAGE_SIZE}
            if cursor:
                page_params["cursor"] = cursor

            data = await self._get("/markets", params=page_params)
            envelope = KalshiMarketsResponse.model_validate(data)
            results.extend(envelope.markets)

            cursor = envelope.cursor or None
            if not cursor or len(envelope.markets) < _PAGE_SIZE:
                break

        return results

    # ------------------------------------------------------------------
    # Domain mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dollar(s: str | None) -> float:
        """Parse a Kalshi dollar-string (e.g. '0.5600') to a float."""
        if not s:
            return 0.0
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def _schema_to_market(self, schema: Any, category: str = "other") -> Market:
        """Convert a validated KalshiMarketSchema to the Market domain object."""
        from freqpred.markets.models import KalshiMarketSchema

        now = datetime.now(UTC)
        s: KalshiMarketSchema = schema
        close_time = datetime.fromisoformat(s.close_time.replace("Z", "+00:00"))
        open_time = (
            datetime.fromisoformat(s.open_time.replace("Z", "+00:00"))
            if s.open_time
            else now
        )

        return Market(
            id=s.ticker,
            platform="kalshi",
            question="\n".join(filter(None, [
                f"{s.title} — {s.yes_sub_title}" if s.yes_sub_title else s.title,
                s.rules_primary or None,
                s.rules_secondary or None,
            ])),
            category=category,
            status=s.status or "active",
            result=s.result or None,
            close_time=close_time,
            yes_bid=s.yes_bid,
            yes_ask=s.yes_ask,
            mid_price=s.mid_price,
            last_price=s.last_price,
            volume_24h=float(s.volume_24h),
            open_interest=float(s.open_interest),
            liquidity=s.liquidity,
            last_fetched_at=now,
            price_updated_at=now,
            metadata_fetched_at=now,
            open_time=open_time,
            metadata={
                "event_ticker": s.event_ticker,
                "subtitle": s.subtitle,
                "status": s.status,
            },
        )

    def _to_market(self, raw: dict[str, Any], category: str = "other") -> Market:
        """Convert a raw Kalshi API dict to a Market domain object via Pydantic validation."""
        from freqpred.markets.models import KalshiMarketSchema

        schema = KalshiMarketSchema.model_validate(raw)
        return self._schema_to_market(schema, category)

    # ------------------------------------------------------------------
    # IMarketClient implementation
    # ------------------------------------------------------------------

    async def list_markets(self, category: str | None = None) -> list[Market]:
        """Fetch all open binary markets, optionally filtered by category.

        Category is inferred from each market's ``event_ticker`` prefix via the
        ``_TICKER_PREFIX_TO_CATEGORY`` table.  Always fetches all open markets in
        one paginated pass (~2 seconds), then filters client-side.  This avoids
        the ``/series`` approach which requires one API call per series and Kalshi
        has thousands of series per category.
        """
        schemas = await self._paginate_markets(
            {"status": "open", "mve_filter": "exclude"}
        )
        result: list[Market] = []
        for s in schemas:
            cat = _infer_category(s.event_ticker)
            if category is None or cat == category.lower():
                result.append(self._schema_to_market(s, cat))
        log.info(
            "kalshi_list_markets",
            category=category,
            total_fetched=len(schemas),
            returned=len(result),
        )
        return result

    async def get_market(self, market_id: str) -> Market:
        """Fetch a single market by ticker."""
        data = await self._get(f"/markets/{market_id}")
        # Try wrapped response first, fall back to bare dict
        if "market" in data:
            envelope = KalshiSingleMarketResponse.model_validate(data)
            return self._schema_to_market(envelope.market)
        return self._to_market(data)

    async def get_market_from_settled(self, market_id: str) -> Market | None:
        """Look up a market via the settled (status=settled) list endpoint.

        Kalshi's GET /markets/{ticker} returns 404 for markets that have been
        purged from the live API after finalization.  The settled list endpoint
        retains them and supports filtering by ``tickers=``.  Returns None if
        the market is not found in the settled list.
        """
        try:
            data = await self._get("/markets", params={"status": "settled", "tickers": market_id, "limit": 1})
        except KalshiAPIError:
            return None
        markets = data.get("markets", [])
        if not markets:
            return None
        return self._to_market(markets[0])

    async def get_orderbook(self, market_id: str) -> dict[str, float]:
        """Return best bid/ask from the Kalshi orderbook.

        Kalshi's orderbook returns yes bids and no bids only. The best YES ask
        is derived as ``1 - best_no_bid`` (binary market identity).
        """
        data = await self._get(
            f"/markets/{market_id}/orderbook", params={"depth": 1}
        )
        ob: dict[str, Any] = data.get("orderbook", {})

        yes_bid = 0.0
        yes_ask = 1.0

        yes_levels: list[Any] = ob.get("yes_dollars", [])
        if yes_levels:
            yes_bid = max(
                float(lv[0]) if isinstance(lv, list) else float(lv.get("price", 0))
                for lv in yes_levels
            )

        no_levels: list[Any] = ob.get("no_dollars", [])
        if no_levels:
            best_no_bid = max(
                float(lv[0]) if isinstance(lv, list) else float(lv.get("price", 0))
                for lv in no_levels
            )
            yes_ask = round(1.0 - best_no_bid, 4)

        return {"yes_bid": yes_bid, "yes_ask": yes_ask}

    async def place_order(self, order: Order) -> Order:
        """Submit a limit order to Kalshi.

        Maps Order fields to the Kalshi API request format and returns the Order
        with exchange_order_id and status populated from the response.
        """
        price_cents = int(round(order.price * 100))
        body: dict[str, Any] = {
            "ticker": order.market_id,
            "action": order.action,  # "buy" | "sell"
            "side": order.direction.lower(),  # "yes" | "no"
            "type": "limit",
            "count": order.contracts,
            # Kalshi requires exactly one of yes_price/no_price (integer cents).
            "yes_price" if order.direction == "YES" else "no_price": price_cents,
        }
        # Only send time_in_force when non-default; omitting the field lets Kalshi
        # apply its default (GTC). Kalshi accepts "fill_or_kill" for immediate exits.
        if order.time_in_force.upper() != "GTC":
            body["time_in_force"] = order.time_in_force
        data = await self._post("/portfolio/orders", body)
        exchange_order = data.get("order", data)
        fee_cents = (exchange_order.get("maker_fees") or 0) + (exchange_order.get("taker_fees") or 0)
        fee_usd = fee_cents / 100
        log.info(
            "kalshi.place_order",
            market_id=order.market_id,
            direction=order.direction,
            contracts=order.contracts,
            exchange_order_id=exchange_order.get("order_id"),
            status=exchange_order.get("status"),
            fee_usd=fee_usd,
        )
        return Order(
            market_id=order.market_id,
            direction=order.direction,
            contracts=order.contracts,
            price=order.price,
            mode=order.mode,
            id=order.id,
            exchange_order_id=exchange_order.get("order_id"),
            status=exchange_order.get("status", "resting"),
            fee_usd=fee_usd,
        )

    async def get_positions(self) -> list[Position]:
        """Fetch all open positions from Kalshi for reconciliation with the local DB."""
        data = await self._get("/portfolio/positions")
        # Kalshi v2 returns market_positions (per-contract) and event_positions (aggregated).
        # We use market_positions for per-ticker reconciliation.
        raw_positions: list[Any] = data.get("market_positions", [])
        result: list[Position] = []
        now = datetime.now(UTC)
        for p in raw_positions:
            net = int(p.get("position", 0))
            if net == 0:
                continue
            direction = "YES" if net > 0 else "NO"
            result.append(
                Position(
                    id=p.get("market_id", p.get("ticker", "")),
                    market_id=p.get("ticker", p.get("market_id", "")),
                    signal_id="",
                    strategy_name="exchange_reconciliation",
                    strategy_version="0",
                    signal_confidence=0.0,
                    signal_edge=0.0,
                    signal_estimated_prob=0.0,
                    direction=direction,
                    contracts=abs(net),
                    entry_price=0.0,
                    entry_time=now,
                    mode="live",
                    status="open",
                )
            )
        log.info("kalshi.get_positions", count=len(result))
        return result

    async def get_balance(self) -> float:
        """Return current available balance in USD.

        Kalshi returns the balance as an integer number of cents.
        """
        data = await self._get("/portfolio/balance")
        cents: int = data.get("balance", 0)
        balance_usd = cents / 100.0
        log.info("kalshi.get_balance", balance_usd=balance_usd)
        return balance_usd

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> KalshiClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
