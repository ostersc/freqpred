"""Kalshi prediction market API adapter."""
from __future__ import annotations

import asyncio
import base64
import time
import uuid
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
    KalshiEventSchema,
    KalshiEventsResponse,
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

_MAX_RETRIES = 3
_PAGE_SIZE = 1000
_EVENTS_PAGE_SIZE = 200  # Kalshi /events max page size


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

    async def _delete(self, path: str) -> Any:
        """Authenticated DELETE request to Kalshi API.

        Uses the same RSA-PSS signing as _get() — method="DELETE".
        Raises KalshiAPIError on non-2xx response.
        """
        url = self._base_url + path
        headers = self._make_auth_headers("DELETE", self._base_path + path)
        resp = await self._http.delete(url, headers=headers)
        if resp.status_code >= 400:
            raise KalshiAPIError(resp.status_code, resp.text)
        return resp.json()

    async def _paginate_events(self) -> list[KalshiEventSchema]:
        """Fetch all open events with nested markets from GET /events.

        Uses ``status=open`` and ``with_nested_markets=true`` so each event
        includes the full list of market objects. Category and series_ticker
        come from the event; market price/volume data come from the nested
        market objects.
        """
        results: list[KalshiEventSchema] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {
                "status": "open",
                "with_nested_markets": "true",
                "limit": _EVENTS_PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor

            data = await self._get("/events", params=params)
            envelope = KalshiEventsResponse.model_validate(data)
            results.extend(envelope.events)

            cursor = envelope.cursor or None
            if not cursor or len(envelope.events) < _EVENTS_PAGE_SIZE:
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

    def _schema_to_market(
        self,
        schema: Any,
        category: str = "other",
        series_ticker: str | None = None,
        settlement_sources: list[dict[str, str]] | None = None,
    ) -> Market:
        """Convert a validated KalshiMarketSchema to the Market domain object."""
        from freqpred.markets.models import KalshiMarketSchema

        now = datetime.now(UTC)
        s: KalshiMarketSchema = schema
        close_time = datetime.fromisoformat(s.close_time.replace("Z", "+00:00"))
        open_time = (
            datetime.fromisoformat(s.open_time.replace("Z", "+00:00"))
            if s.open_time
            else None
        )
        sources = settlement_sources or []

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
            settlement_value=s.settlement_value,
            close_time=close_time,
            yes_bid=s.yes_bid,
            yes_ask=s.yes_ask,
            mid_price=s.mid_price,
            last_price=s.last_price,
            volume_24h=float(s.volume_24h),
            volume_total=float(s.volume_total),
            open_interest=float(s.open_interest),
            yes_bid_size=float(s.yes_bid_size),
            yes_ask_size=float(s.yes_ask_size),
            last_fetched_at=now,
            price_updated_at=now,
            metadata_fetched_at=now,
            open_time=open_time,
            series_ticker=series_ticker,
            settlement_sources=sources,
            metadata={
                "event_ticker": s.event_ticker,
                "series_ticker": series_ticker,
                "subtitle": s.subtitle,
                "status": s.status,
                "settlement_sources": sources,
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
        """Fetch all open markets via GET /events with nested markets.

        Category and series_ticker come directly from the Kalshi API (exact
        strings, e.g. "Elections", "Sports"). Optionally filter by category;
        comparison is exact (case-sensitive) to match Kalshi's values.
        """
        events = await self._paginate_events()
        result: list[Market] = []
        for ev in events:
            cat = ev.category
            series = ev.series_ticker or None
            if category is not None and cat != category:
                continue
            sources = [s.model_dump() for s in ev.settlement_sources]
            for market_schema in ev.markets:
                result.append(
                    self._schema_to_market(market_schema, cat, series, sources)
                )

        total = sum(len(ev.markets) for ev in events)
        log.info(
            "kalshi_list_markets",
            category=category,
            total_fetched=total,
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
        """Look up a single market via the settled list endpoint.

        Kalshi's GET /markets/{ticker} returns 404 for markets that have been
        purged from the live API after finalization.  The settled list endpoint
        retains them and supports filtering by ``tickers=``.  Returns None if
        the market is not found in the settled list.
        """
        results = await self.get_markets_from_settled([market_id])
        return results[0] if results else None

    async def get_markets_from_settled(self, market_ids: list[str]) -> list[Market]:
        """Batch-fetch up to 200 markets from the settled list endpoint.

        Pass multiple tickers as a comma-separated string.  Returns only the
        markets that were found; silently skips any that are absent.
        """
        return await self._get_markets_by_tickers_chunked(market_ids, status="settled")

    async def get_markets_by_tickers(self, market_ids: list[str]) -> list[Market]:
        """Batch-fetch markets of any status via GET /markets?tickers=.

        Chunks requests into groups of up to 200 tickers (Kalshi's max).
        Returns only the markets that were found; silently skips any that
        are absent (true 404s — caller diffs requested vs. returned tickers).
        """
        return await self._get_markets_by_tickers_chunked(market_ids, status=None)

    async def _get_markets_by_tickers_chunked(
        self, market_ids: list[str], *, status: str | None
    ) -> list[Market]:
        if not market_ids:
            return []
        chunk_size = 200
        results: list[Market] = []
        for i in range(0, len(market_ids), chunk_size):
            chunk = market_ids[i : i + chunk_size]
            params: dict[str, Any] = {
                "tickers": ",".join(chunk),
                "limit": len(chunk),
            }
            if status is not None:
                params["status"] = status
            try:
                data = await self._get("/markets", params=params)
            except KalshiAPIError:
                continue
            results.extend(self._to_market(m) for m in data.get("markets", []))
        return results

    async def get_events_by_tickers(
        self, event_tickers: list[str]
    ) -> list[KalshiEventSchema]:
        """Batch-fetch events via GET /events?tickers=, chunked at 200 per call.

        Used for targeted backfills (e.g. re-fetching settlement_sources for a
        known subset of events) without re-paginating the full open-events list.
        Silently skips chunks that error.
        """
        if not event_tickers:
            return []
        chunk_size = 200
        results: list[KalshiEventSchema] = []
        for i in range(0, len(event_tickers), chunk_size):
            chunk = event_tickers[i : i + chunk_size]
            try:
                data = await self._get("/events", params={"tickers": ",".join(chunk)})
            except KalshiAPIError:
                continue
            envelope = KalshiEventsResponse.model_validate(data)
            results.extend(envelope.events)
        return results

    async def get_series_settled_history(self, series_ticker: str) -> list[dict[str, Any]]:
        """Return all settled markets for a series, paginated.

        Each dict has at minimum: ticker, result, yes_sub_title.
        """
        results: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {
                "status": "settled",
                "series_ticker": series_ticker,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = await self._get("/markets", params=params)
            except KalshiAPIError as exc:
                log.warning(
                    "kalshi.series_history.error",
                    series_ticker=series_ticker,
                    status=exc.status_code,
                )
                break

            markets: list[dict[str, Any]] = data.get("markets", [])
            results.extend(markets)

            cursor = data.get("cursor") or None
            if not cursor or len(markets) < 200:
                break

        return results

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

    # CreateOrderV2Request.time_in_force only accepts these three values;
    # "GTC" is freqpred's internal shorthand for good_till_canceled.
    _TIME_IN_FORCE_MAP = {
        "GTC": "good_till_canceled",
        "fill_or_kill": "fill_or_kill",
        "immediate_or_cancel": "immediate_or_cancel",
    }

    @staticmethod
    def _book_side_and_price(order: Order) -> tuple[str, float]:
        """Map (direction, action) to (book_side, yes-leg price).

        CreateOrderV2Request quotes everything on the YES leg: book_side="bid"
        means buy YES, "ask" means sell YES. Buying/selling NO is expressed as
        the opposite YES-leg action at the complementary price (1 - price).
        """
        if order.direction == "YES":
            book_side = "bid" if order.action == "buy" else "ask"
            yes_price = order.price
        else:
            book_side = "ask" if order.action == "buy" else "bid"
            yes_price = 1.0 - order.price
        return book_side, yes_price

    async def place_order(self, order: Order) -> Order:
        """Submit a limit order to Kalshi via the V2 /portfolio/events/orders path."""
        try:
            time_in_force = self._TIME_IN_FORCE_MAP[order.time_in_force]
        except KeyError:
            raise ValueError(
                f"Unsupported time_in_force {order.time_in_force!r}; "
                f"expected one of {sorted(self._TIME_IN_FORCE_MAP)}"
            ) from None
        book_side, yes_price = self._book_side_and_price(order)
        client_order_id = order.client_order_id or str(uuid.uuid4())
        path = "/portfolio/events/orders"
        body: dict[str, Any] = {
            "ticker": order.market_id,
            "client_order_id": client_order_id,
            "side": book_side,
            "type": "limit",
            "count": f"{order.contracts:.2f}",
            "price": f"{yes_price:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
        }
        data = await self._post(path, body)
        exchange_order = data.get("order", data)
        return self._order_from_exchange_payload(order, exchange_order)

    async def get_order(self, order_id: str) -> Order:
        """Fetch the exchange-confirmed state of a single order.

        Returns an Order with status, requested_count, filled_yes/no_count,
        remaining_count, fee_usd, and timestamps populated from Kalshi's
        per-order response.  The returned Order mirrors the original side and
        price but the freqpred-level ``contracts`` field reflects the total
        filled across both sides (yes + no).
        """
        data = await self._get(f"/portfolio/orders/{order_id}")
        exchange_order = data.get("order", data)
        return self._order_from_exchange_payload(None, exchange_order)

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel a resting order on the exchange.

        DELETE /portfolio/events/orders/{order_id} returns CancelOrderV2Response
        — {order_id, client_order_id, reduced_by, ts_ms} — not a full order
        object, so status/price/fees can't be derived from it directly. Instead,
        follow up with GET /portfolio/orders/{order_id} for the authoritative
        terminal state (status="canceled" with whatever fill_count_fp landed
        before cancellation), which map_order_to_status needs.

        The immediately-following GET can 404 on a brief propagation lag
        (observed against demo) even though the DELETE already succeeded, so
        retry a few times with a short backoff before giving up.
        """
        delete_data = await self._delete(f"/portfolio/events/orders/{order_id}")
        log.info(
            "kalshi.cancel_order",
            order_id=order_id,
            client_order_id=delete_data.get("client_order_id"),
            reduced_by=delete_data.get("reduced_by"),
        )
        last_exc: KalshiAPIError | None = None
        for attempt in range(3):
            if attempt > 0:
                await asyncio.sleep(0.5 * attempt)
            try:
                return await self.get_order(order_id)
            except KalshiAPIError as exc:
                if exc.status_code != 404:
                    raise
                last_exc = exc
                log.info(
                    "kalshi.cancel_order.get_order_404_retry",
                    order_id=order_id,
                    attempt=attempt,
                )
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _parse_kalshi_ts(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _fp_int(value: Any) -> int | None:
        """Parse a Kalshi fixed-point contract-count string (e.g. "10.00") to int."""
        if value is None:
            return None
        return round(float(value))

    @staticmethod
    def _dollars(value: Any) -> float | None:
        """Parse a Kalshi dollar-denominated string field (e.g. "0.5600") to float."""
        if value is None:
            return None
        return float(value)

    def _order_from_exchange_payload(
        self,
        request: Order | None,
        exchange_order: dict[str, Any],
    ) -> Order:
        """Build an Order from a Kalshi order response payload.

        Used by place_order and get_order to share parsing. ``request`` is the
        originating Order (place_order only) — get_order passes None and the
        returned Order reflects pure exchange state.

        Two response shapes feed this:
        - CreateOrderV2Response (place_order): order_id, client_order_id,
          fill_count, remaining_count (fp strings, no _fp suffix),
          average_fill_price, average_fee_paid (dollar strings), ts_ms. No
          status/side/ticker — these come from ``request``.
        - GetOrderResponse.order (get_order, and cancel_order via get_order):
          full order object with fill_count_fp/remaining_count_fp/
          initial_count_fp, yes_price_dollars/no_price_dollars,
          taker_fees_dollars/maker_fees_dollars (already totals, not
          per-contract), outcome_side/book_side (canonical) plus deprecated
          side/action (yes/no, buy/sell), status, created_time/last_update_time.
        """
        filled_total = self._fp_int(
            exchange_order.get("fill_count_fp", exchange_order.get("fill_count"))
        ) or 0
        remaining_int = self._fp_int(
            exchange_order.get("remaining_count_fp", exchange_order.get("remaining_count"))
        )
        initial_int = self._fp_int(exchange_order.get("initial_count_fp"))

        maker_fees = self._dollars(exchange_order.get("maker_fees_dollars"))
        taker_fees = self._dollars(exchange_order.get("taker_fees_dollars"))
        if maker_fees is not None or taker_fees is not None:
            fee_usd = (maker_fees or 0.0) + (taker_fees or 0.0)
        else:
            avg_fee = self._dollars(exchange_order.get("average_fee_paid"))
            fee_usd = (avg_fee or 0.0) * filled_total

        if request is not None:
            direction = request.direction
            mode = request.mode
            order_id = request.id
            avg_fill_yes_price = self._dollars(exchange_order.get("average_fill_price"))
            if avg_fill_yes_price is not None:
                price = avg_fill_yes_price if direction == "YES" else 1.0 - avg_fill_yes_price
            else:
                price = request.price
            requested_int = (
                filled_total + remaining_int if remaining_int is not None else request.contracts
            )
        else:
            outcome_side = (
                exchange_order.get("outcome_side") or exchange_order.get("side") or ""
            ).lower()
            direction = "YES" if outcome_side == "yes" else "NO"
            mode = "live"
            order_id = None
            yes_price_dollars = self._dollars(exchange_order.get("yes_price_dollars"))
            no_price_dollars = self._dollars(exchange_order.get("no_price_dollars"))
            if direction == "YES" and yes_price_dollars is not None:
                price = yes_price_dollars
            elif direction == "NO" and no_price_dollars is not None:
                price = no_price_dollars
            else:
                price = 0.0
            requested_int = initial_int if initial_int is not None else (
                filled_total + (remaining_int or 0)
            )

        original_contracts = requested_int or 0
        # For partial fills, ``contracts`` reflects what actually filled so
        # downstream sizing/accounting reads the exchange truth.
        effective_contracts = filled_total if filled_total > 0 else original_contracts

        filled_yes = filled_total if direction == "YES" else None
        filled_no = filled_total if direction == "NO" else None

        status_str = exchange_order.get("status")
        if not status_str:
            # CreateOrderV2Response carries no status field — derive one.
            status_str = (
                "executed" if (remaining_int == 0 and filled_total > 0) else "resting"
            )

        created_time = self._parse_kalshi_ts(exchange_order.get("created_time"))
        if created_time is None and exchange_order.get("ts_ms") is not None:
            created_time = datetime.fromtimestamp(
                int(exchange_order["ts_ms"]) / 1000, tz=UTC
            )

        log.info(
            "kalshi.order_payload",
            exchange_order_id=exchange_order.get("order_id"),
            direction=direction,
            requested=requested_int,
            filled_total=filled_total,
            remaining=remaining_int,
            status=status_str,
            fee_usd=fee_usd,
        )

        return Order(
            market_id=exchange_order.get("ticker", request.market_id if request else ""),
            direction=direction,
            contracts=effective_contracts,
            price=price,
            mode=mode,
            id=order_id,
            exchange_order_id=exchange_order.get("order_id"),
            status=status_str,
            fee_usd=fee_usd,
            requested_count=requested_int,
            filled_yes_count=filled_yes,
            filled_no_count=filled_no,
            remaining_count=remaining_int,
            created_time=created_time,
            last_update_time=self._parse_kalshi_ts(exchange_order.get("last_update_time"))
            or created_time,
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

        Prefers ``balance_dollars`` (centi-cent precision, added May 2026) and
        falls back to legacy ``balance`` (integer cents) for older responses.
        """
        data = await self._get("/portfolio/balance")
        if "balance_dollars" in data:
            balance_usd = float(data["balance_dollars"])
        else:
            cents: int = data.get("balance", 0)
            balance_usd = cents / 100.0
        log.info("kalshi.get_balance", balance_usd=balance_usd)
        return balance_usd

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account_limits(self) -> dict[str, Any]:
        """Return raw account limits payload from GET /account/limits."""
        return await self._get("/account/limits")

    async def upgrade_api_tier(self) -> dict[str, Any]:
        """Promote account to the Advanced API tier via POST /account/api_usage_level/upgrade.

        Returns an empty dict on 201 responses with no body.
        """
        path = "/account/api_usage_level/upgrade"
        url = self._base_url + path
        headers = self._make_auth_headers("POST", self._base_path + path)
        headers["Content-Type"] = "application/json"
        resp = await self._http.post(url, json={}, headers=headers)
        if resp.status_code >= 400:
            raise KalshiAPIError(resp.status_code, resp.text)
        return resp.json() if resp.content else {}

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
