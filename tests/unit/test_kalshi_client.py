"""Unit tests for freqpred.markets.kalshi.KalshiClient."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from freqpred.markets.kalshi import KalshiClient, _infer_category
from freqpred.markets.models import Market

BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MARKET_PAYLOAD = {
    "ticker": "KXPRES-25-DEM",
    "event_ticker": "KXPRES-25",
    "title": "Will the Democratic candidate win the 2025 election?",
    "subtitle": "Democratic candidate wins",
    "status": "active",
    "close_time": "2025-11-05T00:00:00Z",
    "yes_bid_dollars": "0.4500",
    "yes_ask_dollars": "0.4700",
    "no_bid_dollars": "0.5300",
    "no_ask_dollars": "0.5500",
    # Use the _fp suffix field names that the Kalshi v2 API returns.
    "volume_24h_fp": 1200,
    "open_interest_fp": 5000,
}

_MARKET_PAYLOAD_2 = {
    "ticker": "KXTECH-25-AI",
    "event_ticker": "KXTECH-25",
    "title": "Will an AI model pass the bar exam?",
    "subtitle": "AI passes bar",
    "status": "active",
    "close_time": "2025-06-01T00:00:00Z",
    "yes_bid_dollars": "0.6200",
    "yes_ask_dollars": "0.6500",
    "no_bid_dollars": "0.3500",
    "no_ask_dollars": "0.3800",
    "volume_24h_fp": 800,
    "open_interest_fp": 2000,
}


def _make_client(api_key: str = "test-key") -> KalshiClient:
    """Create a KalshiClient with no private key (auth disabled)."""
    return KalshiClient(api_key=api_key, base_url=BASE_URL)


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.headers = {}
    return resp


# ---------------------------------------------------------------------------
# _to_market
# ---------------------------------------------------------------------------

class TestToMarket:
    def test_basic_conversion(self) -> None:
        client = _make_client()
        market = client._to_market(_MARKET_PAYLOAD, category="politics")

        assert market.id == "KXPRES-25-DEM"
        assert market.platform == "kalshi"
        assert market.question == "Will the Democratic candidate win the 2025 election?"
        assert market.category == "politics"
        assert market.yes_bid == pytest.approx(0.45)
        assert market.yes_ask == pytest.approx(0.47)
        assert market.mid_price == pytest.approx(0.46)
        assert market.volume_24h == 1200.0
        assert market.open_interest == 5000.0
        assert isinstance(market.close_time, datetime)
        assert market.close_time.tzinfo is not None

    def test_mid_price_calculation(self) -> None:
        client = _make_client()
        raw = {**_MARKET_PAYLOAD, "yes_bid_dollars": "0.30", "yes_ask_dollars": "0.50"}
        market = client._to_market(raw)
        assert market.mid_price == pytest.approx(0.40)

    def test_zero_prices_give_zero_mid(self) -> None:
        client = _make_client()
        raw = {**_MARKET_PAYLOAD, "yes_bid_dollars": "0.00", "yes_ask_dollars": "0.00"}
        market = client._to_market(raw)
        assert market.mid_price == 0.0

    def test_default_category_is_other(self) -> None:
        client = _make_client()
        market = client._to_market(_MARKET_PAYLOAD)
        assert market.category == "other"

    def test_metadata_contains_event_ticker(self) -> None:
        client = _make_client()
        market = client._to_market(_MARKET_PAYLOAD, category="politics")
        assert market.metadata["event_ticker"] == "KXPRES-25"

    def test_missing_prices_default_to_zero(self) -> None:
        client = _make_client()
        raw = {k: v for k, v in _MARKET_PAYLOAD.items() if "dollars" not in k}
        market = client._to_market(raw)
        assert market.yes_bid == 0.0
        assert market.yes_ask == 0.0

    def test_timestamps_are_utc(self) -> None:
        client = _make_client()
        market = client._to_market(_MARKET_PAYLOAD)
        assert market.last_fetched_at.tzinfo is not None
        assert market.price_updated_at.tzinfo is not None
        assert market.metadata_fetched_at.tzinfo is not None


# ---------------------------------------------------------------------------
# _parse_dollar
# ---------------------------------------------------------------------------

class TestParseDollar:
    def test_valid_string(self) -> None:
        assert KalshiClient._parse_dollar("0.5600") == pytest.approx(0.56)

    def test_none_returns_zero(self) -> None:
        assert KalshiClient._parse_dollar(None) == 0.0

    def test_empty_string_returns_zero(self) -> None:
        assert KalshiClient._parse_dollar("") == 0.0

    def test_invalid_string_returns_zero(self) -> None:
        assert KalshiClient._parse_dollar("N/A") == 0.0

    def test_integer_like_string(self) -> None:
        assert KalshiClient._parse_dollar("1") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# list_markets — no category
# ---------------------------------------------------------------------------

class TestListMarketsNoCategory:
    @pytest.mark.asyncio
    async def test_returns_all_open_markets(self) -> None:
        client = _make_client()
        markets_page = {"markets": [_MARKET_PAYLOAD, _MARKET_PAYLOAD_2], "cursor": ""}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(markets_page)
            result = await client.list_markets()

        assert len(result) == 2
        assert result[0].id == "KXPRES-25-DEM"
        assert result[1].id == "KXTECH-25-AI"

    @pytest.mark.asyncio
    async def test_passes_status_open_param(self) -> None:
        client = _make_client()
        markets_page = {"markets": [], "cursor": ""}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(markets_page)
            await client.list_markets()

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs.args[1] if call_kwargs.args else {}
        # Extract params from call
        if hasattr(call_kwargs, 'kwargs'):
            params = call_kwargs.kwargs.get("params", {})
        assert params.get("status") == "open"

    @pytest.mark.asyncio
    async def test_pagination_follows_cursor(self) -> None:
        client = _make_client()
        # First page returns 1000 items with cursor, second returns 1 with empty cursor
        page1 = {"markets": [_MARKET_PAYLOAD] * 1000, "cursor": "abc123"}
        page2 = {"markets": [_MARKET_PAYLOAD_2], "cursor": ""}

        responses = [_mock_response(page1), _mock_response(page2)]

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = responses
            result = await client.list_markets()

        assert len(result) == 1001
        assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_list(self) -> None:
        client = _make_client()

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response({"markets": [], "cursor": ""})
            result = await client.list_markets()

        assert result == []


# ---------------------------------------------------------------------------
# _infer_category
# ---------------------------------------------------------------------------

class TestInferCategory:
    def test_known_prefix_exact(self) -> None:
        assert _infer_category("KXPRES-25-DEM") == "politics"

    def test_known_prefix_sports(self) -> None:
        assert _infer_category("KXNBA-25-LAL") == "sports"

    def test_longest_prefix_match(self) -> None:
        # KXNBA3PT starts with KXNBA → sports
        assert _infer_category("KXNBA3PT-26MAR") == "sports"

    def test_unknown_prefix_returns_other(self) -> None:
        assert _infer_category("UNKNOWN-TICKER") == "other"

    def test_empty_string_returns_other(self) -> None:
        assert _infer_category("") == "other"

    def test_technology_prefix(self) -> None:
        assert _infer_category("KXTECH-25-AI") == "technology"

    def test_economics_prefix(self) -> None:
        assert _infer_category("KXCPI-25-DEC") == "economics"

    def test_case_insensitive_prefix(self) -> None:
        # event_ticker should be uppercased before lookup
        assert _infer_category("kxpres-25-dem") == "politics"


# ---------------------------------------------------------------------------
# list_markets — with category (prefix-inference approach)
# ---------------------------------------------------------------------------

class TestListMarketsWithCategory:
    @pytest.mark.asyncio
    async def test_filters_markets_by_inferred_category(self) -> None:
        """Only markets whose event_ticker maps to the requested category are returned."""
        client = _make_client()
        # _MARKET_PAYLOAD has event_ticker="KXPRES-25" → politics
        # _MARKET_PAYLOAD_2 has event_ticker="KXTECH-25" → technology
        all_markets = {"markets": [_MARKET_PAYLOAD, _MARKET_PAYLOAD_2], "cursor": ""}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(all_markets)
            result = await client.list_markets(category="politics")

        assert len(result) == 1
        assert result[0].id == "KXPRES-25-DEM"
        assert result[0].category == "politics"

    @pytest.mark.asyncio
    async def test_excludes_other_category_markets(self) -> None:
        """Technology markets are excluded when filtering for politics."""
        client = _make_client()
        all_markets = {"markets": [_MARKET_PAYLOAD, _MARKET_PAYLOAD_2], "cursor": ""}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(all_markets)
            result = await client.list_markets(category="technology")

        assert len(result) == 1
        assert result[0].id == "KXTECH-25-AI"
        assert result[0].category == "technology"

    @pytest.mark.asyncio
    async def test_no_matching_markets_returns_empty(self) -> None:
        """Category with no matching prefixes returns empty list."""
        client = _make_client()
        all_markets = {"markets": [_MARKET_PAYLOAD, _MARKET_PAYLOAD_2], "cursor": ""}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(all_markets)
            result = await client.list_markets(category="sports")

        assert result == []

    @pytest.mark.asyncio
    async def test_never_calls_series_endpoint(self) -> None:
        """The new implementation never calls /series — it always uses /markets."""
        client = _make_client()
        all_markets = {"markets": [], "cursor": ""}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(all_markets)
            await client.list_markets(category="politics")

        for call in mock_get.call_args_list:
            url: str = call.args[0]
            assert "/series" not in url


# ---------------------------------------------------------------------------
# get_market
# ---------------------------------------------------------------------------

class TestGetMarket:
    @pytest.mark.asyncio
    async def test_returns_single_market(self) -> None:
        client = _make_client()
        resp_data = {"market": _MARKET_PAYLOAD}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(resp_data)
            market = await client.get_market("KXPRES-25-DEM")

        assert market.id == "KXPRES-25-DEM"
        assert market.yes_bid == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_hits_correct_endpoint(self) -> None:
        client = _make_client()

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response({"market": _MARKET_PAYLOAD})
            await client.get_market("KXPRES-25-DEM")

        call_url = mock_get.call_args.args[0]
        assert call_url.endswith("/markets/KXPRES-25-DEM")

    @pytest.mark.asyncio
    async def test_handles_unwrapped_response(self) -> None:
        """Some Kalshi responses may not wrap in 'market' key."""
        client = _make_client()

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(_MARKET_PAYLOAD)
            market = await client.get_market("KXPRES-25-DEM")

        assert market.id == "KXPRES-25-DEM"


# ---------------------------------------------------------------------------
# get_orderbook
# ---------------------------------------------------------------------------

class TestGetOrderbook:
    @pytest.mark.asyncio
    async def test_parses_yes_bid_from_yes_dollars(self) -> None:
        client = _make_client()
        ob_data = {
            "orderbook": {
                "yes_dollars": [["0.45", "100"], ["0.44", "50"]],
                "no_dollars": [["0.55", "80"]],
            }
        }

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(ob_data)
            result = await client.get_orderbook("KXPRES-25-DEM")

        assert result["yes_bid"] == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_computes_yes_ask_from_no_bid(self) -> None:
        client = _make_client()
        ob_data = {
            "orderbook": {
                "yes_dollars": [["0.45", "100"]],
                "no_dollars": [["0.55", "80"], ["0.54", "40"]],
            }
        }

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(ob_data)
            result = await client.get_orderbook("KXPRES-25-DEM")

        # yes_ask = 1 - best_no_bid = 1 - 0.55 = 0.45
        assert result["yes_ask"] == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_empty_orderbook_returns_defaults(self) -> None:
        client = _make_client()
        ob_data = {"orderbook": {"yes_dollars": [], "no_dollars": []}}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(ob_data)
            result = await client.get_orderbook("KXPRES-25-DEM")

        assert result["yes_bid"] == 0.0
        assert result["yes_ask"] == 1.0

    @pytest.mark.asyncio
    async def test_handles_dict_level_format(self) -> None:
        """Handles PriceLevelDollars as dict instead of list."""
        client = _make_client()
        ob_data = {
            "orderbook": {
                "yes_dollars": [{"price": "0.42", "count": "50"}],
                "no_dollars": [{"price": "0.58", "count": "30"}],
            }
        }

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _mock_response(ob_data)
            result = await client.get_orderbook("KXPRES-25-DEM")

        assert result["yes_bid"] == pytest.approx(0.42)
        assert result["yes_ask"] == pytest.approx(0.42)  # 1 - 0.58


# ---------------------------------------------------------------------------
# Rate limiting / retry
# ---------------------------------------------------------------------------

class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_retries_on_429(self) -> None:
        client = _make_client()
        # First call returns 429, second returns 200
        resp_429 = MagicMock(spec=httpx.Response)
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0.01"}
        resp_429.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
            "429", request=MagicMock(), response=resp_429
        ))

        resp_200 = _mock_response({"markets": [], "cursor": ""})

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [resp_429, resp_200]
            result = await client.list_markets()

        assert mock_get.call_count == 2
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        client = _make_client()
        resp_429 = MagicMock(spec=httpx.Response)
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0.001"}

        with patch.object(client._http, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = resp_429
            with pytest.raises(RuntimeError, match="failed after"):
                await client.list_markets()


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------

class TestAuthHeaders:
    def test_no_key_returns_empty_headers(self) -> None:
        client = _make_client(api_key="")
        headers = client._make_auth_headers("GET", "/markets")
        assert headers == {}

    def test_no_private_key_returns_empty_headers(self) -> None:
        client = _make_client(api_key="some-key")
        # No private_key_path configured → _private_key is None
        headers = client._make_auth_headers("GET", "/markets")
        assert headers == {}

    def test_with_private_key_returns_three_headers(self, tmp_path) -> None:
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        # Generate a throwaway RSA key for testing
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
        key_pem = private_key.private_bytes(
            Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
        )
        key_file = tmp_path / "test.key"
        key_file.write_bytes(key_pem)

        client = KalshiClient(
            api_key="my-api-key",
            base_url=BASE_URL,
            private_key_path=str(key_file),
        )
        headers = client._make_auth_headers("GET", "/markets")

        assert "KALSHI-ACCESS-KEY" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert headers["KALSHI-ACCESS-KEY"] == "my-api-key"


# ---------------------------------------------------------------------------
# get_active_markets (delegates to list_markets)
# ---------------------------------------------------------------------------

class TestGetActiveMarkets:
    @pytest.mark.asyncio
    async def test_delegates_to_list_markets(self) -> None:
        client = _make_client()

        with patch.object(client, "list_markets", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []
            await client.get_active_markets()

        mock_list.assert_called_once_with()
