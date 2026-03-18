"""Unit tests for freqpred.markets.repository."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from freqpred.markets.models import Market
from freqpred.markets.repository import upsert_market, upsert_markets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_market(**overrides) -> Market:
    now = datetime.now(UTC)
    defaults = dict(
        id="KXPRES-25-DEM",
        platform="kalshi",
        question="Will the Dem candidate win?",
        category="politics",
        close_time=now + timedelta(days=30),
        yes_bid=0.45,
        yes_ask=0.47,
        mid_price=0.46,
        volume_24h=1000.0,
        open_interest=5000.0,
        last_fetched_at=now,
        price_updated_at=now,
        metadata_fetched_at=now,
        metadata={"event_ticker": "KXPRES-25"},
    )
    return Market(**{**defaults, **overrides})


def _make_session() -> AsyncMock:
    """Build a mock AsyncSession."""
    session = AsyncMock()
    session.execute.return_value = MagicMock()
    return session


# ---------------------------------------------------------------------------
# upsert_market
# ---------------------------------------------------------------------------

class TestUpsertMarketNew:
    @pytest.mark.asyncio
    async def test_executes_single_upsert_statement(self) -> None:
        """No pre-SELECT; a single INSERT ON CONFLICT DO UPDATE is issued."""
        session = _make_session()
        await upsert_market(session, _make_market())
        assert session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_does_not_commit(self) -> None:
        """upsert_market does not commit — that is caller's responsibility."""
        session = _make_session()
        await upsert_market(session, _make_market())
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_price_updated_at_in_insert_values_is_now(self) -> None:
        """The INSERT row always carries now() for price_updated_at.

        Preservation of the old timestamp when price is unchanged is handled
        by the SQL CASE expression in the ON CONFLICT clause, not in Python.
        This test verifies the Python row builder passes the expected timestamp.
        """
        from freqpred.markets import repository as repo_module

        fixed_now = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
        captured_rows: list[list[dict]] = []

        original_pg_insert = repo_module.pg_insert

        def capturing_insert(table):
            stmt = original_pg_insert(table)
            original_values = stmt.values

            def values_capture(rows):
                captured_rows.append(rows)
                return original_values(rows)

            stmt.values = values_capture
            return stmt

        with (
            patch.object(repo_module, "pg_insert", side_effect=capturing_insert),
            patch("freqpred.markets.repository.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fixed_now
            await upsert_market(session=_make_session(), market=_make_market())

        assert len(captured_rows) == 1
        assert captured_rows[0][0]["price_updated_at"] == fixed_now


# ---------------------------------------------------------------------------
# upsert_markets (batch)
# ---------------------------------------------------------------------------

class TestUpsertMarkets:
    @pytest.mark.asyncio
    async def test_returns_count(self) -> None:
        session = _make_session()
        markets = [_make_market(id=f"MKT-{i}") for i in range(3)]

        count = await upsert_markets(session, markets)

        assert count == 3

    @pytest.mark.asyncio
    async def test_commits_once(self) -> None:
        session = _make_session()
        markets = [_make_market(id=f"MKT-{i}") for i in range(3)]

        await upsert_markets(session, markets)

        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_batches_large_input(self) -> None:
        """Markets exceeding _BATCH_SIZE should produce multiple execute calls."""
        from freqpred.markets.repository import _BATCH_SIZE

        session = _make_session()
        markets = [_make_market(id=f"MKT-{i}") for i in range(_BATCH_SIZE + 1)]

        await upsert_markets(session, markets)

        assert session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero_without_db_calls(self) -> None:
        session = _make_session()

        count = await upsert_markets(session, [])

        assert count == 0
        session.execute.assert_not_called()
        session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------

class TestKalshiMarketSchema:
    def test_parses_valid_payload(self) -> None:
        from freqpred.markets.models import KalshiMarketSchema

        schema = KalshiMarketSchema.model_validate({
            "ticker": "KXPRES-25-DEM",
            "event_ticker": "KXPRES-25",
            "title": "Will the Dem candidate win?",
            "status": "active",
            "close_time": "2025-11-05T00:00:00Z",
            "yes_bid_dollars": "0.4500",
            "yes_ask_dollars": "0.4700",
            "no_bid_dollars": "0.5300",
            "no_ask_dollars": "0.5500",
            "volume_24h": 1200,
            "open_interest": 5000,
        })

        assert schema.ticker == "KXPRES-25-DEM"
        assert schema.yes_bid == pytest.approx(0.45)
        assert schema.yes_ask == pytest.approx(0.47)
        assert schema.mid_price == pytest.approx(0.46)

    def test_missing_price_fields_default_to_zero(self) -> None:
        from freqpred.markets.models import KalshiMarketSchema

        schema = KalshiMarketSchema.model_validate({
            "ticker": "X",
            "event_ticker": "Y",
            "title": "Test",
            "status": "active",
            "close_time": "2025-11-05T00:00:00Z",
        })

        assert schema.yes_bid == 0.0
        assert schema.yes_ask == 0.0
        assert schema.mid_price == 0.0

    def test_numeric_price_coerced_to_string(self) -> None:
        from freqpred.markets.models import KalshiMarketSchema

        schema = KalshiMarketSchema.model_validate({
            "ticker": "X",
            "event_ticker": "Y",
            "title": "Test",
            "status": "active",
            "close_time": "2025-11-05T00:00:00Z",
            "yes_bid_dollars": 0.45,   # numeric, not string
            "yes_ask_dollars": 0.47,
        })

        assert schema.yes_bid == pytest.approx(0.45)

    def test_none_price_coerced_to_zero(self) -> None:
        from freqpred.markets.models import KalshiMarketSchema

        schema = KalshiMarketSchema.model_validate({
            "ticker": "X",
            "event_ticker": "Y",
            "title": "Test",
            "status": "active",
            "close_time": "2025-11-05T00:00:00Z",
            "yes_bid_dollars": None,
        })

        assert schema.yes_bid == 0.0


class TestKalshiMarketsResponse:
    def test_parses_markets_list(self) -> None:
        from freqpred.markets.models import KalshiMarketsResponse

        resp = KalshiMarketsResponse.model_validate({
            "markets": [
                {
                    "ticker": "MKT-1",
                    "event_ticker": "EVT-1",
                    "title": "Test",
                    "status": "active",
                    "close_time": "2025-11-05T00:00:00Z",
                }
            ],
            "cursor": "abc123",
        })

        assert len(resp.markets) == 1
        assert resp.cursor == "abc123"

    def test_empty_markets_defaults(self) -> None:
        from freqpred.markets.models import KalshiMarketsResponse

        resp = KalshiMarketsResponse.model_validate({})
        assert resp.markets == []
        assert resp.cursor == ""


class TestKalshiSeriesResponse:
    def test_parses_series_list(self) -> None:
        from freqpred.markets.models import KalshiSeriesResponse

        resp = KalshiSeriesResponse.model_validate({
            "series": [
                {"ticker": "KXPRES", "category": "Politics", "title": "Presidential race"},
                {"ticker": "KXSEN", "category": "Politics", "title": "Senate"},
            ]
        })

        assert len(resp.series) == 2
        assert resp.series[0].ticker == "KXPRES"
        assert resp.series[0].category == "Politics"

    def test_empty_series_defaults(self) -> None:
        from freqpred.markets.models import KalshiSeriesResponse

        resp = KalshiSeriesResponse.model_validate({})
        assert resp.series == []
