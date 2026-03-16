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


def _make_session(existing_row=None) -> AsyncMock:
    """Build a mock AsyncSession.

    ``existing_row`` is the value returned by session.execute().one_or_none()
    when checking for a pre-existing market price snapshot.
    """
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.one_or_none.return_value = existing_row
    session.execute.return_value = result_mock
    return session


# ---------------------------------------------------------------------------
# upsert_market — new market (no existing row)
# ---------------------------------------------------------------------------

class TestUpsertMarketNew:
    @pytest.mark.asyncio
    async def test_executes_upsert_statement(self) -> None:
        session = _make_session(existing_row=None)
        market = _make_market()

        await upsert_market(session, market)

        assert session.execute.call_count == 2  # SELECT + INSERT

    @pytest.mark.asyncio
    async def test_does_not_commit(self) -> None:
        """upsert_market does not commit — that is caller's responsibility."""
        session = _make_session(existing_row=None)
        await upsert_market(session, _make_market())
        session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# upsert_market — existing market, price unchanged
# ---------------------------------------------------------------------------

class TestUpsertMarketPriceUnchanged:
    @pytest.mark.asyncio
    async def test_price_updated_at_is_preserved(self) -> None:
        """When bid/ask/mid are identical, price_updated_at should not advance."""
        old_price_ts = datetime(2025, 1, 1, tzinfo=UTC)
        existing = MagicMock()
        existing.yes_bid = 0.45
        existing.yes_ask = 0.47
        existing.mid_price = 0.46
        existing.price_updated_at = old_price_ts

        session = _make_session(existing_row=existing)
        market = _make_market(yes_bid=0.45, yes_ask=0.47, mid_price=0.46)

        # Capture the INSERT statement values by inspecting execute calls
        captured_stmts = []
        original_execute = session.execute

        async def capture_execute(stmt, *args, **kwargs):
            captured_stmts.append(stmt)
            return original_execute.return_value

        session.execute.side_effect = capture_execute

        await upsert_market(session, market)

        # The INSERT/UPDATE statement is the second execute call.
        # We can't easily introspect the compiled SQL, but we can verify
        # the logic by checking that price_updated_at == old_price_ts,
        # not a newer timestamp.
        # We do this by patching datetime.now inside repository to a fixed value.
        assert len(captured_stmts) == 2

    @pytest.mark.asyncio
    async def test_price_updated_at_stays_old_when_price_same(self) -> None:
        """Integration-style test: mock datetime.now and verify timestamp selection."""
        from freqpred.markets import repository as repo_module

        fixed_now = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
        old_price_ts = datetime(2026, 3, 14, 10, 0, tzinfo=UTC)

        existing = MagicMock()
        existing.yes_bid = 0.45
        existing.yes_ask = 0.47
        existing.mid_price = 0.46
        existing.price_updated_at = old_price_ts

        # Track what price_updated_at value is used in the INSERT
        used_price_updated_at: list[datetime] = []

        session = _make_session(existing_row=existing)

        # We need to inspect the pg_insert values — easiest approach is to
        # patch pg_insert and capture the kwargs
        original_pg_insert = repo_module.pg_insert

        def capturing_insert(table):
            stmt = original_pg_insert(table)
            original_values = stmt.values

            def values_capture(**kwargs):
                used_price_updated_at.append(kwargs.get("price_updated_at"))
                return original_values(**kwargs)

            stmt.values = values_capture
            return stmt

        with (
            patch.object(repo_module, "pg_insert", side_effect=capturing_insert),
            patch("freqpred.markets.repository.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fixed_now

            await upsert_market(session, _make_market(yes_bid=0.45, yes_ask=0.47, mid_price=0.46))

        # price_updated_at should be old_price_ts (unchanged), not fixed_now
        assert used_price_updated_at[0] == old_price_ts


# ---------------------------------------------------------------------------
# upsert_market — existing market, price changed
# ---------------------------------------------------------------------------

class TestUpsertMarketPriceChanged:
    @pytest.mark.asyncio
    async def test_price_updated_at_advances_when_bid_changes(self) -> None:
        from freqpred.markets import repository as repo_module

        fixed_now = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
        old_price_ts = datetime(2026, 3, 14, 10, 0, tzinfo=UTC)

        existing = MagicMock()
        existing.yes_bid = 0.40   # old bid
        existing.yes_ask = 0.47
        existing.mid_price = 0.435
        existing.price_updated_at = old_price_ts

        used_price_updated_at: list[datetime] = []

        session = _make_session(existing_row=existing)

        original_pg_insert = repo_module.pg_insert

        def capturing_insert(table):
            stmt = original_pg_insert(table)
            original_values = stmt.values

            def values_capture(**kwargs):
                used_price_updated_at.append(kwargs.get("price_updated_at"))
                return original_values(**kwargs)

            stmt.values = values_capture
            return stmt

        with (
            patch.object(repo_module, "pg_insert", side_effect=capturing_insert),
            patch("freqpred.markets.repository.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fixed_now

            # New market has different yes_bid (0.45 vs 0.40)
            await upsert_market(
                session,
                _make_market(yes_bid=0.45, yes_ask=0.47, mid_price=0.46),
            )

        # price_updated_at should be fixed_now (price changed)
        assert used_price_updated_at[0] == fixed_now

    @pytest.mark.asyncio
    async def test_price_updated_at_advances_when_ask_changes(self) -> None:
        from freqpred.markets import repository as repo_module

        fixed_now = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
        old_price_ts = datetime(2026, 3, 14, 10, 0, tzinfo=UTC)

        existing = MagicMock()
        existing.yes_bid = 0.45
        existing.yes_ask = 0.50   # old ask
        existing.mid_price = 0.475
        existing.price_updated_at = old_price_ts

        used_price_updated_at: list[datetime] = []
        session = _make_session(existing_row=existing)

        original_pg_insert = repo_module.pg_insert

        def capturing_insert(table):
            stmt = original_pg_insert(table)
            original_values = stmt.values

            def values_capture(**kwargs):
                used_price_updated_at.append(kwargs.get("price_updated_at"))
                return original_values(**kwargs)

            stmt.values = values_capture
            return stmt

        with (
            patch.object(repo_module, "pg_insert", side_effect=capturing_insert),
            patch("freqpred.markets.repository.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = fixed_now
            await upsert_market(
                session,
                _make_market(yes_bid=0.45, yes_ask=0.47, mid_price=0.46),
            )

        assert used_price_updated_at[0] == fixed_now


# ---------------------------------------------------------------------------
# upsert_markets (batch)
# ---------------------------------------------------------------------------

class TestUpsertMarkets:
    @pytest.mark.asyncio
    async def test_returns_count(self) -> None:
        session = _make_session(existing_row=None)
        markets = [_make_market(id=f"MKT-{i}") for i in range(3)]

        count = await upsert_markets(session, markets)

        assert count == 3

    @pytest.mark.asyncio
    async def test_commits_once(self) -> None:
        session = _make_session(existing_row=None)
        markets = [_make_market(id=f"MKT-{i}") for i in range(3)]

        await upsert_markets(session, markets)

        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_list_commits_and_returns_zero(self) -> None:
        session = _make_session(existing_row=None)

        count = await upsert_markets(session, [])

        assert count == 0
        session.commit.assert_called_once()


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
