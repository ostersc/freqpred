"""Unit tests for freqpred.markets.watcher."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.markets.kalshi import KalshiClient
from freqpred.markets.watcher import (
    PRICE_MOVE_THRESHOLD,
    MarketWatcher,
    is_stale,
    price_moved,
)


# ---------------------------------------------------------------------------
# is_stale
# ---------------------------------------------------------------------------


class TestIsStale:
    def test_not_stale_when_recently_fetched(self) -> None:
        last_fetched = datetime.now(UTC) - timedelta(seconds=100)
        assert is_stale(last_fetched, polling_interval=300) is False

    def test_not_stale_just_inside_boundary(self) -> None:
        """Just inside 3× polling_interval should not be stale."""
        last_fetched = datetime.now(UTC) - timedelta(seconds=899)
        assert is_stale(last_fetched, polling_interval=300) is False

    def test_stale_when_one_second_past_boundary(self) -> None:
        last_fetched = datetime.now(UTC) - timedelta(seconds=901)
        assert is_stale(last_fetched, polling_interval=300) is True

    def test_stale_for_much_older_timestamp(self) -> None:
        last_fetched = datetime.now(UTC) - timedelta(hours=2)
        assert is_stale(last_fetched, polling_interval=300) is True

    def test_shorter_polling_interval_tightens_threshold(self) -> None:
        """With polling_interval=60, stale threshold is 180 seconds."""
        last_fetched = datetime.now(UTC) - timedelta(seconds=181)
        assert is_stale(last_fetched, polling_interval=60) is True

    def test_shorter_polling_interval_not_stale_inside_threshold(self) -> None:
        last_fetched = datetime.now(UTC) - timedelta(seconds=100)
        assert is_stale(last_fetched, polling_interval=60) is False


# ---------------------------------------------------------------------------
# price_moved
# ---------------------------------------------------------------------------


class TestPriceMoved:
    def test_no_move_returns_false(self) -> None:
        assert price_moved(0.50, 0.50) is False

    def test_small_move_below_threshold_returns_false(self) -> None:
        assert price_moved(0.52, 0.50) is False  # 0.02 < 0.05

    def test_move_exactly_at_threshold_returns_false(self) -> None:
        """Exactly equal to threshold is NOT a trigger (strictly greater than)."""
        assert price_moved(0.549, 0.50) is False  # 0.049 < 0.05

    def test_move_above_threshold_returns_true(self) -> None:
        assert price_moved(0.56, 0.50) is True  # 0.06 > 0.05

    def test_negative_move_above_threshold_returns_true(self) -> None:
        assert price_moved(0.44, 0.50) is True  # abs(-0.06) > 0.05

    def test_negative_move_below_threshold_returns_false(self) -> None:
        assert price_moved(0.47, 0.50) is False  # abs(-0.03) < 0.05

    def test_custom_threshold_respected(self) -> None:
        assert price_moved(0.53, 0.50, threshold=0.02) is True   # 0.03 > 0.02
        assert price_moved(0.53, 0.50, threshold=0.10) is False  # 0.03 < 0.10

    def test_default_threshold_matches_module_constant(self) -> None:
        """Ensure default threshold in price_moved matches module constant."""
        # At threshold + epsilon → triggers
        just_over = PRICE_MOVE_THRESHOLD + 0.001
        assert price_moved(0.50 + just_over, 0.50) is True
        # At threshold - epsilon → does not trigger
        just_under = PRICE_MOVE_THRESHOLD - 0.001
        assert price_moved(0.50 + just_under, 0.50) is False


# ---------------------------------------------------------------------------
# MarketWatcher._detect_stale_markets
# ---------------------------------------------------------------------------


class TestDetectStaleMarkets:
    @pytest.mark.asyncio
    async def test_returns_ids_older_than_cutoff(self) -> None:
        """Stale market IDs are returned from the DB query."""
        watcher = _make_watcher(polling_interval=300)

        stale_ids = ["MKT-OLD-1", "MKT-OLD-2"]
        session = _make_session_with_stale_ids(stale_ids)

        result = await watcher._detect_stale_markets(session)

        assert result == stale_ids

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_stale_markets(self) -> None:
        watcher = _make_watcher(polling_interval=300)
        session = _make_session_with_stale_ids([])

        result = await watcher._detect_stale_markets(session)

        assert result == []


# ---------------------------------------------------------------------------
# MarketWatcher._check_price_move_triggers
# ---------------------------------------------------------------------------


class TestCheckPriceMoveTriggers:
    @pytest.mark.asyncio
    async def test_returns_one_when_price_moved(self) -> None:
        watcher = _make_watcher()

        markets = [_make_market("MKT-1", mid_price=0.60)]
        # Signal mid was 0.50 → delta = 0.10 > threshold
        session = _make_session_with_signal_rows([("MKT-1", 0.60, 0.50)])

        count = await watcher._check_price_move_triggers(session, markets)

        assert count == 1

    @pytest.mark.asyncio
    async def test_returns_zero_when_price_stable(self) -> None:
        watcher = _make_watcher()

        markets = [_make_market("MKT-1", mid_price=0.52)]
        # Signal mid was 0.50 → delta = 0.02 < threshold
        session = _make_session_with_signal_rows([("MKT-1", 0.52, 0.50)])

        count = await watcher._check_price_move_triggers(session, markets)

        assert count == 0

    @pytest.mark.asyncio
    async def test_counts_only_moved_markets_from_batch(self) -> None:
        watcher = _make_watcher()

        markets = [
            _make_market("MKT-1", mid_price=0.60),  # moved (0.10 > 0.05)
            _make_market("MKT-2", mid_price=0.52),  # stable (0.02 < 0.05)
            _make_market("MKT-3", mid_price=0.80),  # moved (0.30 > 0.05)
        ]
        session = _make_session_with_signal_rows([
            ("MKT-1", 0.60, 0.50),
            ("MKT-2", 0.52, 0.50),
            ("MKT-3", 0.80, 0.50),
        ])

        count = await watcher._check_price_move_triggers(session, markets)

        assert count == 2

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_markets_list(self) -> None:
        watcher = _make_watcher()
        session = AsyncMock()

        count = await watcher._check_price_move_triggers(session, [])

        assert count == 0
        session.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# MarketWatcher._sweep_closed_markets
# ---------------------------------------------------------------------------


class TestSweepClosedMarkets:
    @pytest.mark.asyncio
    async def test_makes_one_batched_call_instead_of_n_single_calls(self) -> None:
        market_ids = ["MKT-1", "MKT-2", "MKT-3"]
        client = AsyncMock(spec=KalshiClient)
        client.get_markets_by_tickers.return_value = [
            _make_sweep_market(mid) for mid in market_ids
        ]
        session, factory = _make_sweep_session_factory(market_ids)
        watcher = MarketWatcher(client=client, session_factory=factory)

        with patch("freqpred.markets.watcher.upsert_markets", new_callable=AsyncMock):
            updated = await watcher._sweep_closed_markets()

        client.get_markets_by_tickers.assert_awaited_once_with(market_ids)
        client.get_market.assert_not_called()
        assert updated == 3

    @pytest.mark.asyncio
    async def test_missing_from_batch_falls_back_to_settled(self) -> None:
        market_ids = ["MKT-1", "MKT-2"]
        client = AsyncMock(spec=KalshiClient)
        # Only MKT-1 comes back from the batch call; MKT-2 is "missing".
        client.get_markets_by_tickers.return_value = [_make_sweep_market("MKT-1")]
        client.get_market_from_settled.return_value = _make_sweep_market(
            "MKT-2", result="yes"
        )
        session, factory = _make_sweep_session_factory(market_ids)
        watcher = MarketWatcher(client=client, session_factory=factory)

        with patch(
            "freqpred.markets.watcher.upsert_markets", new_callable=AsyncMock
        ) as mock_upsert:
            updated = await watcher._sweep_closed_markets()

        client.get_market_from_settled.assert_awaited_once_with("MKT-2")
        upserted_ids = {m.id for m in mock_upsert.call_args.args[1]}
        assert upserted_ids == {"MKT-1", "MKT-2"}
        assert updated == 2

    @pytest.mark.asyncio
    async def test_missing_and_not_in_settled_marks_finalized(self) -> None:
        market_ids = ["MKT-1"]
        client = AsyncMock(spec=KalshiClient)
        client.get_markets_by_tickers.return_value = []
        client.get_market_from_settled.return_value = None
        session, factory = _make_sweep_session_factory(market_ids)
        watcher = MarketWatcher(client=client, session_factory=factory)

        with patch("freqpred.markets.watcher.upsert_markets", new_callable=AsyncMock):
            updated = await watcher._sweep_closed_markets()

        assert updated == 0
        # The last execute() call should be the "mark finalized" UPDATE.
        last_stmt = session.execute.call_args.args[0]
        assert "finalized" in str(last_stmt.compile(compile_kwargs={"literal_binds": True}))

    @pytest.mark.asyncio
    async def test_finalized_with_no_result_skipped_from_upsert(self) -> None:
        market_ids = ["MKT-1"]
        client = AsyncMock(spec=KalshiClient)
        client.get_markets_by_tickers.return_value = [
            _make_sweep_market("MKT-1", status="finalized", result=None)
        ]
        session, factory = _make_sweep_session_factory(market_ids)
        watcher = MarketWatcher(client=client, session_factory=factory)

        with patch(
            "freqpred.markets.watcher.upsert_markets", new_callable=AsyncMock
        ) as mock_upsert:
            updated = await watcher._sweep_closed_markets()

        mock_upsert.assert_not_called()
        assert updated == 0


def _make_sweep_market(
    market_id: str, status: str = "finalized", result: str | None = "yes"
) -> MagicMock:
    m = MagicMock()
    m.id = market_id
    m.status = status
    m.result = result
    return m


def _make_sweep_session_factory(market_ids: list[str]) -> tuple[AsyncMock, MagicMock]:
    """Session whose first execute() returns the pending market_ids; later
    execute()/commit() calls (UPDATE ... finalized) are accepted no-ops."""
    select_result = MagicMock()
    select_result.all.return_value = [MagicMock(id=mid) for mid in market_ids]

    session = AsyncMock()
    session.execute.return_value = select_result
    session.commit = AsyncMock()

    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=None)
    return session, factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_watcher(polling_interval: int = 300) -> MarketWatcher:
    return MarketWatcher(
        client=AsyncMock(),
        session_factory=MagicMock(),
        polling_interval=polling_interval,
    )


def _make_market(market_id: str, mid_price: float) -> MagicMock:
    m = MagicMock()
    m.id = market_id
    m.mid_price = mid_price
    return m


def _make_session_with_stale_ids(stale_ids: list[str]) -> AsyncMock:
    """Return a mock session whose execute() yields rows with .id for each stale id."""
    rows = [MagicMock(id=mid) for mid in stale_ids]
    result = MagicMock()
    result.all.return_value = rows

    session = AsyncMock()
    session.execute.return_value = result
    return session


def _make_session_with_signal_rows(
    rows_data: list[tuple[str, float, float]],
) -> AsyncMock:
    """Return a mock session for price-move queries.

    rows_data: list of (market_id, current_mid, signal_mid) tuples.
    """
    rows = []
    for market_id, current_mid, signal_mid in rows_data:
        row = MagicMock()
        row.id = market_id
        row.mid_price = current_mid
        row.market_mid_at_signal = signal_mid
        rows.append(row)

    result = MagicMock()
    result.all.return_value = rows

    session = AsyncMock()
    session.execute.return_value = result
    return session
