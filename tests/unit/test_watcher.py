"""Unit tests for freqpred.markets.watcher."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.markets.watcher import (
    PRICE_MOVE_THRESHOLD,
    SIGNAL_TRIGGER_QUEUE,
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
    async def test_enqueues_market_when_price_moved(self) -> None:
        redis = AsyncMock()
        watcher = _make_watcher(redis=redis)

        markets = [_make_market("MKT-1", mid_price=0.60)]
        # Signal mid was 0.50 → delta = 0.10 > threshold
        session = _make_session_with_signal_rows([("MKT-1", 0.60, 0.50)])

        count = await watcher._check_price_move_triggers(session, markets)

        assert count == 1
        redis.rpush.assert_awaited_once()
        call_args = redis.rpush.call_args
        assert call_args[0][0] == SIGNAL_TRIGGER_QUEUE
        payload = json.loads(call_args[0][1])
        assert payload["market_id"] == "MKT-1"
        assert payload["trigger"] == "price_moved"

    @pytest.mark.asyncio
    async def test_does_not_enqueue_when_price_stable(self) -> None:
        redis = AsyncMock()
        watcher = _make_watcher(redis=redis)

        markets = [_make_market("MKT-1", mid_price=0.52)]
        # Signal mid was 0.50 → delta = 0.02 < threshold
        session = _make_session_with_signal_rows([("MKT-1", 0.52, 0.50)])

        count = await watcher._check_price_move_triggers(session, markets)

        assert count == 0
        redis.rpush.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enqueues_only_moved_markets_from_batch(self) -> None:
        redis = AsyncMock()
        watcher = _make_watcher(redis=redis)

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
        assert redis.rpush.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_markets_list(self) -> None:
        redis = AsyncMock()
        watcher = _make_watcher(redis=redis)
        session = AsyncMock()

        count = await watcher._check_price_move_triggers(session, [])

        assert count == 0
        redis.rpush.assert_not_awaited()
        session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_payload_contains_delta(self) -> None:
        redis = AsyncMock()
        watcher = _make_watcher(redis=redis)

        markets = [_make_market("MKT-1", mid_price=0.60)]
        session = _make_session_with_signal_rows([("MKT-1", 0.60, 0.50)])

        await watcher._check_price_move_triggers(session, markets)

        payload = json.loads(redis.rpush.call_args[0][1])
        assert payload["delta"] == pytest.approx(0.10, abs=1e-4)
        assert payload["current_mid"] == pytest.approx(0.60)
        assert payload["signal_mid"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_watcher(
    polling_interval: int = 300,
    redis: AsyncMock | None = None,
) -> MarketWatcher:
    if redis is None:
        redis = AsyncMock()
    return MarketWatcher(
        client=AsyncMock(),
        session_factory=MagicMock(),
        redis=redis,
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
