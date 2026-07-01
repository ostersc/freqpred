"""Unit tests for freqpred/ingestion/selector.py."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import freqpred.ingestion.models  # noqa: F401 — register CatalystRunRow/CatalystQueryRow
import freqpred.llm.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.ingestion.selector import deactivate_stale_catalysts, select_markets
from freqpred.markets.models import Market

NOW = datetime(2026, 3, 16, 12, 0, 0, tzinfo=UTC)
FUTURE = NOW + timedelta(days=10)
PAST = NOW - timedelta(days=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market(
    market_id: str = "MKT-1",
    category: str = "politics",
    volume_24h: float = 1000.0,
    close_time: datetime = FUTURE,
) -> Market:
    return Market(
        id=market_id,
        platform="kalshi",
        question="Will X happen?",
        category=category,
        close_time=close_time,
        yes_bid=0.40,
        yes_ask=0.44,
        mid_price=0.42,
        volume_24h=volume_24h,
        open_interest=500.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
    )


def _strategy(interested: bool = True) -> MagicMock:
    s = MagicMock()
    s.is_market_interesting.return_value = interested
    return s


# ---------------------------------------------------------------------------
# select_markets
# ---------------------------------------------------------------------------


class TestSelectMarkets:
    def test_single_strategy_interested(self) -> None:
        markets = [_market("A"), _market("B")]
        selected = select_markets(markets, [_strategy(True)])
        assert len(selected) == 2

    def test_single_strategy_not_interested(self) -> None:
        markets = [_market("A"), _market("B")]
        selected = select_markets(markets, [_strategy(False)])
        assert selected == []

    def test_any_strategy_selects(self) -> None:
        """Market is included if *any* strategy says yes."""
        markets = [_market("A")]
        selected = select_markets(markets, [_strategy(False), _strategy(True)])
        assert len(selected) == 1

    def test_no_strategies_returns_empty(self) -> None:
        markets = [_market("A"), _market("B")]
        selected = select_markets(markets, [])
        assert selected == []

    def test_empty_market_list(self) -> None:
        selected = select_markets([], [_strategy(True)])
        assert selected == []

    def test_partial_selection(self) -> None:
        """Strategy interested in only some markets."""
        mkt_a = _market("A", category="politics")
        mkt_b = _market("B", category="sports")

        s = MagicMock()
        s.is_market_interesting.side_effect = lambda m: m.category == "politics"

        selected = select_markets([mkt_a, mkt_b], [s])
        assert selected == [mkt_a]

    def test_each_market_checked_independently(self) -> None:
        markets = [_market("A"), _market("B"), _market("C")]
        calls = []

        s = MagicMock()
        s.is_market_interesting.side_effect = lambda m: (calls.append(m.id) or True)

        select_markets(markets, [s])
        assert sorted(calls) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# deactivate_stale_catalysts
# ---------------------------------------------------------------------------


class TestDeactivateStaleMarkets:
    def _make_run_row(self, market_row: MagicMock) -> MagicMock:
        run = MagicMock()
        run.id = uuid.uuid4()
        run.generation = 1
        run.is_active = True
        run.created_at = NOW
        return run

    def _make_market_row(
        self,
        market_id: str = "MKT-1",
        close_time: datetime = FUTURE,
        category: str = "politics",
        volume_24h: float = 1000.0,
    ) -> MagicMock:
        row = MagicMock()
        row.id = market_id
        row.platform = "kalshi"
        row.question = "Will X happen?"
        row.category = category
        row.close_time = close_time
        row.yes_bid = 0.40
        row.yes_ask = 0.44
        row.mid_price = 0.42
        row.volume_24h = volume_24h
        row.open_interest = 500.0
        row.last_fetched_at = NOW
        row.price_updated_at = NOW
        row.metadata_fetched_at = NOW
        row.current_signal_id = None
        row.metadata_ = {}
        return row

    def _make_session(self, pairs: list[tuple]) -> AsyncMock:
        """pairs: list of (run_row, market_row) tuples returned by execute."""
        session = AsyncMock()
        session.add = MagicMock()  # sync in SQLAlchemy; avoid unawaited-coroutine warning
        result = MagicMock()
        result.all.return_value = pairs
        session.execute.return_value = result
        return session

    @pytest.mark.asyncio
    async def test_closed_market_deactivated(self) -> None:
        market_row = self._make_market_row(close_time=PAST)
        run_row = self._make_run_row(market_row)
        session = self._make_session([(run_row, market_row)])

        result = await deactivate_stale_catalysts(session, [_strategy(True)])

        assert len(result) == 1
        assert result[0] == "MKT-1"
        assert run_row.is_active is False
        session.add.assert_called_once_with(run_row)

    @pytest.mark.asyncio
    async def test_no_strategy_interest_deactivated(self) -> None:
        market_row = self._make_market_row(close_time=FUTURE)
        run_row = self._make_run_row(market_row)
        session = self._make_session([(run_row, market_row)])

        result = await deactivate_stale_catalysts(session, [_strategy(False)])

        assert len(result) == 1
        assert run_row.is_active is False

    @pytest.mark.asyncio
    async def test_active_market_not_deactivated(self) -> None:
        market_row = self._make_market_row(close_time=FUTURE)
        run_row = self._make_run_row(market_row)
        session = self._make_session([(run_row, market_row)])

        with patch("freqpred.ingestion.selector.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            result = await deactivate_stale_catalysts(session, [_strategy(True)])

        assert result == []
        assert run_row.is_active is True
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_markets(self) -> None:
        active_mrow = self._make_market_row("A", FUTURE)
        active_rrow = self._make_run_row(active_mrow)

        closed_mrow = self._make_market_row("B", PAST)
        closed_rrow = self._make_run_row(closed_mrow)

        session = self._make_session([
            (active_rrow, active_mrow),
            (closed_rrow, closed_mrow),
        ])

        with patch("freqpred.ingestion.selector.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            result = await deactivate_stale_catalysts(session, [_strategy(True)])

        assert result == ["B"]
        assert active_rrow.is_active is True
        assert closed_rrow.is_active is False

    @pytest.mark.asyncio
    async def test_no_active_runs_returns_empty_list(self) -> None:
        session = self._make_session([])
        result = await deactivate_stale_catalysts(session, [_strategy(True)])
        assert result == []

    @pytest.mark.asyncio
    async def test_deactivated_ids_contain_all_stale_markets(self) -> None:
        """All deactivated market IDs are returned, not just a count."""
        rows = []
        for mid in ["X", "Y", "Z"]:
            mrow = self._make_market_row(mid, close_time=PAST)
            rrow = self._make_run_row(mrow)
            rows.append((rrow, mrow))
        session = self._make_session(rows)

        result = await deactivate_stale_catalysts(session, [_strategy(True)])

        assert sorted(result) == ["X", "Y", "Z"]
