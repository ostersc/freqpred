"""Unit tests for freqpred.ingestion.scheduler._ensure_catalysts and run_cycle."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.scheduler import _CATALYST_REFRESH_INTERVAL, _ensure_catalysts, run_cycle


@pytest.fixture(autouse=True)
def _mock_gdelt():
    """Prevent real GDELT HTTP calls in every scheduler unit test."""
    with patch(
        "freqpred.ingestion.scheduler.gdelt_fetcher.fetch",
        new_callable=AsyncMock,
        return_value=[],
    ):
        yield


@pytest.fixture(autouse=True)
def _mock_cursors(monkeypatch):
    """Patch fetcher cursor helpers so tests don't need a real DB session."""
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.get_cursor",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.set_cursor",
        AsyncMock(return_value=None),
    )


@pytest.fixture(autouse=True)
def _mock_backoff(monkeypatch):
    """Patch backoff helpers so tests don't need a real DB session."""
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.tick_and_load",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.record_rate_limit",
        AsyncMock(return_value=1),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.record_success",
        AsyncMock(return_value=None),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market_row(market_id: str = "MKT-1", category: str = "technology") -> MagicMock:
    row = MagicMock()
    row.id = market_id
    row.platform = "kalshi"
    row.question = "Will X happen?"
    row.category = category
    row.close_time = datetime.now(UTC) + timedelta(days=7)
    row.yes_bid = 0.4
    row.yes_ask = 0.6
    row.mid_price = 0.5
    row.volume_24h = 1000.0
    row.open_interest = 5000.0
    row.last_fetched_at = datetime.now(UTC)
    row.price_updated_at = datetime.now(UTC)
    row.metadata_fetched_at = datetime.now(UTC)
    row.current_signal_id = None
    row.metadata_ = {}
    return row


def _make_strategy(interested: bool = True) -> MagicMock:
    strategy = MagicMock()
    strategy.is_market_interesting.return_value = interested
    return strategy


def _make_session(market_rows: list, catalyst_rows: list) -> AsyncMock:
    """Build a mock session that returns market_rows on the first execute
    and catalyst_rows (latest-run query) on the second execute."""
    session = AsyncMock()

    markets_result = MagicMock()
    markets_result.scalars.return_value.all.return_value = market_rows

    catalysts_result = MagicMock()
    catalysts_result.all.return_value = catalyst_rows

    session.execute.side_effect = [markets_result, catalysts_result]
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnsureCatalysts:
    @pytest.mark.asyncio
    async def test_no_active_run_triggers_generation(self) -> None:
        """A selected market with no catalyst run should get generation 1."""
        market_row = _make_market_row("MKT-1")
        session = _make_session(market_rows=[market_row], catalyst_rows=[])
        strategy = _make_strategy(interested=True)
        llm_client = AsyncMock()
        embedder = MagicMock()

        with (
            patch("freqpred.ingestion.selector.select_markets", return_value=[MagicMock(id="MKT-1")]),
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock),
            patch("freqpred.ingestion.catalyst_generator.generate_catalysts", new_callable=AsyncMock) as mock_gen,
        ):
            count = await _ensure_catalysts(session, strategy, llm_client, embedder)

        assert count == 1
        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_fresh_active_run_skips_generation(self) -> None:
        """A market whose latest run was created recently should not be regenerated."""
        market_row = _make_market_row("MKT-1")

        fresh_run = MagicMock()
        fresh_run.market_id = "MKT-1"
        fresh_run.latest = datetime.now(UTC) - timedelta(hours=1)

        session = AsyncMock()
        session.execute.side_effect = [
            MagicMock(**{"scalars.return_value.all.return_value": [market_row]}),
            MagicMock(**{"all.return_value": [fresh_run]}),
        ]

        strategy = _make_strategy(interested=True)
        llm_client = AsyncMock()
        embedder = MagicMock()

        with (
            patch("freqpred.ingestion.selector.select_markets", return_value=[MagicMock(id="MKT-1")]),
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock),
            patch("freqpred.ingestion.catalyst_generator.generate_catalysts", new_callable=AsyncMock) as mock_gen,
        ):
            count = await _ensure_catalysts(session, strategy, llm_client, embedder)

        assert count == 0
        mock_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_active_run_triggers_refresh(self) -> None:
        """A market whose latest run is older than _CATALYST_REFRESH_INTERVAL should be refreshed."""
        market_row = _make_market_row("MKT-1")

        stale_run = MagicMock()
        stale_run.market_id = "MKT-1"
        stale_run.latest = datetime.now(UTC) - _CATALYST_REFRESH_INTERVAL - timedelta(minutes=1)

        session = AsyncMock()
        session.execute.side_effect = [
            MagicMock(**{"scalars.return_value.all.return_value": [market_row]}),
            MagicMock(**{"all.return_value": [stale_run]}),
        ]

        strategy = _make_strategy(interested=True)
        llm_client = AsyncMock()
        embedder = MagicMock()

        with (
            patch("freqpred.ingestion.selector.select_markets", return_value=[MagicMock(id="MKT-1")]),
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock),
            patch("freqpred.ingestion.catalyst_generator.generate_catalysts", new_callable=AsyncMock) as mock_gen,
        ):
            count = await _ensure_catalysts(session, strategy, llm_client, embedder)

        assert count == 1
        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_selected_markets_returns_zero(self) -> None:
        """If the strategy finds no interesting markets, nothing should be generated."""
        session = AsyncMock()
        session.execute.return_value = MagicMock(**{"scalars.return_value.all.return_value": []})

        strategy = _make_strategy(interested=False)
        llm_client = AsyncMock()
        embedder = MagicMock()

        with (
            patch("freqpred.ingestion.selector.select_markets", return_value=[]),
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock),
            patch("freqpred.ingestion.catalyst_generator.generate_catalysts", new_callable=AsyncMock) as mock_gen,
        ):
            count = await _ensure_catalysts(session, strategy, llm_client, embedder)

        assert count == 0
        mock_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_newsapi_skipped_when_disabled(self) -> None:
        """run_cycle with newsapi_enabled=False must never call the NewsAPI fetcher."""
        session = AsyncMock()
        embedder = MagicMock()

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[("MKT-1", "politics", ["query one"])],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
            ) as mock_newsapi,
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await run_cycle(
                session=session,
                embedder=embedder,
                newsapi_api_key="some-key",
                newsapi_enabled=False,
            )

        mock_newsapi.assert_not_called()

    @pytest.mark.asyncio
    async def test_generation_error_is_swallowed(self) -> None:
        """A CatalystGenerationError for one market should not abort others."""
        from freqpred.ingestion.catalyst_generator import CatalystGenerationError

        market_rows = [_make_market_row("MKT-1"), _make_market_row("MKT-2")]
        session = AsyncMock()
        session.execute.side_effect = [
            MagicMock(**{"scalars.return_value.all.return_value": market_rows}),
            MagicMock(**{"all.return_value": []}),
        ]

        strategy = _make_strategy(interested=True)
        llm_client = AsyncMock()
        embedder = MagicMock()

        markets = [MagicMock(id="MKT-1"), MagicMock(id="MKT-2")]

        with (
            patch("freqpred.ingestion.selector.select_markets", return_value=markets),
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock),
            patch(
                "freqpred.ingestion.catalyst_generator.generate_catalysts",
                new_callable=AsyncMock,
                side_effect=[CatalystGenerationError("boom"), None],
            ) as mock_gen,
        ):
            count = await _ensure_catalysts(session, strategy, llm_client, embedder)

        # First failed, second succeeded → count = 1
        assert count == 1
        assert mock_gen.call_count == 2
