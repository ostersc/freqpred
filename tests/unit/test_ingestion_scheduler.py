"""Unit tests for freqpred.ingestion.scheduler._ensure_catalysts and run_cycle."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.scheduler import (
    _CATALYST_REFRESH_INTERVAL,
    _ensure_catalysts,
    _subreddits_for_category,
    run_cycle,
)


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


@pytest.fixture(autouse=True)
def _mock_cursors(monkeypatch):
    """Patch fetch cursors so every fetcher is 'due' against mock sessions."""
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.get_cursor",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.set_cursor",
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
    """Build a mock session that returns market_rows on the first execute,
    an empty open-positions result on the second, and catalyst_rows on the third."""
    session = AsyncMock()

    markets_result = MagicMock()
    markets_result.scalars.return_value.all.return_value = market_rows

    open_positions_result = MagicMock()
    open_positions_result.all.return_value = []  # no open positions by default

    catalysts_result = MagicMock()
    catalysts_result.all.return_value = catalyst_rows

    session.execute.side_effect = [markets_result, open_positions_result, catalysts_result]
    return session


def _make_session_factory(session: AsyncMock | None = None) -> MagicMock:
    """Wrap a mock session as a minimal async session factory for run_cycle."""
    if session is None:
        session = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=cm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubredditsForCategory:
    def test_all_politics_strategy_categories_map_to_politics_subs(self) -> None:
        """Every category PoliticsEdgeStrategy trades must reach the politics
        subreddits — an unmapped category silently falls back to r/news, which
        is how Mentions markets spent weeks searching only r/news."""
        for category in ("Politics", "Elections", "Mentions"):
            subs = _subreddits_for_category(category)
            assert "PoliticalDiscussion" in subs, f"{category} missing politics subs"
            assert subs != ["news"]

    def test_unknown_category_falls_back_to_news(self) -> None:
        assert _subreddits_for_category("Climate and Weather Oddities") == ["news"]


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
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock, return_value=[]),
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
            MagicMock(**{"all.return_value": []}),  # open positions
            MagicMock(**{"all.return_value": [fresh_run]}),
        ]

        strategy = _make_strategy(interested=True)
        llm_client = AsyncMock()
        embedder = MagicMock()

        with (
            patch("freqpred.ingestion.selector.select_markets", return_value=[MagicMock(id="MKT-1")]),
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock, return_value=[]),
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
            MagicMock(**{"all.return_value": []}),  # open positions
            MagicMock(**{"all.return_value": [stale_run]}),
        ]

        strategy = _make_strategy(interested=True)
        llm_client = AsyncMock()
        embedder = MagicMock()

        with (
            patch("freqpred.ingestion.selector.select_markets", return_value=[MagicMock(id="MKT-1")]),
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock, return_value=[]),
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
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock, return_value=[]),
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

        close_time = datetime.now(UTC) + timedelta(days=7)
        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[("MKT-1", "politics", "Will tech dominate in 2026?", close_time, [("query one", None)])],
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
                session_factory=_make_session_factory(session),
                embedder=embedder,
                newsapi_api_key="some-key",
                newsapi_enabled=False,
            )

        mock_newsapi.assert_not_called()

    @pytest.mark.asyncio
    async def test_reddit_blocked_trips_rate_limit_and_skips_rest_of_cycle(self) -> None:
        """RedditBlockedError must trip the backoff and stop further Reddit calls.

        This is the wiring that prevents a repeat of the May 2026 silent death:
        the fetcher raises, the scheduler records the rate limit, and the rest
        of the cycle skips Reddit instead of hammering a blocked endpoint.
        """
        from freqpred.ingestion.fetchers.reddit import RedditBlockedError

        session = AsyncMock()
        embedder = MagicMock()
        close_time = datetime.now(UTC) + timedelta(days=7)

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[(
                    "MKT-1",
                    "politics",
                    "Will X happen?",
                    close_time,
                    [("query one", None), ("query two", None)],
                )],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                side_effect=RedditBlockedError("all subreddits 403"),
            ) as mock_reddit,
            patch(
                "freqpred.ingestion.scheduler.record_rate_limit",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_trip,
        ):
            await run_cycle(
                session_factory=_make_session_factory(session),
                embedder=embedder,
            )

        # Tripped exactly once, for the reddit service.
        trip_services = [c.args[1] for c in mock_trip.call_args_list]
        assert trip_services == ["reddit"]
        # Second query in the same cycle must not call Reddit again.
        assert mock_reddit.call_count == 1

    @pytest.mark.asyncio
    async def test_reddit_skipped_when_backed_off_from_previous_cycle(self) -> None:
        """Persistent backoff state must keep Reddit off for the whole cycle."""
        session = AsyncMock()
        embedder = MagicMock()
        close_time = datetime.now(UTC) + timedelta(days=7)

        with (
            patch(
                "freqpred.ingestion.scheduler.tick_and_load",
                new_callable=AsyncMock,
                return_value={"reddit": True},
            ),
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[("MKT-1", "politics", "Will X happen?", close_time, [("q", None)])],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
            ) as mock_reddit,
        ):
            await run_cycle(
                session_factory=_make_session_factory(session),
                embedder=embedder,
            )

        mock_reddit.assert_not_called()

    @pytest.mark.asyncio
    async def test_reddit_skipped_when_cursor_not_due(self, monkeypatch) -> None:
        """A market fetched recently (within reddit_min_fetch_interval_hours)
        must not hit Reddit again — each due market costs subreddits x queries
        unauthenticated requests, which is what drew the 429s."""
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_cursor",
            AsyncMock(return_value=datetime.now(UTC) - timedelta(minutes=10)),
        )
        session = AsyncMock()
        embedder = MagicMock()
        close_time = datetime.now(UTC) + timedelta(days=7)

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[("MKT-1", "politics", "Will X happen?", close_time, [("q", None)])],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
            ) as mock_reddit,
        ):
            await run_cycle(
                session_factory=_make_session_factory(session),
                embedder=embedder,
                reddit_min_fetch_interval_hours=2.0,
            )

        mock_reddit.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("jitter_draw", "expect_fetch"),
        [(0.75, True), (1.25, False)],
        ids=["low-jitter-due", "high-jitter-not-due"],
    )
    async def test_reddit_interval_jitter_desynchronizes_markets(
        self, monkeypatch, jitter_draw: float, expect_fetch: bool
    ) -> None:
        """The due-check interval is jittered +/-25% so markets fetched in the
        same cycle don't all become due together one interval later (idle,
        idle, idle, 390-request burst). A 1.6h-old cursor against a 2h base is
        due only when the jitter draws low."""
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_cursor",
            AsyncMock(return_value=datetime.now(UTC) - timedelta(hours=1.6)),
        )
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.random.uniform",
            lambda lo, hi: jitter_draw,
        )
        session = AsyncMock()
        embedder = MagicMock()
        close_time = datetime.now(UTC) + timedelta(days=7)

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[("MKT-1", "politics", "Will X happen?", close_time, [("q", None)])],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_reddit,
        ):
            await run_cycle(
                session_factory=_make_session_factory(session),
                embedder=embedder,
                reddit_min_fetch_interval_hours=2.0,
            )

        assert mock_reddit.called == expect_fetch

    @pytest.mark.asyncio
    async def test_reddit_fires_once_per_due_market_not_per_query(self) -> None:
        """A due market gets exactly one Reddit search regardless of how many
        catalyst queries it has — Reddit's unauthenticated budget is 1
        request/min per IP, so one rotated query per market per interval."""
        session = AsyncMock()
        embedder = MagicMock()
        close_time = datetime.now(UTC) + timedelta(days=7)
        queries = [(f"query {i}", None) for i in range(5)]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[("MKT-1", "politics", "Will X happen?", close_time, queries)],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_reddit,
        ):
            await run_cycle(
                session_factory=_make_session_factory(session),
                embedder=embedder,
            )

        assert mock_reddit.call_count == 1
        # The query used must be one of the market's catalyst queries.
        assert mock_reddit.call_args.kwargs["query"] in {q for q, _ in queries}

    @pytest.mark.asyncio
    async def test_reddit_cursor_set_after_successful_fetch(self, monkeypatch) -> None:
        set_cursor_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("freqpred.ingestion.scheduler.set_cursor", set_cursor_mock)
        session = AsyncMock()
        embedder = MagicMock()
        close_time = datetime.now(UTC) + timedelta(days=7)

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[("MKT-1", "politics", "Will X happen?", close_time, [("q", None)])],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await run_cycle(
                session_factory=_make_session_factory(session),
                embedder=embedder,
            )

        reddit_cursor_calls = [
            c for c in set_cursor_mock.call_args_list if c.args[1] == "reddit"
        ]
        assert len(reddit_cursor_calls) == 1
        assert reddit_cursor_calls[0].args[2] == "MKT-1"

    @pytest.mark.asyncio
    async def test_per_fetcher_telemetry_success_and_error(self) -> None:
        """Each fetcher reports its own heartbeat: success on a clean call,
        error when it raised — independent of the scheduler-level heartbeat."""
        from freqpred.ingestion.fetchers.reddit import RedditBlockedError

        session = AsyncMock()
        embedder = MagicMock()
        telemetry = AsyncMock()
        close_time = datetime.now(UTC) + timedelta(days=7)

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[("MKT-1", "politics", "Will X happen?", close_time, [("q", None)])],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                side_effect=RedditBlockedError("all subreddits 403"),
            ),
        ):
            await run_cycle(
                session_factory=_make_session_factory(session),
                embedder=embedder,
                telemetry=telemetry,
            )

        success_services = [c.args[0] for c in telemetry.mark_success.call_args_list]
        error_services = [c.args[0] for c in telemetry.mark_error.call_args_list]
        # GDELT is mocked to return [] (autouse fixture) → success heartbeat.
        assert "fetcher_gdelt" in success_services
        # Reddit raised → error heartbeat, no success heartbeat.
        assert "fetcher_reddit" in error_services
        assert "fetcher_reddit" not in success_services

    @pytest.mark.asyncio
    async def test_generation_error_is_swallowed(self) -> None:
        """A CatalystGenerationError for one market should not abort others."""
        from freqpred.ingestion.catalyst_generator import CatalystGenerationError

        market_rows = [_make_market_row("MKT-1"), _make_market_row("MKT-2")]
        session = AsyncMock()
        session.execute.side_effect = [
            MagicMock(**{"scalars.return_value.all.return_value": market_rows}),
            MagicMock(**{"all.return_value": []}),  # open positions
            MagicMock(**{"all.return_value": []}),  # catalyst runs
        ]

        strategy = _make_strategy(interested=True)
        llm_client = AsyncMock()
        embedder = MagicMock()

        markets = [MagicMock(id="MKT-1"), MagicMock(id="MKT-2")]

        with (
            patch("freqpred.ingestion.selector.select_markets", return_value=markets),
            patch("freqpred.ingestion.selector.deactivate_stale_catalysts", new_callable=AsyncMock, return_value=[]),
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
