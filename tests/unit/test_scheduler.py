"""Unit tests for freqpred/ingestion/scheduler.py.

All external dependencies (fetchers, store, DB) are mocked.
No real API calls or DB connections are made.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

from freqpred.ingestion.fetchers.guardian import GuardianRateLimitError
from freqpred.ingestion.fetchers.newsapi import NewsAPIRateLimitError
from freqpred.ingestion.scheduler import (
    _subreddits_for_category,
    run_cycle,
    run_scheduler,
)
from freqpred.ingestion.store import DocumentSkipped, RawDocument, UpsertStatus
from freqpred.rag.models import Document

NOW = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
CLOSE_TIME = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)
FAKE_EMBEDDING = [0.1] * 384


@pytest.fixture(autouse=True)
def _mock_gdelt(monkeypatch):
    """Prevent real GDELT HTTP calls in every scheduler unit test."""
    with patch(
        "freqpred.ingestion.scheduler.gdelt_fetcher.fetch",
        new_callable=AsyncMock,
        return_value=[],
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



@pytest.fixture(autouse=True)
def mock_quota(monkeypatch):
    """Patch quota helpers so scheduler tests don't need a real DB session."""
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.get_window_count",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.increment_window_count",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.get_daily_count",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.get_cursor",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.set_cursor",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.delete_cursors",
        AsyncMock(return_value=None),
    )


@pytest.fixture(autouse=True)
def mock_backoff(monkeypatch):
    """Patch backoff helpers so scheduler tests don't need a real DB session."""
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


def _make_raw_doc(url: str = "https://example.com/article") -> RawDocument:
    return RawDocument(
        source_url=url,
        title="Test",
        body="Body text.",
        source_type="news",
        source_name="Reuters",
        category="economics",
        tags=[],
        published_at=NOW,
        fetched_at=NOW,
    )


def _make_document(url: str = "https://example.com/article") -> Document:
    return Document(
        id=str(uuid.uuid4()),
        source_url=url,
        content_hash="abc123",
        title="Test",
        body="Body text.",
        summary=None,
        source_type="news",
        source_name="Reuters",
        category="economics",
        tags=[],
        published_at=NOW,
        fetched_at=NOW,
        embedding=FAKE_EMBEDDING,
        embedding_model="all-MiniLM-L6-v2",
    )


def _make_session() -> AsyncMock:
    """AsyncMock session with begin_nested properly set up as a sync context manager factory."""
    session = AsyncMock()
    # session.begin_nested() must return an async context manager, not a coroutine.
    # SQLAlchemy's real begin_nested() returns AsyncSessionTransaction which supports
    # `async with` directly (not via awaiting). We replicate that here.
    nested_ctx = MagicMock()
    nested_ctx.__aenter__ = AsyncMock(return_value=nested_ctx)
    nested_ctx.__aexit__ = AsyncMock(return_value=None)
    session.begin_nested = MagicMock(return_value=nested_ctx)
    return session


def _make_session_factory(session: AsyncMock | None = None) -> MagicMock:
    """Wrap a mock session as a minimal async session factory for run_cycle."""
    if session is None:
        session = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=cm)


def _make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)
    return embedder


# ---------------------------------------------------------------------------
# _subreddits_for_category
# ---------------------------------------------------------------------------


class TestSubredditsForCategory:
    def test_known_category(self) -> None:
        subs = _subreddits_for_category("politics")
        assert "politics" in subs
        assert len(subs) >= 1

    def test_unknown_category_returns_news(self) -> None:
        subs = _subreddits_for_category("underwater-basket-weaving")
        assert subs == ["news"]

    def test_case_insensitive(self) -> None:
        assert _subreddits_for_category("Economics") == _subreddits_for_category("economics")


# ---------------------------------------------------------------------------
# run_cycle — no active markets
# ---------------------------------------------------------------------------


class TestRunCycleNoMarkets:
    @pytest.mark.asyncio
    async def test_empty_returns_zero_stats(self) -> None:
        session = AsyncMock()
        embedder = _make_embedder()

        with patch(
            "freqpred.ingestion.scheduler._load_active_market_queries",
            new_callable=AsyncMock,
            return_value=[],
        ):
            stats = await run_cycle(_make_session_factory(session), embedder)

        assert stats["markets_processed"] == 0
        assert stats["docs_fetched"] == 0
        assert stats["docs_stored"] == 0
        assert stats["docs_error"] == 0


# ---------------------------------------------------------------------------
# run_cycle — fetchers called with catalyst query text
# ---------------------------------------------------------------------------


class TestRunCycleFetchersCalled:
    @pytest.mark.asyncio
    async def test_fetchers_called_with_query_text(self) -> None:
        """Fetchers must receive the catalyst query text, not category keywords."""
        session = _make_session()
        embedder = _make_embedder()
        doc = _make_document()
        raw_doc = _make_raw_doc()

        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("Fed rate decision March 2026", None)])]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[raw_doc],
            ) as mock_tavily,
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_newsapi,
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_reddit,
            patch(
                "freqpred.ingestion.scheduler.upsert_document",
                new_callable=AsyncMock,
                return_value=(doc, UpsertStatus.INSERTED),
            ),
        ):
            await run_cycle(
                _make_session_factory(session),
                embedder,
                tavily_api_key="tv-key",
                newsapi_api_key="na-key",
            )

        # Query text must be passed, not a category keyword
        mock_tavily.assert_called_once()
        assert mock_tavily.call_args.kwargs["query"] == "Fed rate decision March 2026"

        mock_newsapi.assert_called_once()
        assert mock_newsapi.call_args.kwargs["query"] == "Fed rate decision March 2026"

        mock_reddit.assert_called_once()
        assert mock_reddit.call_args.kwargs["query"] == "Fed rate decision March 2026"

    @pytest.mark.asyncio
    async def test_tavily_skipped_when_no_key(self) -> None:
        session = AsyncMock()
        embedder = _make_embedder()

        market_queries = [("MKT-1", "politics", CLOSE_TIME, [("some query", None)])]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_tavily,
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await run_cycle(_make_session_factory(session), embedder, tavily_api_key="")  # no key

        mock_tavily.assert_not_called()

    @pytest.mark.asyncio
    async def test_newsapi_skipped_when_no_key(self) -> None:
        session = AsyncMock()
        embedder = _make_embedder()

        market_queries = [("MKT-1", "politics", CLOSE_TIME, [("some query", None)])]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_newsapi,
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await run_cycle(_make_session_factory(session), embedder, newsapi_api_key="")  # no key

        mock_newsapi.assert_not_called()

    @pytest.mark.asyncio
    async def test_reddit_always_runs(self) -> None:
        """Reddit doesn't require a key and must always run."""
        session = AsyncMock()
        embedder = _make_embedder()

        market_queries = [("MKT-1", "politics", CLOSE_TIME, [("some query", None)])]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_reddit,
        ):
            await run_cycle(_make_session_factory(session), embedder, tavily_api_key="", newsapi_api_key="")

        mock_reddit.assert_called_once()


# ---------------------------------------------------------------------------
# run_cycle — error isolation
# ---------------------------------------------------------------------------


class TestRunCycleErrorIsolation:
    @pytest.mark.asyncio
    async def test_tavily_failure_does_not_stop_newsapi_or_reddit(self) -> None:
        """If Tavily raises, NewsAPI and Reddit still run."""
        session = AsyncMock()
        embedder = _make_embedder()

        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Tavily down"),
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_newsapi,
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_reddit,
        ):
            stats = await run_cycle(
                _make_session_factory(session), embedder, tavily_api_key="key", newsapi_api_key="key"
            )

        # NewsAPI and Reddit must still have been called
        mock_newsapi.assert_called_once()
        mock_reddit.assert_called_once()
        # No unhandled exception — cycle completed
        assert stats["markets_processed"] == 1

    @pytest.mark.asyncio
    async def test_newsapi_failure_does_not_stop_reddit(self) -> None:
        session = AsyncMock()
        embedder = _make_embedder()

        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                side_effect=RuntimeError("NewsAPI down"),
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_reddit,
        ):
            stats = await run_cycle(
                _make_session_factory(session), embedder, tavily_api_key="key", newsapi_api_key="key"
            )

        mock_reddit.assert_called_once()
        assert stats["markets_processed"] == 1

    @pytest.mark.asyncio
    async def test_upsert_error_counted_not_raised(self) -> None:
        """A document store error increments docs_error without raising."""
        session = _make_session()
        embedder = _make_embedder()
        raw_doc = _make_raw_doc()

        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[raw_doc],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.upsert_document",
                new_callable=AsyncMock,
                side_effect=RuntimeError("DB error"),
            ),
        ):
            stats = await run_cycle(
                _make_session_factory(session), embedder, tavily_api_key="key"
            )

        assert stats["docs_error"] == 1
        assert stats["docs_stored"] == 0

    @pytest.mark.asyncio
    async def test_document_skipped_not_counted_as_error(self) -> None:
        """DocumentSkipped (empty body) is silently ignored, not counted as an error."""
        session = _make_session()
        embedder = _make_embedder()
        raw_doc = _make_raw_doc()

        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[raw_doc],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.upsert_document",
                new_callable=AsyncMock,
                side_effect=DocumentSkipped("https://example.com/empty"),
            ),
        ):
            stats = await run_cycle(
                _make_session_factory(session), embedder, tavily_api_key="key"
            )

        assert stats["docs_error"] == 0
        assert stats["docs_stored"] == 0


    @pytest.mark.asyncio
    async def test_newsapi_rate_limit_sets_circuit_breaker(self) -> None:
        """NewsAPIRateLimitError from the first query stops NewsAPI for the rest of the cycle."""
        session = AsyncMock()
        embedder = _make_embedder()

        # Two markets, each with one query — NewsAPI should only be attempted once.
        market_queries = [
            ("MKT-1", "economics", CLOSE_TIME, [("query-1", None)]),
            ("MKT-2", "politics", CLOSE_TIME, [("query-2", None)]),
        ]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                side_effect=NewsAPIRateLimitError("rate limited"),
            ) as mock_newsapi,
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_reddit,
        ):
            stats = await run_cycle(
                _make_session_factory(session), embedder, tavily_api_key="key", newsapi_api_key="key"
            )

        # NewsAPI attempted once, then circuit breaker stops it for the second market.
        mock_newsapi.assert_called_once()
        # Reddit still runs for both markets.
        assert mock_reddit.call_count == 2
        assert stats["markets_processed"] == 2


# ---------------------------------------------------------------------------
# run_cycle — multiple markets
# ---------------------------------------------------------------------------


class TestRunCycleMultipleMarkets:
    @pytest.mark.asyncio
    async def test_markets_processed_count(self) -> None:
        """markets_processed reflects number of distinct markets."""
        session = AsyncMock()
        embedder = _make_embedder()

        market_queries = [
            ("MKT-A", "politics", CLOSE_TIME, [("q1", None), ("q2", None)]),
            ("MKT-B", "economics", CLOSE_TIME, [("q3", None)]),
        ]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            stats = await run_cycle(_make_session_factory(session), embedder)

        assert stats["markets_processed"] == 2

    @pytest.mark.asyncio
    async def test_commit_called_once_per_market(self) -> None:
        """run_cycle must commit the session once per market, not once at the end."""
        embedder = _make_embedder()

        market_queries = [
            ("MKT-A", "politics", CLOSE_TIME, [("q1", None)]),
            ("MKT-B", "economics", CLOSE_TIME, [("q2", None)]),
            ("MKT-C", "technology", CLOSE_TIME, [("q3", None)]),
        ]

        # Track each session mock created by the factory so we can assert
        # commit was called on each market session (not the setup session).
        market_sessions: list[AsyncMock] = []

        call_count = 0

        def make_tracking_factory():
            def factory():
                nonlocal call_count
                call_count += 1
                session = _make_session()
                if call_count > 1:  # first call is the setup session
                    market_sessions.append(session)
                cm = MagicMock()
                cm.__aenter__ = AsyncMock(return_value=session)
                cm.__aexit__ = AsyncMock(return_value=None)
                return cm
            return factory

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(make_tracking_factory(), embedder)

        # One market session per market — each must have been committed exactly once.
        assert len(market_sessions) == 3
        for ms in market_sessions:
            ms.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# run_cycle — stats aggregation
# ---------------------------------------------------------------------------


class TestRunCycleStats:
    @pytest.mark.asyncio
    async def test_stats_counted_across_markets(self) -> None:
        session = _make_session()
        embedder = _make_embedder()
        doc = _make_document()

        market_queries = [
            ("MKT-A", "politics", CLOSE_TIME, [("query-a", None)]),
            ("MKT-B", "economics", CLOSE_TIME, [("query-b", None)]),
        ]

        with (
            patch(
                "freqpred.ingestion.scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[_make_raw_doc("https://a.com/1")],
            ),
            patch(
                "freqpred.ingestion.scheduler.newsapi_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[_make_raw_doc("https://b.com/1")],
            ),
            patch(
                "freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.scheduler.upsert_document",
                new_callable=AsyncMock,
                return_value=(doc, UpsertStatus.INSERTED),
            ),
        ):
            stats = await run_cycle(
                _make_session_factory(session), embedder, tavily_api_key="k", newsapi_api_key="k"
            )

        # 2 markets × 2 docs each = 4 total
        assert stats["markets_processed"] == 2
        assert stats["docs_fetched"] == 4
        assert stats["docs_stored"] == 4
        assert stats["docs_error"] == 0


# ---------------------------------------------------------------------------
# run_cycle — adaptive rate limits, per-market cursors, Guardian fetcher
# ---------------------------------------------------------------------------


class TestAdaptiveLimits:
    """Tests for Guardian fetcher, daily cap enforcement, and per-market cursor gates."""

    # ------------------------------------------------------------------
    # Guardian — basic on/off
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_guardian_skipped_when_no_key(self) -> None:
        """Guardian must not be called when guardian_api_key is empty."""
        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [("MKT-1", "politics", CLOSE_TIME, [("some query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.guardian_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]) as mock_guardian,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(_make_session_factory(session), embedder, guardian_api_key="")

        mock_guardian.assert_not_called()

    @pytest.mark.asyncio
    async def test_guardian_skipped_when_disabled(self) -> None:
        """Guardian must not be called when guardian_enabled=False even if key is set."""
        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [("MKT-1", "politics", CLOSE_TIME, [("some query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.guardian_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]) as mock_guardian,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(_make_session_factory(session), embedder, guardian_api_key="key", guardian_enabled=False)

        mock_guardian.assert_not_called()

    @pytest.mark.asyncio
    async def test_guardian_called_with_tv_query_when_set(self) -> None:
        """Guardian must receive the Solr tv_query string when it is set on the catalyst."""
        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [
            ("MKT-1", "politics", CLOSE_TIME,
             [("plain text query", 'trump AND ("tariff" OR "tariffs")')])
        ]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.guardian_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]) as mock_guardian,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(_make_session_factory(session), embedder, guardian_api_key="key")

        mock_guardian.assert_called_once()
        assert mock_guardian.call_args.kwargs["query"] == 'trump AND ("tariff" OR "tariffs")'

    @pytest.mark.asyncio
    async def test_guardian_falls_back_to_query_text_when_no_tv_query(self) -> None:
        """Guardian must use query_text when tv_query is None."""
        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("Fed rate hike 2026", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.guardian_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]) as mock_guardian,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(_make_session_factory(session), embedder, guardian_api_key="key")

        mock_guardian.assert_called_once()
        assert mock_guardian.call_args.kwargs["query"] == "Fed rate hike 2026"

    # ------------------------------------------------------------------
    # Daily cap enforcement at cycle start
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tavily_skipped_when_daily_cap_already_hit(self, monkeypatch) -> None:
        """If get_daily_count returns >= tavily_daily_cap at cycle start, Tavily is skipped."""
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_daily_count",
            AsyncMock(return_value=33),  # exactly at cap
        )
        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]) as mock_tavily,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(_make_session_factory(session), embedder, tavily_api_key="key", tavily_daily_cap=33)

        mock_tavily.assert_not_called()

    @pytest.mark.asyncio
    async def test_guardian_skipped_when_daily_cap_already_hit(self, monkeypatch) -> None:
        """If get_daily_count returns >= guardian_daily_cap at cycle start, Guardian is skipped."""
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_daily_count",
            AsyncMock(return_value=490),  # exactly at cap
        )
        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [("MKT-1", "politics", CLOSE_TIME, [("query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.guardian_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]) as mock_guardian,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(_make_session_factory(session), embedder, guardian_api_key="key", guardian_daily_cap=490)

        mock_guardian.assert_not_called()

    # ------------------------------------------------------------------
    # Per-market cursor gates
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tavily_skipped_for_market_when_cursor_is_recent(self, monkeypatch) -> None:
        """Tavily must be skipped for a market whose cursor is within the fetch interval."""
        # Cursor was set 30 minutes ago (relative to actual now); interval is 1 hour → not due.
        recent = datetime.now(timezone.utc) - timedelta(minutes=30)
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_cursor",
            AsyncMock(return_value=recent),
        )
        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]) as mock_tavily,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(
                _make_session_factory(session), embedder,
                tavily_api_key="key",
                tavily_min_fetch_interval_hours=1.0,
                tavily_daily_cap=33,
            )

        mock_tavily.assert_not_called()

    @pytest.mark.asyncio
    async def test_tavily_runs_for_market_when_cursor_is_stale(self, monkeypatch) -> None:
        """Tavily must run for a market whose cursor is beyond the fetch interval."""
        # Cursor was set 2 hours ago (relative to actual now); interval is 1 hour → due.
        stale = datetime.now(timezone.utc) - timedelta(hours=2)
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_cursor",
            AsyncMock(return_value=stale),
        )
        session = _make_session()
        embedder = _make_embedder()
        doc = _make_document()
        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[_make_raw_doc()]) as mock_tavily,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
            patch("freqpred.ingestion.scheduler.upsert_document",
                  new_callable=AsyncMock, return_value=(doc, UpsertStatus.INSERTED)),
        ):
            await run_cycle(
                _make_session_factory(session), embedder,
                tavily_api_key="key",
                tavily_min_fetch_interval_hours=1.0,
                tavily_daily_cap=33,
            )

        mock_tavily.assert_called_once()

    @pytest.mark.asyncio
    async def test_tavily_runs_for_market_when_cursor_is_none(self, monkeypatch) -> None:
        """Tavily must run for a market that has never been fetched (no cursor)."""
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_cursor",
            AsyncMock(return_value=None),
        )
        session = _make_session()
        embedder = _make_embedder()
        doc = _make_document()
        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[_make_raw_doc()]) as mock_tavily,
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
            patch("freqpred.ingestion.scheduler.upsert_document",
                  new_callable=AsyncMock, return_value=(doc, UpsertStatus.INSERTED)),
        ):
            await run_cycle(_make_session_factory(session), embedder, tavily_api_key="key")

        mock_tavily.assert_called_once()

    # ------------------------------------------------------------------
    # set_cursor written (or not) after fetch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_set_cursor_called_after_successful_tavily_fetch(self, monkeypatch) -> None:
        """set_cursor must be called for a market after Tavily fetches documents."""
        mock_set_cursor = AsyncMock(return_value=None)
        monkeypatch.setattr("freqpred.ingestion.scheduler.set_cursor", mock_set_cursor)
        monkeypatch.setattr("freqpred.ingestion.scheduler.get_cursor", AsyncMock(return_value=None))

        session = _make_session()
        embedder = _make_embedder()
        doc = _make_document()
        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[_make_raw_doc()]),
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
            patch("freqpred.ingestion.scheduler.upsert_document",
                  new_callable=AsyncMock, return_value=(doc, UpsertStatus.INSERTED)),
        ):
            await run_cycle(_make_session_factory(session), embedder, tavily_api_key="key")

        tavily_cursor_calls = [c for c in mock_set_cursor.call_args_list
                               if c.args[1] == "tavily"]
        assert len(tavily_cursor_calls) == 1
        assert tavily_cursor_calls[0].args[2] == "MKT-1"

    @pytest.mark.asyncio
    async def test_set_cursor_not_called_when_tavily_skipped_by_cursor(self, monkeypatch) -> None:
        """set_cursor must not be called for a market that was skipped due to a recent cursor."""
        recent = datetime.now(timezone.utc) - timedelta(minutes=10)
        mock_set_cursor = AsyncMock(return_value=None)
        monkeypatch.setattr("freqpred.ingestion.scheduler.set_cursor", mock_set_cursor)
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_cursor", AsyncMock(return_value=recent)
        )

        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [("MKT-1", "economics", CLOSE_TIME, [("query", None)])]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(
                _make_session_factory(session), embedder,
                tavily_api_key="key",
                tavily_min_fetch_interval_hours=1.0,
            )

        tavily_cursor_calls = [c for c in mock_set_cursor.call_args_list
                               if c.args[1] == "tavily"]
        assert len(tavily_cursor_calls) == 0

    # ------------------------------------------------------------------
    # Mid-cycle cap enforcement across markets
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tavily_stops_subsequent_markets_after_mid_cycle_cap(self, monkeypatch) -> None:
        """Once Tavily hits its daily cap mid-cycle, it must not be called for later markets."""
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_cursor", AsyncMock(return_value=None)
        )
        # daily count starts at 0; cap is 1 → after first market's fetch, cap is hit
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_daily_count", AsyncMock(return_value=0)
        )

        session = _make_session()
        embedder = _make_embedder()
        doc = _make_document()
        market_queries = [
            ("MKT-1", "economics", CLOSE_TIME, [("query 1", None)]),
            ("MKT-2", "economics", CLOSE_TIME, [("query 2", None)]),
        ]

        call_count = 0

        async def fake_tavily(api_key, query, **kwargs):
            nonlocal call_count
            call_count += 1
            return [_make_raw_doc(f"https://example.com/{call_count}")]

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.tavily_fetcher.fetch",
                  side_effect=fake_tavily),
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
            patch("freqpred.ingestion.scheduler.upsert_document",
                  new_callable=AsyncMock, return_value=(doc, UpsertStatus.INSERTED)),
        ):
            await run_cycle(_make_session_factory(session), embedder, tavily_api_key="key", tavily_daily_cap=1)

        # Only one market should have triggered a Tavily call; second was blocked by cap.
        assert call_count == 1

    # ------------------------------------------------------------------
    # Guardian rate limit → circuit breaker
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_guardian_rate_limit_sets_circuit_breaker(self, monkeypatch) -> None:
        """GuardianRateLimitError must set guardian_limit_hit and skip remaining markets."""
        monkeypatch.setattr(
            "freqpred.ingestion.scheduler.get_cursor", AsyncMock(return_value=None)
        )

        session = AsyncMock()
        embedder = _make_embedder()
        market_queries = [
            ("MKT-1", "politics", CLOSE_TIME, [("query 1", None)]),
            ("MKT-2", "politics", CLOSE_TIME, [("query 2", None)]),
        ]

        guardian_call_count = 0

        async def failing_guardian(**kwargs):
            nonlocal guardian_call_count
            guardian_call_count += 1
            raise GuardianRateLimitError("429 Too Many Requests")

        with (
            patch("freqpred.ingestion.scheduler._load_active_market_queries",
                  new_callable=AsyncMock, return_value=market_queries),
            patch("freqpred.ingestion.scheduler.guardian_fetcher.fetch",
                  side_effect=failing_guardian),
            patch("freqpred.ingestion.scheduler.reddit_fetcher.fetch",
                  new_callable=AsyncMock, return_value=[]),
        ):
            await run_cycle(_make_session_factory(session), embedder, guardian_api_key="key")

        # Guardian should have been attempted for MKT-1 only; MKT-2 blocked.
        assert guardian_call_count == 1


# ---------------------------------------------------------------------------
# run_scheduler — background loop
# ---------------------------------------------------------------------------


class TestRunScheduler:
    @pytest.mark.asyncio
    async def test_scheduler_calls_run_cycle_and_sleeps(self) -> None:
        """run_scheduler calls run_cycle then sleeps; we cancel after 1 iter."""
        session = AsyncMock()
        session_factory = MagicMock()
        session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        session_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        session.commit = AsyncMock()
        embedder = _make_embedder()

        call_count = 0

        async def fake_run_cycle(*args, **kwargs) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError
            return {"markets_processed": 0, "docs_fetched": 0, "docs_stored": 0, "docs_error": 0}

        with (
            patch("freqpred.ingestion.scheduler.run_cycle", side_effect=fake_run_cycle),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(asyncio.CancelledError):
                await run_scheduler(
                    session_factory,
                    embedder,
                    interval_seconds=1,
                )

        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_scheduler_continues_after_cycle_error(self) -> None:
        """Cycle-level exceptions are caught; loop continues."""
        session = AsyncMock()
        session_factory = MagicMock()
        session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        session_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        session.commit = AsyncMock()
        embedder = _make_embedder()

        call_count = 0

        async def erroring_cycle(*args, **kwargs) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("cycle blew up")
            raise asyncio.CancelledError

        with (
            patch("freqpred.ingestion.scheduler.run_cycle", side_effect=erroring_cycle),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(asyncio.CancelledError):
                await run_scheduler(
                    session_factory,
                    embedder,
                    interval_seconds=1,
                )

        # Second iteration must have been reached despite first erroring
        assert call_count == 2
