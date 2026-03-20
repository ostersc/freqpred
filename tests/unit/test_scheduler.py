"""Unit tests for freqpred/ingestion/scheduler.py.

All external dependencies (fetchers, store, DB) are mocked.
No real API calls or DB connections are made.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.newsapi import NewsAPIRateLimitError
from freqpred.ingestion.scheduler import (
    _subreddits_for_category,
    run_cycle,
    run_scheduler,
)
from freqpred.ingestion.store import DocumentSkipped, RawDocument
from freqpred.rag.models import Document

NOW = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
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
def mock_cursors(monkeypatch):
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
def mock_quota(monkeypatch):
    """Patch quota helpers so scheduler tests don't need a real DB session."""
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.get_daily_count",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.scheduler.increment_daily_count",
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
            stats = await run_cycle(session, embedder)

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

        market_queries = [("MKT-1", "economics", ["Fed rate decision March 2026"])]

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
                return_value=doc,
            ),
        ):
            await run_cycle(
                session,
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

        market_queries = [("MKT-1", "politics", ["some query"])]

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
            await run_cycle(session, embedder, tavily_api_key="")  # no key

        mock_tavily.assert_not_called()

    @pytest.mark.asyncio
    async def test_newsapi_skipped_when_no_key(self) -> None:
        session = AsyncMock()
        embedder = _make_embedder()

        market_queries = [("MKT-1", "politics", ["some query"])]

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
            await run_cycle(session, embedder, newsapi_api_key="")  # no key

        mock_newsapi.assert_not_called()

    @pytest.mark.asyncio
    async def test_reddit_always_runs(self) -> None:
        """Reddit doesn't require a key and must always run."""
        session = AsyncMock()
        embedder = _make_embedder()

        market_queries = [("MKT-1", "politics", ["some query"])]

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
            await run_cycle(session, embedder, tavily_api_key="", newsapi_api_key="")

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

        market_queries = [("MKT-1", "economics", ["query"])]

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
                session, embedder, tavily_api_key="key", newsapi_api_key="key"
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

        market_queries = [("MKT-1", "economics", ["query"])]

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
                session, embedder, tavily_api_key="key", newsapi_api_key="key"
            )

        mock_reddit.assert_called_once()
        assert stats["markets_processed"] == 1

    @pytest.mark.asyncio
    async def test_upsert_error_counted_not_raised(self) -> None:
        """A document store error increments docs_error without raising."""
        session = _make_session()
        embedder = _make_embedder()
        raw_doc = _make_raw_doc()

        market_queries = [("MKT-1", "economics", ["query"])]

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
                session, embedder, tavily_api_key="key"
            )

        assert stats["docs_error"] == 1
        assert stats["docs_stored"] == 0

    @pytest.mark.asyncio
    async def test_document_skipped_not_counted_as_error(self) -> None:
        """DocumentSkipped (empty body) is silently ignored, not counted as an error."""
        session = _make_session()
        embedder = _make_embedder()
        raw_doc = _make_raw_doc()

        market_queries = [("MKT-1", "economics", ["query"])]

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
                session, embedder, tavily_api_key="key"
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
            ("MKT-1", "economics", ["query-1"]),
            ("MKT-2", "politics", ["query-2"]),
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
                session, embedder, tavily_api_key="key", newsapi_api_key="key"
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
            ("MKT-A", "politics", ["q1", "q2"]),
            ("MKT-B", "economics", ["q3"]),
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
            stats = await run_cycle(session, embedder)

        assert stats["markets_processed"] == 2


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
            ("MKT-A", "politics", ["query-a"]),
            ("MKT-B", "economics", ["query-b"]),
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
                return_value=doc,
            ),
        ):
            stats = await run_cycle(
                session, embedder, tavily_api_key="k", newsapi_api_key="k"
            )

        # 2 markets × 2 docs each = 4 total
        assert stats["markets_processed"] == 2
        assert stats["docs_fetched"] == 4
        assert stats["docs_stored"] == 4
        assert stats["docs_error"] == 0


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
