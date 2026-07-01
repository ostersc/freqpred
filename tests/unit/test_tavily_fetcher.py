"""Unit tests for freqpred/ingestion/fetchers/tavily.py.

All HTTP calls are mocked — no real Tavily API calls.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from freqpred.ingestion.fetchers import is_domain_excluded
from freqpred.ingestion.fetchers.tavily import fetch
from freqpred.ingestion.store import RawDocument

_API_KEY = "test-key"
_QUERY = "election results"
_NOW = datetime(2026, 3, 16, 12, 0, 0, tzinfo=UTC)


def _make_item(
    url: str = "https://example.com/article",
    content: str = "Article body text.",
    title: str = "Article Title",
    published_date: str | None = "2026-03-15T10:00:00",
    raw_content: str | None = None,
) -> dict:
    return {
        "url": url,
        "content": content,
        "raw_content": raw_content,
        "title": title,
        "published_date": published_date,
    }


@pytest.fixture()
def mock_tavily_client():
    with patch("freqpred.ingestion.fetchers.tavily.TavilyClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        yield instance


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_raw_documents(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            _make_item(url="https://a.com/1", content="Body one."),
            _make_item(url="https://b.com/2", content="Body two."),
        ]
    }

    docs = await fetch(_API_KEY, _QUERY)

    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)
    assert docs[0].source_url == "https://a.com/1"
    assert docs[1].source_url == "https://b.com/2"


@pytest.mark.asyncio
async def test_fetch_sets_source_type_news(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [_make_item()]
    }
    docs = await fetch(_API_KEY, _QUERY)
    assert docs[0].source_type == "news"
    assert docs[0].source_name == "Tavily"


@pytest.mark.asyncio
async def test_fetch_prefers_raw_content_over_content(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            _make_item(content="short", raw_content="full raw content here")
        ]
    }
    docs = await fetch(_API_KEY, _QUERY)
    assert docs[0].body == "full raw content here"


@pytest.mark.asyncio
async def test_fetch_parses_published_date(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [_make_item(published_date="2026-03-10T08:30:00")]
    }
    docs = await fetch(_API_KEY, _QUERY)
    assert docs[0].published_at == datetime(2026, 3, 10, 8, 30, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_fetch_naive_published_date_gets_utc(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [_make_item(published_date="2026-03-10T08:30:00")]
    }
    docs = await fetch(_API_KEY, _QUERY)
    assert docs[0].published_at.tzinfo is not None


@pytest.mark.asyncio
async def test_fetch_missing_published_date_stores_none(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [_make_item(published_date=None)]
    }
    docs = await fetch(_API_KEY, _QUERY)
    assert docs[0].published_at is None


@pytest.mark.asyncio
async def test_fetch_invalid_published_date_falls_back(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [_make_item(published_date="not-a-date")]
    }
    docs = await fetch(_API_KEY, _QUERY)
    assert len(docs) == 1  # not skipped, just uses fallback date


@pytest.mark.asyncio
async def test_fetch_passes_max_results(mock_tavily_client):
    mock_tavily_client.search.return_value = {"results": []}
    await fetch(_API_KEY, _QUERY, max_results=5)
    mock_tavily_client.search.assert_called_once()
    call_kwargs = mock_tavily_client.search.call_args
    assert call_kwargs.kwargs.get("max_results") == 5 or call_kwargs.args[1] == 5


# ---------------------------------------------------------------------------
# Skipping invalid results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_missing_url(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            {"url": "", "content": "Some body", "title": "Title"},
            _make_item(url="https://valid.com/"),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY)
    assert len(docs) == 1
    assert docs[0].source_url == "https://valid.com/"


@pytest.mark.asyncio
async def test_fetch_skips_missing_body(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            {"url": "https://example.com/empty", "content": "", "raw_content": None, "title": "T"},
            _make_item(url="https://valid.com/", content="Has body."),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY)
    assert len(docs) == 1
    assert docs[0].source_url == "https://valid.com/"


@pytest.mark.asyncio
async def test_fetch_empty_results(mock_tavily_client):
    mock_tavily_client.search.return_value = {"results": []}
    docs = await fetch(_API_KEY, _QUERY)
    assert docs == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_api_error(mock_tavily_client):
    mock_tavily_client.search.side_effect = RuntimeError("API down")
    docs = await fetch(_API_KEY, _QUERY)
    assert docs == []


# ---------------------------------------------------------------------------
# Domain blacklist — is_domain_excluded helper
# ---------------------------------------------------------------------------


def test_is_domain_excluded_exact_match():
    assert is_domain_excluded("https://kalshi.com/markets/foo", frozenset({"kalshi.com"}))


def test_is_domain_excluded_subdomain():
    assert is_domain_excluded("https://api.kalshi.com/v1/events", frozenset({"kalshi.com"}))


def test_is_domain_excluded_deep_subdomain():
    assert is_domain_excluded("https://x.y.kalshi.com/path", frozenset({"kalshi.com"}))


def test_is_domain_excluded_no_false_positive():
    assert not is_domain_excluded("https://notkalshi.com/article", frozenset({"kalshi.com"}))


def test_is_domain_excluded_unrelated_domain():
    assert not is_domain_excluded("https://reuters.com/news/story", frozenset({"kalshi.com"}))


def test_is_domain_excluded_multiple_domains():
    blacklist = frozenset({"kalshi.com", "polymarket.com"})
    assert is_domain_excluded("https://polymarket.com/market/x", blacklist)
    assert is_domain_excluded("https://kalshi.com/event/y", blacklist)
    assert not is_domain_excluded("https://reuters.com/news", blacklist)


def test_is_domain_excluded_empty_blacklist():
    assert not is_domain_excluded("https://kalshi.com/foo", frozenset())


# ---------------------------------------------------------------------------
# Domain blacklist — fetcher integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_blacklisted_domain(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            _make_item(url="https://kalshi.com/markets/some-event", content="Kalshi market page."),
            _make_item(url="https://reuters.com/article", content="News body."),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, excluded_domains=frozenset({"kalshi.com"}))
    assert len(docs) == 1
    assert docs[0].source_url == "https://reuters.com/article"


@pytest.mark.asyncio
async def test_fetch_skips_subdomain_of_blacklisted_domain(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            _make_item(url="https://api.kalshi.com/v2/events", content="API response."),
            _make_item(url="https://valid.com/article", content="Valid body."),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, excluded_domains=frozenset({"kalshi.com"}))
    assert len(docs) == 1
    assert docs[0].source_url == "https://valid.com/article"


@pytest.mark.asyncio
async def test_fetch_empty_blacklist_keeps_all(mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            _make_item(url="https://kalshi.com/foo", content="Body."),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, excluded_domains=frozenset())
    assert len(docs) == 1
