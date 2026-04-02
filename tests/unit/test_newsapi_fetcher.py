"""Unit tests for freqpred/ingestion/fetchers/newsapi.py.

All HTTP calls are mocked — no real NewsAPI calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.newsapi import _RATE_LIMIT_SLEEP, fetch
from freqpred.ingestion.store import RawDocument

_API_KEY = "test-key"
_QUERY = "election results"
_FROM = datetime(2026, 3, 10, 0, 0, 0, tzinfo=timezone.utc)


def _make_article(
    url: str = "https://example.com/article",
    content: str = "Article body text.",
    description: str = "Short description.",
    title: str = "Article Title",
    source_name: str = "Reuters",
    published_at: str = "2026-03-15T10:00:00Z",
) -> dict:
    return {
        "url": url,
        "content": content,
        "description": description,
        "title": title,
        "source": {"id": None, "name": source_name},
        "publishedAt": published_at,
    }


@pytest.fixture()
def mock_sleep():
    with patch("freqpred.ingestion.fetchers.newsapi.asyncio.sleep", new_callable=AsyncMock) as m:
        yield m


@pytest.fixture()
def mock_newsapi_client(mock_sleep):
    with patch("freqpred.ingestion.fetchers.newsapi.NewsApiClient") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        yield instance


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_raw_documents(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [
            _make_article(url="https://a.com/1"),
            _make_article(url="https://b.com/2"),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)
    assert docs[0].source_url == "https://a.com/1"
    assert docs[1].source_url == "https://b.com/2"


@pytest.mark.asyncio
async def test_fetch_sets_source_type_news(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [_make_article()]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].source_type == "news"


@pytest.mark.asyncio
async def test_fetch_populates_source_name(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [_make_article(source_name="BBC News")]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].source_name == "BBC News"


@pytest.mark.asyncio
async def test_fetch_source_name_defaults_when_missing(mock_newsapi_client):
    article = _make_article()
    article["source"] = {"id": None, "name": None}
    mock_newsapi_client.get_everything.return_value = {"articles": [article]}
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].source_name == "NewsAPI"


@pytest.mark.asyncio
async def test_fetch_parses_published_at_z_suffix(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [_make_article(published_at="2026-03-15T10:00:00Z")]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].published_at == datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_fetch_missing_published_at_stores_none(mock_newsapi_client):
    article = _make_article()
    article["publishedAt"] = None
    mock_newsapi_client.get_everything.return_value = {"articles": [article]}
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].published_at is None


@pytest.mark.asyncio
async def test_fetch_prefers_content_over_description(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [_make_article(content="Full content.", description="Short.")]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].body == "Full content."


@pytest.mark.asyncio
async def test_fetch_falls_back_to_description(mock_newsapi_client):
    article = _make_article(description="Just a description.")
    article["content"] = ""
    mock_newsapi_client.get_everything.return_value = {"articles": [article]}
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].body == "Just a description."


@pytest.mark.asyncio
async def test_fetch_caps_page_size_at_100(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {"articles": []}
    await fetch(_API_KEY, _QUERY, _FROM, max_results=200)
    call_kwargs = mock_newsapi_client.get_everything.call_args.kwargs
    assert call_kwargs["page_size"] == 100


@pytest.mark.asyncio
async def test_fetch_empty_articles(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {"articles": []}
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_enforces_rate_limit(mock_newsapi_client, mock_sleep):
    mock_newsapi_client.get_everything.return_value = {"articles": []}
    await fetch(_API_KEY, _QUERY, _FROM)
    mock_sleep.assert_called_once_with(_RATE_LIMIT_SLEEP)


# ---------------------------------------------------------------------------
# Skipping invalid articles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_missing_url(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [
            {**_make_article(url=""), "url": ""},
            _make_article(url="https://valid.com/"),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert len(docs) == 1
    assert docs[0].source_url == "https://valid.com/"


@pytest.mark.asyncio
async def test_fetch_skips_missing_body(mock_newsapi_client):
    article = _make_article()
    article["content"] = ""
    article["description"] = ""
    mock_newsapi_client.get_everything.return_value = {
        "articles": [
            article,
            _make_article(url="https://valid.com/", content="Has body."),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert len(docs) == 1
    assert docs[0].source_url == "https://valid.com/"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_api_error(mock_newsapi_client):
    mock_newsapi_client.get_everything.side_effect = RuntimeError("API down")
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs == []


# ---------------------------------------------------------------------------
# Domain blacklist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_blacklisted_domain(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [
            _make_article(url="https://kalshi.com/markets/event", content="Kalshi page."),
            _make_article(url="https://reuters.com/article", content="News body."),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM, excluded_domains=frozenset({"kalshi.com"}))
    assert len(docs) == 1
    assert docs[0].source_url == "https://reuters.com/article"


@pytest.mark.asyncio
async def test_fetch_skips_subdomain_of_blacklisted_domain(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [
            _make_article(url="https://api.kalshi.com/v1/markets", content="API body."),
            _make_article(url="https://valid.com/story", content="Valid body."),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM, excluded_domains=frozenset({"kalshi.com"}))
    assert len(docs) == 1
    assert docs[0].source_url == "https://valid.com/story"


@pytest.mark.asyncio
async def test_fetch_empty_blacklist_keeps_all(mock_newsapi_client):
    mock_newsapi_client.get_everything.return_value = {
        "articles": [
            _make_article(url="https://kalshi.com/foo", content="Body."),
        ]
    }
    docs = await fetch(_API_KEY, _QUERY, _FROM, excluded_domains=frozenset())
    assert len(docs) == 1
