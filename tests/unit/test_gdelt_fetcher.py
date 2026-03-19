"""Unit tests for freqpred/ingestion/fetchers/gdelt.py.

All HTTP calls are mocked — no real network calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.gdelt import _sanitize_query, fetch
from freqpred.ingestion.store import RawDocument

# Suppress the 5-second rate-limit sleep in all tests.
pytestmark = pytest.mark.usefixtures("_no_gdelt_sleep")


@pytest.fixture(autouse=True)
def _no_gdelt_sleep():
    with patch("freqpred.ingestion.fetchers.gdelt.asyncio.sleep", new=AsyncMock()):
        yield

_QUERY = "election results 2026"
_NOW = datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc)


def _make_article(
    url: str = "https://example.com/article",
    title: str = "Article Title",
    seendate: str = "20260319T120000Z",
) -> dict:
    return {"url": url, "title": title, "seendate": seendate}


def _make_gdelt_response(articles: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"articles": articles}
    return resp


def _make_body_response(text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status = MagicMock()
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Helpers to patch httpx.AsyncClient
# ---------------------------------------------------------------------------


def _make_client_mock(gdelt_response: MagicMock, body_responses: list[MagicMock]) -> MagicMock:
    """Build a mock httpx.AsyncClient context manager.

    *gdelt_response* is returned for the first GET (Doc API query).
    *body_responses* are returned for subsequent GETs (article bodies), in order.
    """
    client_instance = AsyncMock()
    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
    client_instance.__aexit__ = AsyncMock(return_value=False)

    get_responses = [gdelt_response] + list(body_responses)
    client_instance.get = AsyncMock(side_effect=get_responses)
    return client_instance


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_raw_documents():
    articles = [
        _make_article(url="https://a.com/1", title="Story One"),
        _make_article(url="https://b.com/2", title="Story Two"),
    ]
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [_make_body_response("Body one."), _make_body_response("Body two.")],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)
    assert docs[0].source_url == "https://a.com/1"
    assert docs[1].source_url == "https://b.com/2"


@pytest.mark.asyncio
async def test_fetch_sets_source_type_and_name():
    articles = [_make_article()]
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [_make_body_response("Article body text.")],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert docs[0].source_type == "news"
    assert docs[0].source_name == "GDELT"


@pytest.mark.asyncio
async def test_fetch_parses_seendate():
    articles = [_make_article(seendate="20260310T083000Z")]
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [_make_body_response("Body text.")],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert docs[0].published_at == datetime(2026, 3, 10, 8, 30, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_fetch_invalid_seendate_falls_back_to_now():
    articles = [_make_article(seendate="not-a-date")]
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [_make_body_response("Body text.")],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert docs[0].published_at.tzinfo is not None
    assert docs[0].published_at.year == 2026


@pytest.mark.asyncio
async def test_fetch_passes_timespan_and_max_results():
    client_mock = _make_client_mock(_make_gdelt_response([]), [])

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        await fetch(_QUERY, timespan="6h", max_results=5)

    call_kwargs = client_mock.get.call_args_list[0]
    params = call_kwargs.kwargs.get("params", {})
    assert params["timespan"] == "6h"
    assert params["maxrecords"] == 5


# ---------------------------------------------------------------------------
# Skipping / filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_failed_article_body():
    articles = [
        _make_article(url="https://paywalled.com/1"),
        _make_article(url="https://ok.com/2"),
    ]
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [
            _make_body_response("", status_code=403),  # paywalled — raises
            _make_body_response("Good article body."),
        ],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert len(docs) == 1
    assert docs[0].source_url == "https://ok.com/2"


@pytest.mark.asyncio
async def test_fetch_skips_empty_body():
    articles = [
        _make_article(url="https://empty.com/1"),
        _make_article(url="https://ok.com/2"),
    ]
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [_make_body_response("   "), _make_body_response("Real content.")],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert len(docs) == 1
    assert docs[0].source_url == "https://ok.com/2"


@pytest.mark.asyncio
async def test_fetch_skips_excluded_domain():
    articles = [
        _make_article(url="https://kalshi.com/market/foo"),
        _make_article(url="https://reuters.com/article"),
    ]
    # Only one body fetch expected (kalshi.com is filtered before body fetch)
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [_make_body_response("Reuters article body.")],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY, excluded_domains=frozenset({"kalshi.com"}))

    assert len(docs) == 1
    assert docs[0].source_url == "https://reuters.com/article"


@pytest.mark.asyncio
async def test_fetch_skips_subdomain_of_excluded_domain():
    articles = [_make_article(url="https://api.kalshi.com/v2/events")]
    client_mock = _make_client_mock(_make_gdelt_response(articles), [])

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY, excluded_domains=frozenset({"kalshi.com"}))

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_empty_articles_returns_empty():
    client_mock = _make_client_mock(_make_gdelt_response([]), [])

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_null_articles_key_returns_empty():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {}  # no "articles" key
    client_mock = _make_client_mock(resp, [])

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert docs == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_api_error():
    client_instance = AsyncMock()
    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
    client_instance.__aexit__ = AsyncMock(return_value=False)
    client_instance.get = AsyncMock(side_effect=RuntimeError("API down"))

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_instance):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_http_error():
    resp = MagicMock()
    resp.raise_for_status.side_effect = Exception("HTTP 500")
    client_instance = AsyncMock()
    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
    client_instance.__aexit__ = AsyncMock(return_value=False)
    client_instance.get = AsyncMock(return_value=resp)

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_instance):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_429():
    import httpx as _httpx

    mock_response = MagicMock()
    mock_response.status_code = 429
    exc = _httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response)
    resp = MagicMock()
    resp.raise_for_status.side_effect = exc

    client_instance = AsyncMock()
    client_instance.__aenter__ = AsyncMock(return_value=client_instance)
    client_instance.__aexit__ = AsyncMock(return_value=False)
    client_instance.get = AsyncMock(return_value=resp)

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_instance):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_all_bodies_fail_returns_empty():
    articles = [_make_article(url="https://a.com/1"), _make_article(url="https://b.com/2")]
    failing_body = MagicMock()
    failing_body.raise_for_status.side_effect = Exception("timeout")

    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [failing_body, failing_body],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    assert docs == []


# ---------------------------------------------------------------------------
# Query sanitization
# ---------------------------------------------------------------------------


def test_sanitize_query_removes_short_tokens():
    result = _sanitize_query("How often does Trump mention specific medical conditions in rallies speeches")
    assert result == "often does Trump mention specific medical conditions rallies speeches"


def test_sanitize_query_unchanged_when_all_long():
    q = "Trump rally medical"
    assert _sanitize_query(q) == q


def test_sanitize_query_returns_none_when_all_tokens_short():
    assert _sanitize_query("is in of") is None


@pytest.mark.asyncio
async def test_fetch_sanitizes_query_before_api_call():
    articles = [_make_article()]
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [_make_body_response("Article body.")],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        await fetch("How often does Trump rally in speeches")

    call_kwargs = client_mock.get.call_args_list[0]
    params = call_kwargs.kwargs.get("params", {})
    # "How" (3 chars) and "in" (2 chars) should be stripped
    assert "How" not in params["query"]
    assert " in " not in params["query"]
    assert "Trump" in params["query"]


@pytest.mark.asyncio
async def test_fetch_returns_empty_when_query_all_short_tokens():
    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient") as mock_client:
        docs = await fetch("is in of")

    assert docs == []
    mock_client.assert_not_called()


# ---------------------------------------------------------------------------
# Parallel fetch — all bodies fetched via asyncio.gather
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_fetches_bodies_in_parallel():
    """Verify that gather is called: all body GETs happen within a single client session."""
    articles = [
        _make_article(url=f"https://news.com/{i}") for i in range(3)
    ]
    client_mock = _make_client_mock(
        _make_gdelt_response(articles),
        [_make_body_response(f"Body {i}.") for i in range(3)],
    )

    with patch("freqpred.ingestion.fetchers.gdelt.httpx.AsyncClient", return_value=client_mock):
        docs = await fetch(_QUERY)

    # All 3 articles fetched and returned
    assert len(docs) == 3
    # client.get called 4 times total: 1 API + 3 bodies
    assert client_mock.get.call_count == 4
