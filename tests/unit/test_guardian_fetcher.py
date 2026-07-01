"""Unit tests for freqpred/ingestion/fetchers/guardian.py.

All HTTP calls are mocked — no real Guardian API calls.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.guardian import (
    GuardianRateLimitError,
    _sanitize_query,
    _strip_html,
    fetch,
)
from freqpred.ingestion.store import RawDocument

_API_KEY = "test-guardian-key"
_QUERY = "UK election results"
_FROM = datetime(2026, 3, 10, 0, 0, 0, tzinfo=UTC)


def _make_item(
    url: str = "https://www.theguardian.com/world/2026/story",
    headline: str = "Article Headline",
    body: str = "<p>Full article body text.</p>",
    published_at: str = "2026-03-15T10:00:00Z",
) -> dict:
    return {
        "id": "world/2026/story",
        "type": "article",
        "webTitle": headline,
        "webUrl": url,
        "webPublicationDate": published_at,
        "fields": {
            "headline": headline,
            "body": body,
        },
    }


def _make_response(items: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"response": {"results": items}}
    return mock


@pytest.fixture()
def mock_sleep():
    with patch("freqpred.ingestion.fetchers.guardian.asyncio.sleep", new_callable=AsyncMock) as m:
        yield m


@pytest.fixture()
def mock_http(mock_sleep):
    """Patches httpx.AsyncClient so no real HTTP is made."""
    with patch("freqpred.ingestion.fetchers.guardian.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        yield mock_client


# ---------------------------------------------------------------------------
# _strip_html helper
# ---------------------------------------------------------------------------


def test_strip_html_removes_tags():
    assert _strip_html("<p>Hello <b>world</b>.</p>") == "Hello world ."


def test_strip_html_unescapes_entities():
    assert _strip_html("&amp; &lt;b&gt;") == "& <b>"


def test_strip_html_collapses_whitespace():
    assert _strip_html("<p>  lots   of   space  </p>") == "lots of space"


def test_strip_html_empty_string():
    assert _strip_html("") == ""


# ---------------------------------------------------------------------------
# _sanitize_query helper
# ---------------------------------------------------------------------------


def test_sanitize_query_strips_site_prefix():
    assert _sanitize_query('site:truthsocial.com Trump "golden dome"') == 'Trump "golden dome"'


def test_sanitize_query_strips_site_prefix_case_insensitive():
    assert _sanitize_query("SITE:example.com foo bar") == "foo bar"


def test_sanitize_query_strips_multiple_site_tokens():
    assert _sanitize_query("site:a.com site:b.com climate policy") == "climate policy"


def test_sanitize_query_passthrough_clean_query():
    assert _sanitize_query('Trump AND "golden dome"') == 'Trump AND "golden dome"'


def test_sanitize_query_empty_after_strip_returns_empty():
    assert _sanitize_query("site:foo.com") == ""


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_raw_documents(mock_http):
    mock_http.get.return_value = _make_response([
        _make_item(url="https://www.theguardian.com/story/1"),
        _make_item(url="https://www.theguardian.com/story/2"),
    ])
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)


@pytest.mark.asyncio
async def test_fetch_sets_source_type_news(mock_http):
    mock_http.get.return_value = _make_response([_make_item()])
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].source_type == "news"


@pytest.mark.asyncio
async def test_fetch_sets_source_name_guardian(mock_http):
    mock_http.get.return_value = _make_response([_make_item()])
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].source_name == "The Guardian"


@pytest.mark.asyncio
async def test_fetch_strips_html_from_body(mock_http):
    mock_http.get.return_value = _make_response([
        _make_item(body="<p>This is <b>bold</b> text.</p>")
    ])
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert "<" not in docs[0].body
    assert "This is bold text." in docs[0].body


@pytest.mark.asyncio
async def test_fetch_uses_fields_headline_as_title(mock_http):
    mock_http.get.return_value = _make_response([
        _make_item(headline="My Headline")
    ])
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].title == "My Headline"


@pytest.mark.asyncio
async def test_fetch_parses_published_at(mock_http):
    mock_http.get.return_value = _make_response([
        _make_item(published_at="2026-03-15T10:00:00Z")
    ])
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].published_at == datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_fetch_missing_published_at_stores_none(mock_http):
    item = _make_item()
    item["webPublicationDate"] = None
    mock_http.get.return_value = _make_response([item])
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs[0].published_at is None


@pytest.mark.asyncio
async def test_fetch_empty_results(mock_http):
    mock_http.get.return_value = _make_response([])
    docs = await fetch(_API_KEY, _QUERY, _FROM)
    assert docs == []


@pytest.mark.asyncio
async def test_fetch_passes_from_date_param(mock_http):
    mock_http.get.return_value = _make_response([])
    await fetch(_API_KEY, _QUERY, _FROM)
    call_kwargs = mock_http.get.call_args
    params = call_kwargs.kwargs.get("params", call_kwargs.args[1] if len(call_kwargs.args) > 1 else {})
    assert params["from-date"] == "2026-03-10"


@pytest.mark.asyncio
async def test_fetch_no_from_date_omits_param(mock_http):
    mock_http.get.return_value = _make_response([])
    await fetch(_API_KEY, _QUERY, from_date=None)
    call_kwargs = mock_http.get.call_args
    params = call_kwargs.kwargs.get("params", {})
    assert "from-date" not in params


@pytest.mark.asyncio
async def test_fetch_passes_show_fields_body(mock_http):
    mock_http.get.return_value = _make_response([])
    await fetch(_API_KEY, _QUERY)
    params = mock_http.get.call_args.kwargs.get("params", {})
    assert "body" in params.get("show-fields", "")


@pytest.mark.asyncio
async def test_fetch_respects_max_results(mock_http):
    mock_http.get.return_value = _make_response([])
    await fetch(_API_KEY, _QUERY, max_results=50)
    params = mock_http.get.call_args.kwargs.get("params", {})
    assert params["page-size"] == 50


@pytest.mark.asyncio
async def test_fetch_caps_page_size_at_200(mock_http):
    mock_http.get.return_value = _make_response([])
    await fetch(_API_KEY, _QUERY, max_results=500)
    params = mock_http.get.call_args.kwargs.get("params", {})
    assert params["page-size"] == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_enforces_rate_limit_sleep(mock_http, mock_sleep):
    mock_http.get.return_value = _make_response([])
    await fetch(_API_KEY, _QUERY)
    mock_sleep.assert_called_once_with(1.0)


@pytest.mark.asyncio
async def test_fetch_raises_guardian_rate_limit_error_on_429(mock_http):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_http.get.return_value = mock_resp
    with pytest.raises(GuardianRateLimitError):
        await fetch(_API_KEY, _QUERY)


# ---------------------------------------------------------------------------
# Skipping invalid items
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_missing_url(mock_http):
    items = [
        {**_make_item(url=""), "webUrl": ""},
        _make_item(url="https://www.theguardian.com/valid"),
    ]
    mock_http.get.return_value = _make_response(items)
    docs = await fetch(_API_KEY, _QUERY)
    assert len(docs) == 1
    assert docs[0].source_url == "https://www.theguardian.com/valid"


@pytest.mark.asyncio
async def test_fetch_skips_missing_body(mock_http):
    item = _make_item()
    item["fields"]["body"] = ""
    mock_http.get.return_value = _make_response([
        item,
        _make_item(url="https://www.theguardian.com/valid", body="<p>Has content.</p>"),
    ])
    docs = await fetch(_API_KEY, _QUERY)
    assert len(docs) == 1
    assert docs[0].source_url == "https://www.theguardian.com/valid"


@pytest.mark.asyncio
async def test_fetch_skips_body_that_is_only_whitespace_after_strip(mock_http):
    item = _make_item(body="<p>   </p>")
    mock_http.get.return_value = _make_response([item])
    docs = await fetch(_API_KEY, _QUERY)
    assert docs == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_http_error(mock_http):
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_http.get.return_value = mock_resp
    docs = await fetch(_API_KEY, _QUERY)
    assert docs == []


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_network_exception(mock_http):
    mock_http.get.side_effect = Exception("connection refused")
    docs = await fetch(_API_KEY, _QUERY)
    assert docs == []


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_json_error(mock_http):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("bad json")
    mock_http.get.return_value = mock_resp
    docs = await fetch(_API_KEY, _QUERY)
    assert docs == []


# ---------------------------------------------------------------------------
# Domain blacklist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_blacklisted_domain(mock_http):
    mock_http.get.return_value = _make_response([
        _make_item(url="https://kalshi.com/markets/event"),
        _make_item(url="https://www.theguardian.com/valid"),
    ])
    docs = await fetch(_API_KEY, _QUERY, excluded_domains=frozenset({"kalshi.com"}))
    assert len(docs) == 1
    assert docs[0].source_url == "https://www.theguardian.com/valid"


@pytest.mark.asyncio
async def test_fetch_empty_blacklist_keeps_all(mock_http):
    mock_http.get.return_value = _make_response([
        _make_item(url="https://kalshi.com/foo"),
    ])
    docs = await fetch(_API_KEY, _QUERY, excluded_domains=frozenset())
    assert len(docs) == 1
