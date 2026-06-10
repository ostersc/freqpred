"""Unit tests for freqpred/ingestion/fetchers/reddit.py (RSS-based).

All HTTP calls are mocked — no real Reddit calls.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.reddit import _MAX_AGE_DAYS, RedditBlockedError, fetch
from freqpred.ingestion.store import RawDocument

_SUBREDDITS = ["politics"]
_QUERY = "election results"

_NOW = datetime.now(timezone.utc)
_RECENT = (_NOW - timedelta(days=1)).isoformat()
_OLD = (_NOW - timedelta(days=_MAX_AGE_DAYS + 1)).isoformat()


def _make_entry(
    title: str = "Test Post",
    content: str = "<p>Some content about the election.</p>",
    published: str | None = None,
    href: str = "https://www.reddit.com/r/politics/comments/abc/test_post/",
) -> str:
    published_el = f"<published>{published or _RECENT}</published>"
    return f"""
    <entry>
        <title>{title}</title>
        <link href="{href}"/>
        {published_el}
        <content type="html">{content}</content>
    </entry>
    """


def _make_feed(entries: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        + "".join(entries)
        + "</feed>"
    )


def _make_http_response(entries: list[str], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = _make_feed(entries)
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture()
def mock_httpx():
    """Patch httpx.AsyncClient and return the mock client instance."""
    mock_client = AsyncMock()

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("freqpred.ingestion.fetchers.reddit.httpx.AsyncClient", return_value=mock_cm):
        yield mock_client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_raw_documents(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([
        _make_entry(href="https://www.reddit.com/r/politics/comments/1/a/"),
        _make_entry(href="https://www.reddit.com/r/politics/comments/2/b/"),
    ])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)


@pytest.mark.asyncio
async def test_fetch_sets_source_type_reddit(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([_make_entry()])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].source_type == "reddit"


@pytest.mark.asyncio
async def test_fetch_sets_source_name_with_subreddit(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([_make_entry()])

    docs = await fetch(["politics"], _QUERY)

    assert docs[0].source_name == "r/politics"


@pytest.mark.asyncio
async def test_fetch_sets_source_url_from_link(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([
        _make_entry(href="https://www.reddit.com/r/politics/comments/abc/test/")
    ])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].source_url == "https://www.reddit.com/r/politics/comments/abc/test/"


@pytest.mark.asyncio
async def test_fetch_body_strips_html(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([
        _make_entry(content="&lt;p&gt;Self &lt;b&gt;text&lt;/b&gt; content.&lt;/p&gt;")
    ])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].body == "Self text content."


@pytest.mark.asyncio
async def test_fetch_body_strips_submitted_by_footer(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([
        _make_entry(
            content="&lt;p&gt;Real content here.&lt;/p&gt; submitted by /u/someuser [link] [comments]"
        )
    ])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].body == "Real content here."


@pytest.mark.asyncio
async def test_fetch_body_falls_back_to_title(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([
        _make_entry(title="Just a link title", content="")
    ])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].body == "Just a link title"


@pytest.mark.asyncio
async def test_fetch_published_at_is_utc(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([_make_entry(published=_RECENT)])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].published_at == datetime.fromisoformat(_RECENT)
    assert docs[0].published_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Filtering: recency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_old_posts(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([
        _make_entry(published=_OLD, href="https://www.reddit.com/r/p/comments/old/"),
        _make_entry(published=_RECENT, href="https://www.reddit.com/r/p/comments/new/"),
    ])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 1
    assert "new" in docs[0].source_url


# ---------------------------------------------------------------------------
# Filtering: empty body / missing link
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_empty_body_and_title(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([
        _make_entry(title="", content=""),
        _make_entry(content="Has content.", href="https://www.reddit.com/r/p/comments/valid/"),
    ])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 1
    assert docs[0].source_url == "https://www.reddit.com/r/p/comments/valid/"


@pytest.mark.asyncio
async def test_fetch_skips_entry_without_link(mock_httpx):
    feed = _make_feed([
        "<entry><title>No link</title><content>body</content></entry>",
        _make_entry(href="https://www.reddit.com/r/p/comments/ok/"),
    ])
    resp = MagicMock()
    resp.status_code = 200
    resp.text = feed
    resp.raise_for_status = MagicMock()
    mock_httpx.get.return_value = resp

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 1
    assert docs[0].source_url == "https://www.reddit.com/r/p/comments/ok/"


# ---------------------------------------------------------------------------
# Multiple subreddits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_aggregates_across_subreddits(mock_httpx):
    mock_httpx.get.side_effect = [
        _make_http_response([_make_entry(href="https://www.reddit.com/r/a/comments/1/")]),
        _make_http_response([_make_entry(href="https://www.reddit.com/r/b/comments/2/")]),
    ]

    docs = await fetch(["a", "b"], _QUERY)

    assert len(docs) == 2


@pytest.mark.asyncio
async def test_fetch_calls_correct_subreddit_rss_path(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([])

    await fetch(["worldnews"], _QUERY)

    call_args = mock_httpx.get.call_args
    path = call_args[0][0]
    assert "worldnews" in path
    assert path.endswith("search.rss")


# ---------------------------------------------------------------------------
# Error handling: per-subreddit skips vs blanket blocking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_404_is_silent_skip_not_failure(mock_httpx):
    """404 = subreddit doesn't exist — never raises, even when it's the only one."""
    mock_httpx.get.return_value = _make_http_response([], status_code=404)

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_single_403_among_successes_continues(mock_httpx):
    """One blocked subreddit must not prevent fetching from the others."""
    mock_httpx.get.side_effect = [
        _make_http_response([], status_code=403),
        _make_http_response([_make_entry(href="https://www.reddit.com/r/b/comments/1/")]),
    ]

    docs = await fetch(["restricted", "b"], _QUERY)

    assert len(docs) == 1


@pytest.mark.asyncio
async def test_fetch_blanket_403_raises_blocked_error(mock_httpx):
    """Every subreddit 403ing means Reddit is blocking us — raise, don't die silently."""
    mock_httpx.get.side_effect = [
        _make_http_response([], status_code=403),
        _make_http_response([], status_code=403),
    ]

    with pytest.raises(RedditBlockedError):
        await fetch(["a", "b"], _QUERY)


@pytest.mark.asyncio
async def test_fetch_blanket_transport_error_raises_blocked_error(mock_httpx):
    """DNS/connect failures on every subreddit are a blanket failure too."""
    mock_httpx.get.side_effect = Exception("nodename nor servname provided")

    with pytest.raises(RedditBlockedError):
        await fetch(["a", "b"], _QUERY)


@pytest.mark.asyncio
async def test_fetch_all_failed_except_404_still_raises(mock_httpx):
    """404s are excluded from the blanket count: 404 + 403 with no success → blocked."""
    mock_httpx.get.side_effect = [
        _make_http_response([], status_code=404),
        _make_http_response([], status_code=403),
    ]

    with pytest.raises(RedditBlockedError):
        await fetch(["gone", "blocked"], _QUERY)


@pytest.mark.asyncio
async def test_fetch_transport_error_then_success_does_not_raise(mock_httpx):
    mock_httpx.get.side_effect = [
        Exception("rate limited"),
        _make_http_response([_make_entry(href="https://www.reddit.com/r/b/comments/2/")]),
    ]

    docs = await fetch(["a", "b"], _QUERY)

    assert len(docs) == 1


@pytest.mark.asyncio
async def test_fetch_empty_subreddits_returns_empty(mock_httpx):
    docs = await fetch([], _QUERY)
    assert docs == []
    mock_httpx.get.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_empty_feed_returns_empty(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([])

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs == []
