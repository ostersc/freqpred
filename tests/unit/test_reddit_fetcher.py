"""Unit tests for freqpred/ingestion/fetchers/reddit.py.

All HTTP calls are mocked — no real Reddit API calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from freqpred.ingestion.fetchers.reddit import _MAX_AGE_DAYS, _MIN_SCORE, fetch
from freqpred.ingestion.store import RawDocument

_SUBREDDITS = ["politics"]
_QUERY = "election results"

_NOW = datetime.now(timezone.utc)
_RECENT_TS = (_NOW - timedelta(days=1)).timestamp()
_OLD_TS = (_NOW - timedelta(days=_MAX_AGE_DAYS + 1)).timestamp()


def _make_post(
    title: str = "Test Post",
    selftext: str = "Some content about the election.",
    score: int = 100,
    created_utc: float | None = None,
    permalink: str = "/r/politics/comments/abc/test_post/",
    subreddit: str = "politics",
) -> dict:
    return {
        "kind": "t3",
        "data": {
            "title": title,
            "selftext": selftext,
            "score": score,
            "created_utc": created_utc if created_utc is not None else _RECENT_TS,
            "permalink": permalink,
            "subreddit": subreddit,
        },
    }


def _make_response(posts: list[dict]) -> dict:
    return {"data": {"children": posts}}


def _make_http_response(posts: list[dict], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = _make_response(posts)
    resp.raise_for_status = MagicMock()
    resp.aread = AsyncMock()
    return resp


def _make_stream_cm(resp: MagicMock) -> MagicMock:
    """Wrap a response mock in an async context manager for client.stream()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.fixture()
def mock_httpx(monkeypatch):
    """Patch httpx.AsyncClient and return a mock that yields a configured instance."""
    mock_client = AsyncMock()
    # stream() is not awaited — it returns an async context manager directly.
    # Replace the AsyncMock attribute with a plain MagicMock so calling it
    # returns a context manager rather than a coroutine.
    mock_client.stream = MagicMock()

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
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([
        _make_post(permalink="/r/politics/comments/1/a/"),
        _make_post(permalink="/r/politics/comments/2/b/"),
    ]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)


@pytest.mark.asyncio
async def test_fetch_sets_source_type_reddit(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([_make_post()]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].source_type == "reddit"


@pytest.mark.asyncio
async def test_fetch_sets_source_name_with_subreddit(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(
        _make_http_response([_make_post(subreddit="politics")])
    )

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].source_name == "r/politics"


@pytest.mark.asyncio
async def test_fetch_sets_source_url_from_permalink(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([
        _make_post(permalink="/r/politics/comments/abc/test/")
    ]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].source_url == "https://reddit.com/r/politics/comments/abc/test/"


@pytest.mark.asyncio
async def test_fetch_body_uses_selftext(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([
        _make_post(selftext="Self text content.", title="Title")
    ]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].body == "Self text content."


@pytest.mark.asyncio
async def test_fetch_body_falls_back_to_title(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([
        _make_post(selftext="", title="Just a link title")
    ]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs[0].body == "Just a link title"


@pytest.mark.asyncio
async def test_fetch_published_at_is_utc(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([_make_post(created_utc=_RECENT_TS)]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    expected = datetime.fromtimestamp(_RECENT_TS, tz=timezone.utc)
    assert docs[0].published_at == expected


# ---------------------------------------------------------------------------
# Filtering: upvote score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_low_score(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([
        _make_post(score=_MIN_SCORE - 1, permalink="/r/p/comments/low/"),
        _make_post(score=_MIN_SCORE, permalink="/r/p/comments/high/"),
    ]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 1
    assert docs[0].source_url == "https://reddit.com/r/p/comments/high/"


@pytest.mark.asyncio
async def test_fetch_includes_exact_min_score(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([_make_post(score=_MIN_SCORE)]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 1


# ---------------------------------------------------------------------------
# Filtering: recency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_old_posts(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([
        _make_post(created_utc=_OLD_TS, permalink="/r/p/comments/old/"),
        _make_post(created_utc=_RECENT_TS, permalink="/r/p/comments/new/"),
    ]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 1
    assert "new" in docs[0].source_url


# ---------------------------------------------------------------------------
# Filtering: empty body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_empty_body(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([
        _make_post(selftext="", title=""),
        _make_post(selftext="Has content.", permalink="/r/p/comments/valid/"),
    ]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert len(docs) == 1
    assert docs[0].source_url == "https://reddit.com/r/p/comments/valid/"


# ---------------------------------------------------------------------------
# Multiple subreddits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_aggregates_across_subreddits(mock_httpx):
    mock_httpx.stream.side_effect = [
        _make_stream_cm(_make_http_response([_make_post(permalink="/r/a/comments/1/", subreddit="a")])),
        _make_stream_cm(_make_http_response([_make_post(permalink="/r/b/comments/2/", subreddit="b")])),
    ]

    docs = await fetch(["a", "b"], _QUERY)

    assert len(docs) == 2


@pytest.mark.asyncio
async def test_fetch_calls_correct_subreddit_path(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([]))

    await fetch(["worldnews"], _QUERY)

    call_args = mock_httpx.stream.call_args
    # stream("GET", path, ...) — path is the second positional arg
    assert "worldnews" in call_args[0][1]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_http_error(mock_httpx):
    mock_httpx.stream.side_effect = Exception("connection refused")

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_skips_subreddit_on_http_error_continues(mock_httpx):
    mock_httpx.stream.side_effect = [
        Exception("rate limited"),
        _make_stream_cm(_make_http_response([_make_post(permalink="/r/b/comments/2/")])),
    ]

    docs = await fetch(["a", "b"], _QUERY)

    assert len(docs) == 1


@pytest.mark.asyncio
async def test_fetch_empty_subreddits_returns_empty(mock_httpx):
    docs = await fetch([], _QUERY)
    assert docs == []
    mock_httpx.stream.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_empty_posts_returns_empty(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([]))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_403_returns_empty_silently(mock_httpx):
    """403s should be swallowed quietly (subreddit restricted/rate-limited)."""
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([], status_code=403))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_404_returns_empty_silently(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([], status_code=404))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_429_returns_empty_silently(mock_httpx):
    mock_httpx.stream.return_value = _make_stream_cm(_make_http_response([], status_code=429))

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_continues_after_403(mock_httpx):
    """403 on first subreddit should not prevent fetching from the second."""
    mock_httpx.stream.side_effect = [
        _make_stream_cm(_make_http_response([], status_code=403)),
        _make_stream_cm(_make_http_response([_make_post(permalink="/r/b/comments/1/")])),
    ]

    docs = await fetch(["restricted", "b"], _QUERY)

    assert len(docs) == 1
