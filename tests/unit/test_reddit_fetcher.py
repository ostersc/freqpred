"""Unit tests for freqpred/ingestion/fetchers/reddit.py (multireddit RSS).

All HTTP calls are mocked — no real Reddit calls.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.reddit import _MAX_AGE_DAYS, RedditBlockedError, fetch
from freqpred.ingestion.store import RawDocument

_SUBREDDITS = ["politics"]
_QUERY = "election results"

_NOW = datetime.now(UTC)
_RECENT = (_NOW - timedelta(days=1)).isoformat()
_OLD = (_NOW - timedelta(days=_MAX_AGE_DAYS + 1)).isoformat()


def _make_entry(
    title: str = "Test Post",
    content: str = "<p>Some content about the election.</p>",
    published: str | None = None,
    href: str = "https://www.reddit.com/r/politics/comments/abc/test_post/",
    subreddit: str | None = "politics",
) -> str:
    published_el = f"<published>{published or _RECENT}</published>"
    category_el = (
        f'<category term="{subreddit}" label="r/{subreddit}"/>' if subreddit else ""
    )
    return f"""
    <entry>
        {category_el}
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


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    """Disable the inter-request throttle so unit tests don't sleep."""
    monkeypatch.setattr("freqpred.ingestion.fetchers.reddit._REQUEST_SPACING_SECONDS", 0)


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
async def test_fetch_source_name_from_entry_category(mock_httpx):
    """In a multireddit search the entry's <category term> identifies its
    actual subreddit — required for per-source Brier attribution."""
    mock_httpx.get.return_value = _make_http_response([
        _make_entry(subreddit="PoliticalDiscussion",
                    href="https://www.reddit.com/r/PoliticalDiscussion/comments/1/a/"),
        _make_entry(subreddit="Conservative",
                    href="https://www.reddit.com/r/Conservative/comments/2/b/"),
    ])

    docs = await fetch(["politics", "PoliticalDiscussion", "Conservative"], _QUERY)

    assert {d.source_name for d in docs} == {"r/PoliticalDiscussion", "r/Conservative"}


@pytest.mark.asyncio
async def test_fetch_source_name_falls_back_to_first_subreddit(mock_httpx):
    mock_httpx.get.return_value = _make_http_response([_make_entry(subreddit=None)])

    docs = await fetch(["politics", "Conservative"], _QUERY)

    assert docs[0].source_name == "r/politics"


@pytest.mark.asyncio
async def test_fetch_single_multireddit_request(mock_httpx):
    """N subreddits must cost exactly one HTTP request — unauthenticated
    tolerance is ~10 req/min, so request count is the scarce resource."""
    mock_httpx.get.return_value = _make_http_response([])

    await fetch(["politics", "PoliticalDiscussion", "neutralpolitics"], _QUERY)

    assert mock_httpx.get.call_count == 1
    path = mock_httpx.get.call_args[0][0]
    assert path == "/r/politics+PoliticalDiscussion+neutralpolitics/search.rss"


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
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_404_is_silent_skip_not_failure(mock_httpx):
    """404 = no such subreddit(s) — a config issue, never raises."""
    mock_httpx.get.return_value = _make_http_response([], status_code=404)

    docs = await fetch(_SUBREDDITS, _QUERY)

    assert docs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 429])
async def test_fetch_blocked_status_raises(mock_httpx, status):
    """403/429 on the search means Reddit is blocking us — raise so the
    scheduler trips backoff, never die silently."""
    mock_httpx.get.return_value = _make_http_response([], status_code=status)

    with pytest.raises(RedditBlockedError):
        await fetch(_SUBREDDITS, _QUERY)


@pytest.mark.asyncio
async def test_fetch_transport_error_raises(mock_httpx):
    mock_httpx.get.side_effect = Exception("nodename nor servname provided")

    with pytest.raises(RedditBlockedError):
        await fetch(["a", "b"], _QUERY)


@pytest.mark.asyncio
async def test_fetch_unparseable_feed_raises(mock_httpx):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "not xml at all"
    resp.raise_for_status = MagicMock()
    mock_httpx.get.return_value = resp

    with pytest.raises(RedditBlockedError):
        await fetch(_SUBREDDITS, _QUERY)


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


# ---------------------------------------------------------------------------
# Request throttling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_throttle_spaces_consecutive_requests(mock_httpx, monkeypatch):
    """Back-to-back fetch calls must sleep to honor the global spacing —
    Reddit's unauthenticated tolerance is ~10 requests/min per IP."""
    import freqpred.ingestion.fetchers.reddit as reddit_mod

    monkeypatch.setattr(reddit_mod, "_REQUEST_SPACING_SECONDS", 61.0)
    monkeypatch.setattr(reddit_mod, "_last_request_at", 0.0)

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(reddit_mod.asyncio, "sleep", fake_sleep)
    mock_httpx.get.return_value = _make_http_response([])

    await fetch(["a"], _QUERY)
    await fetch(["b"], _QUERY)
    await fetch(["c"], _QUERY)

    # First call goes through immediately; the next two must wait.
    assert len(sleeps) == 2
    assert all(0 < s <= 61.0 for s in sleeps)
