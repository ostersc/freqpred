"""Unit tests for freqpred/ingestion/fetchers/truthsocial.py.

All truthbrush.Api calls are mocked — no real network calls.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.truthsocial import (
    LoginErrorException,
    _strip_html,
    fetch_account,
    fetch_search,
)
from freqpred.ingestion.store import RawDocument

_NOW = datetime.now(UTC)
_RECENT_DT = _NOW - timedelta(hours=1)
_OLD_DT = _NOW - timedelta(hours=72)


def _make_status(
    url: str = "https://truthsocial.com/@realDonaldTrump/123",
    content: str = "<p>Hello world</p>",
    created_at: str | None = None,
    account_username: str = "realDonaldTrump",
    reblog: dict | None = None,
) -> dict:
    return {
        "id": "123",
        "url": url,
        "content": content,
        "created_at": created_at or _RECENT_DT.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "account": {"username": account_username, "display_name": account_username},
        "reblog": reblog,
    }


def _make_api(statuses: list[dict] | None = None, search_pages: list[dict] | None = None) -> MagicMock:
    """Return a mock truthbrush.Api object."""
    api = MagicMock()
    api.pull_statuses.return_value = iter(statuses or [])
    api.search.return_value = iter(search_pages or [{"statuses": statuses or []}])
    return api


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------


def test_strip_html_removes_tags():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_empty_string():
    assert _strip_html("") == ""


def test_strip_html_plain_text():
    assert _strip_html("no tags here") == "no tags here"


# ---------------------------------------------------------------------------
# fetch_search — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_search_returns_raw_documents():
    api = _make_api(statuses=[_make_status()])
    docs = await fetch_search(api, query="election")
    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)


@pytest.mark.asyncio
async def test_fetch_search_source_type_social():
    api = _make_api(statuses=[_make_status()])
    docs = await fetch_search(api, query="election")
    assert docs[0].source_type == "social"


@pytest.mark.asyncio
async def test_fetch_search_source_name_truthsocial():
    api = _make_api(statuses=[_make_status()])
    docs = await fetch_search(api, query="election")
    assert docs[0].source_name == "TruthSocial"


@pytest.mark.asyncio
async def test_fetch_search_body_strips_html():
    api = _make_api(statuses=[_make_status(content="<p>Hello <b>world</b></p>")])
    docs = await fetch_search(api, query="election")
    assert docs[0].body == "Hello world"


@pytest.mark.asyncio
async def test_fetch_search_url_set():
    api = _make_api(statuses=[_make_status(url="https://truthsocial.com/@user/456")])
    docs = await fetch_search(api, query="election")
    assert docs[0].source_url == "https://truthsocial.com/@user/456"


@pytest.mark.asyncio
async def test_fetch_search_title_includes_username():
    api = _make_api(statuses=[_make_status(account_username="testuser")])
    docs = await fetch_search(api, query="election")
    assert "@testuser" in docs[0].title


# ---------------------------------------------------------------------------
# fetch_search — filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_search_skips_old_posts():
    old_status = _make_status(
        url="https://truthsocial.com/@u/old",
        created_at=_OLD_DT.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    )
    recent_status = _make_status(url="https://truthsocial.com/@u/recent")
    api = MagicMock()
    api.search.return_value = iter([{"statuses": [old_status, recent_status]}])

    created_after = _NOW - timedelta(hours=48)
    docs = await fetch_search(api, query="test", created_after=created_after)

    assert len(docs) == 1
    assert docs[0].source_url == "https://truthsocial.com/@u/recent"


@pytest.mark.asyncio
async def test_fetch_search_skips_reblogs():
    reblogged = _make_status(reblog={"id": "999"})
    normal = _make_status(url="https://truthsocial.com/@u/normal")
    api = MagicMock()
    api.search.return_value = iter([{"statuses": [reblogged, normal]}])

    docs = await fetch_search(api, query="test")

    assert len(docs) == 1
    assert docs[0].source_url == "https://truthsocial.com/@u/normal"


@pytest.mark.asyncio
async def test_fetch_search_skips_missing_url():
    no_url = _make_status(url="")
    valid = _make_status(url="https://truthsocial.com/@u/valid")
    api = MagicMock()
    api.search.return_value = iter([{"statuses": [no_url, valid]}])

    docs = await fetch_search(api, query="test")

    assert len(docs) == 1
    assert docs[0].source_url == "https://truthsocial.com/@u/valid"


@pytest.mark.asyncio
async def test_fetch_search_skips_empty_body():
    empty_body = _make_status(content="<p></p>")
    valid = _make_status(url="https://truthsocial.com/@u/v", content="<p>Real content</p>")
    api = MagicMock()
    api.search.return_value = iter([{"statuses": [empty_body, valid]}])

    docs = await fetch_search(api, query="test")

    assert len(docs) == 1


@pytest.mark.asyncio
async def test_fetch_search_skips_excluded_domain():
    excluded = _make_status(url="https://kalshi.com/@u/123")
    valid = _make_status(url="https://truthsocial.com/@u/123")
    api = MagicMock()
    api.search.return_value = iter([{"statuses": [excluded, valid]}])

    docs = await fetch_search(api, query="test", excluded_domains=frozenset({"kalshi.com"}))

    assert len(docs) == 1
    assert docs[0].source_url == "https://truthsocial.com/@u/123"


@pytest.mark.asyncio
async def test_fetch_search_empty_results():
    api = _make_api(statuses=[])
    docs = await fetch_search(api, query="test")
    assert docs == []


@pytest.mark.asyncio
async def test_fetch_search_uses_default_lookback_when_no_created_after():
    """Without created_after, posts older than 48h are excluded."""
    old_status = _make_status(
        url="https://truthsocial.com/@u/old",
        created_at=(_NOW - timedelta(hours=50)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    )
    api = MagicMock()
    api.search.return_value = iter([{"statuses": [old_status]}])

    with patch("freqpred.ingestion.fetchers.truthsocial.datetime") as mock_dt:
        mock_dt.now.return_value = _NOW
        mock_dt.fromisoformat.side_effect = datetime.fromisoformat
        docs = await fetch_search(api, query="test")

    assert docs == []


# ---------------------------------------------------------------------------
# fetch_search — error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_search_returns_empty_on_error():
    api = MagicMock()
    api.search.side_effect = RuntimeError("network error")

    docs = await fetch_search(api, query="test")

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_search_propagates_login_error():
    api = MagicMock()
    api.search.side_effect = LoginErrorException("bad credentials")

    with pytest.raises(LoginErrorException):
        await fetch_search(api, query="test")


# ---------------------------------------------------------------------------
# fetch_account — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_account_returns_raw_documents():
    api = MagicMock()
    api.pull_statuses.return_value = iter([_make_status()])

    docs = await fetch_account(api, username="realDonaldTrump", created_after=_OLD_DT)

    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)


@pytest.mark.asyncio
async def test_fetch_account_source_type_social():
    api = MagicMock()
    api.pull_statuses.return_value = iter([_make_status()])

    docs = await fetch_account(api, username="realDonaldTrump", created_after=_OLD_DT)

    assert docs[0].source_type == "social"


@pytest.mark.asyncio
async def test_fetch_account_passes_created_after_to_truthbrush():
    api = MagicMock()
    api.pull_statuses.return_value = iter([])

    cutoff = _NOW - timedelta(hours=6)
    await fetch_account(api, username="user", created_after=cutoff)

    api.pull_statuses.assert_called_once_with("user", created_after=cutoff)


@pytest.mark.asyncio
async def test_fetch_account_skips_reblogs():
    reblog_status = _make_status(reblog={"id": "999"})
    api = MagicMock()
    api.pull_statuses.return_value = iter([reblog_status])

    docs = await fetch_account(api, username="user", created_after=_OLD_DT)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_account_excludes_domains():
    api = MagicMock()
    api.pull_statuses.return_value = iter([
        _make_status(url="https://kalshi.com/@u/1"),
        _make_status(url="https://truthsocial.com/@u/2"),
    ])

    docs = await fetch_account(
        api, username="user", created_after=_OLD_DT, excluded_domains=frozenset({"kalshi.com"})
    )

    assert len(docs) == 1
    assert docs[0].source_url == "https://truthsocial.com/@u/2"


# ---------------------------------------------------------------------------
# fetch_account — error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_account_returns_empty_on_error():
    api = MagicMock()
    api.pull_statuses.side_effect = RuntimeError("network error")

    docs = await fetch_account(api, username="user", created_after=_OLD_DT)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_account_propagates_login_error():
    api = MagicMock()
    api.pull_statuses.side_effect = LoginErrorException("bad credentials")

    with pytest.raises(LoginErrorException):
        await fetch_account(api, username="user", created_after=_OLD_DT)
