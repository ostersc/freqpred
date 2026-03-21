"""Unit tests for freqpred/ingestion/fetchers/tv_archive.py.

All HTTP calls are mocked — no real network calls.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.tv_archive import _strip_highlight_markers, _build_filter_map, fetch
from freqpred.ingestion.store import RawDocument

_QUERY = 'trump AND ("communist" OR "communism")'
_NOW = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_CLOSE_TIME = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(
    identifier: str = "CNN_20260320_140000_NewsRoom",
    title: str = "News Room : CNN : March 20, 2026",
    date: str = "2026-03-20T00:00:00Z",
    href: str = "/details/CNN_20260320_140000_NewsRoom/start/885/end/945?q=trump",
    highlight_text: str = "what's motivating president {{{trump}}} to mention {{{communism}}}",
    subject: list | None = None,
) -> dict:
    return {
        "fields": {
            "identifier": identifier,
            "title": title,
            "date": date,
            "creator": ["CNN"],
            "__href__": href,
            "subject": subject or ["trump", "communism"],
        },
        "highlight": {
            "text": [highlight_text],
        },
    }


def _make_response(hits: list[dict], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=MagicMock(status_code=status_code),
        )
    else:
        resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "response": {
            "body": {
                "hits": {
                    "hits": hits,
                }
            }
        }
    }
    return resp


def _make_client_mock(response: MagicMock) -> MagicMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def test_strip_highlight_markers_removes_braces():
    assert _strip_highlight_markers("president {{{trump}}} said {{{communism}}}") == "president trump said communism"


def test_strip_highlight_markers_no_markers():
    assert _strip_highlight_markers("plain text") == "plain text"


def test_build_filter_map_uses_close_time():
    close = datetime(2026, 3, 30, tzinfo=timezone.utc)
    fm = _build_filter_map(close)
    assert fm["language"] == {"English": "inc"}
    assert fm["program"] == {"News": "inc"}
    # end month should be 2026-03 (March, since close is in March)
    assert "2026-03" in fm["date"].values() or "2026-03" in fm["date"]


def test_build_filter_map_no_close_time_uses_now():
    fm = _build_filter_map(None)
    # Should still produce a valid date range
    assert "date" in fm
    date_range = fm["date"]
    assert any(op in ("gte", "lte") for op in date_range.values())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_raw_documents():
    hits = [_make_hit(title="Clip One"), _make_hit(title="Clip Two", href="/details/CNN/start/100/end/160?q=trump")]
    client = _make_client_mock(_make_response(hits))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY, close_time=_CLOSE_TIME)

    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)


@pytest.mark.asyncio
async def test_fetch_sets_source_type_and_name():
    client = _make_client_mock(_make_response([_make_hit()]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs[0].source_type == "tv_transcript"
    assert docs[0].source_name == "TVArchive"


@pytest.mark.asyncio
async def test_fetch_strips_highlight_markers_from_body():
    hit = _make_hit(highlight_text="{{{trump}}} mentioned {{{communism}}} on air")
    client = _make_client_mock(_make_response([hit]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert "{{{" not in docs[0].body
    assert "}}}" not in docs[0].body
    assert "trump" in docs[0].body
    assert "communism" in docs[0].body


@pytest.mark.asyncio
async def test_fetch_builds_source_url_from_href():
    hit = _make_hit(href="/details/CNN_20260320/start/885/end/945?q=trump")
    client = _make_client_mock(_make_response([hit]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs[0].source_url == "https://archive.org/details/CNN_20260320/start/885/end/945?q=trump"


@pytest.mark.asyncio
async def test_fetch_parses_broadcast_date():
    hit = _make_hit(date="2026-03-15T00:00:00Z")
    client = _make_client_mock(_make_response([hit]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs[0].published_at == datetime(2026, 3, 15, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_fetch_uses_subjects_as_tags():
    hit = _make_hit(subject=["trump", "communism", "china"])
    client = _make_client_mock(_make_response([hit]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert "trump" in docs[0].tags
    assert "communism" in docs[0].tags


# ---------------------------------------------------------------------------
# Skipping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_skips_hit_without_highlight():
    hit_no_highlight = {
        "fields": {
            "identifier": "CNN_foo",
            "title": "CNN",
            "date": "2026-03-20T00:00:00Z",
            "__href__": "/details/CNN_foo/start/0/end/60",
            "subject": [],
        },
        "highlight": {"text": []},
    }
    client = _make_client_mock(_make_response([hit_no_highlight]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_skips_hit_without_href():
    hit_no_href = {
        "fields": {
            "identifier": "CNN_foo",
            "title": "CNN",
            "date": "2026-03-20T00:00:00Z",
            "__href__": "",
            "subject": [],
        },
        "highlight": {"text": ["some text about {{{trump}}}"]},
    }
    client = _make_client_mock(_make_response([hit_no_href]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_empty_hits_returns_empty():
    client = _make_client_mock(_make_response([]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_respects_max_results():
    hits = [_make_hit(identifier=f"CNN_{i}", href=f"/details/CNN_{i}/start/0/end/60?q=t") for i in range(5)]
    client = _make_client_mock(_make_response(hits))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY, max_results=3)

    assert len(docs) <= 3


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_timeout():
    import httpx
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_http_error():
    client = _make_client_mock(_make_response([], status_code=503))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_returns_empty_on_non_json_response():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.side_effect = ValueError("not json")
    resp.text = "Internal Server Error"
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=resp)

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        docs = await fetch(_QUERY)

    assert docs == []


@pytest.mark.asyncio
async def test_fetch_sends_query_in_params():
    """Verify the query string is sent as user_query parameter."""
    client = _make_client_mock(_make_response([]))

    with patch("freqpred.ingestion.fetchers.tv_archive.httpx.AsyncClient", return_value=client):
        await fetch(_QUERY, close_time=_CLOSE_TIME)

    call_kwargs = client.get.call_args
    params = call_kwargs.kwargs.get("params", {})
    assert params["user_query"] == _QUERY
    assert params["service_backend"] == "tvs"
