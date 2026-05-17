"""Unit tests for FactBase phrase frequency fetcher."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.fetchers.factbase import (
    FactbasePhraseCache,
    FactbaseSearchTerms,
    fetch_phrase_frequency,
    extract_search_terms,
)


# ---------------------------------------------------------------------------
# FactbasePhraseCache
# ---------------------------------------------------------------------------


def test_phrase_cache_mark_and_check() -> None:
    cache = FactbasePhraseCache()
    cache.mark_ready("market-1")
    assert cache.is_ready("market-1") is True


def test_phrase_cache_unknown_not_ready() -> None:
    cache = FactbasePhraseCache()
    assert cache.is_ready("market-unknown") is False


def test_phrase_cache_idempotent_mark() -> None:
    cache = FactbasePhraseCache()
    cache.mark_ready("market-1")
    cache.mark_ready("market-1")
    assert cache.is_ready("market-1") is True


# ---------------------------------------------------------------------------
# extract_search_terms — Haiku mock
# ---------------------------------------------------------------------------


def _mock_llm_client(terms: list[str], display_phrase: str) -> MagicMock:
    response = MagicMock()
    response.content = json.dumps({"display_phrase": display_phrase, "terms": terms})
    response.llm_query_id = 1
    client = MagicMock()
    client.complete = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_extract_terms_single_phrase() -> None:
    client = _mock_llm_client(
        terms=["witch hunt", "witch hunts", "witch hunt's"],
        display_phrase="witch hunt",
    )
    result = await extract_search_terms(
        "Will Trump say 'witch hunt' this week?", client, market_id="m1"
    )
    assert result is not None
    assert result.display_phrase == "witch hunt"
    assert "witch hunt" in result.match_terms
    assert result.api_query == '"witch hunt" OR "witch hunts" OR "witch hunt\'s"'


@pytest.mark.asyncio
async def test_extract_terms_slash_variants() -> None:
    client = _mock_llm_client(
        terms=["Communist", "Communism", "Communists", "Communist's"],
        display_phrase="Communist / Communism",
    )
    result = await extract_search_terms(
        "Will Trump say 'Communist / Communism'?", client, market_id="m2"
    )
    assert result is not None
    assert result.display_phrase == "Communist / Communism"
    assert "Communist" in result.match_terms
    assert "Communism" in result.match_terms
    assert '"Communist" OR "Communism" OR "Communists" OR "Communist\'s"' == result.api_query


@pytest.mark.asyncio
async def test_extract_terms_returns_none_when_no_phrase() -> None:
    client = _mock_llm_client(terms=[], display_phrase="")
    result = await extract_search_terms("Some market with no phrase", client)
    assert result is None


@pytest.mark.asyncio
async def test_extract_terms_logs_llm_query() -> None:
    client = _mock_llm_client(
        terms=["test"],
        display_phrase="test",
    )
    await extract_search_terms("Will Trump say 'test'?", client, market_id="m3")
    client.complete.assert_awaited_once()
    call_kwargs = client.complete.call_args
    assert call_kwargs.kwargs.get("query_type") == "factbase_phrase_extract"


@pytest.mark.asyncio
async def test_extract_terms_handles_llm_error() -> None:
    from freqpred.llm.client import LLMError

    client = MagicMock()
    client.complete = AsyncMock(side_effect=LLMError("timeout"))
    result = await extract_search_terms("Will Trump say 'test'?", client)
    assert result is None


# ---------------------------------------------------------------------------
# fetch_phrase_frequency — HTTP mock
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)


def _make_segment(days_ago: int, speaker: str = "Donald Trump", text: str = "test phrase here") -> dict:
    dt = _NOW - timedelta(days=days_ago)
    return {
        "date": dt.isoformat(),
        "speaker": speaker,
        "text": text,
        "record_type": "speech",
    }


def _make_http_response(segments: list[dict], total_pages: int = 1) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={
        "data": segments,
        "meta": {"total_pages": total_pages},
    })
    return resp


@pytest.mark.asyncio
async def test_compute_windows_in_market() -> None:
    market_open = _NOW - timedelta(days=3)
    segments = [
        _make_segment(1),   # in market, in 7d, 30d, 365d
        _make_segment(5),   # NOT in market (before open), in 7d, 30d, 365d
        _make_segment(40),  # in 30d, 365d only
        _make_segment(400), # older than 365d — won't appear (pagination cutoff)
    ]

    search_terms = FactbaseSearchTerms(
        display_phrase="test",
        api_query='"test"',
        match_terms=["test"],
    )

    mock_resp = _make_http_response(segments[:3])  # exclude >365d (cutoff logic)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        data = await fetch_phrase_frequency(search_terms, "trump", market_open)

    assert data.in_market_count == 1
    assert data.count_7d == 2
    assert data.count_30d == 2
    assert data.count_365d == 3


@pytest.mark.asyncio
async def test_top_quotes_trump_only() -> None:
    market_open = _NOW - timedelta(days=60)
    segments = [
        _make_segment(1, speaker="Donald Trump", text="I will not stand for this"),
        _make_segment(2, speaker="Joe Biden", text="Something else"),
        _make_segment(3, speaker="Donald Trump", text="The witch hunt continues"),
    ]

    search_terms = FactbaseSearchTerms(
        display_phrase="witch hunt",
        api_query='"witch hunt"',
        match_terms=["witch hunt"],
    )

    mock_resp = _make_http_response(segments)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        data = await fetch_phrase_frequency(search_terms, "trump", market_open)

    assert len(data.top_quotes) == 2
    assert all(q["text"] != "Something else" for q in data.top_quotes)


@pytest.mark.asyncio
async def test_fetch_empty_returns_zeros() -> None:
    search_terms = FactbaseSearchTerms(
        display_phrase="test",
        api_query='"test"',
        match_terms=["test"],
    )

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        data = await fetch_phrase_frequency(search_terms, "trump", None)

    assert data.in_market_count == 0
    assert data.count_7d == 0
    assert data.count_30d == 0
    assert data.count_365d == 0
    assert data.top_quotes == []
