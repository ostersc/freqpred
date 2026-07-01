"""Unit tests for freqpred/ingestion/social_summarizer.py.

LLMClient is mocked — no real API or DB calls.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.ingestion.social_summarizer import summarize
from freqpred.ingestion.store import RawDocument
from freqpred.llm.client import LLMError
from freqpred.llm.models import LLMResponse

_TOPIC = "US presidential election"
_MODEL = "claude-haiku-4-5-20251001"

_VALID_SUMMARY = {
    "sentiment": "mixed",
    "key_claims": ["Polls show tight race", "Turnout expected to be high"],
    "notable_threads": ["r/politics megathread on results"],
    "overall_signal": "Slight lean toward incumbent based on swing state sentiment.",
}


def _make_raw_doc(i: int = 1) -> RawDocument:
    now = datetime(2026, 3, 16, 12, 0, 0, tzinfo=UTC)
    return RawDocument(
        source_url=f"https://reddit.com/r/politics/comments/{i}/",
        title=f"Post {i} about the election",
        body=f"Body text for post {i}. Lots of opinions here.",
        source_type="reddit",
        source_name="r/politics",
        category="",
        tags=[],
        published_at=now,
        fetched_at=now,
    )


def _make_llm_client(content: str = "", raises: Exception | None = None) -> MagicMock:
    """Return a mock LLMClient."""
    client = MagicMock()
    if raises:
        client.complete = AsyncMock(side_effect=raises)
    else:
        client.complete = AsyncMock(
            return_value=LLMResponse(
                content=content,
                model=_MODEL,
                tokens_input=300,
                tokens_output=100,
                cost_usd=0.00035,
                latency_ms=450,
                llm_query_id=1,
            )
        )
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_returns_raw_document():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    result = await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    assert isinstance(result, RawDocument)


@pytest.mark.asyncio
async def test_summarize_source_type_is_reddit_summary():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    result = await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    assert result.source_type == "reddit_summary"


@pytest.mark.asyncio
async def test_summarize_title_contains_topic():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    result = await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    assert _TOPIC in result.title


@pytest.mark.asyncio
async def test_summarize_body_is_valid_json():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    result = await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    parsed = json.loads(result.body)
    assert isinstance(parsed, dict)


@pytest.mark.asyncio
async def test_summarize_body_has_required_keys():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    result = await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    parsed = json.loads(result.body)
    assert "sentiment" in parsed
    assert "key_claims" in parsed
    assert "notable_threads" in parsed
    assert "overall_signal" in parsed


@pytest.mark.asyncio
async def test_summarize_source_url_is_unique():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    docs = [_make_raw_doc(1)]
    r1 = await summarize(docs, _TOPIC, client, _MODEL)
    r2 = await summarize(docs, _TOPIC, client, _MODEL)
    assert r1.source_url != r2.source_url


# ---------------------------------------------------------------------------
# Audit: LLMClient.complete() is called with expected args
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_calls_llm_client_with_correct_query_type():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    client.complete.assert_called_once()
    kwargs = client.complete.call_args
    assert kwargs.args[2] == "social_summarization"  # positional: prompt, model, query_type


@pytest.mark.asyncio
async def test_summarize_calls_llm_client_with_strategy():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    kwargs = client.complete.call_args.kwargs
    assert kwargs["strategy"] == "social_summarizer"


@pytest.mark.asyncio
async def test_summarize_passes_system_prompt():
    client = _make_llm_client(content=json.dumps(_VALID_SUMMARY))
    await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    kwargs = client.complete.call_args.kwargs
    assert kwargs.get("system") is not None


# ---------------------------------------------------------------------------
# Failure path — API error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_on_api_error_returns_fallback_doc():
    client = _make_llm_client(raises=LLMError("API down"))
    result = await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    assert result.source_type == "reddit_summary"
    parsed = json.loads(result.body)
    assert parsed["sentiment"] == "unknown"
    assert "Summarization failed" in parsed["overall_signal"]


@pytest.mark.asyncio
async def test_summarize_on_api_error_does_not_raise():
    """summarize() must never propagate LLMError — it returns a fallback."""
    client = _make_llm_client(raises=LLMError("network error"))
    result = await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    assert isinstance(result, RawDocument)


# ---------------------------------------------------------------------------
# JSON parse failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_on_invalid_json_uses_raw_body():
    client = _make_llm_client(content="This is not JSON at all.")
    result = await summarize([_make_raw_doc(1)], _TOPIC, client, _MODEL)
    assert result.body == "This is not JSON at all."
