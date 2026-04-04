"""Unit tests for freqpred/ingestion/body_summarizer.py.

LLMClient is mocked — no real API or DB calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.ingestion.body_summarizer import _MAX_SUMMARY_CHARS, _MODEL, summarize_body
from freqpred.ingestion.store import RawDocument
from freqpred.llm.client import LLMError
from freqpred.llm.models import LLMResponse

_NOW = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)
_QUERY = "Federal Reserve interest rate decision March 2026"
_MARKET = "Will the Federal Reserve raise rates before June 2026?\nIf the Federal Reserve raises rates before June 2026 this market resolves Yes."
_LONG_BODY = "a" * 2000  # well over the 1000-char threshold


def _make_raw_doc(body: str = _LONG_BODY) -> RawDocument:
    return RawDocument(
        source_url="https://example.com/article",
        title="Fed Rate Decision Article",
        body=body,
        source_type="news",
        source_name="Reuters",
        category="economics",
        tags=[],
        published_at=_NOW,
        fetched_at=_NOW,
    )


def _make_llm_client(content: str = "Summary text.", raises: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if raises:
        client.complete = AsyncMock(side_effect=raises)
    else:
        client.complete = AsyncMock(
            return_value=LLMResponse(
                content=content,
                model=_MODEL,
                tokens_input=200,
                tokens_output=50,
                cost_usd=0.0001,
                latency_ms=300,
                llm_query_id=1,
            )
        )
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_body_returns_string():
    client = _make_llm_client(content="Fed raised rates by 25bps.")
    result = await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_summarize_body_strips_whitespace():
    client = _make_llm_client(content="  Fed raised rates.  ")
    result = await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    assert result == "Fed raised rates."


@pytest.mark.asyncio
async def test_summarize_body_truncates_to_max_chars():
    long_response = "x" * (_MAX_SUMMARY_CHARS + 100)
    client = _make_llm_client(content=long_response)
    result = await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    assert result is not None
    assert len(result) <= _MAX_SUMMARY_CHARS


@pytest.mark.asyncio
async def test_summarize_body_short_summary_not_truncated():
    short = "Fed held rates steady."
    client = _make_llm_client(content=short)
    result = await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    assert result == short


# ---------------------------------------------------------------------------
# Audit: LLMClient.complete() is called with expected args
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_body_calls_llm_once():
    client = _make_llm_client()
    await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    client.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_summarize_body_uses_body_summarization_query_type():
    client = _make_llm_client()
    await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    args = client.complete.call_args
    assert args.args[2] == "body_summarization"


@pytest.mark.asyncio
async def test_summarize_body_uses_body_summarizer_strategy():
    client = _make_llm_client()
    await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    kwargs = client.complete.call_args.kwargs
    assert kwargs.get("strategy") == "body_summarizer"


@pytest.mark.asyncio
async def test_summarize_body_passes_system_prompt():
    client = _make_llm_client()
    await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    kwargs = client.complete.call_args.kwargs
    assert kwargs.get("system") is not None


@pytest.mark.asyncio
async def test_summarize_body_includes_market_question_in_prompt():
    client = _make_llm_client()
    await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    prompt = client.complete.call_args.args[0]
    assert "Federal Reserve" in prompt


@pytest.mark.asyncio
async def test_summarize_body_includes_query_text_in_prompt():
    client = _make_llm_client()
    await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    prompt = client.complete.call_args.args[0]
    assert _QUERY in prompt


# ---------------------------------------------------------------------------
# Failure path — LLMError never propagates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_body_on_llm_error_returns_none():
    client = _make_llm_client(raises=LLMError("API down"))
    result = await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    assert result is None


@pytest.mark.asyncio
async def test_summarize_body_on_llm_error_does_not_raise():
    client = _make_llm_client(raises=LLMError("network error"))
    # Must not raise
    result = await summarize_body(_make_raw_doc(), _QUERY, _MARKET, client)
    assert result is None
