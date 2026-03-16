"""Unit tests for freqpred/ingestion/social_summarizer.py.

Anthropic SDK and SQLAlchemy session are mocked — no real API or DB calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.social_summarizer import summarize
from freqpred.ingestion.store import RawDocument
from freqpred.llm.models import LLMQueryRow

_API_KEY = "test-anthropic-key"
_TOPIC = "US presidential election"
_MODEL = "claude-haiku-4-5-20251001"

_VALID_SUMMARY = {
    "sentiment": "mixed",
    "key_claims": ["Polls show tight race", "Turnout expected to be high"],
    "notable_threads": ["r/politics megathread on results"],
    "overall_signal": "Slight lean toward incumbent based on swing state sentiment.",
}


def _make_raw_doc(
    i: int = 1,
    source_type: str = "reddit",
) -> RawDocument:
    now = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
    return RawDocument(
        source_url=f"https://reddit.com/r/politics/comments/{i}/",
        title=f"Post {i} about the election",
        body=f"Body text for post {i}. Lots of opinions here.",
        source_type=source_type,
        source_name="r/politics",
        category="",
        tags=[],
        published_at=now,
        fetched_at=now,
    )


@pytest.fixture()
def mock_anthropic():
    """Patch anthropic.AsyncAnthropic and return a mock messages response."""
    with patch(
        "freqpred.ingestion.social_summarizer.anthropic.AsyncAnthropic"
    ) as MockClient:
        instance = AsyncMock()
        MockClient.return_value = instance

        # Build a realistic response object
        content_block = MagicMock()
        content_block.text = json.dumps(_VALID_SUMMARY)
        usage = MagicMock()
        usage.input_tokens = 300
        usage.output_tokens = 100
        response = MagicMock()
        response.content = [content_block]
        response.usage = usage

        instance.messages.create = AsyncMock(return_value=response)
        yield instance


@pytest.fixture()
def mock_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_returns_raw_document(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1), _make_raw_doc(2)]
    result = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    assert isinstance(result, RawDocument)


@pytest.mark.asyncio
async def test_summarize_source_type_is_reddit_summary(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    result = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    assert result.source_type == "reddit_summary"


@pytest.mark.asyncio
async def test_summarize_title_contains_topic(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    result = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    assert _TOPIC in result.title


@pytest.mark.asyncio
async def test_summarize_body_is_valid_json(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    result = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    parsed = json.loads(result.body)
    assert isinstance(parsed, dict)


@pytest.mark.asyncio
async def test_summarize_body_has_required_keys(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    result = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    parsed = json.loads(result.body)
    assert "sentiment" in parsed
    assert "key_claims" in parsed
    assert "notable_threads" in parsed
    assert "overall_signal" in parsed


@pytest.mark.asyncio
async def test_summarize_source_url_is_unique(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    r1 = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)
    r2 = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    assert r1.source_url != r2.source_url


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_writes_audit_row(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    mock_session.add.assert_called_once()
    row = mock_session.add.call_args[0][0]
    assert isinstance(row, LLMQueryRow)


@pytest.mark.asyncio
async def test_summarize_audit_row_query_type(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    row = mock_session.add.call_args[0][0]
    assert row.query_type == "social_summarization"


@pytest.mark.asyncio
async def test_summarize_audit_row_strategy(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    row = mock_session.add.call_args[0][0]
    assert row.strategy == "social_summarizer"


@pytest.mark.asyncio
async def test_summarize_audit_row_success_true(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    row = mock_session.add.call_args[0][0]
    assert row.success is True


@pytest.mark.asyncio
async def test_summarize_audit_row_flush_called(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_audit_row_has_token_counts(mock_anthropic, mock_session):
    docs = [_make_raw_doc(1)]
    await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    row = mock_session.add.call_args[0][0]
    assert row.tokens_input == 300
    assert row.tokens_output == 100
    assert row.tokens_total == 400


# ---------------------------------------------------------------------------
# Failure path — API error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_on_api_error_still_writes_audit_row(mock_session):
    with patch(
        "freqpred.ingestion.social_summarizer.anthropic.AsyncAnthropic"
    ) as MockClient:
        instance = AsyncMock()
        MockClient.return_value = instance
        instance.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

        docs = [_make_raw_doc(1)]
        result = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    mock_session.add.assert_called_once()
    row = mock_session.add.call_args[0][0]
    assert isinstance(row, LLMQueryRow)
    assert row.success is False
    assert row.error_message == "API down"


@pytest.mark.asyncio
async def test_summarize_on_api_error_returns_fallback_doc(mock_session):
    with patch(
        "freqpred.ingestion.social_summarizer.anthropic.AsyncAnthropic"
    ) as MockClient:
        instance = AsyncMock()
        MockClient.return_value = instance
        instance.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

        docs = [_make_raw_doc(1)]
        result = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    assert result.source_type == "reddit_summary"
    parsed = json.loads(result.body)
    assert parsed["sentiment"] == "unknown"
    assert "Summarization failed" in parsed["overall_signal"]


# ---------------------------------------------------------------------------
# JSON parse failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_on_invalid_json_uses_raw_body(mock_session):
    with patch(
        "freqpred.ingestion.social_summarizer.anthropic.AsyncAnthropic"
    ) as MockClient:
        instance = AsyncMock()
        MockClient.return_value = instance

        content_block = MagicMock()
        content_block.text = "This is not JSON at all."
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 20
        response = MagicMock()
        response.content = [content_block]
        response.usage = usage
        instance.messages.create = AsyncMock(return_value=response)

        docs = [_make_raw_doc(1)]
        result = await summarize(docs, _TOPIC, mock_session, _API_KEY, _MODEL)

    assert result.body == "This is not JSON at all."
    # Audit row should still be success=True (LLM responded, just bad JSON)
    row = mock_session.add.call_args[0][0]
    assert row.success is True
