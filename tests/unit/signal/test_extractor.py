"""Unit tests for freqpred/signal/extractor.py (T101).

The properties that matter, in order of how much they'd cost to get wrong:

1. Extraction failure must fail *open* — a bad API minute must degrade to the
   old 500-char cut, never drop a document and never raise.
2. The skip gate must key on ``body``, not ``summary or body``. Gating on the
   summary hands the decision back to the ingestion-time gate this change
   exists to escape.
3. The cache key is ``(document, market, prompt_version)``; a prompt-version
   change must re-extract rather than serve text written under different
   instructions.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import freqpred.ingestion.models  # noqa: F401 — registers mappers
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.llm.client import LLMError
from freqpred.markets.models import Market
from freqpred.rag.models import Document
from freqpred.signal.extractor import (
    EXTRACT_PROMPT_VERSION,
    MIN_EXTRACT_BODY_CHARS,
    DocumentExtract,
    build_extract_prompt,
    extract_document,
    extract_for_documents,
    needs_extraction,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
MARKET_ID = "KXTRUMPSAY-26AUG17-MELA"

_LONG_BODY = "Trump appeared with the first lady. " * 40  # ~1,400 chars


def _make_market() -> Market:
    return Market(
        id=MARKET_ID,
        platform="kalshi",
        question='Will Trump say "Melania" before Aug 17, 2026?',
        category="Mentions",
        close_time=datetime(2026, 8, 17, tzinfo=UTC),
        yes_bid=0.53,
        yes_ask=0.55,
        mid_price=0.54,
        volume_24h=1000.0,
        open_interest=500.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
        series_ticker="KXTRUMPSAY",
    )


def _make_document(
    body: str = _LONG_BODY, summary: str | None = None, doc_id: str | None = None
) -> Document:
    did = doc_id or str(uuid.uuid4())
    return Document(
        id=did,
        source_url=f"https://example.com/{did}",
        content_hash="abc123",
        title="Trump at the White House",
        body=body,
        source_type="news",
        source_name="The Guardian",
        category="Mentions",
        tags=[],
        published_at=NOW,
        fetched_at=NOW,
        embedding=[],
        embedding_model="",
        summary=summary,
    )


def _tool_response(relevance: str, extract: str) -> MagicMock:
    response = MagicMock()
    response.content = json.dumps({"relevance": relevance, "extract": extract})
    response.model = "claude-haiku-4-5-20251001"
    return response


# ---------------------------------------------------------------------------
# Single-document extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_returns_parsed_relevance_and_text() -> None:
    client = MagicMock()
    client.complete = AsyncMock(
        return_value=_tool_response("direct", "Trump named Melania on Tuesday.")
    )

    result = await extract_document(
        client, _make_market(), _make_document(), model="claude-haiku-4-5-20251001"
    )

    assert result.relevance == "direct"
    assert result.extract == "Trump named Melania on Tuesday."
    assert result.prompt_version == EXTRACT_PROMPT_VERSION
    assert result.fallback is False
    # Forced tool use, not a plain-text sentinel: a sentinel proved unreliable,
    # leaking "**Relevance:**" preambles into the extract.
    assert client.complete.await_args.kwargs["json_tool"]["name"] == "record_extract"


@pytest.mark.asyncio
async def test_extract_falls_back_to_raw_cut_on_llm_error() -> None:
    client = MagicMock()
    client.complete = AsyncMock(side_effect=LLMError("upstream 529"))
    doc = _make_document()

    result = await extract_document(
        client, _make_market(), doc, model="claude-haiku-4-5-20251001"
    )

    assert result.fallback is True
    assert result.extract == doc.body[:MIN_EXTRACT_BODY_CHARS].strip()
    # Never "none": a failure must not remove the document from the prompt.
    assert result.relevance != "none"


@pytest.mark.asyncio
async def test_extract_falls_back_on_malformed_tool_input() -> None:
    client = MagicMock()
    bad = MagicMock()
    bad.content = json.dumps({"relevance": "very relevant", "extract": "x"})
    bad.model = "claude-haiku-4-5-20251001"
    client.complete = AsyncMock(return_value=bad)

    result = await extract_document(
        client, _make_market(), _make_document(), model="claude-haiku-4-5-20251001"
    )

    assert result.fallback is True


@pytest.mark.asyncio
async def test_extract_skipped_for_short_bodies() -> None:
    client = MagicMock()
    client.complete = AsyncMock()

    result = await extract_document(
        client,
        _make_market(),
        _make_document(body="Short body."),
        model="claude-haiku-4-5-20251001",
    )

    client.complete.assert_not_awaited()
    assert result.fallback is True
    assert result.extract == "Short body."


def test_skip_gate_keys_on_body_not_summary() -> None:
    """A 486-char ingestion summary must not mask a 43,000-char article.

    The summary was written against whichever question triggered the fetch —
    deferring to it here would reinstate the exact gate T101 removes.
    """
    doc = _make_document(body="x" * 43_000, summary="A short ingestion summary.")
    assert needs_extraction(doc) is True


def test_skip_gate_false_for_genuinely_short_body() -> None:
    assert needs_extraction(_make_document(body="x" * 200)) is False


def test_extract_prompt_carries_the_market_question_and_document() -> None:
    market = _make_market()
    prompt = build_extract_prompt(market, _make_document())
    assert market.question in prompt
    # The catalyst framing: an earlier resolution-only draft produced a false
    # "none" on a scheduling article.
    assert "background signals" in prompt
    assert "contextual" in prompt


# ---------------------------------------------------------------------------
# Batch extraction + cache
# ---------------------------------------------------------------------------


class _FakeSession:
    """Minimal AsyncSession stand-in returning pre-seeded cache rows."""

    def __init__(self, cached_rows: list | None = None) -> None:
        self._cached_rows = cached_rows or []
        self.executed: list = []
        self.commits = 0

    async def execute(self, stmt):  # noqa: ANN001, ANN201
        self.executed.append(stmt)
        result = MagicMock()
        # Only the SELECT is read for values; the INSERT's return is ignored.
        result.scalars.return_value = self._cached_rows
        return result

    async def commit(self) -> None:
        self.commits += 1


def _cache_row(doc_id: str, relevance: str = "direct", extract: str = "cached text"):
    row = MagicMock()
    row.document_id = uuid.UUID(doc_id)
    row.relevance = relevance
    row.extract = extract
    row.model_used = "claude-haiku-4-5-20251001"
    row.prompt_version = EXTRACT_PROMPT_VERSION
    return row


@pytest.mark.asyncio
async def test_extract_cached_by_document_market_prompt_version() -> None:
    """A cached triple hits the DB, not the LLM."""
    doc = _make_document()
    session = _FakeSession(cached_rows=[_cache_row(str(doc.id))])
    client = MagicMock()
    client.complete = AsyncMock()

    result = await extract_for_documents(
        session, client, _make_market(), [doc], model="claude-haiku-4-5-20251001"
    )

    client.complete.assert_not_awaited()
    assert result[str(doc.id)].extract == "cached text"


@pytest.mark.asyncio
async def test_cache_miss_extracts_and_persists() -> None:
    doc = _make_document()
    session = _FakeSession(cached_rows=[])  # nothing cached
    client = MagicMock()
    client.complete = AsyncMock(return_value=_tool_response("contextual", "fresh text"))

    result = await extract_for_documents(
        session, client, _make_market(), [doc], model="claude-haiku-4-5-20251001"
    )

    client.complete.assert_awaited_once()
    assert result[str(doc.id)].extract == "fresh text"
    # Extracts already paid for must survive a later failure in the same run.
    assert session.commits == 1


@pytest.mark.asyncio
async def test_prompt_version_change_forces_reextract() -> None:
    """The cache SELECT is filtered on the *current* extraction prompt version.

    A row written under an older version therefore never matches, which is what
    stops the pipeline serving text produced under different instructions.
    """
    doc = _make_document()
    stale = _cache_row(str(doc.id))
    stale.prompt_version = "extract-v0"
    # The real query filters server-side; simulate the filter missing the row.
    session = _FakeSession(cached_rows=[])
    client = MagicMock()
    client.complete = AsyncMock(return_value=_tool_response("direct", "re-extracted"))

    result = await extract_for_documents(
        session, client, _make_market(), [doc], model="claude-haiku-4-5-20251001"
    )

    client.complete.assert_awaited_once()
    assert result[str(doc.id)].prompt_version == EXTRACT_PROMPT_VERSION

    # And the SELECT really does constrain on the current version.
    select_sql = str(session.executed[0])
    assert "prompt_version" in select_sql


@pytest.mark.asyncio
async def test_fallbacks_are_not_persisted() -> None:
    """Caching a failure would pin it to the pair until the version changes."""
    doc = _make_document()
    session = _FakeSession(cached_rows=[])
    client = MagicMock()
    client.complete = AsyncMock(side_effect=LLMError("upstream 529"))

    result = await extract_for_documents(
        session, client, _make_market(), [doc], model="claude-haiku-4-5-20251001"
    )

    assert result[str(doc.id)].fallback is True
    assert session.commits == 0  # nothing written
    assert len(session.executed) == 1  # the cache SELECT only


@pytest.mark.asyncio
async def test_returns_an_entry_for_every_document() -> None:
    """Callers index the map unconditionally, so it must be total."""
    cached_doc = _make_document()
    fresh_doc = _make_document()
    short_doc = _make_document(body="tiny")
    session = _FakeSession(cached_rows=[_cache_row(str(cached_doc.id))])
    client = MagicMock()
    client.complete = AsyncMock(return_value=_tool_response("direct", "fresh"))

    result = await extract_for_documents(
        session,
        client,
        _make_market(),
        [cached_doc, fresh_doc, short_doc],
        model="claude-haiku-4-5-20251001",
    )

    assert set(result) == {str(cached_doc.id), str(fresh_doc.id), str(short_doc.id)}
    assert isinstance(result[str(short_doc.id)], DocumentExtract)


@pytest.mark.asyncio
async def test_no_documents_makes_no_query() -> None:
    session = _FakeSession()
    client = MagicMock()
    client.complete = AsyncMock()

    assert (
        await extract_for_documents(
            session, client, _make_market(), [], model="claude-haiku-4-5-20251001"
        )
        == {}
    )
    assert session.executed == []
