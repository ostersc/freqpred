"""Unit tests for freqpred/signal/ (cache, llm, pipeline).

All DB, embedder, and LLM API calls are mocked — no external dependencies.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.signal.cache import should_skip
from freqpred.signal.llm import build_prompt, parse_signal_response
from freqpred.signal.pipeline import SignalPipeline
from freqpred.signal.models import Signal

# Ensure all ORM relationships resolve before any test runs
import freqpred.ingestion.models   # noqa: F401
import freqpred.llm.models         # noqa: F401
import freqpred.markets.models     # noqa: F401
import freqpred.rag.models         # noqa: F401
import freqpred.signal.models      # noqa: F401

NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
FAKE_HASH = "a" * 64
FAKE_SIGNAL_ID = str(uuid.uuid4())
FAKE_MARKET_ID = "MARKET-001"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_market(
    mid_price: float = 0.50,
    current_signal_id: str | None = None,
) -> MagicMock:
    """Return a minimal Market-like object."""
    from freqpred.markets.models import Market

    return Market(
        id=FAKE_MARKET_ID,
        platform="kalshi",
        question="Will the Fed raise rates in June 2026?",
        category="economics",
        close_time=datetime(2026, 6, 30, tzinfo=timezone.utc),
        yes_bid=0.48,
        yes_ask=0.52,
        mid_price=mid_price,
        volume_24h=1000.0,
        open_interest=500.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
        current_signal_id=current_signal_id,
    )


def _make_document(doc_id: str | None = None) -> MagicMock:
    from freqpred.rag.models import Document

    did = doc_id or str(uuid.uuid4())
    return Document(
        id=did,
        source_url=f"https://example.com/{did}",
        content_hash="abc123",
        title="Fed rate hike article",
        body="The Federal Reserve is expected to raise rates.",
        source_type="news",
        source_name="Reuters",
        category="economics",
        tags=["fed", "rates"],
        published_at=NOW,
        fetched_at=NOW,
        embedding=[0.1] * 1024,
        embedding_model="voyage-3",
        summary=None,
    )


def _make_session_scalar(return_value) -> AsyncMock:
    """Mock session whose execute().scalar_one_or_none() returns *return_value*."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = return_value
    session.execute = AsyncMock(return_value=result)
    return session


def _make_llm_client(content: str = "", error: Exception | None = None) -> MagicMock:
    from freqpred.llm.models import LLMResponse

    client = AsyncMock()
    if error:
        from freqpred.llm.client import LLMError
        client.complete = AsyncMock(side_effect=LLMError("api error"))
    else:
        response = LLMResponse(
            content=content,
            model="claude-sonnet-4-6",
            tokens_input=100,
            tokens_output=50,
            cost_usd=0.002,
            latency_ms=300,
            llm_query_id=1,
        )
        client.complete = AsyncMock(return_value=response)
    return client


def _make_session_factory(session: AsyncMock) -> MagicMock:
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory


def _valid_llm_json(
    probability: float = 0.70,
    confidence: float = 0.80,
    direction: str = "YES",
    doc_ids: list[str] | None = None,
) -> str:
    return json.dumps({
        "probability": probability,
        "confidence": confidence,
        "direction": direction,
        "reasoning": "Strong evidence suggests YES.",
        "evidence_used": doc_ids or [],
    })


# ---------------------------------------------------------------------------
# cache.should_skip
# ---------------------------------------------------------------------------


class TestShouldSkip:
    @pytest.mark.asyncio
    async def test_no_current_signal_returns_false(self) -> None:
        session = _make_session_scalar(None)
        assert await should_skip(session, None, FAKE_HASH) is False

    @pytest.mark.asyncio
    async def test_matching_hash_returns_true(self) -> None:
        session = _make_session_scalar(FAKE_HASH)
        assert await should_skip(session, FAKE_SIGNAL_ID, FAKE_HASH) is True

    @pytest.mark.asyncio
    async def test_different_hash_returns_false(self) -> None:
        session = _make_session_scalar("b" * 64)
        assert await should_skip(session, FAKE_SIGNAL_ID, FAKE_HASH) is False

    @pytest.mark.asyncio
    async def test_signal_not_found_returns_false(self) -> None:
        session = _make_session_scalar(None)
        assert await should_skip(session, FAKE_SIGNAL_ID, FAKE_HASH) is False

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_false(self) -> None:
        session = _make_session_scalar(FAKE_HASH)
        assert await should_skip(session, "not-a-uuid", FAKE_HASH) is False

    @pytest.mark.asyncio
    async def test_none_hash_returns_false(self) -> None:
        """If the stored hash is None for some reason, treat as no prior evidence."""
        session = _make_session_scalar(None)
        assert await should_skip(session, FAKE_SIGNAL_ID, FAKE_HASH) is False


# ---------------------------------------------------------------------------
# llm.parse_signal_response
# ---------------------------------------------------------------------------


class TestParseSignalResponse:
    def test_valid_json_returns_dict(self) -> None:
        result = parse_signal_response(_valid_llm_json())
        assert result is not None
        assert result["probability"] == 0.70
        assert result["confidence"] == 0.80
        assert result["direction"] == "YES"
        assert "reasoning" in result
        assert "evidence_used" in result

    def test_invalid_json_returns_none(self) -> None:
        assert parse_signal_response("not json at all") is None

    def test_missing_field_returns_none(self) -> None:
        data = {"probability": 0.5, "confidence": 0.6, "direction": "YES", "reasoning": "ok"}
        # missing evidence_used
        assert parse_signal_response(json.dumps(data)) is None

    def test_invalid_direction_returns_none(self) -> None:
        result = parse_signal_response(_valid_llm_json(direction="MAYBE"))
        assert result is None

    def test_direction_case_normalized(self) -> None:
        data = {
            "probability": 0.5,
            "confidence": 0.5,
            "direction": "yes",  # lowercase
            "reasoning": "ok",
            "evidence_used": [],
        }
        result = parse_signal_response(json.dumps(data))
        assert result is not None
        assert result["direction"] == "YES"

    def test_probability_clamped_above_one(self) -> None:
        result = parse_signal_response(_valid_llm_json(probability=1.5))
        assert result is not None
        assert result["probability"] == 1.0

    def test_probability_clamped_below_zero(self) -> None:
        result = parse_signal_response(_valid_llm_json(probability=-0.1))
        assert result is not None
        assert result["probability"] == 0.0

    def test_markdown_fence_stripped(self) -> None:
        content = "```json\n" + _valid_llm_json() + "\n```"
        result = parse_signal_response(content)
        assert result is not None
        assert result["probability"] == 0.70

    def test_direction_no_is_valid(self) -> None:
        result = parse_signal_response(_valid_llm_json(direction="NO"))
        assert result is not None
        assert result["direction"] == "NO"

    def test_direction_skip_is_valid(self) -> None:
        result = parse_signal_response(_valid_llm_json(direction="SKIP"))
        assert result is not None
        assert result["direction"] == "SKIP"

    def test_non_numeric_probability_returns_none(self) -> None:
        data = {
            "probability": "high",
            "confidence": 0.5,
            "direction": "YES",
            "reasoning": "ok",
            "evidence_used": [],
        }
        assert parse_signal_response(json.dumps(data)) is None


# ---------------------------------------------------------------------------
# llm.build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_market_question(self) -> None:
        market = _make_market()
        prompt = build_prompt(market, [])
        assert market.question in prompt

    def test_does_not_include_mid_price(self) -> None:
        market = _make_market(mid_price=0.65)
        prompt = build_prompt(market, [])
        assert "0.6500" not in prompt
        assert "0.65" not in prompt

    def test_no_docs_shows_no_evidence(self) -> None:
        market = _make_market()
        prompt = build_prompt(market, [])
        assert "No evidence available" in prompt

    def test_docs_included(self) -> None:
        market = _make_market()
        doc = _make_document()
        prompt = build_prompt(market, [doc])
        assert doc.title in prompt
        assert doc.id in prompt

    def test_summary_preferred_over_body(self) -> None:
        market = _make_market()
        doc = _make_document()
        doc.summary = "SHORT SUMMARY"
        doc.body = "LONG BODY CONTENT THAT SHOULD NOT APPEAR"
        prompt = build_prompt(market, [doc])
        assert "SHORT SUMMARY" in prompt

    def test_multiple_docs_numbered(self) -> None:
        market = _make_market()
        docs = [_make_document() for _ in range(3)]
        prompt = build_prompt(market, docs)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt


# ---------------------------------------------------------------------------
# SignalPipeline.analyze — full pipeline tests
# ---------------------------------------------------------------------------


class TestSignalPipelineAnalyze:
    def _make_pipeline(
        self,
        docs: list,
        llm_content: str = "",
        llm_error: Exception | None = None,
        current_signal_hash: str | None = None,
    ) -> tuple[SignalPipeline, AsyncMock, AsyncMock]:
        """Return (pipeline, mock_session, mock_llm_client)."""
        from freqpred.rag.retriever import compute_retrieval_hash

        doc_ids = [d.id for d in docs]
        new_hash = compute_retrieval_hash(doc_ids)

        # Session: should_skip query returns current_signal_hash (or None)
        session = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()

        scalar_result = MagicMock()
        scalar_result.scalar_one_or_none.return_value = current_signal_hash
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = current_signal_hash
        session.execute = AsyncMock(return_value=execute_result)

        session_factory = _make_session_factory(session)

        embedder = AsyncMock()
        embedder.embed_text = AsyncMock(return_value=[0.1] * 1024)

        llm_client = _make_llm_client(content=llm_content, error=llm_error)

        pipeline = SignalPipeline(
            session_factory=session_factory,
            embedder=embedder,
            llm_client=llm_client,
        )

        return pipeline, session, llm_client

    @pytest.mark.asyncio
    async def test_returns_signal_on_new_evidence(self) -> None:
        doc = _make_document()
        pipeline, _, _ = self._make_pipeline(
            docs=[doc],
            llm_content=_valid_llm_json(doc_ids=[doc.id]),
            current_signal_hash=None,  # no prior signal
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            result = await pipeline.analyze(_make_market())

        assert isinstance(result, Signal)
        assert result.market_id == FAKE_MARKET_ID
        assert result.direction == "YES"
        assert 0.0 <= result.estimated_probability <= 1.0

    @pytest.mark.asyncio
    async def test_returns_none_when_evidence_unchanged(self) -> None:
        """Second call with same docs → same hash → None returned, no LLM call."""
        from freqpred.rag.retriever import compute_retrieval_hash

        doc = _make_document()
        doc_hash = compute_retrieval_hash([doc.id])

        pipeline, _, llm_client = self._make_pipeline(
            docs=[doc],
            llm_content=_valid_llm_json(),
            current_signal_hash=doc_hash,  # current signal already has this hash
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            result = await pipeline.analyze(_make_market(current_signal_id=FAKE_SIGNAL_ID))

        assert result is None
        llm_client.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_new_document_triggers_llm_call(self) -> None:
        """Different hash from prior signal triggers new LLM call."""
        doc = _make_document()
        pipeline, _, llm_client = self._make_pipeline(
            docs=[doc],
            llm_content=_valid_llm_json(doc_ids=[doc.id]),
            current_signal_hash="different_hash" + "x" * 50,  # won't match
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            result = await pipeline.analyze(_make_market(current_signal_id=FAKE_SIGNAL_ID))

        assert result is not None
        llm_client.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_error_returns_none(self) -> None:
        doc = _make_document()
        pipeline, _, _ = self._make_pipeline(
            docs=[doc],
            llm_error=Exception("api timeout"),
            current_signal_hash=None,
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            result = await pipeline.analyze(_make_market())

        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self) -> None:
        doc = _make_document()
        pipeline, _, _ = self._make_pipeline(
            docs=[doc],
            llm_content="this is not json",
            current_signal_hash=None,
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            result = await pipeline.analyze(_make_market())

        assert result is None

    @pytest.mark.asyncio
    async def test_signal_written_to_session(self) -> None:
        doc = _make_document()
        pipeline, session, _ = self._make_pipeline(
            docs=[doc],
            llm_content=_valid_llm_json(doc_ids=[doc.id]),
            current_signal_hash=None,
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            await pipeline.analyze(_make_market())

        # session.add should have been called (signal row + document link)
        assert session.add.call_count >= 1

    @pytest.mark.asyncio
    async def test_market_current_signal_id_updated(self) -> None:
        """Market.current_signal_id update is issued via session.execute."""
        doc = _make_document()
        pipeline, session, _ = self._make_pipeline(
            docs=[doc],
            llm_content=_valid_llm_json(doc_ids=[doc.id]),
            current_signal_hash=None,
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            await pipeline.analyze(_make_market())

        # session.execute is called at least once (the market update via UPDATE statement)
        # should_skip returns early (no DB call) when current_signal_id is None
        assert session.execute.await_count >= 1

    @pytest.mark.asyncio
    async def test_session_committed(self) -> None:
        doc = _make_document()
        pipeline, session, _ = self._make_pipeline(
            docs=[doc],
            llm_content=_valid_llm_json(doc_ids=[doc.id]),
            current_signal_hash=None,
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            await pipeline.analyze(_make_market())

        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edge_computed_correctly(self) -> None:
        doc = _make_document()
        pipeline, _, _ = self._make_pipeline(
            docs=[doc],
            llm_content=_valid_llm_json(probability=0.70),
            current_signal_hash=None,
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[doc])):
            result = await pipeline.analyze(_make_market(mid_price=0.50))

        assert result is not None
        assert abs(result.edge - 0.20) < 1e-6

    @pytest.mark.asyncio
    async def test_no_docs_still_runs(self) -> None:
        """Empty retrieval still triggers LLM (hash of empty list is valid)."""
        pipeline, _, llm_client = self._make_pipeline(
            docs=[],
            llm_content=_valid_llm_json(),
            current_signal_hash=None,
        )

        with patch("freqpred.signal.pipeline.retrieve", new=AsyncMock(return_value=[])):
            result = await pipeline.analyze(_make_market())

        # LLM should still be called even with no evidence
        llm_client.complete.assert_awaited_once()
        assert result is not None
