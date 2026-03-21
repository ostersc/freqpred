"""Unit tests for freqpred/ingestion/catalyst_generator.py.

All LLM calls, DB operations, and embedder calls are mocked.
No real API calls are made.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.ingestion.catalyst_generator import (
    CatalystGenerationError,
    _build_prompt,
    _parse_queries,
    generate_catalysts,
)
from freqpred.llm.client import LLMError
from freqpred.llm.models import LLMResponse
from freqpred.markets.models import Market
from freqpred.rag.models import Document

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models        # noqa: F401
import freqpred.signal.models     # noqa: F401

NOW = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
FUTURE = NOW + timedelta(days=14)
FAKE_QUERIES = ["February CPI release 2026", "Fed Powell testimony", "January jobs report"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market(market_id: str = "MKT-1") -> Market:
    return Market(
        id=market_id,
        platform="kalshi",
        question="Will the Fed raise rates at the March 2026 meeting?",
        category="economics",
        close_time=FUTURE,
        yes_bid=0.30,
        yes_ask=0.34,
        mid_price=0.32,
        volume_24h=5000.0,
        open_interest=1000.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
    )


def _make_llm_client(
    queries: list[str] = FAKE_QUERIES,
    raises: Exception | None = None,
    llm_query_id: int = 42,
) -> MagicMock:
    """Return a mock LLMClient."""
    import json
    client = MagicMock()
    if raises:
        client.complete = AsyncMock(side_effect=raises)
    else:
        client.complete = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(queries),
                model="claude-haiku-4-5-20251001",
                tokens_input=120,
                tokens_output=40,
                cost_usd=0.000256,
                latency_ms=300,
                llm_query_id=llm_query_id,
            )
        )
    return client


def _make_session(prior_run_row=None) -> AsyncMock:
    """Build a mock AsyncSession."""
    session = AsyncMock()
    session.add = MagicMock()

    call_count = 0

    async def execute_side_effect(*args, **kwargs):
        nonlocal call_count
        result = MagicMock()
        call_count += 1
        if call_count == 1:
            result.scalar_one_or_none.return_value = prior_run_row
        else:
            result.scalar_one_or_none.return_value = None
        return result

    session.execute.side_effect = execute_side_effect
    session.flush = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# _parse_queries
# ---------------------------------------------------------------------------


class TestParseQueries:
    # New object format (preferred)

    def test_object_array_returns_pairs(self) -> None:
        text = '[{"query_text": "q one", "tv_query": "q AND one"}, {"query_text": "q two", "tv_query": null}]'
        pairs, err = _parse_queries(text)
        assert pairs == [("q one", "q AND one"), ("q two", None)]
        assert err is None

    def test_object_array_empty_tv_query_coerced_to_none(self) -> None:
        text = '[{"query_text": "query", "tv_query": "  "}]'
        pairs, err = _parse_queries(text)
        assert pairs == [("query", None)]
        assert err is None

    def test_object_array_missing_tv_query_key(self) -> None:
        text = '[{"query_text": "query"}]'
        pairs, err = _parse_queries(text)
        assert pairs == [("query", None)]
        assert err is None

    # Legacy plain-string format (still supported)

    def test_legacy_string_array_returns_pairs_with_none_tv(self) -> None:
        queries, err = _parse_queries('["query one", "query two", "query three"]')
        assert queries == [("query one", None), ("query two", None), ("query three", None)]
        assert err is None

    def test_legacy_json_with_leading_text(self) -> None:
        text = 'Here are the queries:\n["q1", "q2"]'
        queries, err = _parse_queries(text)
        assert queries == [("q1", None), ("q2", None)]
        assert err is None

    def test_legacy_empty_strings_filtered_out(self) -> None:
        queries, err = _parse_queries('["q1", "", "  ", "q2"]')
        assert queries == [("q1", None), ("q2", None)]
        assert err is None

    # Error cases

    def test_invalid_json_returns_error(self) -> None:
        queries, err = _parse_queries("not json at all")
        assert queries == []
        assert err is not None

    def test_json_object_not_array_returns_error(self) -> None:
        queries, err = _parse_queries('{"key": "value"}')
        assert queries == []
        assert err is not None

    def test_array_of_non_strings_returns_error(self) -> None:
        queries, err = _parse_queries("[1, 2, 3]")
        assert queries == []
        assert err is not None

    def test_empty_string_input(self) -> None:
        queries, err = _parse_queries("")
        assert queries == []
        assert err is not None


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_first_run_prompt_contains_market_info(self) -> None:
        market = _market()
        prompt = _build_prompt(market, rag_docs=[])
        assert market.question in prompt
        assert market.category in prompt
        assert "2026" in prompt  # close date

    def test_first_run_prompt_has_no_rag_section(self) -> None:
        prompt = _build_prompt(_market(), rag_docs=[])
        assert "Recent relevant news" not in prompt

    def test_rerun_prompt_includes_doc_titles(self) -> None:
        doc = Document(
            id=str(uuid.uuid4()),
            source_url="https://example.com/cpi",
            content_hash="abc",
            title="February CPI Comes In Hot",
            body="Inflation rose more than expected...",
            source_type="news",
            source_name="Reuters",
            category="economics",
            tags=[],
            published_at=NOW,
            fetched_at=NOW,
            embedding=[0.1] * 384,
            embedding_model="all-MiniLM-L6-v2",
        )
        prompt = _build_prompt(_market(), rag_docs=[doc])
        assert "Recent relevant news" in prompt
        assert "February CPI Comes In Hot" in prompt

    def test_doc_body_truncated(self) -> None:
        long_body = "x" * 2000
        doc = Document(
            id=str(uuid.uuid4()),
            source_url="https://example.com/long",
            content_hash="abc",
            title="Long Article",
            body=long_body,
            source_type="news",
            source_name="AP",
            category="economics",
            tags=[],
            published_at=NOW,
            fetched_at=NOW,
            embedding=[0.1] * 384,
            embedding_model="all-MiniLM-L6-v2",
        )
        prompt = _build_prompt(_market(), rag_docs=[doc])
        assert long_body not in prompt


# ---------------------------------------------------------------------------
# generate_catalysts
# ---------------------------------------------------------------------------


class TestGenerateCatalysts:
    @pytest.mark.asyncio
    async def test_first_run_creates_run_and_queries(self) -> None:
        session = _make_session(prior_run_row=None)
        client = _make_llm_client(llm_query_id=42)

        result = await generate_catalysts(_market(), session, client)

        assert result.market_id == "MKT-1"
        assert result.generation == 1
        assert result.is_active is True
        assert result.llm_query_id == 42

        # CatalystRunRow + 3 CatalystQueryRow = 4 session.add calls
        assert session.add.call_count == 4

    @pytest.mark.asyncio
    async def test_rerun_increments_generation(self) -> None:
        prior = MagicMock()
        prior.generation = 2
        prior.id = uuid.uuid4()

        session = _make_session(prior_run_row=prior)
        client = _make_llm_client(llm_query_id=99)

        result = await generate_catalysts(_market(), session, client)

        assert result.generation == 3

    @pytest.mark.asyncio
    async def test_llm_api_error_raises_catalyst_error(self) -> None:
        """LLMError from the client propagates as CatalystGenerationError."""
        session = _make_session(prior_run_row=None)
        client = _make_llm_client(raises=LLMError("API down"))

        with pytest.raises(CatalystGenerationError):
            await generate_catalysts(_market(), session, client)

    @pytest.mark.asyncio
    async def test_unparseable_response_raises_catalyst_error(self) -> None:
        """If LLM returns bad JSON, generate_catalysts raises CatalystGenerationError."""
        import json
        session = _make_session(prior_run_row=None)
        # Return a valid LLMResponse but with unparseable content
        client = MagicMock()
        client.complete = AsyncMock(
            return_value=LLMResponse(
                content="not valid json",
                model="claude-haiku-4-5-20251001",
                tokens_input=50,
                tokens_output=10,
                cost_usd=0.0,
                latency_ms=100,
                llm_query_id=7,
            )
        )

        with pytest.raises(CatalystGenerationError):
            await generate_catalysts(_market(), session, client)

    @pytest.mark.asyncio
    async def test_llm_client_called_with_correct_query_type(self) -> None:
        session = _make_session(prior_run_row=None)
        client = _make_llm_client()

        await generate_catalysts(_market(), session, client)

        client.complete.assert_called_once()
        args = client.complete.call_args.args
        assert args[2] == "catalyst_generation"

    @pytest.mark.asyncio
    async def test_llm_client_called_with_market_id(self) -> None:
        session = _make_session(prior_run_row=None)
        client = _make_llm_client()

        await generate_catalysts(_market("MKT-99"), session, client)

        kwargs = client.complete.call_args.kwargs
        assert kwargs["market_id"] == "MKT-99"

    @pytest.mark.asyncio
    async def test_rerun_uses_embedder_for_rag(self) -> None:
        """On a re-run, the embedder is called to retrieve RAG context."""
        from unittest.mock import patch

        prior = MagicMock()
        prior.generation = 1
        prior.id = uuid.uuid4()

        session = _make_session(prior_run_row=prior)
        client = _make_llm_client()
        embedder = AsyncMock()
        embedder.embed_text.return_value = [0.1] * 384

        with patch("freqpred.rag.retriever.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = []
            await generate_catalysts(_market(), session, client, embedder=embedder)

        mock_retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_first_run_no_rag_call(self) -> None:
        """On a first run (no prior), embedder/RAG should NOT be called."""
        from unittest.mock import patch

        session = _make_session(prior_run_row=None)
        client = _make_llm_client()
        embedder = AsyncMock()

        with patch("freqpred.rag.retriever.retrieve", new_callable=AsyncMock) as mock_retrieve:
            await generate_catalysts(_market(), session, client, embedder=embedder)

        mock_retrieve.assert_not_called()
        embedder.embed_text.assert_not_called()
