"""Unit tests for freqpred/ingestion/catalyst_generator.py.

All LLM calls, DB operations, and embedder calls are mocked.
No real API calls are made.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.catalyst_generator import (
    CatalystGenerationError,
    _build_prompt,
    _parse_queries,
    generate_catalysts,
)
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


def _make_anthropic_response(queries: list[str]) -> MagicMock:
    """Build a mock Anthropic message response containing a JSON array."""
    msg = MagicMock()
    content_block = MagicMock()
    content_block.text = json.dumps(queries)
    msg.content = [content_block]
    msg.usage.input_tokens = 120
    msg.usage.output_tokens = 40
    return msg


def _make_session(prior_run_row=None, has_run_today: bool = False) -> AsyncMock:
    """Build a mock AsyncSession."""
    session = AsyncMock()
    session.add = MagicMock()

    call_count = 0

    async def execute_side_effect(*args, **kwargs):
        nonlocal call_count
        result = MagicMock()
        call_count += 1
        if call_count == 1:
            # _get_latest_run query
            result.scalar_one_or_none.return_value = prior_run_row
        elif call_count == 2:
            # log_llm_query flush triggers auto-id
            result.scalar_one_or_none.return_value = None
        else:
            result.scalar_one_or_none.return_value = None
        return result

    session.execute.side_effect = execute_side_effect

    # flush does nothing but must be awaitable
    session.flush = AsyncMock()

    return session


def _make_anthropic_client(queries: list[str]) -> MagicMock:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=_make_anthropic_response(queries))
    return client


# ---------------------------------------------------------------------------
# _parse_queries
# ---------------------------------------------------------------------------


class TestParseQueries:
    def test_clean_json_array(self) -> None:
        queries, err = _parse_queries('["query one", "query two", "query three"]')
        assert queries == ["query one", "query two", "query three"]
        assert err is None

    def test_json_with_leading_text(self) -> None:
        text = 'Here are the queries:\n["q1", "q2"]'
        queries, err = _parse_queries(text)
        assert queries == ["q1", "q2"]
        assert err is None

    def test_empty_strings_filtered_out(self) -> None:
        queries, err = _parse_queries('["q1", "", "  ", "q2"]')
        assert queries == ["q1", "q2"]
        assert err is None

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
            embedding=[0.1] * 1024,
            embedding_model="voyage-3",
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
            embedding=[0.1] * 1024,
            embedding_model="voyage-3",
        )
        prompt = _build_prompt(_market(), rag_docs=[doc])
        # 500 char limit + "..." — prompt should not contain the full 2000 chars
        assert long_body not in prompt


# ---------------------------------------------------------------------------
# generate_catalysts
# ---------------------------------------------------------------------------


class TestGenerateCatalysts:
    @pytest.mark.asyncio
    async def test_first_run_creates_run_and_queries(self) -> None:
        session = _make_session(prior_run_row=None)
        client = _make_anthropic_client(FAKE_QUERIES)

        with patch("freqpred.ingestion.catalyst_generator.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 42  # fake llm_query_id
            result = await generate_catalysts(_market(), session, client)

        assert result.market_id == "MKT-1"
        assert result.generation == 1
        assert result.is_active is True
        assert result.llm_query_id == 42

        # CatalystRunRow + 3 CatalystQueryRow adds = 4 calls to session.add
        assert session.add.call_count == 4

    @pytest.mark.asyncio
    async def test_rerun_increments_generation(self) -> None:
        prior = MagicMock()
        prior.generation = 2
        prior.id = uuid.uuid4()

        session = _make_session(prior_run_row=prior)
        client = _make_anthropic_client(FAKE_QUERIES)

        with patch("freqpred.ingestion.catalyst_generator.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 99
            result = await generate_catalysts(_market(), session, client)

        assert result.generation == 3

    @pytest.mark.asyncio
    async def test_llm_always_logged_on_failure(self) -> None:
        """Even when the LLM returns unparseable output, audit row is written."""
        session = _make_session(prior_run_row=None)
        client = MagicMock()
        bad_msg = MagicMock()
        bad_msg.content = [MagicMock(text="not valid json")]
        bad_msg.usage.input_tokens = 50
        bad_msg.usage.output_tokens = 10
        client.messages.create = AsyncMock(return_value=bad_msg)

        with patch("freqpred.ingestion.catalyst_generator.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 7
            with pytest.raises(CatalystGenerationError):
                await generate_catalysts(_market(), session, client)

        mock_log.assert_called_once()
        # success=False should be passed
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_llm_api_error_logged_and_raises(self) -> None:
        session = _make_session(prior_run_row=None)
        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("freqpred.ingestion.catalyst_generator.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 5
            with pytest.raises(CatalystGenerationError):
                await generate_catalysts(_market(), session, client)

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_rerun_uses_embedder_for_rag(self) -> None:
        """On a re-run, the embedder is called to retrieve RAG context."""
        prior = MagicMock()
        prior.generation = 1
        prior.id = uuid.uuid4()

        session = _make_session(prior_run_row=prior)
        client = _make_anthropic_client(FAKE_QUERIES)
        embedder = AsyncMock()
        embedder.embed_text.return_value = [0.1] * 1024

        with patch("freqpred.llm.audit.log_llm_query", new_callable=AsyncMock) as mock_log, \
             patch("freqpred.rag.retriever.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_log.return_value = 10
            mock_retrieve.return_value = []
            await generate_catalysts(_market(), session, client, embedder=embedder)

        mock_retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_first_run_no_rag_call(self) -> None:
        """On a first run (no prior), embedder/RAG should NOT be called."""
        session = _make_session(prior_run_row=None)
        client = _make_anthropic_client(FAKE_QUERIES)
        embedder = AsyncMock()

        with patch("freqpred.llm.audit.log_llm_query", new_callable=AsyncMock) as mock_log, \
             patch("freqpred.rag.retriever.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_log.return_value = 1
            await generate_catalysts(_market(), session, client, embedder=embedder)

        mock_retrieve.assert_not_called()
        embedder.embed_text.assert_not_called()
