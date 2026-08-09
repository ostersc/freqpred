"""Unit tests for freqpred/llm/client.py.

All Anthropic API calls and DB writes are mocked — no real API calls made.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.llm.audit import LLMBudgetExceededError
from freqpred.llm.client import LLMClient, LLMConsecutiveErrorsError, LLMError
from freqpred.llm.models import LLMResponse

MODEL = "claude-haiku-4-5-20251001"
PROMPT = "Will the Fed raise rates?"
QUERY_TYPE = "market_analysis"
FAKE_CONTENT = "Probability is approximately 0.35."
FAKE_QUERY_ID = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_response(
    content: str = FAKE_CONTENT,
    input_tokens: int = 100,
    output_tokens: int = 30,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> MagicMock:
    msg = MagicMock()
    block = MagicMock()
    block.text = content
    msg.content = [block]
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    msg.usage.cache_read_input_tokens = cache_read_tokens
    msg.usage.cache_creation_input_tokens = cache_creation_tokens
    # A real Anthropic-direct response carries no cost field; MagicMock would
    # otherwise invent one and make this look like an OpenRouter response.
    msg.usage.cost = None
    return msg


def _make_client(
    anthropic_response=None,
    api_error: Exception | None = None,
    daily_spend_cap_usd: float | None = None,
) -> tuple[LLMClient, MagicMock]:
    """Return (LLMClient, mock_anthropic_client)."""
    anth = MagicMock()
    anth.messages = MagicMock()
    if api_error:
        anth.messages.create = AsyncMock(side_effect=api_error)
    else:
        anth.messages.create = AsyncMock(return_value=anthropic_response or _make_anthropic_response())

    session_factory = MagicMock()
    # Session context manager
    session = AsyncMock()
    session.commit = AsyncMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    client = LLMClient(
        anth,
        session_factory,
        default_strategy="test_strategy",
        daily_spend_cap_usd=daily_spend_cap_usd,
    )
    return client, anth


def _make_openrouter_response(
    content: str = FAKE_CONTENT,
    input_tokens: int = 100,
    output_tokens: int = 30,
    cost: float = 3.65e-06,
) -> MagicMock:
    """An OpenRouter response: Anthropic-shaped, plus a reported dollar cost."""
    msg = _make_anthropic_response(content, input_tokens, output_tokens)
    msg.usage.cost = cost
    return msg


def _make_routed_client(
    anthropic_response=None,
    openrouter_response=None,
    with_openrouter: bool = True,
) -> tuple[LLMClient, MagicMock, MagicMock | None]:
    """Return (LLMClient, mock_anthropic_client, mock_openrouter_client)."""
    anth = MagicMock()
    anth.messages = MagicMock()
    anth.messages.create = AsyncMock(
        return_value=anthropic_response or _make_anthropic_response()
    )

    router = None
    if with_openrouter:
        router = MagicMock()
        router.messages = MagicMock()
        router.messages.create = AsyncMock(
            return_value=openrouter_response or _make_openrouter_response()
        )

    session_factory = MagicMock()
    session = AsyncMock()
    session.commit = AsyncMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    client = LLMClient(
        anth,
        session_factory,
        default_strategy="test_strategy",
        openrouter_client=router,
    )
    return client, anth, router


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestComplete:
    @pytest.mark.asyncio
    async def test_returns_llm_response_on_success(self) -> None:
        client, _ = _make_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = FAKE_QUERY_ID
            result = await client.complete(PROMPT, MODEL, QUERY_TYPE)

        assert isinstance(result, LLMResponse)
        assert result.content == FAKE_CONTENT
        assert result.model == MODEL
        assert result.tokens_input == 100
        assert result.tokens_output == 30
        assert result.llm_query_id == FAKE_QUERY_ID
        assert result.cost_usd > 0
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_audit_row_written_on_success(self) -> None:
        client, _ = _make_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = FAKE_QUERY_ID
            await client.complete(PROMPT, MODEL, QUERY_TYPE, market_id="MKT-1")

        mock_log.assert_called_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["success"] is True
        assert kwargs["model_used"] == MODEL
        assert kwargs["query_type"] == QUERY_TYPE
        assert kwargs["market_id"] == "MKT-1"
        assert kwargs.get("error_message") is None
        assert kwargs["tokens_input"] == 100
        assert kwargs["tokens_output"] == 30

    @pytest.mark.asyncio
    async def test_strategy_passed_through(self) -> None:
        client, _ = _make_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 1
            await client.complete(PROMPT, MODEL, QUERY_TYPE, strategy="ConservativeDefault")

        assert mock_log.call_args.kwargs["strategy"] == "ConservativeDefault"

    @pytest.mark.asyncio
    async def test_default_strategy_used_when_not_provided(self) -> None:
        client, _ = _make_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 1
            await client.complete(PROMPT, MODEL, QUERY_TYPE)

        assert mock_log.call_args.kwargs["strategy"] == "test_strategy"

    @pytest.mark.asyncio
    async def test_exactly_one_audit_row_per_call(self) -> None:
        client, _ = _make_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 1
            await client.complete(PROMPT, MODEL, QUERY_TYPE)
            await client.complete(PROMPT, MODEL, QUERY_TYPE)

        assert mock_log.call_count == 2


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestCompleteFailure:
    @pytest.mark.asyncio
    async def test_raises_llm_error_on_api_failure(self) -> None:
        client, _ = _make_client(api_error=RuntimeError("API timeout"))
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 5
            with pytest.raises(LLMError, match="API timeout"):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)

    @pytest.mark.asyncio
    async def test_audit_row_written_on_failure(self) -> None:
        client, _ = _make_client(api_error=RuntimeError("network error"))
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 7
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE, market_id="MKT-2")

        mock_log.assert_called_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["success"] is False
        assert "network error" in kwargs["error_message"]
        assert kwargs["market_id"] == "MKT-2"

    @pytest.mark.asyncio
    async def test_failure_row_has_zero_tokens_and_cost(self) -> None:
        client, _ = _make_client(api_error=ValueError("bad request"))
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 3
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)

        kwargs = mock_log.call_args.kwargs
        assert kwargs["tokens_input"] == 0
        assert kwargs["tokens_output"] == 0
        assert kwargs["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


class TestCostCalculation:
    @pytest.mark.asyncio
    async def test_haiku_cost_computed_correctly(self) -> None:
        resp = _make_anthropic_response(input_tokens=1_000_000, output_tokens=1_000_000)
        client, _ = _make_client(anthropic_response=resp)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 1
            result = await client.complete(PROMPT, "claude-haiku-4-5-20251001", QUERY_TYPE)

        # 1.00/M input + 5.00/M output = $6.00
        assert abs(result.cost_usd - 6.00) < 1e-6

    @pytest.mark.asyncio
    async def test_sonnet_cost_computed_correctly(self) -> None:
        resp = _make_anthropic_response(input_tokens=1_000_000, output_tokens=1_000_000)
        client, _ = _make_client(anthropic_response=resp)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 1
            result = await client.complete(PROMPT, "claude-sonnet-4-6", QUERY_TYPE)

        # 3.00/M input + 15.00/M output = $18.00
        assert abs(result.cost_usd - 18.00) < 1e-6

    @pytest.mark.asyncio
    async def test_opus_4_7_priced_at_its_own_rate_not_sonnet_default(self) -> None:
        """claude-opus-4-7 must have its own pricing entry, not silently fall
        back to the Sonnet-level default — that fallback previously undercounted
        every Opus 4.7 call by ~40%."""
        resp = _make_anthropic_response(input_tokens=1_000_000, output_tokens=1_000_000)
        client, _ = _make_client(anthropic_response=resp)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 1
            result = await client.complete(PROMPT, "claude-opus-4-7", QUERY_TYPE)

        # 5.00/M input + 25.00/M output = $30.00
        assert abs(result.cost_usd - 30.00) < 1e-6

    @pytest.mark.asyncio
    async def test_opus_5_priced_at_its_own_rate_not_sonnet_default(self) -> None:
        """claude-opus-5 must have its own pricing entry, not silently fall
        back to the Sonnet-level default."""
        resp = _make_anthropic_response(input_tokens=1_000_000, output_tokens=1_000_000)
        client, _ = _make_client(anthropic_response=resp)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 1
            result = await client.complete(PROMPT, "claude-opus-5", QUERY_TYPE)

        # 5.00/M input + 25.00/M output = $30.00
        assert abs(result.cost_usd - 30.00) < 1e-6

    @pytest.mark.asyncio
    async def test_cache_tokens_included_in_cost(self) -> None:
        """Cache write/read tokens must be billed, not silently dropped —
        cache_system=True calls previously counted only the uncached
        input_tokens, undercounting real Anthropic billing."""
        resp = _make_anthropic_response(
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        client, _ = _make_client(anthropic_response=resp)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 1
            result = await client.complete(PROMPT, "claude-sonnet-4-6", QUERY_TYPE)

        # sonnet input rate $3.00/M: cache write @2x = $6.00, cache read @0.1x = $0.30
        assert abs(result.cost_usd - 6.30) < 1e-6

    @pytest.mark.asyncio
    async def test_unknown_model_falls_back_to_default_and_warns(self) -> None:
        """A model string with no pricing entry must still produce a cost
        (via the default rate) and must log a warning so a future model swap
        without a matching pricing entry is caught immediately instead of
        silently mispricing for days."""
        resp = _make_anthropic_response(input_tokens=1_000_000, output_tokens=1_000_000)
        client, _ = _make_client(anthropic_response=resp)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log, patch(
            "freqpred.llm.audit.log"
        ) as mock_audit_log:
            mock_log.return_value = 1
            result = await client.complete(PROMPT, "claude-opus-9000-hypothetical", QUERY_TYPE)

        # falls back to _DEFAULT_COST_PER_TOKEN = 3.00/M input + 15.00/M output = $18.00
        assert abs(result.cost_usd - 18.00) < 1e-6
        mock_audit_log.warning.assert_called_once()
        assert mock_audit_log.warning.call_args.args[0] == "llm.unknown_model_pricing"


# ---------------------------------------------------------------------------
# Budget circuit breaker
# ---------------------------------------------------------------------------


class TestBudgetCircuitBreaker:
    @pytest.mark.asyncio
    async def test_raises_when_cap_exceeded(self) -> None:
        client, _ = _make_client(daily_spend_cap_usd=10.0)
        with patch("freqpred.llm.client.get_daily_spend_usd", new_callable=AsyncMock, return_value=10.0):
            with pytest.raises(LLMBudgetExceededError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)

    @pytest.mark.asyncio
    async def test_raises_when_spend_exceeds_cap(self) -> None:
        client, _ = _make_client(daily_spend_cap_usd=5.0)
        with patch("freqpred.llm.client.get_daily_spend_usd", new_callable=AsyncMock, return_value=7.32):
            with pytest.raises(LLMBudgetExceededError, match="5.00"):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)

    @pytest.mark.asyncio
    async def test_no_api_call_when_budget_exceeded(self) -> None:
        client, anth = _make_client(daily_spend_cap_usd=10.0)
        with patch("freqpred.llm.client.get_daily_spend_usd", new_callable=AsyncMock, return_value=10.0):
            with pytest.raises(LLMBudgetExceededError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
        anth.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_call_when_under_cap(self) -> None:
        client, _ = _make_client(daily_spend_cap_usd=10.0)
        with patch("freqpred.llm.client.get_daily_spend_usd", new_callable=AsyncMock, return_value=9.99):
            with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
                mock_log.return_value = 1
                result = await client.complete(PROMPT, MODEL, QUERY_TYPE)
        assert isinstance(result, LLMResponse)

    @pytest.mark.asyncio
    async def test_no_cap_check_when_cap_is_none(self) -> None:
        """When daily_spend_cap_usd is None, no spend check is performed."""
        client, _ = _make_client(daily_spend_cap_usd=None)
        with patch("freqpred.llm.client.get_daily_spend_usd", new_callable=AsyncMock) as mock_spend:
            with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock) as mock_log:
                mock_log.return_value = 1
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
        mock_spend.assert_not_called()


# ---------------------------------------------------------------------------
# Consecutive-error circuit breaker
# ---------------------------------------------------------------------------


def _make_error_client(max_consecutive_errors: int = 3) -> LLMClient:
    """Return an LLMClient wired to always fail the Anthropic API call."""
    anth = MagicMock()
    anth.messages = MagicMock()
    anth.messages.create = AsyncMock(side_effect=Exception("api down"))

    session_factory = MagicMock()
    session = AsyncMock()
    session.commit = AsyncMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    return LLMClient(
        anth,
        session_factory,
        max_consecutive_errors=max_consecutive_errors,
    )


class TestConsecutiveErrorCircuitBreaker:
    @pytest.mark.asyncio
    async def test_fires_after_threshold(self) -> None:
        """Third consecutive failure raises LLMConsecutiveErrorsError."""
        client = _make_error_client(max_consecutive_errors=3)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
            with pytest.raises(LLMConsecutiveErrorsError, match="3 consecutive"):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)

    @pytest.mark.asyncio
    async def test_below_threshold_raises_llm_error(self) -> None:
        """Failures below threshold raise LLMError, not the CB exception."""
        client = _make_error_client(max_consecutive_errors=3)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
        assert client._consecutive_errors == 2

    @pytest.mark.asyncio
    async def test_resets_on_success(self) -> None:
        """Counter resets to 0 after a successful call; subsequent failure is LLMError, not CB."""
        anth = MagicMock()
        anth.messages = MagicMock()
        # First two calls fail, third succeeds, fourth fails
        anth.messages.create = AsyncMock(
            side_effect=[
                Exception("fail 1"),
                Exception("fail 2"),
                _make_anthropic_response(),
                Exception("fail 4"),
            ]
        )
        session_factory = MagicMock()
        session = AsyncMock()
        session.commit = AsyncMock()
        session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        client = LLMClient(anth, session_factory, max_consecutive_errors=3)

        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
            # Success — counter resets
            await client.complete(PROMPT, MODEL, QUERY_TYPE)
            assert client._consecutive_errors == 0
            # One more failure — counter = 1, not CB
            with pytest.raises(LLMError):
                await client.complete(PROMPT, MODEL, QUERY_TYPE)
            assert client._consecutive_errors == 1


# ---------------------------------------------------------------------------
# OpenRouter transport routing
# ---------------------------------------------------------------------------


OPENROUTER_MODEL = "deepseek/deepseek-v3"


class TestTransportRouting:
    """Which client actually receives the call — the wiring, not the helper.

    _transport_for() returning the right object proves nothing on its own;
    these assert that complete() sends the request through it.
    """

    @pytest.mark.asyncio
    async def test_openrouter_slug_goes_to_openrouter_client(self) -> None:
        client, anth, router = _make_routed_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE)

        router.messages.create.assert_awaited_once()
        anth.messages.create.assert_not_awaited()
        assert router.messages.create.await_args.kwargs["model"] == OPENROUTER_MODEL

    @pytest.mark.asyncio
    async def test_anthropic_id_stays_on_anthropic_client(self) -> None:
        client, anth, router = _make_routed_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            await client.complete(PROMPT, MODEL, QUERY_TYPE)

        anth.messages.create.assert_awaited_once()
        router.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slug_without_openrouter_client_raises(self) -> None:
        """Better a named misconfiguration than an 'unknown model' from Anthropic."""
        client, anth, _ = _make_routed_client(with_openrouter=False)
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
                await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE)

        anth.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_forcing_survives_the_hop(self) -> None:
        """The signal path depends on json_tool; it must reach OpenRouter intact."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"probability": 0.5}
        response = _make_openrouter_response()
        response.content = [tool_block]
        client, _, router = _make_routed_client(openrouter_response=response)
        tool = {"name": "submit_analysis", "description": "d", "input_schema": {}}
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE, json_tool=tool)

        kwargs = router.messages.create.await_args.kwargs
        assert kwargs["tools"][0]["name"] == "submit_analysis"
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_analysis"}


class TestCostAccounting:
    @pytest.mark.asyncio
    async def test_openrouter_cost_is_the_reported_figure(self) -> None:
        """Not calculate_cost, which has no rates for a non-Anthropic slug."""
        client, _, _ = _make_routed_client(
            openrouter_response=_make_openrouter_response(cost=0.00042)
        )
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            with patch("freqpred.llm.client.calculate_cost") as mock_calc:
                result = await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE)

        assert result.cost_usd == pytest.approx(0.00042)
        mock_calc.assert_not_called()

    @pytest.mark.asyncio
    async def test_anthropic_cost_still_uses_price_table(self) -> None:
        client, _, _ = _make_routed_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            with patch("freqpred.llm.client.calculate_cost", return_value=0.99) as mock_calc:
                result = await client.complete(PROMPT, MODEL, QUERY_TYPE)

        mock_calc.assert_called_once()
        assert result.cost_usd == pytest.approx(0.99)

    @pytest.mark.asyncio
    async def test_reported_cost_reaches_the_audit_row(self) -> None:
        """Spend tracking is only as good as what gets written to llm_queries."""
        client, _, _ = _make_routed_client(
            openrouter_response=_make_openrouter_response(cost=0.00123)
        )
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1) as mock_log:
            await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE)

        assert mock_log.await_args.kwargs["cost_usd"] == pytest.approx(0.00123)
        assert mock_log.await_args.kwargs["model_used"] == OPENROUTER_MODEL


class TestToolContract:
    """A forced tool_choice that the model ignores must not read as success."""

    @staticmethod
    def _no_tool_response(blocks: list) -> MagicMock:
        msg = _make_openrouter_response()
        msg.content = blocks
        msg.stop_reason = "end_turn"
        return msg

    @staticmethod
    def _text_block(text: str) -> MagicMock:
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    @staticmethod
    def _thinking_block() -> MagicMock:
        block = MagicMock(spec=["type", "thinking"])
        block.type = "thinking"
        block.thinking = "pondering"
        return block

    @pytest.mark.asyncio
    async def test_prose_instead_of_tool_call_raises(self) -> None:
        """deepseek/deepseek-v3.2 answers in prose and stops with end_turn."""
        client, _, _ = _make_routed_client(
            openrouter_response=self._no_tool_response([self._text_block("I cannot know that.")])
        )
        tool = {"name": "submit_analysis", "description": "d", "input_schema": {}}
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            with pytest.raises(LLMError, match="ignored the forced tool_choice"):
                await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE, json_tool=tool)

    @pytest.mark.asyncio
    async def test_thinking_only_response_raises(self) -> None:
        """tencent/hy3 spends the budget thinking and emits no tool block.

        The old fallback produced "" here, which is the quietest possible
        failure: an empty signal logged as a success.
        """
        client, _, _ = _make_routed_client(
            openrouter_response=self._no_tool_response([self._thinking_block()])
        )
        tool = {"name": "submit_analysis", "description": "d", "input_schema": {}}
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            with pytest.raises(LLMError):
                await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE, json_tool=tool)

    @pytest.mark.asyncio
    async def test_violation_is_audited_as_failure_with_its_real_cost(self) -> None:
        """The call still spent money; the row must say so, and say it failed."""
        client, _, _ = _make_routed_client(
            openrouter_response=self._no_tool_response([self._text_block("prose")])
        )
        tool = {"name": "submit_analysis", "description": "d", "input_schema": {}}
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1) as mock_log:
            with pytest.raises(LLMError):
                await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE, json_tool=tool)

        assert mock_log.await_count == 1
        kwargs = mock_log.await_args.kwargs
        assert kwargs["success"] is False
        assert kwargs["cost_usd"] > 0
        assert "forced tool_choice" in kwargs["error_message"]

    @pytest.mark.asyncio
    async def test_honored_tool_call_still_succeeds(self) -> None:
        client, _, router = _make_routed_client()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"probability": 0.5}
        response = _make_openrouter_response()
        response.content = [tool_block]
        router.messages.create = AsyncMock(return_value=response)

        tool = {"name": "submit_analysis", "description": "d", "input_schema": {}}
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1) as mock_log:
            result = await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE, json_tool=tool)

        assert result.content == '{"probability": 0.5}'
        assert mock_log.await_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_no_tool_requested_is_unaffected(self) -> None:
        """Plain text completions must not be caught by the new guard."""
        client, _, _ = _make_routed_client()
        with patch("freqpred.llm.client.log_llm_query", new_callable=AsyncMock, return_value=1):
            result = await client.complete(PROMPT, OPENROUTER_MODEL, QUERY_TYPE)

        assert result.content == FAKE_CONTENT
