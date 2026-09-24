"""Unit tests for the TypeSafe System One transport and LLMClient.system_one.

No real API calls: the transport is driven through an httpx MockTransport and
DB writes are mocked.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.llm.audit import calculate_cost
from freqpred.llm.client import LLMClient, LLMError
from freqpred.llm.typesafe import (
    JEV_INPUT_PER_MTOK,
    JEV_MODEL,
    SystemOneResponse,
    TypeSafeError,
    TypeSafeTransport,
    audit_prompt,
    choice,
    noul,
    register_jev_pricing,
    score,
)

FAKE_QUERY_ID = 7


def _payload(answers: dict, *, input_tokens: int = 447, output_tokens: int = 23) -> dict:
    return {
        "model": JEV_MODEL,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _transport(
    handler=None, *, status: int = 200, payload: dict | None = None
) -> TypeSafeTransport:
    calls: list[httpx.Request] = []

    def _default(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=payload or _payload({"p": {"type": "noul", "noul": 0.7}}))

    mock = httpx.MockTransport(handler or _default)
    transport = TypeSafeTransport("key", http_client=httpx.AsyncClient(transport=mock))
    transport.recorded = calls  # type: ignore[attr-defined]
    return transport


def _client(transport: TypeSafeTransport | None, *, daily_spend_cap_usd: float | None = None):
    session = AsyncMock()
    session.commit = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return LLMClient(
        MagicMock(),
        session_factory,
        default_strategy="test_strategy",
        typesafe_transport=transport,
        daily_spend_cap_usd=daily_spend_cap_usd,
    )


class TestQuestionBuilders:
    def test_noul_carries_both_criteria(self):
        q = noul("Will it resolve YES?", when_true="it happens", when_false="it does not")
        assert q["type"] == "noul"
        assert q["criteria"] == {"true": "it happens", "false": "it does not"}

    def test_score_requires_at_least_two_levels(self):
        with pytest.raises(ValueError):
            score("How strong?", ["only one"])

    def test_choice_requires_at_least_two_options(self):
        with pytest.raises(ValueError):
            choice("Which?", {"only": None})


class TestSystemOneResponse:
    def _response(self, answers: dict) -> SystemOneResponse:
        return SystemOneResponse(
            model=JEV_MODEL, answers=answers, tokens_input=10, tokens_output=0,
            latency_ms=100, raw={},
        )

    def test_reads_a_noul_probability(self):
        assert self._response({"p": {"type": "noul", "noul": 0.92}}).noul("p") == 0.92

    @pytest.mark.parametrize("bad", [45, 1.5, -0.1, "0.5", None])
    def test_rejects_a_noul_outside_zero_to_one(self, bad):
        # A provider typing a percentage as a "number" would otherwise read as
        # certainty — the deepseek failure mode, which cost a live signal once.
        with pytest.raises(TypeSafeError):
            self._response({"p": {"type": "noul", "noul": bad}}).noul("p")

    def test_rejects_a_wrong_answer_type(self):
        with pytest.raises(TypeSafeError, match="expected 'noul'"):
            self._response({"p": {"type": "score", "score": 1.0}}).noul("p")

    def test_rejects_a_missing_question(self):
        with pytest.raises(TypeSafeError, match="no answer"):
            self._response({"p": {"type": "noul", "noul": 0.5}}).noul("absent")

    def test_confidence_is_none_on_a_noul(self):
        # Documented API behaviour: only choice and score carry confidence.
        assert self._response({"p": {"type": "noul", "noul": 0.5}}).confidence("p") is None

    def test_confidence_is_read_from_a_score(self):
        r = self._response({"s": {"type": "score", "score": 1.6, "confidence": 0.78}})
        assert r.confidence("s") == 0.78


class TestTransport:
    @pytest.mark.asyncio
    async def test_posts_model_state_and_questions(self):
        transport = _transport()
        await transport.system_one(
            state={"q": "will it?"}, questions={"p": noul("?", when_true="t", when_false="f")}
        )
        body = json.loads(transport.recorded[0].content)
        assert body["model"] == JEV_MODEL
        assert body["state"] == {"q": "will it?"}
        assert set(body["questions"]) == {"p"}

    @pytest.mark.asyncio
    async def test_sends_the_bearer_token(self):
        transport = _transport()
        await transport.system_one(state="x", questions={"p": noul("?", when_true="t", when_false="f")})
        assert transport.recorded[0].headers["Authorization"] == "Bearer key"

    @pytest.mark.asyncio
    async def test_reports_usage_and_latency(self):
        transport = _transport(payload=_payload({"p": {"type": "noul", "noul": 0.7}}))
        response = await transport.system_one(state="x", questions={"p": noul("?", when_true="t", when_false="f")})
        assert (response.tokens_input, response.tokens_output) == (447, 23)
        assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_raises_on_a_validation_error(self):
        transport = _transport(status=422, payload={"error": "malformed question"})
        with pytest.raises(TypeSafeError, match="422"):
            await transport.system_one(state="x", questions={"p": noul("?", when_true="t", when_false="f")})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 529])
    async def test_retries_then_succeeds(self, status):
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(1)
            if len(seen) == 1:
                return httpx.Response(status)
            return httpx.Response(200, json=_payload({"p": {"type": "noul", "noul": 0.6}}))

        with patch("freqpred.llm.typesafe.asyncio.sleep", new=AsyncMock()):
            response = await _transport(handler).system_one(
                state="x", questions={"p": noul("?", when_true="t", when_false="f")}
            )
        assert response.noul("p") == 0.6
        assert len(seen) == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_repeated_overload(self):
        with patch("freqpred.llm.typesafe.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(TypeSafeError):
                await _transport(lambda r: httpx.Response(529)).system_one(
                    state="x", questions={"p": noul("?", when_true="t", when_false="f")}
                )

    @pytest.mark.asyncio
    async def test_rejects_an_empty_question_map(self):
        with pytest.raises(ValueError):
            await _transport().system_one(state="x", questions={})


class TestPricing:
    def test_jev_is_priced_from_its_own_rate_not_the_sonnet_fallback(self):
        # calculate_cost falls back to Sonnet ($3/$15 per Mtok) for unknown
        # models. Unregistered, Jev would read ~70x more expensive than it is —
        # inverting the one comparison it is being evaluated on.
        register_jev_pricing()
        expected = 32_000 * JEV_INPUT_PER_MTOK / 1_000_000
        assert calculate_cost(JEV_MODEL, 32_000, 500) == pytest.approx(expected)
        assert calculate_cost(JEV_MODEL, 32_000, 500) < calculate_cost(
            "claude-sonnet-4-6", 32_000, 500
        ) / 50

    def test_output_tokens_are_free(self):
        register_jev_pricing()
        assert calculate_cost(JEV_MODEL, 100, 0) == calculate_cost(JEV_MODEL, 100, 10_000)


class TestAuditPrompt:
    def test_serializes_the_whole_request(self):
        stored = json.loads(audit_prompt({"a": 1}, {"p": noul("?", when_true="t", when_false="f")}))
        assert stored["state"] == {"a": 1}
        assert stored["questions"]["p"]["type"] == "noul"


class TestClientSystemOne:
    @pytest.mark.asyncio
    async def test_writes_one_audit_row_on_success(self):
        client = _client(_transport())
        with patch("freqpred.llm.client.log_llm_query", new=AsyncMock(return_value=FAKE_QUERY_ID)) as logged:
            response, query_id = await client.system_one(
                state="x",
                questions={"p": noul("?", when_true="t", when_false="f")},
                query_type="model_eval",
            )
        assert query_id == FAKE_QUERY_ID
        assert response.noul("p") == 0.7
        assert logged.await_count == 1
        kwargs = logged.await_args.kwargs
        assert kwargs["success"] is True
        assert kwargs["query_type"] == "model_eval"
        assert kwargs["model_used"] == JEV_MODEL

    @pytest.mark.asyncio
    async def test_audits_the_registered_jev_cost(self):
        client = _client(_transport())
        with patch("freqpred.llm.client.log_llm_query", new=AsyncMock(return_value=FAKE_QUERY_ID)) as logged:
            await client.system_one(
                state="x",
                questions={"p": noul("?", when_true="t", when_false="f")},
                query_type="model_eval",
            )
        expected = 447 * JEV_INPUT_PER_MTOK / 1_000_000
        assert logged.await_args.kwargs["cost_usd"] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_writes_a_failure_row_and_reraises(self):
        # Hard constraint: a failed call is still a logged call.
        client = _client(_transport(status=422, payload={"error": "bad"}))
        with patch("freqpred.llm.client.log_llm_query", new=AsyncMock(return_value=FAKE_QUERY_ID)) as logged:
            with pytest.raises(TypeSafeError):
                await client.system_one(
                    state="x",
                    questions={"p": noul("?", when_true="t", when_false="f")},
                    query_type="model_eval",
                )
        assert logged.await_count == 1
        assert logged.await_args.kwargs["success"] is False
        assert "422" in logged.await_args.kwargs["error_message"]

    @pytest.mark.asyncio
    async def test_raises_when_no_transport_is_configured(self):
        with pytest.raises(LLMError, match="TYPESAFE_API_KEY"):
            await _client(None).system_one(
                state="x",
                questions={"p": noul("?", when_true="t", when_false="f")},
                query_type="model_eval",
            )

    @pytest.mark.asyncio
    async def test_respects_the_shared_daily_spend_cap(self):
        from freqpred.llm.audit import LLMBudgetExceededError

        client = _client(_transport(), daily_spend_cap_usd=1.0)
        with patch("freqpred.llm.client.get_daily_spend_usd", new=AsyncMock(return_value=5.0)):
            with pytest.raises(LLMBudgetExceededError):
                await client.system_one(
                    state="x",
                    questions={"p": noul("?", when_true="t", when_false="f")},
                    query_type="model_eval",
                )
