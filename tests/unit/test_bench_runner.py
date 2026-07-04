"""Unit tests for freqpred/bench/runner.py — audited execution, reps, estimate.

The LLM client is always mocked; no real API calls.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.bench.runner import estimate_run, run_benchmark
from freqpred.bench.scenarios import ModelOutput, Scenario
from freqpred.llm.client import LLMClient, LLMConsecutiveErrorsError, LLMError
from freqpred.llm.models import LLMResponse
from freqpred.signal.llm import SIGNAL_ANALYSIS_TOOL, SYSTEM_PROMPT

FROZEN_CLOSE = datetime(2026, 6, 1, tzinfo=UTC)


def _scenario(scenario_id: str = "s1", prompt: str = "the frozen prompt") -> Scenario:
    return Scenario(
        id=scenario_id,
        source="db",
        market_id=f"MKT-{scenario_id}",
        market_question="Will it happen?",
        close_time=FROZEN_CLOSE,
        outcome=1.0,
        prompt=prompt,
        incumbent=ModelOutput(
            model="incumbent-model",
            prior=0.6,
            posterior=0.65,
            confidence=0.7,
            direction="YES",
            updates_count=0,
            tokens_input=1200,
        ),
        yes_bid=0.48,
        yes_ask=0.50,
        mid_price=0.49,
    )


def _response(posterior: float = 0.70, direction: str = "YES") -> LLMResponse:
    content = json.dumps(
        {
            "prior": 0.6,
            "prior_basis": "basis",
            "updates_applied": [],
            "posterior": posterior,
            "probability": posterior,
            "confidence": 0.8,
            "direction": direction,
            "reasoning": "reasoning",
        }
    )
    return LLMResponse(
        content=content,
        model="candidate-model",
        tokens_input=1200,
        tokens_output=150,
        cost_usd=0.012,
        latency_ms=900,
        llm_query_id=1,
        thinking_tokens=50,
    )


def _mock_client(side_effect=None, return_value=None) -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(side_effect=side_effect, return_value=return_value)
    return client


async def test_all_candidate_calls_go_through_llm_client_audit() -> None:
    """Every call must hit LLMClient.complete with the audit fields —
    query_type='model_eval', the frozen prompt verbatim, the production
    system prompt and tool schema, and adaptive thinking."""
    client = _mock_client(return_value=_response())
    scenarios = [_scenario("s1", prompt="prompt one"), _scenario("s2", prompt="prompt two")]

    run = await run_benchmark(client, scenarios, candidate_model="candidate-model", reps=1)

    assert client.complete.await_count == 2
    for call, scenario in zip(client.complete.await_args_list, scenarios, strict=True):
        args, kwargs = call
        assert args[0] == scenario.prompt  # verbatim
        assert args[1] == "candidate-model"
        assert kwargs["query_type"] == "model_eval"
        assert kwargs["system"] == SYSTEM_PROMPT
        assert kwargs["json_tool"] == SIGNAL_ANALYSIS_TOOL
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert run.stopped_early is None
    assert all(r.point_estimate is not None for r in run.scenario_runs)


async def test_prompt_mode_defaults_to_incumbent_model() -> None:
    """candidate_model=None → each scenario runs on its incumbent's model."""
    client = _mock_client(return_value=_response())
    await run_benchmark(client, [_scenario()], candidate_model=None, reps=1)
    assert client.complete.await_args_list[0][0][1] == "incumbent-model"


async def test_reps_aggregation_reports_per_scenario_spread() -> None:
    client = _mock_client(
        side_effect=[_response(0.60), _response(0.70), _response(0.80)]
    )
    run = await run_benchmark(client, [_scenario()], candidate_model="m", reps=3)

    scenario_run = run.scenario_runs[0]
    assert len(scenario_run.reps) == 3
    point = scenario_run.point_estimate
    assert point.posterior == pytest.approx(0.70)  # mean over reps
    assert scenario_run.posterior_spread == pytest.approx(0.20)  # max - min
    assert point.cost_usd == pytest.approx(3 * 0.012)  # total spend across reps


async def test_modal_direction_across_reps() -> None:
    client = _mock_client(
        side_effect=[
            _response(0.55, "YES"),
            _response(0.45, "NO"),
            _response(0.60, "YES"),
        ]
    )
    run = await run_benchmark(client, [_scenario()], candidate_model="m", reps=3)
    assert run.scenario_runs[0].point_estimate.direction == "YES"


async def test_consecutive_errors_stop_run_early() -> None:
    """LLMConsecutiveErrorsError aborts the whole run, keeping partial results."""
    client = _mock_client(
        side_effect=[_response(), LLMConsecutiveErrorsError("3 consecutive errors")]
    )
    scenarios = [_scenario("s1"), _scenario("s2"), _scenario("s3")]
    run = await run_benchmark(client, scenarios, candidate_model="m", reps=1)

    assert run.stopped_early == "3 consecutive errors"
    assert client.complete.await_count == 2  # s3 never attempted
    assert run.scenario_runs[0].point_estimate is not None
    assert run.scenario_runs[1].reps == []  # the failed scenario recorded nothing


async def test_single_llm_error_records_error_and_continues() -> None:
    client = _mock_client(side_effect=[LLMError("transient"), _response()])
    run = await run_benchmark(
        client, [_scenario("s1"), _scenario("s2")], candidate_model="m", reps=1
    )
    assert run.stopped_early is None
    assert run.scenario_runs[0].point_estimate is None
    assert run.scenario_runs[0].reps[0].error == "transient"
    assert run.scenario_runs[1].point_estimate is not None


async def test_unparseable_candidate_response_records_error() -> None:
    bad = _response()
    bad.content = "prose, not a tool call"
    client = _mock_client(return_value=bad)
    run = await run_benchmark(client, [_scenario()], candidate_model="m", reps=1)
    assert run.scenario_runs[0].point_estimate is None
    assert "parse_failed" in run.scenario_runs[0].reps[0].error


def test_estimate_only_makes_no_llm_calls() -> None:
    """estimate_run is pure — it takes no client and projects from audit rows."""
    scenarios = [_scenario("s1"), _scenario("s2")]
    estimate = estimate_run(scenarios, reps=3)
    assert estimate["scenarios"] == 2
    assert estimate["total_calls"] == 6
    assert estimate["projected_input_tokens"] == 2 * 1200 * 3
    assert estimate["input_tokens_from_audit_rows"] == 2

    # Falls back to a chars/4 heuristic when the audit row lacks token counts.
    no_tokens = _scenario("s3", prompt="x" * 400)
    no_tokens.incumbent.tokens_input = None
    estimate = estimate_run([no_tokens], reps=1)
    assert estimate["projected_input_tokens"] == 100
    assert estimate["input_tokens_from_audit_rows"] == 0


async def test_cost_summary_reports_per_call_deltas() -> None:
    from freqpred.bench.report import cost_summary

    # Incumbent audit rows: cost 0.01 / 1500ms (from _scenario). Candidate:
    # 0.012 / 900ms per call.
    client = _mock_client(return_value=_response())
    scenario = _scenario()
    scenario.incumbent.cost_usd = 0.01
    scenario.incumbent.latency_ms = 1500
    run = await run_benchmark(client, [scenario], candidate_model="m", reps=2)

    cost = cost_summary(run)
    assert cost["incumbent_mean_cost_usd"] == pytest.approx(0.01)
    assert cost["candidate_mean_cost_usd"] == pytest.approx(0.012)
    assert cost["cost_delta_usd_per_call"] == pytest.approx(0.002)
    assert cost["latency_delta_ms_per_call"] == pytest.approx(900 - 1500)
    assert cost["candidate_calls_priced"] == 2
    assert cost["candidate_total_cost_usd"] == pytest.approx(0.024)


async def test_cost_summary_handles_prompt_mode_incumbents_without_costs() -> None:
    from freqpred.bench.report import cost_summary

    client = _mock_client(return_value=_response())
    scenario = _scenario()
    scenario.incumbent.cost_usd = None  # fixture incumbents carry no economics
    scenario.incumbent.latency_ms = None
    run = await run_benchmark(client, [scenario], candidate_model="m", reps=1)

    cost = cost_summary(run)
    assert cost["incumbent_mean_cost_usd"] is None
    assert cost["cost_delta_usd_per_call"] is None
    assert cost["candidate_mean_cost_usd"] == pytest.approx(0.012)


async def test_thinking_none_omits_thinking_parameter() -> None:
    """Pre-4.6 candidates (e.g. Haiku 4.5) reject adaptive thinking — thinking=None
    must pass None through so LLMClient omits the parameter entirely."""
    client = _mock_client(return_value=_response())
    await run_benchmark(client, [_scenario()], candidate_model="claude-haiku-4-5-20251001", reps=1, thinking=None)
    assert client.complete.await_args_list[0][1]["thinking"] is None


def test_estimate_projects_dollar_cost_from_pricing_table() -> None:
    """The estimate reports typical/max USD using the production pricing table."""
    scenario = _scenario()
    scenario.incumbent.tokens_output = 200

    # claude-sonnet-4-6 pricing: $3/M input, $15/M output.
    est = estimate_run(
        [scenario], reps=1, candidate_model="claude-sonnet-4-6", thinking={"type": "adaptive", "display": "summarized"}
    )
    # typical: 1200 in + 200*3 thinking-scaled out = 1200*3e-6 + 600*15e-6
    assert est["projected_cost_typical_usd"] == pytest.approx(0.0036 + 0.009, abs=1e-4)
    # max: 1200 in + 4096 out
    assert est["projected_cost_max_usd"] == pytest.approx(0.0036 + 4096 * 15e-6, abs=1e-4)

    # thinking off → no output multiplier
    est_off = estimate_run(
        [scenario], reps=1, candidate_model="claude-sonnet-4-6", thinking=None
    )
    assert est_off["projected_cost_typical_usd"] == pytest.approx(0.0036 + 0.003, abs=1e-4)

    # reps scale everything linearly
    est_reps = estimate_run(
        [scenario], reps=3, candidate_model="claude-sonnet-4-6", thinking={"type": "adaptive", "display": "summarized"}
    )
    assert est_reps["projected_cost_typical_usd"] == pytest.approx(3 * (0.0036 + 0.009), abs=1e-3)


# ---------------------------------------------------------------------------
# Eval cache — identical experiments are never re-billed
# ---------------------------------------------------------------------------


async def test_eval_cache_hit_skips_api_and_miss_populates(tmp_path) -> None:
    from freqpred.bench.eval_cache import EvalCache

    cache = EvalCache(tmp_path)
    client = _mock_client(return_value=_response())
    scenario = _scenario("s1", prompt="cached prompt")

    first = await run_benchmark(
        client, [scenario], candidate_model="candidate-model", reps=1, cache=cache
    )
    assert client.complete.await_count == 1
    assert first.scenario_runs[0].reps[0].cached is False

    # Second run: same model + thinking + system + prompt → zero API calls.
    client2 = _mock_client(return_value=_response())
    second = await run_benchmark(
        client2, [scenario], candidate_model="candidate-model", reps=1, cache=cache
    )
    assert client2.complete.await_count == 0
    rep = second.scenario_runs[0].reps[0]
    assert rep.cached is True
    assert rep.output.posterior == pytest.approx(0.70)

    # Different model → miss; different thinking → miss.
    client3 = _mock_client(return_value=_response())
    await run_benchmark(
        client3, [scenario], candidate_model="other-model", reps=1, cache=cache
    )
    assert client3.complete.await_count == 1
    client4 = _mock_client(return_value=_response())
    await run_benchmark(
        client4, [scenario], candidate_model="candidate-model", reps=1,
        thinking=None, cache=cache,
    )
    assert client4.complete.await_count == 1


async def test_eval_cache_reps_extend_not_duplicate(tmp_path) -> None:
    """reps=3 after a cached reps=1 run makes exactly 2 fresh calls and keeps
    per-scenario spread meaningful (cached rep + fresh reps, not 3 copies)."""
    from freqpred.bench.eval_cache import EvalCache

    cache = EvalCache(tmp_path)
    scenario = _scenario("s1")
    await run_benchmark(
        _mock_client(return_value=_response(0.70)), [scenario],
        candidate_model="candidate-model", reps=1, cache=cache,
    )

    client = _mock_client(return_value=_response(0.80))
    run = await run_benchmark(
        client, [scenario], candidate_model="candidate-model", reps=3, cache=cache
    )
    assert client.complete.await_count == 2
    flags = [r.cached for r in run.scenario_runs[0].reps]
    assert flags == [True, False, False]
    # All three now cached for the next run.
    client2 = _mock_client(return_value=_response())
    await run_benchmark(
        client2, [scenario], candidate_model="candidate-model", reps=3, cache=cache
    )
    assert client2.complete.await_count == 0


def test_estimate_counts_cache_hits_as_free(tmp_path) -> None:
    from freqpred.bench.eval_cache import EvalCache, cache_key

    cache = EvalCache(tmp_path)
    cached_scenario = _scenario("s1", prompt="already evaluated")
    fresh_scenario = _scenario("s2", prompt="never evaluated")
    thinking = None
    key = cache_key(
        "candidate-model", thinking, SYSTEM_PROMPT, cached_scenario.prompt
    )
    cache.append(key, cached_scenario.incumbent, None)

    estimate = estimate_run(
        [cached_scenario, fresh_scenario], 1,
        candidate_model="candidate-model", thinking=thinking, cache=cache,
    )
    assert estimate["cached_calls"] == 1
    assert estimate["fresh_calls"] == 1
    no_cache = estimate_run(
        [cached_scenario, fresh_scenario], 1,
        candidate_model="candidate-model", thinking=thinking,
    )
    assert estimate["projected_cost_typical_usd"] < no_cache["projected_cost_typical_usd"]


async def test_budget_exceeded_stops_gracefully_with_partial_results() -> None:
    """Hitting the daily spend cap mid-run must stop with partial results and
    stopped_early set — never propagate as a crash (the artifact and summary
    still get written from what completed)."""
    from freqpred.llm.audit import LLMBudgetExceededError

    client = _mock_client(
        side_effect=[_response(), LLMBudgetExceededError("cap reached")]
    )
    scenarios = [_scenario("s1"), _scenario("s2"), _scenario("s3")]
    run = await run_benchmark(client, scenarios, candidate_model="candidate-model", reps=1)
    assert run.stopped_early == "cap reached"
    assert len(run.scenario_runs[0].reps) == 1  # s1 completed
    assert run.scenario_runs[0].reps[0].output is not None
