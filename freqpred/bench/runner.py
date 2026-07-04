"""Candidate execution for the benchmark harness (T93).

Every candidate call goes through ``LLMClient.complete()`` — logged to
``llm_queries`` with ``query_type="model_eval"`` and counted against the daily
spend cap. Candidates never receive web tools: frozen prompt text plus
pretrained knowledge only, which is what keeps recent-resolution scoring
contamination-free.

Adaptive thinking is requested by default with a summarized display (same
rationale as the original ``compare_model_signals.py``): without it, a model
whose thinking defaults to on would reason invisibly, making posterior
surprises unexplainable. Pre-4.6 models (e.g. Haiku 4.5) reject adaptive
thinking with a 400 — pass ``thinking=None`` for those, which omits the
parameter and runs the model's default behavior (matching how production
signal calls invoke the LLM).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean

import structlog

from freqpred.bench.eval_cache import EvalCache, cache_key
from freqpred.bench.scenarios import ModelOutput, Scenario
from freqpred.llm.client import LLMClient, LLMConsecutiveErrorsError, LLMError
from freqpred.signal.llm import SIGNAL_ANALYSIS_TOOL, SYSTEM_PROMPT, parse_signal_response

log = structlog.get_logger(__name__)

# Matches SignalPipeline._LLM_MAX_TOKENS — thinking output shares the same
# max_tokens budget as the final tool call, so candidates get the same
# headroom production gives the live model.
_CANDIDATE_MAX_TOKENS = 4096

# Default thinking config for candidates (4.6+/5-family models).
DEFAULT_THINKING: dict = {"type": "adaptive", "display": "summarized"}


@dataclass
class CandidateRep:
    """One candidate call for one scenario (a single repetition)."""

    output: ModelOutput | None
    error: str | None = None
    thinking_tokens: int | None = None
    cached: bool = False  # served from the eval cache — no API spend this run


@dataclass
class ScenarioRun:
    """All repetitions for one scenario plus the derived point estimate."""

    scenario: Scenario
    reps: list[CandidateRep] = field(default_factory=list)

    @property
    def successful_outputs(self) -> list[ModelOutput]:
        return [r.output for r in self.reps if r.output is not None]

    @property
    def point_estimate(self) -> ModelOutput | None:
        """Mean posterior/confidence over successful reps; modal direction.

        A conclusion that flips between reps is no conclusion — the spread is
        reported alongside so reviewers can see per-scenario stability.
        """
        outputs = self.successful_outputs
        if not outputs:
            return None
        directions = Counter(o.direction for o in outputs)
        modal_direction, _ = directions.most_common(1)[0]
        costs = [o.cost_usd for o in outputs if o.cost_usd is not None]
        latencies = [o.latency_ms for o in outputs if o.latency_ms is not None]
        return ModelOutput(
            model=outputs[0].model,
            prior=mean(o.prior for o in outputs),
            posterior=mean(o.posterior for o in outputs),
            confidence=mean(o.confidence for o in outputs),
            direction=modal_direction,
            updates_count=round(mean(o.updates_count for o in outputs)),
            reasoning=outputs[0].reasoning,
            tokens_input=outputs[0].tokens_input,
            tokens_output=outputs[0].tokens_output,
            cost_usd=sum(costs) if costs else None,  # total spend across reps
            latency_ms=round(mean(latencies)) if latencies else None,
        )

    @property
    def posterior_spread(self) -> float | None:
        """max - min posterior across successful reps (0.0 when reps == 1)."""
        outputs = self.successful_outputs
        if not outputs:
            return None
        posteriors = [o.posterior for o in outputs]
        return max(posteriors) - min(posteriors)


@dataclass
class BenchmarkRun:
    scenario_runs: list[ScenarioRun] = field(default_factory=list)
    stopped_early: str | None = None  # LLMConsecutiveErrorsError message


async def _call_candidate(
    llm_client: LLMClient, scenario: Scenario, model: str, thinking: dict | None
) -> CandidateRep:
    try:
        response = await llm_client.complete(
            scenario.prompt,
            model,
            query_type="model_eval",
            system=SYSTEM_PROMPT,
            cache_system=True,
            market_id=scenario.market_id,
            signal_id=scenario.id if scenario.source == "db" else None,
            strategy=f"model_eval:{model}",
            max_tokens=_CANDIDATE_MAX_TOKENS,
            json_tool=SIGNAL_ANALYSIS_TOOL,
            thinking=thinking,
        )
    except LLMConsecutiveErrorsError:
        raise
    except LLMError as exc:
        return CandidateRep(output=None, error=str(exc))

    parsed = parse_signal_response(response.content)
    if parsed is None:
        return CandidateRep(
            output=None, error=f"parse_failed: {response.content[:200]!r}"
        )

    return CandidateRep(
        output=ModelOutput(
            model=model,
            prior=parsed["prior"],
            posterior=parsed["posterior"],
            confidence=parsed["confidence"],
            direction=parsed["direction"],
            updates_count=len(parsed["updates_applied"]),
            reasoning=parsed["reasoning"],
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        ),
        thinking_tokens=response.thinking_tokens,
    )


async def run_benchmark(
    llm_client: LLMClient,
    scenarios: list[Scenario],
    *,
    candidate_model: str | None,
    reps: int = 1,
    thinking: dict | None = DEFAULT_THINKING,
    cache: EvalCache | None = None,
) -> BenchmarkRun:
    """Run the candidate over every scenario, *reps* times each.

    ``candidate_model=None`` (prompt mode) evaluates each scenario's incumbent
    model against the re-rendered prompt — isolating the prompt change.
    ``thinking=None`` omits the thinking parameter (required for pre-4.6
    models like Haiku 4.5, which 400 on adaptive thinking).
    ``cache`` reuses prior evaluations of the identical experiment (same
    model + thinking + system prompt + prompt) at zero API cost, and stores
    every fresh successful call for future runs.
    ``LLMConsecutiveErrorsError`` stops the whole run early with partial
    results preserved.
    """
    run = BenchmarkRun()
    for scenario in scenarios:
        model = candidate_model or scenario.incumbent.model
        scenario_run = ScenarioRun(scenario=scenario)
        run.scenario_runs.append(scenario_run)

        key = cache_key(model, thinking, SYSTEM_PROMPT, scenario.prompt) if cache else ""
        cached_reps = cache.load(key) if cache else []
        for rep_index in range(reps):
            if rep_index < len(cached_reps):
                record = cached_reps[rep_index]
                scenario_run.reps.append(
                    CandidateRep(
                        output=ModelOutput(**record["output"]),
                        thinking_tokens=record.get("thinking_tokens"),
                        cached=True,
                    )
                )
                continue
            try:
                rep = await _call_candidate(llm_client, scenario, model, thinking)
            except LLMConsecutiveErrorsError as exc:
                run.stopped_early = str(exc)
                log.error("bench.stopped_early", error=str(exc))
                return run
            scenario_run.reps.append(rep)
            if cache is not None and rep.output is not None:
                cache.append(key, rep.output, rep.thinking_tokens)
    return run


# Observed multiplier on output tokens when adaptive thinking is on: thinking
# shares the output budget, and candidates run ~3x the incumbent's thinking-free
# output (~600 vs ~200 tokens in observed model_eval runs).
_THINKING_OUTPUT_MULTIPLIER = 3
_FALLBACK_OUTPUT_TOKENS = 200


def estimate_run(
    scenarios: list[Scenario],
    reps: int,
    *,
    candidate_model: str | None = None,
    thinking: dict | None = DEFAULT_THINKING,
    cache: EvalCache | None = None,
) -> dict:
    """Cost preview without any LLM calls.

    Input tokens are projected from each scenario's incumbent audit row (the
    candidate sees the same prompt). Output tokens are projected from the
    incumbent's observed output, scaled up when adaptive thinking is on.
    Scenarios already present in the eval cache are counted as free hits.
    Dollar figures use the same pricing table as the production audit trail
    (freqpred.llm.audit.calculate_cost) — rough by construction: they ignore
    prompt-cache discounts, and the ``max`` bound assumes every call exhausts
    max_tokens. Actuals land in ``llm_queries`` as the run executes.
    """
    from freqpred.llm.audit import calculate_cost  # noqa: PLC0415 — avoid module cycle

    known_inputs = [
        s.incumbent.tokens_input for s in scenarios if s.incumbent.tokens_input
    ]
    output_multiplier = _THINKING_OUTPUT_MULTIPLIER if thinking else 1

    typical_cost = 0.0
    max_cost = 0.0
    total_inputs = 0
    fresh_calls = 0
    cached_calls = 0
    for scenario in scenarios:
        model = candidate_model or scenario.incumbent.model
        cached = 0
        if cache is not None:
            key = cache_key(model, thinking, SYSTEM_PROMPT, scenario.prompt)
            cached = min(cache.count(key), reps)
        cached_calls += cached
        fresh = reps - cached
        fresh_calls += fresh
        if fresh == 0:
            continue
        input_tokens = scenario.incumbent.tokens_input or len(scenario.prompt) // 4
        typical_output = (
            scenario.incumbent.tokens_output or _FALLBACK_OUTPUT_TOKENS
        ) * output_multiplier
        total_inputs += input_tokens * fresh
        typical_cost += calculate_cost(model, input_tokens, typical_output) * fresh
        max_cost += calculate_cost(model, input_tokens, _CANDIDATE_MAX_TOKENS) * fresh

    return {
        "scenarios": len(scenarios),
        "reps": reps,
        "total_calls": len(scenarios) * reps,
        "cached_calls": cached_calls,
        "fresh_calls": fresh_calls,
        "projected_input_tokens": total_inputs,
        "input_tokens_from_audit_rows": len(known_inputs),
        "max_output_tokens_per_call": _CANDIDATE_MAX_TOKENS,
        "projected_cost_typical_usd": round(typical_cost, 4),
        "projected_cost_max_usd": round(max_cost, 4),
    }
