"""Artifact building and console formatting for benchmark runs (T93)."""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from freqpred.bench.runner import BenchmarkRun
from freqpred.bench.scoring import PairScore, aggregate, score_pair
from freqpred.signal.llm import PROMPT_VERSION

if TYPE_CHECKING:
    from freqpred.strategy.base import IPredictionStrategy

ARTIFACT_SCHEMA_VERSION = 2  # v2: stake-weighted trade metrics (TradeDecision.stake/pnl)


def score_run(
    run: BenchmarkRun,
    *,
    min_edge: float,
    min_confidence: float,
    strategy: IPredictionStrategy,
    bankroll: float,
) -> list[PairScore]:
    """Score every scenario that produced at least one successful candidate rep.

    *strategy* sizes each would-trade via its own ``position_size()``, so the
    stake-weighted metrics reflect the sizing logic that would run live.
    """
    scores: list[PairScore] = []
    for scenario_run in run.scenario_runs:
        point = scenario_run.point_estimate
        if point is None:
            continue
        scores.append(
            score_pair(
                scenario_run.scenario,
                point,
                min_edge=min_edge,
                min_confidence=min_confidence,
                strategy=strategy,
                bankroll=bankroll,
            )
        )
    return scores


def cost_summary(run: BenchmarkRun) -> dict:
    """Mean per-call cost/latency for incumbent vs candidate, where known.

    Incumbent numbers come from the historical audit rows; prompt-mode
    scenarios (fixtures) don't carry them, so counts of priced/timed calls are
    reported alongside. Candidate cost is normalized per call (reps divide the
    per-scenario total).
    """
    inc_costs = [
        r.scenario.incumbent.cost_usd
        for r in run.scenario_runs
        if r.scenario.incumbent.cost_usd is not None
    ]
    inc_latencies = [
        r.scenario.incumbent.latency_ms
        for r in run.scenario_runs
        if r.scenario.incumbent.latency_ms is not None
    ]
    # Cached reps carry their original cost/latency for reference but were
    # free this run — only fresh calls count as spend.
    fresh_outputs = [
        rep.output
        for r in run.scenario_runs
        for rep in r.reps
        if rep.output is not None and not rep.cached
    ]
    cached_calls = sum(
        1 for r in run.scenario_runs for rep in r.reps if rep.cached
    )
    cand_costs = [o.cost_usd for o in fresh_outputs if o.cost_usd is not None]
    cand_latencies = [o.latency_ms for o in fresh_outputs if o.latency_ms is not None]

    def _mean(values: list) -> float | None:
        return sum(values) / len(values) if values else None

    inc_mean_cost = _mean(inc_costs)
    cand_mean_cost = _mean(cand_costs)
    inc_mean_latency = _mean(inc_latencies)
    cand_mean_latency = _mean(cand_latencies)
    return {
        "incumbent_mean_cost_usd": inc_mean_cost,
        "candidate_mean_cost_usd": cand_mean_cost,
        "cost_delta_usd_per_call": (
            cand_mean_cost - inc_mean_cost
            if inc_mean_cost is not None and cand_mean_cost is not None
            else None
        ),
        "incumbent_mean_latency_ms": inc_mean_latency,
        "candidate_mean_latency_ms": cand_mean_latency,
        "latency_delta_ms_per_call": (
            cand_mean_latency - inc_mean_latency
            if inc_mean_latency is not None and cand_mean_latency is not None
            else None
        ),
        "incumbent_calls_priced": len(inc_costs),
        "candidate_calls_priced": len(cand_costs),
        "candidate_cached_calls": cached_calls,
        "candidate_total_cost_usd": sum(cand_costs) if cand_costs else 0.0,
    }


def build_artifact(
    run: BenchmarkRun,
    scores: list[PairScore],
    *,
    mode: str,
    candidate_model: str | None,
    training_cutoff: datetime,
    excluded_contaminated: int,
    config: dict,
    run_started_at: datetime,
) -> dict:
    """Schema-versioned JSON artifact so runs stay comparable over time."""
    summary = aggregate(scores)
    summary["cost"] = cost_summary(run)
    scores_by_id = {s.scenario_id: s for s in scores}

    scenario_records = []
    for scenario_run in run.scenario_runs:
        scenario = scenario_run.scenario
        record: dict = {
            "scenario_id": scenario.id,
            "source": scenario.source,
            "market_id": scenario.market_id,
            "market_question": scenario.market_question,
            "close_time": scenario.close_time.isoformat(),
            "outcome": scenario.outcome,
            "contaminated": scenario.contaminated,
            "notes": scenario.notes,
            "prices_at_signal": {
                "yes_bid": scenario.yes_bid,
                "yes_ask": scenario.yes_ask,
                "mid": scenario.mid_price,
            },
            "incumbent": asdict(scenario.incumbent),
            "reps": [
                {
                    "output": asdict(rep.output) if rep.output else None,
                    "error": rep.error,
                    "thinking_tokens": rep.thinking_tokens,
                    "cached": rep.cached,
                }
                for rep in scenario_run.reps
            ],
            "posterior_spread": scenario_run.posterior_spread,
        }
        point = scenario_run.point_estimate
        record["candidate_point_estimate"] = asdict(point) if point else None
        score = scores_by_id.get(scenario.id)
        if score is not None:
            record["score"] = asdict(score)
        scenario_records.append(record)

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": mode,
        "candidate_model": candidate_model,
        "prompt_version": PROMPT_VERSION,
        "training_cutoff": training_cutoff.isoformat(),
        "excluded_contaminated_scenarios": excluded_contaminated,
        "config": config,
        "run_started_at": run_started_at.isoformat(),
        "run_finished_at": datetime.now(UTC).isoformat(),
        "stopped_early": run.stopped_early,
        "summary": summary,
        "scenarios": scenario_records,
    }


def format_summary(summary: dict, *, candidate_label: str) -> str:
    """Human-readable aggregate block, decision-rule numbers first."""
    if summary.get("n_scenarios", 0) == 0:
        return "No scenarios were scored — nothing to summarize.\n"

    lines = [
        "=" * 78,
        f"  Benchmark summary — candidate: {candidate_label}",
        "=" * 78,
        f"  Scenarios scored     : {summary['n_scenarios']} across "
        f"{summary.get('n_markets', summary['n_scenarios'])} markets"
        + (
            f"  (CONTAMINATED: {summary['n_contaminated']})"
            if summary["n_contaminated"]
            else ""
        ),
        "",
        "  Adopt/reject gate (paired Brier, negative delta = candidate better;",
        "  clustered by market — signals on one market share its outcome):",
        f"    mean Brier — incumbent : {summary['incumbent_mean_brier']:.4f}",
        f"    mean Brier — candidate : {summary['candidate_mean_brier']:.4f}",
        f"    paired delta           : mean={summary['brier_delta_mean']:+.4f}  "
        f"median={summary['brier_delta_median']:+.4f}",
        f"    cluster bootstrap 95% CI: [{summary['brier_delta_ci95'][0]:+.4f}, "
        f"{summary['brier_delta_ci95'][1]:+.4f}]"
        + ("  ← excludes zero" if summary["brier_delta_significant"] else "  (includes zero — not significant)"),
        f"    per-market winners     : candidate={summary['candidate_wins']}  "
        f"incumbent={summary['incumbent_wins']}  ties={summary['ties']}  "
        f"(sign test p={summary['sign_test_p']:.3f})",
        "",
        f"  Log loss  : incumbent={summary['incumbent_mean_log_loss']:.4f}  "
        f"candidate={summary['candidate_mean_log_loss']:.4f}",
        f"  Direction : incumbent {summary['incumbent_direction_accuracy'][0]}/"
        f"{summary['incumbent_direction_accuracy'][1]} correct  |  candidate "
        f"{summary['candidate_direction_accuracy'][0]}/"
        f"{summary['candidate_direction_accuracy'][1]} correct  (SKIP excluded)",
    ]

    regimes = summary.get("regimes") or {}
    if regimes:
        lines += [
            "",
            "  By market regime (frozen mid at signal vs outcome — a hedging model",
            "  loses on favorites and wins on upsets; check both before concluding):",
        ]
        labels = {
            "favorite": "favorites (market called it)",
            "upset": "upsets (market was wrong)",
            "coin_flip": "coin flips (mid 0.4–0.6)",
        }
        for regime, label in labels.items():
            seg = regimes.get(regime)
            if seg is None:
                continue
            lines.append(
                f"    {label:32s}: n={seg['n']:<4d} "
                f"brier inc={seg['incumbent_mean_brier']:.4f} "
                f"cand={seg['candidate_mean_brier']:.4f} "
                f"Δ={seg['brier_delta_mean']:+.4f}  "
                f"(cand wins {seg['candidate_wins']}/{seg['n']})"
            )

    trades = summary["trade_decisions"]
    inc_ev = trades["incumbent_mean_ev_per_trade"]
    cand_ev = trades["candidate_mean_ev_per_trade"]

    def _usd(value: float | None, sign: str = "") -> str:
        return f"${value:{sign},.2f}" if value is not None else "n/a"

    def _ratio(value: float | None) -> str:
        return f"{value:+.4f}" if value is not None else "n/a"

    common = trades["common_trades"]
    lines += [
        "",
        "  Degradation guard (entry-faithful: one entry per market per model, at",
        "  that model's first gate-clearing signal — frozen prices, not a portfolio sim):",
        f"    markets entered : incumbent={trades['incumbent_would_trade']}  "
        f"candidate={trades['candidate_would_trade']}",
        "    mean EV/entry  : incumbent="
        + (f"{inc_ev:+.4f}" if inc_ev is not None else "n/a")
        + "  candidate="
        + (f"{cand_ev:+.4f}" if cand_ev is not None else "n/a"),
        "    Sized by the strategy's own position_size from each side's posterior +",
        "    confidence (zero existing exposure, no assessment):",
        f"      total stake        : incumbent={_usd(trades['incumbent_total_stake'])}  "
        f"candidate={_usd(trades['candidate_total_stake'])}",
        f"      settled P&L        : incumbent={_usd(trades['incumbent_stake_weighted_pnl'], '+')}  "
        f"candidate={_usd(trades['candidate_stake_weighted_pnl'], '+')}",
        f"      P&L per $1 staked  : incumbent={_ratio(trades['incumbent_pnl_per_dollar_staked'])}  "
        f"candidate={_ratio(trades['candidate_pnl_per_dollar_staked'])}",
        f"      common trades      : n={common['n']}  mean stake "
        f"incumbent={_usd(common['incumbent_mean_stake'])}  "
        f"candidate={_usd(common['candidate_mean_stake'])}",
    ]
    if trades["disagreements"]:
        lines.append("    disagreements (exactly one side enters the market):")
        for d in trades["disagreements"]:
            ev = d["trade_ev"]
            ev_str = f"{ev:+.4f}" if ev is not None else "n/a"
            when = (d["entry_signal_time"] or "")[:16]
            lines.append(
                f"      {d['market_id']}: only {d['entered_by']} enters "
                f"(at {when}, edge={d['entry_edge']:+.3f}, "
                f"outcome={'YES' if d['outcome'] == 1.0 else 'NO'}, EV={ev_str})"
            )
    else:
        lines.append("    disagreements : none — both sides enter the same markets")

    cost = summary.get("cost")
    if cost:
        def _fmt(value: float | None, spec: str, unit: str = "") -> str:
            return f"{format(value, spec)}{unit}" if value is not None else "n/a"

        lines += [
            "",
            "  Cost / latency (per call; incumbent from historical audit rows):",
            f"    mean cost    : incumbent={_fmt(cost['incumbent_mean_cost_usd'], '.4f')}"
            f"  candidate={_fmt(cost['candidate_mean_cost_usd'], '.4f')}"
            f"  Δ={_fmt(cost['cost_delta_usd_per_call'], '+.4f', ' USD')}",
            f"    mean latency : incumbent={_fmt(cost['incumbent_mean_latency_ms'], '.0f', 'ms')}"
            f"  candidate={_fmt(cost['candidate_mean_latency_ms'], '.0f', 'ms')}"
            f"  Δ={_fmt(cost['latency_delta_ms_per_call'], '+.0f', 'ms')}",
            f"    candidate total spend this run: ${cost['candidate_total_cost_usd']:.4f}"
            + (
                f"  ({cost['candidate_cached_calls']} call(s) served from eval cache)"
                if cost.get("candidate_cached_calls")
                else ""
            ),
        ]

    lines.append("")
    return "\n".join(lines)
