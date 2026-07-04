"""Benchmark a model or prompt candidate against resolved-market outcomes.

Supersedes scripts/compare_model_signals.py. Builds a scenario bank of frozen
historical decisions on markets whose outcome is now known, runs the candidate
over each scenario, and reports paired calibration statistics (the
adopt/reject gate) plus trade-decision metrics (the degradation guard).

Modes:
  Model mode (default) — resolved markets from the DB; each scenario replays
    the exact stored prompt (Signal.raw_context) verbatim, so it isolates a
    MODEL change. Requires --candidate-model.
  Prompt mode (--prompt-mode) — T66 replay fixtures re-rendered through the
    CURRENT build_prompt/SYSTEM_PROMPT, so it measures a PROMPT change.
    Without --candidate-model, each scenario runs on its incumbent model
    (prompt change in isolation); with it, both change at once.

Contamination guard: --training-cutoff is REQUIRED. Scenarios whose market
closed on or before the candidate's training-data cutoff are excluded — the
outcome may be in the training data, so scoring against it proves nothing.
--include-contaminated keeps them, loudly flagged.

Interpreting results — the adopt/reject decision rule:
  1. Adopt only if the paired Brier delta is significant (bootstrap 95% CI
     excludes zero, or sign test p < 0.05). A better raw mean on a small
     noisy sample is not evidence.
  2. Guard: would-trade rate, disagreement table, per-trade EV, and the
     stake-weighted P&L (sized by --strategy's own position_size; its config
     also supplies the default gates) must not degrade — a candidate can be
     better calibrated but too timid to trade, or right as often yet
     overconfident exactly when wrong.
  3. Tiebreaker: cost and latency.
  4. Ambiguous → keep the incumbent; it has live calibration history.

Every candidate call goes through LLMClient (logged to llm_queries as
query_type="model_eval", counted against the daily spend cap); candidates
never get web tools. Use --estimate-only to preview call volume first.

Usage:
    uv run python scripts/benchmark_signals.py --candidate-model claude-sonnet-5 \\
        --training-cutoff 2026-03-01 --limit 50 --json-out benchmarks/sonnet5.json
    uv run python scripts/benchmark_signals.py --prompt-mode \\
        --training-cutoff 2026-03-01 --reps 3
    uv run python scripts/benchmark_signals.py --candidate-model claude-sonnet-5 \\
        --training-cutoff 2026-03-01 --estimate-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import anthropic

import freqpred.ingestion.models  # noqa: F401 — registers mapper
import freqpred.llm.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
from freqpred.bench import (
    aggregate,
    build_artifact,
    build_db_scenarios,
    build_fixture_scenarios,
    cost_summary,
    estimate_run,
    filter_contaminated,
    format_summary,
    run_benchmark,
    sample_markets,
    sample_per_market,
    score_run,
)
from freqpred.bench.eval_cache import DEFAULT_CACHE_DIR, EvalCache
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.llm.client import LLMClient
from freqpred.replay.fixtures import DEFAULT_FIXTURE_DIR
from freqpred.signal.llm import PROMPT_VERSION
from freqpred.strategy.loader import load_strategy

_DEFAULT_STRATEGY = "PoliticsEdgeStrategy"  # matches `fixtures record-bank`
_DEFAULT_BANKROLL = 1000.0                  # matches `fixtures record`


def _print_scenario_block(record_index: int, scenario_run, score) -> None:
    scenario = scenario_run.scenario
    flag = "  [CONTAMINATED]" if scenario.contaminated else ""
    print(f"\n{'─' * 78}")
    print(f"  [{record_index}] {scenario.market_id} — {scenario.market_question[:70]}{flag}")
    print(
        f"      closed={scenario.close_time.date()}  "
        f"outcome={'YES' if scenario.outcome == 1.0 else 'NO'}  "
        f"prices at signal: bid={scenario.yes_bid:.2f} ask={scenario.yes_ask:.2f}"
    )
    inc = scenario.incumbent
    print(
        f"      incumbent  ({inc.model}): posterior={inc.posterior:.3f} "
        f"conf={inc.confidence:.2f} dir={inc.direction}"
    )
    point = scenario_run.point_estimate
    if point is None:
        errors = [r.error for r in scenario_run.reps if r.error]
        print(f"      candidate : ALL REPS FAILED ({errors[:1]})")
        return
    spread = scenario_run.posterior_spread or 0.0
    spread_note = f" (spread {spread:.3f} over {len(scenario_run.reps)} reps)" if len(scenario_run.reps) > 1 else ""
    print(
        f"      candidate  ({point.model}): posterior={point.posterior:.3f} "
        f"conf={point.confidence:.2f} dir={point.direction}{spread_note}"
    )
    if score is not None:
        inc_trade = "trade" if score.incumbent_trade.would_trade else "pass"
        cand_trade = "trade" if score.candidate_trade.would_trade else "pass"
        print(
            f"      brier: inc={score.incumbent_brier:.4f} cand={score.candidate_brier:.4f} "
            f"Δ={score.brier_delta:+.4f}  |  decision: inc={inc_trade} cand={cand_trade}"
        )


async def main(args: argparse.Namespace) -> None:
    run_started_at = datetime.now(UTC)
    config = load_config()
    if not config.database.url:
        raise SystemExit("ERROR: DATABASE_URL not configured.")

    # The strategy supplies the sizing logic for stake-weighted metrics and
    # the default would-trade gates, so the guard mirrors what would run live.
    strategy = load_strategy(args.strategy)
    min_edge = args.min_edge if args.min_edge is not None else strategy.config.min_edge
    min_confidence = (
        args.min_confidence
        if args.min_confidence is not None
        else strategy.config.min_confidence
    )

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)
    try:
        async with session_factory() as session:
            if args.prompt_mode:
                mode = "prompt"
                scenarios, skipped = await build_fixture_scenarios(
                    session, args.fixtures
                )
                for reason in skipped:
                    print(f"  skip: {reason}")
            else:
                mode = "model"
                if not args.candidate_model:
                    raise SystemExit(
                        "ERROR: --candidate-model is required in model mode "
                        "(without --prompt-mode there is no other candidate axis)."
                    )
                scenarios = await build_db_scenarios(
                    session, market_id=args.market_id
                )

        if not scenarios:
            raise SystemExit("No usable scenarios found — nothing to benchmark.")

        kept, excluded = filter_contaminated(
            scenarios,
            args.training_cutoff,
            include_contaminated=args.include_contaminated,
        )
        if excluded:
            print(
                f"Excluded {len(excluded)} scenario(s) with close_time <= "
                f"{args.training_cutoff.date()} (candidate training-data cutoff)."
            )
        contaminated_kept = sum(1 for s in kept if s.contaminated)
        if contaminated_kept:
            print(
                f"\n{'!' * 78}\n"
                f"  WARNING: {contaminated_kept} CONTAMINATED scenario(s) kept via "
                "--include-contaminated.\n"
                "  Their outcomes may be inside the candidate's training data — "
                "scores on them\n  are NOT evidence of forecasting skill.\n"
                f"{'!' * 78}\n"
            )
        if not kept:
            raise SystemExit(
                "All scenarios fall inside the training cutoff — nothing "
                "contamination-free to benchmark."
            )

        # Market selection is a seeded random sample (no recency bias), then
        # per-market signal sampling picks the decision points to score.
        kept, n_markets = sample_markets(kept, args.limit, seed=args.seed)
        try:
            kept = sample_per_market(kept, args.per_market)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

        thinking = (
            None
            if args.thinking == "none"
            else {"type": "adaptive", "display": "summarized"}
        )
        cache = None if args.no_cache else EvalCache(DEFAULT_CACHE_DIR)

        candidate_label = args.candidate_model or f"prompt {PROMPT_VERSION} (incumbent models)"
        print(
            f"Mode={mode}  candidate={candidate_label}  "
            f"scenarios={len(kept)} across {n_markets} markets "
            f"(--per-market {args.per_market}, seed={args.seed})  reps={args.reps}\n"
            f"Trade gates: min_edge={min_edge}  min_confidence={min_confidence}  "
            f"sizing: {args.strategy}.position_size @ bankroll=${args.bankroll:,.0f}"
        )

        if args.estimate_only:
            estimate = estimate_run(
                kept,
                args.reps,
                candidate_model=args.candidate_model,
                thinking=thinking,
                cache=cache,
            )
            print("\nEstimate only — no LLM calls made:")
            for key, value in estimate.items():
                print(f"  {key:28s}: {value}")
            print(
                f"  daily spend cap             : "
                f"${config.risk.max_daily_llm_spend_usd:.2f} (enforced per call)"
            )
            print(
                "  (typical = incumbent-derived output sizes; max = every call "
                "exhausting max_tokens; both ignore prompt-cache discounts)"
            )
            return

        if not config.anthropic.api_key:
            raise SystemExit("ERROR: ANTHROPIC_API_KEY not configured.")
        llm_client = LLMClient(
            anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
            session_factory,
            default_strategy="model_eval",
            prompt_version=PROMPT_VERSION,
            daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
            max_consecutive_errors=config.risk.max_consecutive_llm_errors,
        )

        run = await run_benchmark(
            llm_client,
            kept,
            candidate_model=args.candidate_model,
            reps=args.reps,
            thinking=thinking,
            cache=cache,
        )
        if run.stopped_early:
            print(f"\nStopped early — {run.stopped_early}")

        scores = score_run(
            run,
            min_edge=min_edge,
            min_confidence=min_confidence,
            strategy=strategy,
            bankroll=args.bankroll,
        )
        scores_by_id = {s.scenario_id: s for s in scores}
        for i, scenario_run in enumerate(run.scenario_runs, start=1):
            _print_scenario_block(
                i, scenario_run, scores_by_id.get(scenario_run.scenario.id)
            )

        summary = aggregate(scores)
        summary["cost"] = cost_summary(run)
        print("\n" + format_summary(summary, candidate_label=candidate_label))

        if args.json_out:
            artifact = build_artifact(
                run,
                scores,
                mode=mode,
                candidate_model=args.candidate_model,
                training_cutoff=args.training_cutoff,
                excluded_contaminated=len(excluded),
                config={
                    "reps": args.reps,
                    "limit": args.limit,
                    "per_market": args.per_market,
                    "seed": args.seed,
                    "min_edge": min_edge,
                    "min_confidence": min_confidence,
                    "strategy": args.strategy,
                    "bankroll": args.bankroll,
                    "cache": not args.no_cache,
                    "thinking": args.thinking,
                    "include_contaminated": args.include_contaminated,
                    "fixtures": str(args.fixtures) if args.prompt_mode else None,
                },
                run_started_at=run_started_at,
            )
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(artifact, indent=2, default=str))
            print(f"Wrote artifact to {args.json_out}")
    finally:
        await engine.dispose()


def _parse_cutoff(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidate-model", default=None,
        help="Anthropic model ID to evaluate (required in model mode; optional in "
             "prompt mode, where it defaults to each scenario's incumbent model).",
    )
    parser.add_argument(
        "--training-cutoff", type=_parse_cutoff, required=True,
        help="Candidate training-data cutoff (YYYY-MM-DD). Scenarios whose market "
             "closed on or before this date are excluded as contaminated.",
    )
    parser.add_argument(
        "--prompt-mode", action="store_true",
        help="Benchmark the CURRENT prompt template by re-rendering T66 fixture "
             "inputs instead of replaying stored prompts verbatim.",
    )
    parser.add_argument(
        "--fixtures", type=Path, default=DEFAULT_FIXTURE_DIR,
        help=f"Fixture directory for --prompt-mode (default: {DEFAULT_FIXTURE_DIR}).",
    )
    parser.add_argument(
        "--include-contaminated", action="store_true",
        help="Keep scenarios inside the training cutoff, loudly flagged.",
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Max MARKETS, sampled randomly with --seed — not first/last N "
             "(default: 50; scenarios per market set by --per-market).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the random market sample — same seed, same markets (default: 42).",
    )
    parser.add_argument(
        "--per-market", default="spread:3",
        help="Which of each market's signals to score: 'last' (final pre-resolution "
             "signal — favorites-heavy, least tradeable), 'all' (every signal), or "
             "'spread:K' (K signals evenly spaced across the market's timeline, "
             "first and last included). Default: spread:3. Statistics are "
             "market-clustered either way.",
    )
    parser.add_argument(
        "--reps", type=int, default=1,
        help="Candidate calls per scenario; >1 reports per-scenario posterior spread.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip the eval cache (benchmarks/.eval_cache) — every call hits the "
             "API even if the identical experiment (model+thinking+prompts) was "
             "already evaluated.",
    )
    parser.add_argument(
        "--market-id", default=None, help="Restrict model mode to a single market."
    )
    parser.add_argument(
        "--strategy", default=_DEFAULT_STRATEGY,
        help="Strategy name or .py path whose position_size() sizes the "
             "stake-weighted metrics and whose config supplies the default "
             f"would-trade gates (default: {_DEFAULT_STRATEGY}).",
    )
    parser.add_argument(
        "--bankroll", type=float, default=_DEFAULT_BANKROLL,
        help=f"Bankroll passed to position_size (default: {_DEFAULT_BANKROLL:.0f}).",
    )
    parser.add_argument(
        "--min-edge", type=float, default=None,
        help="Edge threshold for the would-trade gate "
             "(default: the strategy's config.min_edge).",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="Confidence threshold for the would-trade gate "
             "(default: the strategy's config.min_confidence).",
    )
    parser.add_argument(
        "--thinking", choices=["adaptive", "none"], default="adaptive",
        help="Thinking config for candidate calls. 'adaptive' (default) suits "
             "4.6+/5-family models; use 'none' for pre-4.6 candidates like "
             "claude-haiku-4-5, which reject adaptive thinking with a 400 "
             "(production signal calls also pass no thinking config).",
    )
    parser.add_argument(
        "--estimate-only", action="store_true",
        help="Print scenario counts and a token projection without calling the API.",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Write the full schema-versioned artifact to this path.",
    )
    asyncio.run(main(parser.parse_args()))
