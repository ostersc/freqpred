#!/usr/bin/env python
"""Benchmark TypeSafe Jev against the incumbent signal model and the free baselines.

Three arms over the frozen prompt bank:

  A            the rendered prompt verbatim as state — same information the
               incumbent saw, so the only variable is calibrated output
  B            structured state with fuller evidence AND the market price
  B-noprice    the same structured state with the price withheld — the only arm
               that can produce an estimate independent of the price, and
               therefore the only one that can produce tradeable edge

Every arm is scored against the incumbent and against poisson_30d, poisson_365d,
constant and market_mid (``freqpred.bench.baselines``). The bar is market_mid:
measured over 333 markets the incumbent beats Poisson but loses to the price,
so clearing Poisson alone proves nothing tradeable.

Scenarios use each fixture's FROZEN rendered prompt, not a re-render through the
current template: the bank is signal-v11 and HEAD is signal-v12, and re-rendering
would change the prompt axis as well as the model axis.

Concurrency carries the guards T103 specified and never shipped: a pre-flight
scope assertion before any spend, a per-run spend cap, an abort when actual
spend runs past the projection, a heartbeat carrying markets touched and spend,
and deterministic output ordering independent of completion order.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from unittest.mock import MagicMock

from freqpred.bench.baselines import parse_factbase, score_all_baselines
from freqpred.bench.scenarios import ModelOutput, Scenario, sample_markets, sample_per_market
from freqpred.bench.scoring import brier, cluster_bootstrap_mean_ci, sign_test_p
from freqpred.db import make_engine, make_session_factory
from freqpred.llm.client import LLMClient
from freqpred.llm.typesafe import JEV_INPUT_PER_MTOK, JEV_MODEL, TypeSafeTransport, noul, score

ARMS = ("A", "B", "B_noprice")
SCHEMA_VERSION = 1

# Measured on real bank scenarios 2026-09-18, used for the pre-flight projection.
_EST_TOKENS = {"A": 3_200, "B": 7_000, "B_noprice": 7_000}
_OVERRUN_ABORT_RATIO = 1.25


@dataclass
class ArmResult:
    arm: str
    scenario_id: str
    market_id: str
    outcome: float
    probability: float
    confidence: float | None
    tokens_input: int
    cost_usd: float
    latency_ms: int


def load_scenarios(bank: Path, outcomes: dict[str, float]) -> list[Scenario]:
    """Build scenarios from the bank's frozen prompts. Deterministic order."""
    from freqpred.signal.llm import parse_signal_response

    scenarios: list[Scenario] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(bank.glob("*.json")):
        fx = json.loads(path.read_text())
        inp, exp = fx["inputs"], fx["expectations"]
        market = inp["market"]
        if market["id"] not in outcomes:
            continue
        key = (market["id"], inp["now"])
        if key in seen:
            continue
        seen.add(key)
        parsed = parse_signal_response(inp["llm_response"])
        if parsed is None:
            continue
        scenarios.append(
            Scenario(
                id=fx["name"], source="fixture", market_id=market["id"],
                market_question=market["question"],
                close_time=datetime.fromisoformat(market["close_time"]),
                outcome=outcomes[market["id"]],
                prompt=exp["rendered_prompt"],
                incumbent=ModelOutput(
                    model="claude-sonnet-4-6", prior=parsed["prior"],
                    posterior=parsed["posterior"], confidence=parsed["confidence"],
                    direction=parsed["direction"],
                    updates_count=len(parsed["updates_applied"]), reasoning="",
                ),
                yes_bid=market["yes_bid"], yes_ask=market["yes_ask"],
                mid_price=market["mid_price"],
                signal_time=datetime.fromisoformat(inp["now"]),
            )
        )
        scenarios[-1].notes.append(str(path))
    return scenarios


def build_state(scenario: Scenario, arm: str, doc_chars: int, docs_per_scenario: int):
    """The state for one arm. Arm A is the frozen prompt; B arms are structured."""
    if arm == "A":
        return scenario.prompt

    fixture = json.loads(Path(scenario.notes[-1]).read_text())
    inp = fixture["inputs"]
    fb = parse_factbase(scenario.prompt)
    state: dict = {
        "market_question": scenario.market_question,
        "close_time": scenario.close_time.isoformat(),
        "as_of": scenario.signal_time.isoformat() if scenario.signal_time else None,
        "days_remaining": fb.days_to_close if fb else None,
        "base_rates": {
            "occurrences_since_market_opened": fb.in_market_count if fb else None,
            "occurrences_last_7_days": fb.count_7d if fb else None,
            "occurrences_last_30_days": fb.count_30d if fb else None,
            "occurrences_last_365_days": fb.count_365d if fb else None,
            "poisson_baseline_from_30d_rate": fb.poisson_30d if fb else None,
            "poisson_baseline_from_365d_rate": fb.poisson_365d if fb else None,
        },
        "series_history": inp.get("series_history"),
        "evidence": [
            {
                "title": d.get("title"),
                "source": d.get("source_name"),
                "published": d.get("published_at"),
                "text": (d.get("full_body") or d.get("body") or d.get("summary") or "")[:doc_chars],
            }
            for d in (inp.get("documents") or [])[:docs_per_scenario]
        ],
    }
    if arm == "B":
        state["market_price"] = {
            "yes_bid": scenario.yes_bid, "yes_ask": scenario.yes_ask,
            "mid": scenario.mid_price,
            "note": "the market's own implied probability that this resolves YES",
        }
    return state


QUESTIONS = {
    "resolves_yes": noul(
        "Will this market resolve YES — will the described event occur at or before the close time?",
        when_true="The event occurs at or before the close time",
        when_false="The event does not occur before the close time",
    ),
    "evidence_strength": score(
        "How strong and decisive is the evidence in the state for judging this question?",
        ["No usable evidence", "Weak or ambiguous evidence", "Clear, decisive evidence"],
    ),
}


async def run_arm(client, scenario, arm, sem, state_kwargs) -> ArmResult | None:
    async with sem:
        try:
            response, _ = await client.system_one(
                state=build_state(scenario, arm, **state_kwargs),
                questions=QUESTIONS, query_type="model_eval",
                market_id=scenario.market_id, prompt_version=f"jev-arm-{arm}",
            )
            return ArmResult(
                arm=arm, scenario_id=scenario.id, market_id=scenario.market_id,
                outcome=scenario.outcome, probability=response.noul("resolves_yes"),
                confidence=response.confidence("evidence_strength"),
                tokens_input=response.tokens_input,
                cost_usd=response.tokens_input * JEV_INPUT_PER_MTOK / 1e6,
                latency_ms=response.latency_ms,
            )
        except Exception as exc:  # audited inside system_one; surfaced here
            print(f"  ! {arm} {scenario.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None


def summarise(results: list[ArmResult], scenarios: list[Scenario]) -> dict:
    """Per-arm Brier against outcome, plus paired comparison to the incumbent."""
    by_id = {s.id: s for s in scenarios}
    out: dict = {}
    for arm in ARMS:
        rows = [r for r in results if r.arm == arm]
        if not rows:
            continue
        arm_scen = [by_id[r.scenario_id] for r in rows]
        prob = {r.scenario_id: r.probability for r in rows}
        by_market: dict[str, list[float]] = {}
        for r in rows:
            inc = by_id[r.scenario_id].incumbent.posterior
            by_market.setdefault(r.market_id, []).append(
                brier(r.probability, r.outcome) - brier(inc, r.outcome)
            )
        lo, hi = cluster_bootstrap_mean_ci(by_market)
        means = {m: mean(v) for m, v in by_market.items()}
        wins = sum(1 for d in means.values() if d < 0)
        losses = sum(1 for d in means.values() if d > 0)
        out[arm] = {
            "n_scenarios": len(rows),
            "n_markets": len(by_market),
            "jev_mean_brier": mean(brier(r.probability, r.outcome) for r in rows),
            "incumbent_mean_brier": mean(
                brier(by_id[r.scenario_id].incumbent.posterior, r.outcome) for r in rows
            ),
            "brier_delta_vs_incumbent_ci95": [lo, hi],
            "jev_beats_incumbent": hi < 0,
            "sign_test_p": sign_test_p(wins, losses),
            "baselines": score_all_baselines(arm_scen, lambda s, _p=prob: _p[s.id]),
            "mean_tokens_input": mean(r.tokens_input for r in rows),
            "mean_latency_ms": mean(r.latency_ms for r in rows),
            "total_cost_usd": sum(r.cost_usd for r in rows),
        }
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="benchmarks/prompt_bank", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="max markets (default: all)")
    ap.add_argument("--per-market", default="spread:3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--doc-chars", type=int, default=4000)
    ap.add_argument("--docs-per-scenario", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-spend-usd", type=float, default=3.0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    arms = tuple(a for a in args.arms.split(",") if a)
    engine = make_engine(os.environ["DATABASE_URL"])
    session_factory = make_session_factory(engine)

    from sqlalchemy import text
    async with session_factory() as session:
        rows = (await session.execute(text(
            "SELECT id, result FROM markets WHERE status='finalized' AND result IN ('yes','no')"
        ))).all()
    outcomes = {r[0]: (1.0 if r[1] == "yes" else 0.0) for r in rows}

    scenarios = load_scenarios(args.bank, outcomes)
    scenarios, n_markets = sample_markets(scenarios, args.limit, seed=args.seed)
    scenarios = sample_per_market(scenarios, args.per_market)
    scenarios.sort(key=lambda s: s.id)

    # --- Pre-flight scope assertion, BEFORE any spend. A count beyond the
    # requested scope is the bug, not a detail to notice 14 minutes in.
    touched = {s.market_id for s in scenarios}
    if args.limit is not None and len(touched) > args.limit:
        raise SystemExit(
            f"SCOPE: {len(touched)} markets in the work set exceeds --limit {args.limit}"
        )

    projection = sum(
        len(scenarios) * _EST_TOKENS[a] * JEV_INPUT_PER_MTOK / 1e6 for a in arms
    )
    print(f"markets      : {len(touched)} (sampled from {n_markets})")
    print(f"scenarios    : {len(scenarios)}  x {len(arms)} arms = {len(scenarios)*len(arms)} calls")
    print(f"projection   : ${projection:.4f}   cap ${args.max_spend_usd:.2f}")
    if projection > args.max_spend_usd:
        raise SystemExit(f"SPEND: projection ${projection:.4f} exceeds --max-spend-usd")
    if args.estimate_only:
        await engine.dispose()
        return 0

    transport = TypeSafeTransport(os.environ["TYPESAFE_API_KEY"])
    sem = asyncio.Semaphore(args.concurrency)
    state_kwargs = {"doc_chars": args.doc_chars, "docs_per_scenario": args.docs_per_scenario}
    started = time.monotonic()
    results: list[ArmResult] = []

    async with transport:
        client = LLMClient(MagicMock(), session_factory, default_strategy="jev_bench",
                           typesafe_transport=transport)
        tasks = [
            asyncio.create_task(run_arm(client, s, a, sem, state_kwargs))
            for a in arms for s in scenarios
        ]
        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            done += 1
            if result is not None:
                results.append(result)
            spend = sum(r.cost_usd for r in results)
            if spend > projection * _OVERRUN_ABORT_RATIO and spend > 0.05:
                for t in tasks:
                    t.cancel()
                raise SystemExit(
                    f"SPEND OVERRUN: ${spend:.4f} is past {_OVERRUN_ABORT_RATIO}x the "
                    f"${projection:.4f} projection — aborting"
                )
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} calls  "
                      f"{len({r.market_id for r in results})} markets  "
                      f"${spend:.4f}  {time.monotonic()-started:.0f}s", flush=True)

    results.sort(key=lambda r: (r.arm, r.scenario_id))   # deterministic artifact
    summary = summarise(results, scenarios)
    artifact = {
        "schema_version": SCHEMA_VERSION, "model": JEV_MODEL,
        "run_started_at": datetime.now(UTC).isoformat(),
        "config": vars(args) | {"bank": str(args.bank), "json_out": str(args.json_out)},
        "n_markets": len(touched), "n_scenarios": len(scenarios),
        "summary": summary,
        "results": [vars(r) for r in results],
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(artifact, indent=2, default=str))
        print(f"\nartifact -> {args.json_out}")

    print(f"\n{'arm':<11}{'Brier':>8}{'incumb':>9}{'mkt_mid':>9}{'pois30':>9}"
          f"{'vs incumbent CI95':>26}{'tok':>7}{'ms':>6}{'cost':>9}")
    for arm in arms:
        s = summary.get(arm)
        if not s:
            continue
        bl = s["baselines"]
        lo, hi = s["brier_delta_vs_incumbent_ci95"]
        print(f"{arm:<11}{s['jev_mean_brier']:>8.4f}{s['incumbent_mean_brier']:>9.4f}"
              f"{bl['market_mid']['baseline_mean_brier']:>9.4f}"
              f"{bl['poisson_30d']['baseline_mean_brier']:>9.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>26}"
              f"{s['mean_tokens_input']:>7.0f}{s['mean_latency_ms']:>6.0f}"
              f"{s['total_cost_usd']:>9.4f}")
    print("\nvs market_mid (the bar): negative delta = Jev better")
    for arm in arms:
        s = summary.get(arm)
        if not s:
            continue
        m = s["baselines"]["market_mid"]
        lo, hi = m["brier_delta_ci95"]
        verdict = ("JEV BETTER" if hi < 0 else "market better" if lo > 0 else "tie (CI spans zero)")
        print(f"  {arm:<11} delta={m['brier_delta_mean']:+.4f} CI[{lo:+.4f},{hi:+.4f}]  {verdict}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
