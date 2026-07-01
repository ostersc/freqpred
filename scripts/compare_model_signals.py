"""Compare a candidate Claude model against the model behind each open market's
latest real signal, replaying the exact stored prompt so both models see
identical input.

For every active market, finds the most recent Signal that actually called
the LLM (skips price-moved repricing clones, which never call the LLM and
carry no llm_query_id), recovers that signal's full prior/posterior/
confidence/reasoning/document-citations from the llm_queries audit row, then
replays the same prompt (Signal.raw_context) to the candidate model and
captures the same fields plus latency/tokens/cost. Every candidate-model call
goes through LLMClient.complete(), so it is logged to llm_queries like any
other LLM call (query_type="model_eval") and counts toward the daily spend
cap.

The candidate-model call requests adaptive thinking with a summarized
display, since omitting `thinking` defaults to different behavior per model
(e.g. Sonnet 4.6 defaults to no thinking; Sonnet 5 defaults to adaptive
thinking with the reasoning hidden) — without requesting it explicitly, the
candidate model's reasoning chain would be invisible, making any prior/
posterior surprises unexplainable.

With --resolved, compares against the most recently RESOLVED markets instead
of active ones, scoring each model's posterior against the actual outcome
(Brier score + a binary "called it right" check, matching the convention in
freqpred.metrics.calibration) rather than against market price. This is a
genuine accuracy check, not just an agreement-with-the-crowd check. No
training-data contamination risk for recent resolutions: both models' training
cutoffs predate any market that closed after the cutoff (the event hadn't
happened yet when the training data was collected), and the candidate-model
call is never given web_search/web_fetch tools — it only ever sees frozen
pretrained knowledge plus the literal text in the replayed prompt. The script
prints each market's close_time so this is easy to eyeball.

Usage:
    uv run python scripts/compare_model_signals.py --model claude-sonnet-5
    uv run python scripts/compare_model_signals.py --model claude-sonnet-5 --limit 5
    uv run python scripts/compare_model_signals.py --model claude-sonnet-5 --market-id KXSOMETHING-26JUN30
    uv run python scripts/compare_model_signals.py --model claude-sonnet-5 --json-out /tmp/cmp.json
    uv run python scripts/compare_model_signals.py --model claude-sonnet-5 --resolved --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import anthropic
from sqlalchemy import select
from sqlalchemy.orm import aliased

import freqpred.ingestion.models  # noqa: F401 — registers mapper
import freqpred.rag.models        # noqa: F401 — registers mapper
import freqpred.llm.models        # noqa: F401 — registers mapper

from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.llm.client import LLMClient, LLMConsecutiveErrorsError, LLMError
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import MarketRow
from freqpred.signal.llm import PROMPT_VERSION, SIGNAL_ANALYSIS_TOOL, SYSTEM_PROMPT, parse_signal_response
from freqpred.signal.models import SignalRow


@dataclass
class ModelResult:
    model: str
    prior: float
    prior_basis: str
    posterior: float
    confidence: float
    direction: str
    reasoning: str
    updates_applied: list[dict] = field(default_factory=list)
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    thinking_text: str | None = None
    thinking_tokens: int | None = None


@dataclass
class ComparisonRow:
    market_id: str
    market_question: str
    market_category: str
    market_mid_price: float
    market_mid_at_signal: float
    signal_id: str
    signal_created_at: datetime
    old: ModelResult
    new: ModelResult | None = None
    error: str | None = None
    # Populated only in --resolved mode: 1.0 if the market resolved YES, else 0.0.
    actual_outcome: float | None = None
    market_close_time: datetime | None = None


async def fetch_markets_with_latest_llm_signal(
    session, market_id: str | None, limit: int, resolved: bool = False
) -> list[tuple[MarketRow, SignalRow]]:
    """Return (market, signal) pairs for the `limit` most recently LLM-analyzed
    markets, using each market's latest LLM-backed signal — not necessarily
    Market.current_signal_id, which can point at a price-moved repricing clone
    that never called the LLM.

    Scans from signals, not markets: only a small fraction of markets are ever
    selected for LLM analysis (most are high-frequency, auto-listed strikes the
    strategy's is_market_interesting() never picks up), so starting from "all
    markets" and checking each one for a signal mostly produces skips.
    DISTINCT ON gets one (the most recent) signal per market in a single
    query; the outer query then sorts the deduped set and applies the limit.

    resolved=False (default): active markets, ordered by signal recency (most
    recently analyzed first) — matches what the live signal generator would
    see right now.
    resolved=True: finalized markets with a known result, ordered by
    close_time (most recently resolved first, since there's no dedicated
    resolution timestamp on Market) — each market's *last pre-resolution*
    signal is the one to score against the actual outcome.
    """
    dedup_stmt = (
        select(SignalRow, MarketRow.close_time.label("market_close_time"))
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(SignalRow.llm_query_id.isnot(None))
    )
    if resolved:
        # result.in_(("yes", "no")) — not just isnot(None) — to exclude "scalar"
        # markets (non-binary outcomes), which would otherwise be silently
        # miscategorized as a "no" resolution by the 1.0/0.0 outcome mapping.
        dedup_stmt = dedup_stmt.where(MarketRow.status == "finalized", MarketRow.result.in_(("yes", "no")))
    else:
        dedup_stmt = dedup_stmt.where(MarketRow.status == "active")
    if market_id:
        dedup_stmt = dedup_stmt.where(SignalRow.market_id == market_id)
    dedup_stmt = dedup_stmt.distinct(SignalRow.market_id).order_by(
        SignalRow.market_id, SignalRow.created_at.desc()
    )

    deduped = dedup_stmt.subquery()
    signal_alias = aliased(SignalRow, deduped)
    order_col = deduped.c.market_close_time if resolved else signal_alias.created_at
    signals = (
        await session.execute(
            select(signal_alias).order_by(order_col.desc()).limit(limit)
        )
    ).scalars().all()

    if not signals:
        return []

    markets_by_id = {
        m.id: m
        for m in (
            await session.execute(
                select(MarketRow).where(MarketRow.id.in_([s.market_id for s in signals]))
            )
        ).scalars().all()
    }
    return [(markets_by_id[s.market_id], s) for s in signals if s.market_id in markets_by_id]


async def recover_old_model_detail(session, signal: SignalRow) -> ModelResult | None:
    """Recover the full old-model record from the llm_queries audit row.

    signal.llm_query_id is guaranteed non-null by the caller's query, but the
    row could in principle be missing or its response unparseable — handled
    here as a None return rather than a crash.
    """
    result = await session.execute(
        select(LLMQueryRow).where(LLMQueryRow.id == signal.llm_query_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None

    parsed = parse_signal_response(row.response)
    if parsed is None:
        return None

    return ModelResult(
        model=row.model_used,
        prior=parsed["prior"],
        prior_basis=parsed["prior_basis"],
        posterior=parsed["posterior"],
        confidence=parsed["confidence"],
        direction=parsed["direction"],
        reasoning=parsed["reasoning"],
        updates_applied=parsed["updates_applied"],
        tokens_input=row.tokens_input,
        tokens_output=row.tokens_output,
        cost_usd=row.cost_usd,
        latency_ms=row.latency_ms,
    )


# Higher than production's 1024 — adaptive thinking output shares the same
# max_tokens budget as the final tool call, so the cap needs headroom beyond
# what a thinking-free, JSON-only response requires.
_NEW_MODEL_MAX_TOKENS = 4096


async def call_new_model(
    llm_client: LLMClient,
    signal: SignalRow,
    market_id: str,
    candidate_model: str,
) -> tuple[ModelResult | None, str | None]:
    """Replay signal.raw_context to candidate_model. Returns (result, error).

    Requests adaptive thinking with a summarized display so the candidate
    model's reasoning chain is visible for diagnosis — without it, a model
    whose thinking defaults to on (e.g. Claude Sonnet 5, unlike Sonnet 4.6)
    would think invisibly, making prior/posterior surprises unexplainable.
    """
    try:
        response = await llm_client.complete(
            signal.raw_context,
            candidate_model,
            query_type="model_eval",
            system=SYSTEM_PROMPT,
            cache_system=True,
            market_id=market_id,
            signal_id=str(signal.id),
            strategy=f"model_eval:{candidate_model}",
            max_tokens=_NEW_MODEL_MAX_TOKENS,
            json_tool=SIGNAL_ANALYSIS_TOOL,
            thinking={"type": "adaptive", "display": "summarized"},
        )
    except LLMError as exc:
        return None, str(exc)

    parsed = parse_signal_response(response.content)
    if parsed is None:
        return None, f"parse_failed: {response.content[:200]!r}"

    return ModelResult(
        model=candidate_model,
        prior=parsed["prior"],
        prior_basis=parsed["prior_basis"],
        posterior=parsed["posterior"],
        confidence=parsed["confidence"],
        direction=parsed["direction"],
        reasoning=parsed["reasoning"],
        updates_applied=parsed["updates_applied"],
        tokens_input=response.tokens_input,
        tokens_output=response.tokens_output,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        thinking_text=response.thinking,
        thinking_tokens=response.thinking_tokens,
    ), None


def compute_deltas(
    old: ModelResult, new: ModelResult, market_mid_at_signal: float, actual_outcome: float | None = None
) -> dict:
    cost_delta = (
        new.cost_usd - old.cost_usd
        if old.cost_usd is not None and new.cost_usd is not None
        else None
    )
    latency_delta = (
        new.latency_ms - old.latency_ms
        if old.latency_ms is not None and new.latency_ms is not None
        else None
    )
    old_tokens = (
        old.tokens_input + old.tokens_output
        if old.tokens_input is not None and old.tokens_output is not None
        else None
    )
    new_tokens = (
        new.tokens_input + new.tokens_output
        if new.tokens_input is not None and new.tokens_output is not None
        else None
    )
    tokens_delta = new_tokens - old_tokens if old_tokens is not None and new_tokens is not None else None

    # Distance of each model's posterior from where the market was actually
    # pricing the contract at signal time — a crude "agreement with the
    # market" check, not a calibration metric (the market can be wrong too).
    old_distance_to_mid = round(abs(old.posterior - market_mid_at_signal), 4)
    new_distance_to_mid = round(abs(new.posterior - market_mid_at_signal), 4)
    if old_distance_to_mid < new_distance_to_mid:
        closer_to_market = "old"
    elif new_distance_to_mid < old_distance_to_mid:
        closer_to_market = "new"
    else:
        closer_to_market = "tie"

    result = {
        "posterior_delta": round(new.posterior - old.posterior, 4),
        "confidence_delta": round(new.confidence - old.confidence, 4),
        "direction_match": new.direction == old.direction,
        "cost_delta_usd": round(cost_delta, 6) if cost_delta is not None else None,
        "latency_delta_ms": latency_delta,
        "tokens_total_delta": tokens_delta,
        "old_distance_to_mid": old_distance_to_mid,
        "new_distance_to_mid": new_distance_to_mid,
        "closer_to_market": closer_to_market,
    }

    if actual_outcome is not None:
        # Brier score component per model — matches freqpred.metrics.calibration's
        # convention: estimated_probability is always P(YES) regardless of
        # direction, so no direction-based flip is needed here.
        old_brier = round((old.posterior - actual_outcome) ** 2, 4)
        new_brier = round((new.posterior - actual_outcome) ** 2, 4)
        if old_brier < new_brier:
            better_calibrated = "old"
        elif new_brier < old_brier:
            better_calibrated = "new"
        else:
            better_calibrated = "tie"

        def _called_it_right(direction: str) -> bool | None:
            if direction == "YES":
                return actual_outcome == 1.0
            if direction == "NO":
                return actual_outcome == 0.0
            return None  # SKIP — no directional call to grade

        result.update({
            "old_brier": old_brier,
            "new_brier": new_brier,
            "better_calibrated": better_calibrated,
            "old_called_it_right": _called_it_right(old.direction),
            "new_called_it_right": _called_it_right(new.direction),
        })

    return result


def _preview(text: str, n: int = 150) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")


def print_market_block(row: ComparisonRow) -> None:
    print(f"\n{'─' * 78}")
    print(f"  {row.market_id}  —  {row.market_question[:80]}")
    print(f"{'─' * 78}")
    print(
        f"  mid now={row.market_mid_price:.3f}  mid at signal={row.market_mid_at_signal:.3f}  "
        f"signal_id={row.signal_id}  signal_created={row.signal_created_at}"
    )
    if row.actual_outcome is not None:
        print(
            f"  RESOLVED: outcome={'YES' if row.actual_outcome == 1.0 else 'NO'}  "
            f"market closed={row.market_close_time}"
        )

    if row.error:
        print(f"  ERROR (new model): {row.error}")
        return

    old, new = row.old, row.new
    assert new is not None

    print(f"\n  {old.model} (current)")
    print(
        f"    prior={old.prior:.3f}  posterior={old.posterior:.3f}  "
        f"confidence={old.confidence:.2f}  direction={old.direction}"
    )
    if old.tokens_input is not None:
        print(
            f"    tokens={old.tokens_input}in/{old.tokens_output}out  "
            f"cost=${old.cost_usd:.4f}  latency={old.latency_ms}ms"
        )
    print(f"    prior_basis: {_preview(old.prior_basis)}")
    print(f"    reasoning  : {_preview(old.reasoning)}")

    print(f"\n  {new.model} (candidate)")
    print(
        f"    prior={new.prior:.3f}  posterior={new.posterior:.3f}  "
        f"confidence={new.confidence:.2f}  direction={new.direction}"
    )
    print(
        f"    tokens={new.tokens_input}in/{new.tokens_output}out  "
        f"cost=${new.cost_usd:.4f}  latency={new.latency_ms}ms"
    )
    print(f"    prior_basis: {_preview(new.prior_basis)}")
    print(f"    reasoning  : {_preview(new.reasoning)}")
    if new.thinking_text:
        print(f"    thinking   : ({new.thinking_tokens} tokens) {_preview(new.thinking_text, n=300)}")
    elif new.thinking_tokens == 0:
        print("    thinking   : model used 0 thinking tokens (adaptive thinking chose not to reason)")
    elif new.thinking_tokens:
        print(f"    thinking   : {new.thinking_tokens} thinking tokens used but no text returned (check display setting)")
    else:
        print("    thinking   : unavailable (thinking_tokens not reported)")

    deltas = compute_deltas(old, new, row.market_mid_at_signal, row.actual_outcome)
    direction_note = "same" if deltas["direction_match"] else "CHANGED"
    print(
        f"\n  Δ posterior={deltas['posterior_delta']:+.4f}  "
        f"Δ confidence={deltas['confidence_delta']:+.4f}  direction={direction_note}"
    )
    if deltas["cost_delta_usd"] is not None:
        print(
            f"  Δ cost=${deltas['cost_delta_usd']:+.4f}  "
            f"Δ latency={deltas['latency_delta_ms']:+d}ms  "
            f"Δ tokens={deltas['tokens_total_delta']:+d}"
        )
    print(
        f"  vs market mid ({row.market_mid_at_signal:.3f}): "
        f"old off by {deltas['old_distance_to_mid']:.4f}  "
        f"new off by {deltas['new_distance_to_mid']:.4f}  "
        f"closer={deltas['closer_to_market']}"
    )
    if row.actual_outcome is not None:
        old_call = deltas["old_called_it_right"]
        new_call = deltas["new_called_it_right"]
        old_call_str = "—" if old_call is None else ("right" if old_call else "WRONG")
        new_call_str = "—" if new_call is None else ("right" if new_call else "WRONG")
        print(
            f"  vs actual outcome: old brier={deltas['old_brier']:.4f} ({old_call_str})  "
            f"new brier={deltas['new_brier']:.4f} ({new_call_str})  "
            f"better_calibrated={deltas['better_calibrated']}"
        )


def print_summary(rows: list[ComparisonRow], candidate_model: str) -> None:
    compared = [r for r in rows if r.error is None]
    errored = [r for r in rows if r.error is not None]

    print(f"\n{'=' * 78}")
    print(f"  Model comparison summary — candidate: {candidate_model}")
    print(f"{'=' * 78}")
    print(f"  Markets evaluated : {len(rows)}")
    print(f"    succeeded       : {len(compared)}")
    print(f"    errored         : {len(errored)}")

    if not compared:
        print()
        return

    deltas = [compute_deltas(r.old, r.new, r.market_mid_at_signal, r.actual_outcome) for r in compared]
    posterior_deltas = [d["posterior_delta"] for d in deltas]
    confidence_deltas = [d["confidence_delta"] for d in deltas]
    direction_matches = sum(1 for d in deltas if d["direction_match"])
    costs_old = [r.old.cost_usd for r in compared if r.old.cost_usd is not None]
    costs_new = [r.new.cost_usd for r in compared]
    latencies_old = [r.old.latency_ms for r in compared if r.old.latency_ms is not None]
    latencies_new = [r.new.latency_ms for r in compared]
    old_distances = [d["old_distance_to_mid"] for d in deltas]
    new_distances = [d["new_distance_to_mid"] for d in deltas]
    closer_counts = {
        label: sum(1 for d in deltas if d["closer_to_market"] == label)
        for label in ("old", "new", "tie")
    }

    print(f"\n  Posterior delta   : mean={mean(posterior_deltas):+.4f}  median={median(posterior_deltas):+.4f}")
    print(f"  Confidence delta  : mean={mean(confidence_deltas):+.4f}  median={median(confidence_deltas):+.4f}")
    print(f"  Direction agreement: {direction_matches}/{len(compared)} ({direction_matches / len(compared):.0%})")
    print(f"\n  Total cost — old  : ${sum(costs_old):.4f}  ({len(costs_old)} priced)")
    print(f"  Total cost — new  : ${sum(costs_new):.4f}")
    print(f"  Mean latency — old: {mean(latencies_old):.0f}ms  ({len(latencies_old)} timed)")
    print(f"  Mean latency — new: {mean(latencies_new):.0f}ms")
    print(f"\n  Mean |posterior - market mid at signal|: old={mean(old_distances):.4f}  new={mean(new_distances):.4f}")
    print(
        f"  Closer to market mid: old={closer_counts['old']}  new={closer_counts['new']}  "
        f"tie={closer_counts['tie']}  (of {len(compared)})"
    )

    if all(r.actual_outcome is not None for r in compared):
        old_briers = [d["old_brier"] for d in deltas]
        new_briers = [d["new_brier"] for d in deltas]
        better_counts = {
            label: sum(1 for d in deltas if d["better_calibrated"] == label)
            for label in ("old", "new", "tie")
        }
        old_graded = [d["old_called_it_right"] for d in deltas if d["old_called_it_right"] is not None]
        new_graded = [d["new_called_it_right"] for d in deltas if d["new_called_it_right"] is not None]
        old_correct = sum(1 for c in old_graded if c)
        new_correct = sum(1 for c in new_graded if c)

        print(f"\n  Mean Brier score (vs actual outcome): old={mean(old_briers):.4f}  new={mean(new_briers):.4f}")
        print(
            f"  Better calibrated: old={better_counts['old']}  new={better_counts['new']}  "
            f"tie={better_counts['tie']}  (of {len(compared)})"
        )
        print(
            f"  Direction correct (excludes SKIP): "
            f"old={old_correct}/{len(old_graded)}  new={new_correct}/{len(new_graded)}"
        )

    if errored:
        print("\n  Errors:")
        for r in errored:
            print(f"    {r.market_id}: {r.error}")
    print()


def to_json_dict(rows: list[ComparisonRow], candidate_model: str, run_started_at: datetime) -> dict:
    out_rows = []
    for r in rows:
        d = {
            "market_id": r.market_id,
            "market_question": r.market_question,
            "market_category": r.market_category,
            "market_mid_price": r.market_mid_price,
            "market_mid_at_signal": r.market_mid_at_signal,
            "signal_id": r.signal_id,
            "signal_created_at": r.signal_created_at.isoformat(),
            "actual_outcome": r.actual_outcome,
            "market_close_time": r.market_close_time.isoformat() if r.market_close_time else None,
            "old": asdict(r.old),
            "new": asdict(r.new) if r.new else None,
            "error": r.error,
        }
        if r.new is not None:
            d["deltas"] = compute_deltas(r.old, r.new, r.market_mid_at_signal, r.actual_outcome)
        out_rows.append(d)

    return {
        "candidate_model": candidate_model,
        "run_started_at": run_started_at.isoformat(),
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "rows": out_rows,
    }


async def main(
    candidate_model: str, market_id: str | None, limit: int, json_out: Path | None, resolved: bool
) -> None:
    run_started_at = datetime.now(timezone.utc)
    config = load_config()

    if not config.database.url:
        raise SystemExit("ERROR: DATABASE_URL not configured.")
    if not config.anthropic.api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not configured.")

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        session_factory,
        default_strategy="model_eval",
        prompt_version=PROMPT_VERSION,
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
        max_consecutive_errors=config.risk.max_consecutive_llm_errors,
    )

    try:
        async with session_factory() as session:
            scope = "resolved" if resolved else "active"
            print(f"Scanning {scope} markets (limit={limit})...")
            pairs = await fetch_markets_with_latest_llm_signal(session, market_id, limit, resolved=resolved)

        print(f"Found {len(pairs)} market(s) with an LLM-backed signal. Evaluating against {candidate_model}...")

        rows: list[ComparisonRow] = []
        for market, signal in pairs:
            async with session_factory() as session:
                old = await recover_old_model_detail(session, signal)
            if old is None:
                print(f"  skip {market.id}: could not recover old-model detail from llm_queries")
                continue

            try:
                new, error = await call_new_model(llm_client, signal, market.id, candidate_model)
            except LLMConsecutiveErrorsError as exc:
                print(f"\nStopping early — {exc}")
                break

            row = ComparisonRow(
                market_id=market.id,
                market_question=market.question,
                market_category=market.category,
                market_mid_price=market.mid_price,
                market_mid_at_signal=signal.market_mid_at_signal,
                signal_id=str(signal.id),
                signal_created_at=signal.created_at,
                old=old,
                new=new,
                error=error,
                actual_outcome=(1.0 if market.result == "yes" else 0.0) if resolved else None,
                market_close_time=market.close_time if resolved else None,
            )
            rows.append(row)
            print_market_block(row)

        print_summary(rows, candidate_model)

        if json_out:
            json_out.write_text(json.dumps(to_json_dict(rows, candidate_model, run_started_at), indent=2, default=str))
            print(f"Wrote full comparison to {json_out}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Candidate Anthropic model ID, e.g. claude-sonnet-5")
    parser.add_argument("--market-id", default=None, help="Evaluate a single market instead of scanning all open markets")
    parser.add_argument("--limit", type=int, default=20, help="Max markets to evaluate (caps cost; default: 20)")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write full structured comparison JSON")
    parser.add_argument(
        "--resolved", action="store_true",
        help="Compare against the most recently RESOLVED markets instead of active ones, scoring each "
             "model's posterior against the actual outcome (calibration check) instead of market price.",
    )
    args = parser.parse_args()

    asyncio.run(main(args.model, args.market_id, args.limit, args.json_out, args.resolved))
