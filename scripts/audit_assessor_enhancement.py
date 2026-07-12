"""Live-Opus audit: does adding point-in-time edge-calibration data to the
assessor's prompt improve its trust_score's correlation with actual outcomes?

Ad hoc /goal deliverable, not a tracked T-task. Reuses real production code
paths rather than reimplementing them:
  - freqpred.metrics.assessment._load_source_breakdown /
    _load_similar_market_summary / _build_prompt_payload / _SYSTEM_PROMPT /
    _ASSESSMENT_TOOL — the actual assessor prompt construction.
  - freqpred.replay.recorder._reconstruct_prices — recovers (yes_bid, yes_ask)
    at signal time from the signal's own stored fields, so the live call sees
    the market as it looked when the signal fired, not its (leaky) settled
    state today.
  - freqpred.llm.client.LLMClient with query_type="model_eval" — the same
    non-production query_type scripts/benchmark_signals.py uses for candidate
    calls, so this never writes to signal_assessments (no pollution of the
    real sizing-decision audit trail) but still logs real spend to
    llm_queries under a clearly-labeled bucket.

One live call per sampled signal per arm. Two arms, selectable via --arms:
  current    — the live production package: production _SYSTEM_PROMPT +
               production section loaders + production _PROMPT_VERSION, on
               the production judgment model (config.anthropic.
               judgment_model). Needs no maintenance; it is whatever shipped.
  challenger — the proposed package: defined per experiment via the
               CHALLENGER block near the top of this script (version string,
               system prompt, payload builder, optional judgment-model
               override for model-swap experiments). Undefined by default;
               the run fails loudly if requested without being defined.
Both arms share the same PIT base payload and the same PIT edge-band
calibration, so the paired contrast isolates exactly the challenger's change.
market.days_to_close is corrected to be relative to the signal's own
created_at in both arms (the real function uses wall-clock "now", which for
a since-resolved market leaks "this already closed" — a point-in-time bug
for this audit, not a feature under test).

POINT-IN-TIME REVISION (v2): the first run of this audit used the production
_load_similar_market_summary/_load_source_breakdown verbatim against current
DB state. That leaked outcomes: the sampled markets have since finalized, so
the "exact question" Brier history included the assessed market's OWN
resolution — control correlation jumped to 0.85 vs 0.08 for the real
production assessments, which is answer-leakage, not skill, and it saturated
both conditions. v2 replaces both loaders with point-in-time copies that
(a) exclude the assessed market entirely and (b) only admit markets closed /
positions exited / snapshots computed before the signal's created_at. In
production the equivalent leak cannot occur (the assessed market is never
finalized when the assessor runs), so the PIT copies approximate the
production information set, not a new behavior.

Remaining accepted scope limit: phrase_data is passed as None for both
conditions (production sometimes passes FactBase phrase-frequency context) —
its count_365d field would leak future occurrences for a point-in-time
signal, so it's dropped entirely rather than reconstructed.

CURRENT-VS-CHALLENGER REVISION (v4): earlier revisions pinned each
historical package (assessment-v4/v5/v6) as its own frozen arm to settle the
T94/T95 adoption decisions (results: README → "Auditing the sizing
assessor" reference runs; result CSVs remain in scripts/.audit_output/).
Going forward the harness only ever needs to answer one question — does the
proposed assessor beat the one in production? — so it now runs exactly the
two arms above. --reuse-csv imports per-signal columns for arms NOT being
re-run from a previous run over the same seed/sample, so a budget-
constrained day pays only for the new arm(s).

Note: the shipped edge_band_calibration loader reads the
edge_calibration_scores snapshot table, which has no as_of filter
(production always wants the latest snapshot — fine there, leaky here).
Both arms therefore use this script's PIT _edge_band_calibration
computation; it is byte-identical across the two arms, so it cancels out of
the current-vs-challenger contrast.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import uuid
from datetime import datetime

import anthropic
import pandas as pd
from sqlalchemy import select

import freqpred.ingestion.models  # noqa: F401 — registers mapper
import freqpred.rag.models  # noqa: F401
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.llm.client import LLMClient
from freqpred.markets.models import Market, MarketRow
from freqpred.metrics.assessment import (
    _ASSESSMENT_TOOL,
    _PROMPT_VERSION,
    _SYSTEM_PROMPT,
    _brier_from_rows,
    _build_prompt_payload,
    _clamp_multiplier,
    _load_market_reevaluation_history,
    _parse_assessment_response,
    _question_first_line,
    _trust_score_to_multiplier,
)
from freqpred.metrics.models import SignalAssessmentRow, SourceQualityScoreRow
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.replay.recorder import _reconstruct_prices
from freqpred.signal.models import Signal, SignalRow
from freqpred.strategy.loader import load_strategy

SEED = 42
SAMPLE_N = 30
MAX_PER_MARKET = 2
JUDGMENT_QUERY_TYPE = "model_eval"  # never "signal_assessment" — no DB pollution
EDGE_BINS = [-100, 0, 15, 40, 200]
EDGE_LABELS = ["<0", "0-15", "15-40", ">40"]
ARM_NAMES = ("current", "challenger")

# ---------------------------------------------------------------------------
# CHALLENGER definition — edit this block when screening a proposed change.
#
# The `current` arm always runs the live production package (production
# _SYSTEM_PROMPT + production section loaders) and needs no maintenance.
# The `challenger` arm is whatever you are proposing: set the three hooks
# below before running with challenger in --arms. Leaving them as None makes
# the run fail loudly instead of silently measuring current-vs-current.
#
#   CHALLENGER_VERSION       — prompt_version logged to llm_queries, e.g.
#                              "assessment-v7-audit-pit-challenger"
#   CHALLENGER_SYSTEM_PROMPT — the proposed system prompt, verbatim
#   CHALLENGER_MODEL         — judgment model override for the challenger arm
#                              only, e.g. "claude-opus-4-8"; None = same model
#                              as production (config.anthropic.judgment_model)
#   _challenger_payload      — builds the proposed payload from the same
#                              PIT base payload + PIT edge calibration the
#                              current arm gets, so the contrast isolates
#                              exactly what you changed
#
# One axis per experiment. For a model-only swap: set CHALLENGER_MODEL, set
# CHALLENGER_SYSTEM_PROMPT = _SYSTEM_PROMPT (production, unchanged), and make
# _challenger_payload mirror the current arm exactly:
#
#     base_payload["edge_band_calibration"] = edge_calib
#     base_payload["market_reevaluation_history"] = (
#         await _load_market_reevaluation_history(session, signal)
#     )
#     return base_payload
# ---------------------------------------------------------------------------
# No active experiment. Last run (2026-07-12): judgment-model swap
# opus-4-7 -> opus-4-8 (v6 package unchanged, baseline = adoption run's t95
# columns via --reuse-csv). Result: wash — corr +0.588 vs +0.569, CI
# (-0.175, +0.157), and 4.8 produced zero size_up where 4.7 had one (on a
# winner). Stayed on opus-4-7. CSV: assessor_audit_pit_opus48.csv.
CHALLENGER_VERSION: str | None = None
CHALLENGER_SYSTEM_PROMPT: str | None = None
CHALLENGER_MODEL: str | None = None


async def _challenger_payload(
    session, signal: Signal, base_payload: dict, edge_calib: dict | None
) -> dict:
    """Build the challenger arm's payload. Edit per experiment.

    For a model-only swap, mirror the current arm's payload exactly:
        base_payload["edge_band_calibration"] = edge_calib
        base_payload["market_reevaluation_history"] = (
            await _load_market_reevaluation_history(session, signal)
        )
        return base_payload
    """
    raise NotImplementedError(
        "define the challenger package (CHALLENGER_VERSION, "
        "CHALLENGER_SYSTEM_PROMPT, _challenger_payload) before running "
        "with the challenger arm"
    )


def _edge_band(edge_pct: float) -> str:
    for lo, hi, label in zip(EDGE_BINS[:-1], EDGE_BINS[1:], EDGE_LABELS, strict=True):
        if lo <= edge_pct < hi:
            return label
    return EDGE_LABELS[-1]


async def _pick_sample(session) -> list[uuid.UUID]:
    result = await session.execute(
        select(SignalAssessmentRow.signal_id, SignalRow.market_id, SignalRow.created_at)
        .join(SignalRow, SignalRow.id == SignalAssessmentRow.signal_id)
        .where(SignalAssessmentRow.llm_query_id.is_not(None))
    )
    rows = result.all()
    rng = random.Random(SEED)
    rows = list(rows)
    rng.shuffle(rows)
    per_market: dict[str, int] = {}
    picked: list[uuid.UUID] = []
    for signal_id, market_id, _ in rows:
        if per_market.get(market_id, 0) >= MAX_PER_MARKET:
            continue
        per_market[market_id] = per_market.get(market_id, 0) + 1
        picked.append(signal_id)
        if len(picked) >= SAMPLE_N:
            break
    return picked


def _row_to_signal(row: SignalRow) -> Signal:
    return Signal(
        id=str(row.id),
        market_id=row.market_id,
        estimated_probability=row.estimated_probability,
        confidence=row.confidence,
        edge=row.edge,
        market_mid_at_signal=row.market_mid_at_signal,
        direction=row.direction,
        reasoning=row.reasoning,
        sources=list(row.sources or []),
        retrieval_hash=row.retrieval_hash,
        model_used=row.model_used,
        prompt_version=row.prompt_version,
        trigger=row.trigger,
        created_at=row.created_at,
        raw_context=row.raw_context,
        market_ask_at_signal=row.market_ask_at_signal,
        social_sentiment_summary=row.social_sentiment_summary,
    )


def _row_to_market_at_signal_time(row: MarketRow, signal: Signal) -> Market:
    """Historical Market snapshot: prices reconstructed at signal time, never
    the (settled, leaky) current DB state."""
    yes_bid, yes_ask = _reconstruct_prices(
        signal.direction,
        signal.market_mid_at_signal,
        signal.market_ask_at_signal,
        signal.estimated_probability,
        signal.edge,
    )
    return Market(
        id=row.id,
        platform=row.platform,
        question=row.question,
        category=row.category,
        close_time=row.close_time,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        mid_price=signal.market_mid_at_signal,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=None,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
        open_time=row.open_time,
        status=row.status,
        result=None,  # never pass settlement result into the prompt
        settlement_value=None,
        last_price=row.last_price,
        yes_bid_size=row.yes_bid_size,
        yes_ask_size=row.yes_ask_size,
        series_ticker=row.series_ticker,
        volume_total=row.volume_total,
        settlement_sources=[],
    )


def _fix_days_to_close(payload: dict, market: Market, signal_created_at: datetime) -> dict:
    days = (market.close_time - signal_created_at).total_seconds() / 86400
    payload["market"]["days_to_close"] = round(days, 1)
    return payload


async def _load_source_breakdown_pit(session, signal: Signal, market: Market, as_of: datetime) -> list[dict]:
    """Point-in-time copy of assessment._load_source_breakdown: only source-
    quality snapshots computed before *as_of* are eligible."""
    from sqlalchemy import case, func, or_  # noqa: PLC0415

    counts_result = await session.execute(
        select(DocumentRow.source_name, func.count(DocumentRow.id).label("doc_count"))
        .join(DocumentMarketLinkRow, DocumentMarketLinkRow.document_id == DocumentRow.id)
        .where(DocumentMarketLinkRow.signal_id == uuid.UUID(signal.id))
        .group_by(DocumentRow.source_name)
        .order_by(func.count(DocumentRow.id).desc(), DocumentRow.source_name.asc())
    )
    counts = [(row.source_name, int(row.doc_count)) for row in counts_result.all()]
    total_docs = sum(count for _, count in counts)
    if total_docs == 0:
        return []
    breakdown: list[dict] = []
    for source_name, doc_count in counts:
        snapshot_result = await session.execute(
            select(SourceQualityScoreRow)
            .where(
                SourceQualityScoreRow.source_name == source_name,
                SourceQualityScoreRow.computed_at < as_of,  # PIT
                or_(
                    SourceQualityScoreRow.market_category == market.category,
                    SourceQualityScoreRow.market_category.is_(None),
                ),
            )
            .order_by(
                case((SourceQualityScoreRow.market_category == market.category, 0), else_=1),
                SourceQualityScoreRow.computed_at.desc(),
            )
            .limit(1)
        )
        snapshot = snapshot_result.scalar_one_or_none()
        if snapshot is None:
            continue
        breakdown.append(
            {
                "source_name": source_name,
                "document_share": round(doc_count / total_docs, 6),
                "doc_count": doc_count,
                "weighted_brier": float(snapshot.weighted_brier),
                "overall_brier": float(snapshot.overall_brier),
                "delta_vs_overall": float(snapshot.weighted_brier - snapshot.overall_brier),
                "n_signals": int(snapshot.n_signals),
                "total_doc_uses": int(snapshot.total_doc_uses),
                "market_category_used": snapshot.market_category,
                "lookback_days": int(snapshot.lookback_days),
                "computed_at": snapshot.computed_at.isoformat(),
            }
        )
    return breakdown


async def _load_similar_market_summary_pit(
    session, market: Market, strategy_name: str, as_of: datetime, *, min_signals: int, min_trades: int
) -> dict:
    """Point-in-time copy of assessment._load_similar_market_summary.

    Two filters added everywhere: the assessed market itself is excluded, and
    only markets closed (or positions exited) before *as_of* are admitted —
    matching what the assessor could actually have known at signal time.
    """
    from sqlalchemy import case  # noqa: PLC0415

    from freqpred.markets.models import PositionRow  # noqa: PLC0415

    if not market.series_ticker:
        return {"available": False, "reason": "missing_series_ticker"}

    signal_rows_result = await session.execute(
        select(
            SignalRow.estimated_probability,
            case((MarketRow.result == "yes", 1), else_=0).label("resolution"),
            MarketRow.question,
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(
            MarketRow.series_ticker == market.series_ticker,
            MarketRow.status == "finalized",
            MarketRow.result.is_not(None),
            MarketRow.id != market.id,          # PIT: never the assessed market
            MarketRow.close_time < as_of,       # PIT: resolved before signal
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
    )
    signal_rows = [
        (float(r.estimated_probability), int(r.resolution), r.question)
        for r in signal_rows_result.all()
    ]
    family_pairs = [(p, y) for p, y, _ in signal_rows]
    family_brier = _brier_from_rows(family_pairs)

    overall_result = await session.execute(
        select(
            SignalRow.estimated_probability,
            case((MarketRow.result == "yes", 1), else_=0).label("resolution"),
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(
            MarketRow.status == "finalized",
            MarketRow.result.is_not(None),
            MarketRow.id != market.id,          # PIT
            MarketRow.close_time < as_of,       # PIT
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
    )
    overall_pairs = [(float(r.estimated_probability), int(r.resolution)) for r in overall_result.all()]
    overall_brier = _brier_from_rows(overall_pairs)

    first_line = _question_first_line(market.question)
    exact_pairs = [(p, y) for p, y, q in signal_rows if _question_first_line(q) == first_line]
    exact_brier = _brier_from_rows(exact_pairs)

    strategy_rows_result = await session.execute(
        select(PositionRow.pnl_pct, PositionRow.pnl)
        .join(MarketRow, MarketRow.id == PositionRow.market_id)
        .where(
            PositionRow.strategy_name == strategy_name,
            PositionRow.status == "closed",
            PositionRow.exit_time < as_of,      # PIT
            MarketRow.id != market.id,          # PIT
            MarketRow.series_ticker == market.series_ticker,
        )
    )
    strategy_rows = [(float(r.pnl_pct), float(r.pnl)) for r in strategy_rows_result.all()]

    baseline_result = await session.execute(
        select(PositionRow.pnl_pct, PositionRow.pnl).where(
            PositionRow.strategy_name == strategy_name,
            PositionRow.status == "closed",
            PositionRow.exit_time < as_of,      # PIT
            PositionRow.market_id != market.id,  # PIT
        )
    )
    baseline_rows = [(float(r.pnl_pct), float(r.pnl)) for r in baseline_result.all()]

    family_signal_count = len(family_pairs)
    family_trade_count = len(strategy_rows)
    strategy_mean = (
        sum(p for p, _ in strategy_rows) / family_trade_count if family_trade_count > 0 else None
    )
    baseline_mean = (
        sum(p for p, _ in baseline_rows) / len(baseline_rows) if baseline_rows else None
    )
    win_rate = (
        sum(1 for _, pnl in strategy_rows if pnl > 0.0) / family_trade_count
        if family_trade_count > 0
        else None
    )
    available = family_signal_count >= min_signals or family_trade_count >= min_trades
    summary: dict = {
        "available": available,
        "match_rule": "series_ticker",
        "series_ticker": market.series_ticker,
        "family_match": {
            "resolved_signals": family_signal_count,
            "family_signal_brier": family_brier,
            "family_signal_delta_vs_overall": (
                family_brier - overall_brier
                if family_brier is not None and overall_brier is not None
                else None
            ),
        },
        "exact_question_subset": {
            "resolved_signals": len(exact_pairs),
            "signal_brier": exact_brier,
            "signal_delta_vs_overall": (
                exact_brier - overall_brier
                if exact_brier is not None and overall_brier is not None
                else None
            ),
            "small_sample": 0 < len(exact_pairs) < min_signals,
        },
        "strategy_trade_history": {
            "strategy_name": strategy_name,
            "closed_trades": family_trade_count,
            "win_rate": win_rate,
            "mean_pnl_pct": strategy_mean,
            "delta_vs_strategy_overall_mean_pnl_pct": (
                strategy_mean - baseline_mean
                if strategy_mean is not None and baseline_mean is not None
                else None
            ),
        },
        "minimums": {"min_signals": min_signals, "min_trades": min_trades},
    }
    if not available:
        summary["reason"] = "insufficient_matched_history"
    return summary


def _edge_band_calibration(calib_df: pd.DataFrame, signal: Signal, as_of: datetime) -> dict:
    edge_pct = signal.edge * 100.0
    band = _edge_band(edge_pct)
    prior = calib_df[calib_df["close_time"] < as_of]
    same_band = prior[prior["edge_band"] == band]
    same_band_dir = same_band[same_band["direction"] == signal.direction]

    def _summarize(df: pd.DataFrame) -> dict:
        if len(df) == 0:
            return {"n_signals": 0, "n_markets": 0}
        return {
            "n_signals": int(len(df)),
            "n_markets": int(df["market_id"].nunique()),
            "hit_rate": round(float(df["hit"].mean()), 3),
            "avg_market_implied_p": round(float(df["market_p_side"].mean()), 3),
            "avg_model_implied_p": round(float(df["p_side"].mean()), 3),
        }

    return {
        "description": (
            "Empirical calibration for signals whose edge fell in the same "
            "band as this one, computed ONLY from markets that had already "
            "closed before this signal fired (no lookahead). hit_rate is "
            "what actually happened; avg_model_implied_p is what the model "
            "claimed at signal time — a large gap between them for this "
            "band is a documented failure mode, not a hypothetical one."
        ),
        "this_signal_edge_band": band,
        "all_directions": _summarize(same_band),
        "same_direction_only": _summarize(same_band_dir),
    }


async def main(
    arms: set[str], reuse_csv: str | None, out_path: str, dry_run: bool = False
) -> None:
    unknown = arms - set(ARM_NAMES)
    if unknown:
        raise SystemExit(f"ERROR: unknown arm(s) {sorted(unknown)}; valid: {ARM_NAMES}")
    if "challenger" in arms and (CHALLENGER_VERSION is None or CHALLENGER_SYSTEM_PROMPT is None):
        raise SystemExit(
            "ERROR: challenger arm requested but no challenger package is defined — "
            "set CHALLENGER_VERSION, CHALLENGER_SYSTEM_PROMPT, and _challenger_payload "
            "(see the CHALLENGER block at the top of this script)."
        )
    config = load_config()
    if not config.database.url:
        raise SystemExit("ERROR: DATABASE_URL not configured.")
    if not config.anthropic.api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not configured.")

    judgment_model = config.anthropic.judgment_model
    strategy = load_strategy("PoliticsEdgeStrategy")

    print("Loading point-in-time calibration reference set from DB...")
    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    async with session_factory() as session:
        result = await session.execute(
            select(
                SignalRow.market_id,
                SignalRow.direction,
                SignalRow.edge,
                SignalRow.estimated_probability,
                SignalRow.market_ask_at_signal,
                SignalRow.created_at,
                MarketRow.close_time,
                MarketRow.result,
                MarketRow.status,
            )
            .join(MarketRow, MarketRow.id == SignalRow.market_id)
            .where(
                SignalRow.trigger == "scheduled",
                SignalRow.confidence >= 0.60,
            )
        )
        rows = result.all()

    calib_records = []
    for r in rows:
        resolved = r.status == "finalized" and r.result in ("yes", "no")
        hit = None
        p_side = r.estimated_probability if r.direction == "YES" else 1.0 - r.estimated_probability
        market_p_side = r.market_ask_at_signal
        if resolved:
            hit = (r.direction == "YES" and r.result == "yes") or (r.direction == "NO" and r.result == "no")
        calib_records.append(
            {
                "market_id": r.market_id,
                "direction": r.direction,
                "edge_band": _edge_band(r.edge * 100.0),
                "close_time": r.close_time,
                "resolved": resolved,
                "hit": hit,
                "p_side": p_side,
                "market_p_side": market_p_side,
            }
        )
    calib_df = pd.DataFrame(calib_records)
    calib_df = calib_df[calib_df["resolved"]].copy()
    print(f"  point-in-time calibration pool: {len(calib_df)} resolved signals")

    reused: dict[str, dict] | None = None
    if reuse_csv:
        # The prior run's signals ARE the sample — pairing requires identical
        # signals, and re-drawing with the seed is not reproducible once the
        # assessment population has grown since that run.
        prev_df = pd.read_csv(reuse_csv)
        sample_ids = [uuid.UUID(str(s)) for s in prev_df["signal_id"]]
        reused = {
            str(r["signal_id"]): {k: v for k, v in r.items() if k != "signal_id"}
            for _, r in prev_df.iterrows()
        }
        print(f"Sample = the {len(sample_ids)} signals from {reuse_csv} (paired reuse)")
    else:
        async with session_factory() as session:
            sample_ids = await _pick_sample(session)
        print(f"Sampled {len(sample_ids)} signals (seed={SEED}, max {MAX_PER_MARKET}/market)")
    print(f"Live arms this run: {sorted(arms)}")

    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        session_factory,
        default_strategy="model_eval",
        prompt_version="assessor-audit-pit-v3",
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
        max_consecutive_errors=config.risk.max_consecutive_llm_errors,
    )

    out_rows = []
    async with session_factory() as session:
        for i, signal_id in enumerate(sample_ids, start=1):
            sig_row = (
                await session.execute(select(SignalRow).where(SignalRow.id == signal_id))
            ).scalar_one()
            market_row = (
                await session.execute(select(MarketRow).where(MarketRow.id == sig_row.market_id))
            ).scalar_one()
            existing = (
                await session.execute(
                    select(SignalAssessmentRow).where(SignalAssessmentRow.signal_id == signal_id)
                )
            ).scalar_one()

            signal = _row_to_signal(sig_row)
            market = _row_to_market_at_signal_time(market_row, signal)

            outcome_result = market_row.result
            hit = (signal.direction == "YES" and outcome_result == "yes") or (
                signal.direction == "NO" and outcome_result == "no"
            )

            source_breakdown = await _load_source_breakdown_pit(
                session, signal, market, signal.created_at
            )
            similar_market_summary = await _load_similar_market_summary_pit(
                session,
                market,
                strategy.config.name,
                signal.created_at,
                min_signals=strategy.config.similar_market_min_signals,
                min_trades=strategy.config.similar_market_min_trades,
            )
            base_payload = _build_prompt_payload(
                signal,
                market,
                strategy.config.name,
                source_breakdown,
                similar_market_summary,
                scale_min=strategy.config.assessment_scale_min,
                scale_max=strategy.config.assessment_scale_max,
                phrase_data=None,
            )
            base_payload = _fix_days_to_close(base_payload, market, signal.created_at)

            async def _call(
                payload: dict,
                label: str,
                *,
                system: str,
                version: str,
                model: str | None = None,
                _market=market,
                _signal=signal,
                _i=i,
            ) -> dict | None:
                prompt = json.dumps(payload, indent=2, sort_keys=True)
                if dry_run:  # payload built + serialized; skip the paid call
                    return None
                try:
                    resp = await llm_client.complete(
                        prompt=prompt,
                        model=model or judgment_model,
                        query_type=JUDGMENT_QUERY_TYPE,
                        system=system,
                        market_id=_market.id,
                        signal_id=_signal.id,
                        strategy=f"audit_{label}",
                        prompt_version=version,
                        max_tokens=768,
                        json_tool=_ASSESSMENT_TOOL,
                    )
                    parsed = _parse_assessment_response(resp.content)
                    mult = _clamp_multiplier(
                        _trust_score_to_multiplier(
                            parsed["trust_score"],
                            scale_min=strategy.config.assessment_scale_min,
                            scale_max=strategy.config.assessment_scale_max,
                        ),
                        scale_min=strategy.config.assessment_scale_min,
                        scale_max=strategy.config.assessment_scale_max,
                    )
                    return {
                        "trust_score": parsed["trust_score"],
                        "multiplier": mult,
                        "verdict": parsed["verdict"],
                    }
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{_i}/{len(sample_ids)}] {label} call failed: {exc}")
                    return None

            arm_results: dict[str, dict | None] = {}
            edge_calib = _edge_band_calibration(calib_df, signal, signal.created_at)
            if "current" in arms:
                current_payload = json.loads(json.dumps(base_payload))  # deep copy
                current_payload["edge_band_calibration"] = edge_calib
                current_payload["market_reevaluation_history"] = (
                    await _load_market_reevaluation_history(session, signal)
                )
                arm_results["current"] = await _call(
                    current_payload,
                    "current",
                    system=_SYSTEM_PROMPT,
                    version=f"{_PROMPT_VERSION}-audit-pit-current",
                )
            if "challenger" in arms:
                challenger_payload = await _challenger_payload(
                    session,
                    signal,
                    json.loads(json.dumps(base_payload)),  # deep copy
                    edge_calib,
                )
                arm_results["challenger"] = await _call(
                    challenger_payload,
                    "challenger",
                    system=CHALLENGER_SYSTEM_PROMPT,
                    version=CHALLENGER_VERSION,
                    model=CHALLENGER_MODEL,
                )

            row_out: dict = {
                "signal_id": str(signal_id),
                "market_id": signal.market_id,
                "direction": signal.direction,
                "edge_pct": signal.edge * 100.0,
                "confidence": signal.confidence,
                "hit": hit,
                "existing_trust_score": existing.trust_score,
                "existing_multiplier": existing.size_multiplier,
            }
            if reused is not None:
                fresh_prefixes = tuple(f"{a}_" for a in arms)
                for col, val in reused[str(signal_id)].items():
                    if col in row_out or col.startswith(fresh_prefixes):
                        continue
                    row_out[col] = val
            for arm, res in arm_results.items():
                row_out[f"{arm}_trust_score"] = res["trust_score"] if res else None
                row_out[f"{arm}_multiplier"] = res["multiplier"] if res else None
                row_out[f"{arm}_verdict"] = res["verdict"] if res else None
            out_rows.append(row_out)

            live_bits = " ".join(
                f"{a}={r['trust_score']:.2f}" if r else f"{a}=FAIL"
                for a, r in arm_results.items()
            )
            print(
                f"  [{i}/{len(sample_ids)}] {signal.market_id[:30]:30s} "
                f"hit={hit} existing={existing.trust_score:.2f} {live_bits}"
            )

    out_df = pd.DataFrame(out_rows)
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paired live-Opus assessor audit (PIT). See module docstring."
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARM_NAMES),
        help="Comma-separated arms to run live this invocation (default: all).",
    )
    parser.add_argument(
        "--reuse-csv",
        default=None,
        help="Prior run's CSV (same seed/sample); its columns for arms not in "
        "--arms are carried into the output instead of re-bought.",
    )
    parser.add_argument(
        "--out",
        default="scripts/.audit_output/assessor_audit_pit.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and serialize every arm payload but make no LLM calls.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            {a.strip() for a in args.arms.split(",") if a.strip()},
            args.reuse_csv,
            args.out,
            dry_run=args.dry_run,
        )
    )
