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

Two live calls per sampled signal, same judgment model (config.anthropic.
judgment_model, i.e. whatever the assessor already runs in production):
  A) CONTROL   — real _build_prompt_payload, unmodified, except market.
                 days_to_close is corrected to be relative to the signal's
                 own created_at (the real function uses wall-clock "now",
                 which for a since-resolved market leaks "this already
                 closed" — a point-in-time bug for this audit, not a feature
                 under test).
  B) ENHANCED  — same payload plus two new sections: edge_band_calibration
                 (hit rate / market-implied p / model-implied p for this
                 signal's own edge bucket, computed ONLY from markets that
                 had already closed before this signal's created_at) and
                 market_reevaluation_history (prior signal count / direction
                 consistency / edge trend for this same market, from signals
                 created before this one).

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
"""
from __future__ import annotations

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


def _market_reevaluation_history(all_signals_df: pd.DataFrame, signal: Signal) -> dict:
    prior = all_signals_df[
        (all_signals_df["market_id"] == signal.market_id)
        & (all_signals_df["created_at"] < signal.created_at)
    ].sort_values("created_at")
    if len(prior) == 0:
        return {"prior_signal_count": 0, "note": "First signal on this market."}
    directions = prior["direction"].unique().tolist()
    edges = prior["edge"].tolist()
    trend = None
    if len(edges) >= 2:
        trend = "increasing" if edges[-1] > edges[0] else "decreasing_or_flat"
    return {
        "prior_signal_count": int(len(prior)),
        "prior_directions": directions,
        "direction_consistent_with_this_signal": all(d == signal.direction for d in directions),
        "edge_trend_across_prior_signals": trend,
        "note": (
            "A market re-signaled many times with a consistent direction and "
            "growing edge, as the market keeps moving further from the "
            "model's estimate, is the specific pattern behind this "
            "strategy's worst historical misses — the model held its ground "
            "while the market kept disagreeing more, and the market was "
            "usually right."
        ),
    }


async def main() -> None:
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
    all_signals_df = pd.DataFrame(
        [
            {
                "market_id": r.market_id,
                "direction": r.direction,
                "edge": r.edge * 100.0,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    )
    calib_df = pd.DataFrame(calib_records)
    calib_df = calib_df[calib_df["resolved"]].copy()
    print(f"  point-in-time calibration pool: {len(calib_df)} resolved signals")

    async with session_factory() as session:
        sample_ids = await _pick_sample(session)
    print(f"Sampled {len(sample_ids)} signals (seed={SEED}, max {MAX_PER_MARKET}/market)")

    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        session_factory,
        default_strategy="model_eval",
        prompt_version=f"{_PROMPT_VERSION}-audit",
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

            enhanced_payload = json.loads(json.dumps(base_payload))  # deep copy
            enhanced_payload["edge_band_calibration"] = _edge_band_calibration(
                calib_df, signal, signal.created_at
            )
            enhanced_payload["market_reevaluation_history"] = _market_reevaluation_history(
                all_signals_df, signal
            )

            async def _call(
                payload: dict, label: str, *, _market=market, _signal=signal, _i=i
            ) -> dict | None:
                prompt = json.dumps(payload, indent=2, sort_keys=True)
                try:
                    resp = await llm_client.complete(
                        prompt=prompt,
                        model=judgment_model,
                        query_type=JUDGMENT_QUERY_TYPE,
                        system=_SYSTEM_PROMPT,
                        market_id=_market.id,
                        signal_id=_signal.id,
                        strategy=f"audit_{label}",
                        prompt_version=f"{_PROMPT_VERSION}-audit-pit-{label}",
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
                    return {"trust_score": parsed["trust_score"], "multiplier": mult}
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{_i}/{len(sample_ids)}] {label} call failed: {exc}")
                    return None

            control = await _call(base_payload, "control")
            enhanced = await _call(enhanced_payload, "enhanced")

            out_rows.append(
                {
                    "signal_id": str(signal_id),
                    "market_id": signal.market_id,
                    "direction": signal.direction,
                    "edge_pct": signal.edge * 100.0,
                    "confidence": signal.confidence,
                    "hit": hit,
                    "existing_trust_score": existing.trust_score,
                    "existing_multiplier": existing.size_multiplier,
                    "control_trust_score": control["trust_score"] if control else None,
                    "control_multiplier": control["multiplier"] if control else None,
                    "enhanced_trust_score": enhanced["trust_score"] if enhanced else None,
                    "enhanced_multiplier": enhanced["multiplier"] if enhanced else None,
                }
            )
            print(
                f"  [{i}/{len(sample_ids)}] {signal.market_id[:30]:30s} "
                f"hit={hit} existing={existing.trust_score:.2f} "
                f"control={control['trust_score'] if control else None} "
                f"enhanced={enhanced['trust_score'] if enhanced else None}"
            )

    out_df = pd.DataFrame(out_rows)
    out_path = "scripts/.audit_output/assessor_enhancement_audit_pit.csv"
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
