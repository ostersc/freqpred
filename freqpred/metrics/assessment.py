"""Signal assessment: source quality + similar-market trust."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.metrics.models import SignalAssessment, SignalAssessmentRow, SourceQualityScoreRow
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.signal.models import SignalRow

if TYPE_CHECKING:
    from freqpred.llm.client import LLMClient
    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal
    from freqpred.strategy.base import IPredictionStrategy

log = structlog.get_logger(__name__)

_PROMPT_VERSION = "assessment-v4"
_QUERY_TYPE = "signal_assessment"
_LOOKBACK_DAYS = 90
_MAX_FACTORS = 5
_MAX_WARNINGS = 3
_SYSTEM_PROMPT = """\
You are a risk and sizing judge for a prediction-market trading system.

The trade direction and base edge have already been decided upstream.
Do not re-predict the market outcome.
Do not change the trade direction.
Do not discuss exits or stop losses.
Your task is only to judge how much we should trust the base position sizing based on:
1. evidence-source quality history, and
2. historical performance in similar markets.

Guidelines:
- Be conservative when sample sizes are small, mixed, or noisy.
- Prefer neutral output when the data is weak or conflicting.
- Exact-family history is more important than broad analogies.
- If source quality and similar-market history disagree, explain the conflict and stay closer to neutral unless one side clearly has stronger data.
- Treat this as a sizing-confidence judgment, not a market-prediction task.
- If you populate the warnings field with any concern, your trust_score MUST be ≤ 0.5. Warnings and a trust_score above 0.5 are contradictory — do not do both.
- Unusually large edge (> 0.4) is a warning sign of stale/illiquid pricing, not genuine alpha. Treat it as a reason to stay neutral or go below 0.5 unless you have strong calibration data that supports the edge.

Return valid JSON only with exactly these keys:
{
  "trust_score": number,
  "reasoning": string,
  "key_factors": [string],
  "warnings": [string]
}

Output rules:
- trust_score must be between 0.0 and 1.0; 0.5 means neutral, below 0.5 means lower confidence, above 0.5 means higher confidence
- reasoning must be 1-3 concise sentences
- key_factors must contain 1-5 short strings
- warnings must contain 0-3 short strings
"""


def _question_first_line(question: str) -> str:
    return question.split("\n", 1)[0].strip()


def _neutral_assessment(signal_id: str) -> SignalAssessment:
    return SignalAssessment(
        signal_id=signal_id,
        trust_score=0.5,
        size_multiplier=1.0,
        verdict="neutral",
        reasoning="Assessment skipped because there was not enough historical trust data.",
        key_factors=[],
        warnings=[],
        source_breakdown=[],
        similar_market_summary={},
        llm_query_id=None,
    )


def _brier_from_rows(rows: list[tuple[float, int]]) -> float | None:
    if not rows:
        return None
    return sum((float(p) - float(y)) ** 2 for p, y in rows) / len(rows)


def _trust_score_to_multiplier(
    trust_score: float,
    *,
    scale_min: float,
    scale_max: float,
) -> float:
    score = min(max(trust_score, 0.0), 1.0)
    if score <= 0.5:
        ratio = score / 0.5 if score > 0.0 else 0.0
        return scale_min + ratio * (1.0 - scale_min)
    ratio = (score - 0.5) / 0.5
    return 1.0 + ratio * (scale_max - 1.0)


def _clamp_multiplier(multiplier: float, *, scale_min: float, scale_max: float) -> float:
    lower = min(scale_min, scale_max)
    upper = max(scale_min, scale_max)
    return min(max(multiplier, lower), upper)


def _parse_assessment_response(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")

    trust_score = data.get("trust_score")
    reasoning = data.get("reasoning")
    key_factors = data.get("key_factors", [])
    warnings = data.get("warnings", [])

    if not isinstance(trust_score, (int, float)):
        raise ValueError("trust_score must be numeric")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be a non-empty string")
    if not isinstance(key_factors, list) or not all(isinstance(v, str) for v in key_factors):
        raise ValueError("key_factors must be a list of strings")
    if not isinstance(warnings, list) or not all(isinstance(v, str) for v in warnings):
        raise ValueError("warnings must be a list of strings")

    data["trust_score"] = min(max(float(trust_score), 0.0), 1.0)
    data["reasoning"] = reasoning.strip()
    data["key_factors"] = [v.strip() for v in key_factors if v.strip()][:_MAX_FACTORS]
    data["warnings"] = [v.strip() for v in warnings if v.strip()][:_MAX_WARNINGS]

    # Derive verdict from trust_score — it is not provided by the LLM.
    # Dead band ±0.05 around neutral prevents borderline scores (e.g. 0.52) from
    # showing as a green "Size up" badge when the assessment is essentially neutral.
    ts = data["trust_score"]
    if ts < 0.45:
        data["verdict"] = "size_down"
    elif ts > 0.55:
        data["verdict"] = "size_up"
    else:
        data["verdict"] = "neutral"

    return data


async def _load_source_breakdown(
    session: AsyncSession,
    signal: "Signal",
    market: "Market",
) -> list[dict[str, Any]]:
    counts_result = await session.execute(
        select(
            DocumentRow.source_name,
            func.count(DocumentRow.id).label("doc_count"),
        )
        .join(DocumentMarketLinkRow, DocumentMarketLinkRow.document_id == DocumentRow.id)
        .where(DocumentMarketLinkRow.signal_id == uuid.UUID(signal.id))
        .group_by(DocumentRow.source_name)
        .order_by(func.count(DocumentRow.id).desc(), DocumentRow.source_name.asc())
    )
    counts = [(row.source_name, int(row.doc_count)) for row in counts_result.all()]
    total_docs = sum(count for _, count in counts)
    if total_docs == 0:
        return []

    breakdown: list[dict[str, Any]] = []
    for source_name, doc_count in counts:
        snapshot_result = await session.execute(
            select(SourceQualityScoreRow)
            .where(
                SourceQualityScoreRow.source_name == source_name,
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


async def _load_similar_market_summary(
    session: AsyncSession,
    market: "Market",
    strategy_name: str,
    *,
    min_signals: int,
    min_trades: int,
) -> dict[str, Any]:
    if not market.series_ticker:
        return {
            "available": False,
            "reason": "missing_series_ticker",
        }

    from freqpred.markets.models import MarketRow, PositionRow  # noqa: PLC0415

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
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
    )
    signal_rows = [
        (float(row.estimated_probability), int(row.resolution), row.question)
        for row in signal_rows_result.all()
    ]
    family_pairs = [(p, y) for p, y, _ in signal_rows]
    family_brier = _brier_from_rows(family_pairs)

    overall_signal_rows_result = await session.execute(
        select(
            SignalRow.estimated_probability,
            case((MarketRow.result == "yes", 1), else_=0).label("resolution"),
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(
            MarketRow.status == "finalized",
            MarketRow.result.is_not(None),
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
    )
    overall_signal_pairs = [
        (float(row.estimated_probability), int(row.resolution))
        for row in overall_signal_rows_result.all()
    ]
    overall_brier = _brier_from_rows(overall_signal_pairs)

    first_line = _question_first_line(market.question)
    exact_pairs = [
        (p, y)
        for p, y, question in signal_rows
        if _question_first_line(question) == first_line
    ]
    exact_brier = _brier_from_rows(exact_pairs)

    strategy_rows_result = await session.execute(
        select(PositionRow.pnl_pct, PositionRow.pnl)
        .join(MarketRow, MarketRow.id == PositionRow.market_id)
        .where(
            PositionRow.strategy_name == strategy_name,
            PositionRow.status == "closed",
            MarketRow.series_ticker == market.series_ticker,
        )
    )
    strategy_rows = [(float(row.pnl_pct), float(row.pnl)) for row in strategy_rows_result.all()]

    strategy_baseline_result = await session.execute(
        select(PositionRow.pnl_pct, PositionRow.pnl)
        .where(
            PositionRow.strategy_name == strategy_name,
            PositionRow.status == "closed",
        )
    )
    strategy_baseline_rows = [
        (float(row.pnl_pct), float(row.pnl))
        for row in strategy_baseline_result.all()
    ]

    family_signal_count = len(family_pairs)
    family_trade_count = len(strategy_rows)
    strategy_mean = (
        sum(pnl_pct for pnl_pct, _ in strategy_rows) / family_trade_count
        if family_trade_count > 0
        else None
    )
    strategy_baseline_mean = (
        sum(pnl_pct for pnl_pct, _ in strategy_baseline_rows) / len(strategy_baseline_rows)
        if strategy_baseline_rows
        else None
    )
    win_rate = (
        sum(1 for _, pnl in strategy_rows if pnl > 0.0) / family_trade_count
        if family_trade_count > 0
        else None
    )

    available = family_signal_count >= min_signals or family_trade_count >= min_trades
    summary: dict[str, Any] = {
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
                strategy_mean - strategy_baseline_mean
                if strategy_mean is not None and strategy_baseline_mean is not None
                else None
            ),
        },
        "minimums": {
            "min_signals": min_signals,
            "min_trades": min_trades,
        },
    }
    if not available:
        summary["reason"] = "insufficient_matched_history"
    return summary


def _build_prompt_payload(
    signal: "Signal",
    market: "Market",
    strategy_name: str,
    source_breakdown: list[dict[str, Any]],
    similar_market_summary: dict[str, Any],
    scale_min: float = 0.80,
    scale_max: float = 1.20,
) -> dict[str, Any]:
    weighted_delta = None
    if source_breakdown:
        weighted_delta = sum(
            float(item["document_share"]) * float(item["delta_vs_overall"])
            for item in source_breakdown
        )

    notes: list[str] = []
    exact_subset = similar_market_summary.get("exact_question_subset", {})
    if exact_subset.get("small_sample"):
        notes.append("Exact-question history is a small sample.")
    if source_breakdown and similar_market_summary.get("available"):
        notes.append("Weight family-level history more than the exact-question subset unless the exact sample is substantial.")

    now = datetime.now(tz=timezone.utc)
    days_to_close = (market.close_time - now).total_seconds() / 86400

    # Show the LLM what its trust_score output actually does to position size.
    # Scores are linearly interpolated: 0.0 → scale_min, 0.5 → 1.0x, 1.0 → scale_max.
    score_examples = {
        "0.0 (maximum concern)": f"{scale_min:.2f}x",
        "0.25 (significant concern)": f"{(scale_min + (1.0 - scale_min) * 0.5):.2f}x",
        "0.45 (mild concern, neutral zone)": f"{(scale_min + (1.0 - scale_min) * 0.9):.2f}x",
        "0.50 (neutral — no adjustment)": "1.00x",
        "0.55 (mild confidence, neutral zone)": f"{(1.0 + (scale_max - 1.0) * 0.1):.2f}x",
        "0.75 (solid confidence)": f"{(1.0 + (scale_max - 1.0) * 0.5):.2f}x",
        "1.0 (maximum confidence)": f"{scale_max:.2f}x",
    }

    return {
        "task": "Assess whether the trade should be sized down, left neutral, or sized up relative to the base Kelly target.",
        "sizing_scale": {
            "description": "Your trust_score maps to a position-size multiplier via linear interpolation. Use this table to calibrate your output.",
            "score_to_multiplier_examples": score_examples,
            "neutral_dead_band": "Scores between 0.45 and 0.55 will display as 'neutral' — reserve values outside this range for cases where you have meaningful signal.",
        },
        "market": {
            "market_id": market.id,
            "series_ticker": market.series_ticker,
            "category": market.category,
            "question": market.question,
            "close_time": market.close_time.isoformat(),
            "days_to_close": round(days_to_close, 1),
        },
        "trade_context": {
            "strategy_name": strategy_name,
            "direction": signal.direction,
            "estimated_probability": signal.estimated_probability,
            "market_mid_at_signal": signal.market_mid_at_signal,
            "edge": signal.edge,
            "confidence": signal.confidence,
        },
        "market_liquidity": {
            "yes_bid": market.yes_bid,
            "yes_ask": market.yes_ask,
            "spread": round(market.yes_ask - market.yes_bid, 4),
            "yes_bid_size_dollars": market.yes_bid_size,
            "yes_ask_size_dollars": market.yes_ask_size,
            "volume_24h": market.volume_24h,
            "open_interest": market.open_interest,
            "price_updated_at": market.price_updated_at.isoformat(),
            "note": "Wide spread, very low volume_24h, or small book depth (yes_bid_size/yes_ask_size) suggests thin/illiquid pricing — a large edge in this context is likely artificial.",
        },
        "source_quality_summary": {
            "lookback_days": _LOOKBACK_DAYS,
            "sources_present": source_breakdown,
            "weighted_delta_vs_overall": weighted_delta,
        },
        "similar_market_summary": similar_market_summary,
        "sample_quality": {
            "source_quality_available": bool(source_breakdown),
            "similar_market_available": bool(similar_market_summary.get("available")),
            "notes": notes,
        },
    }


def _row_to_assessment(row: SignalAssessmentRow) -> SignalAssessment:
    return SignalAssessment(
        signal_id=str(row.signal_id),
        trust_score=row.trust_score,
        size_multiplier=row.size_multiplier,
        verdict=row.verdict,
        reasoning=row.reasoning,
        key_factors=list(row.key_factors or []),
        warnings=list(row.warnings or []),
        source_breakdown=list(row.source_breakdown or []),
        similar_market_summary=dict(row.similar_market_summary or {}),
        llm_query_id=row.llm_query_id,
        created_at=row.created_at,
    )


async def _load_prior_assessment_by_hash(
    session: AsyncSession,
    signal: "Signal",
    market: "Market",
) -> SignalAssessmentRow | None:
    """Return the most recent non-neutral assessment for the same retrieval_hash on this market.

    Used to carry forward the original sizing decision to price_moved clone signals,
    which have no DocumentMarketLink rows and would otherwise always return neutral.
    """
    result = await session.execute(
        select(SignalAssessmentRow)
        .join(SignalRow, SignalAssessmentRow.signal_id == SignalRow.id)
        .where(
            SignalRow.market_id == market.id,
            SignalRow.retrieval_hash == signal.retrieval_hash,
            SignalAssessmentRow.signal_id != uuid.UUID(signal.id),
            SignalAssessmentRow.verdict != "neutral",
        )
        .order_by(SignalAssessmentRow.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def assess_signal_context(
    session: AsyncSession,
    signal: "Signal",
    market: "Market",
    strategy: "IPredictionStrategy",
    llm_client: "LLMClient",
    judgment_model: str,
) -> SignalAssessment:
    """Return and persist one signal assessment row for trade sizing."""
    source_breakdown = await _load_source_breakdown(session, signal, market)
    similar_market_summary = await _load_similar_market_summary(
        session,
        market,
        strategy.config.name,
        min_signals=strategy.config.similar_market_min_signals,
        min_trades=strategy.config.similar_market_min_trades,
    )

    neutral = _neutral_assessment(signal.id)
    neutral.source_breakdown = source_breakdown
    neutral.similar_market_summary = similar_market_summary

    if not source_breakdown and not similar_market_summary.get("available"):
        # For price_moved clones, the evidence set is unchanged — carry forward the
        # original signal's sizing decision rather than defaulting to neutral (1.0),
        # which would undo any assessment-driven size reduction from the first entry.
        if signal.trigger == "price_moved":
            prior = await _load_prior_assessment_by_hash(session, signal, market)
            if prior is not None:
                row = SignalAssessmentRow(
                    signal_id=uuid.UUID(signal.id),
                    trust_score=prior.trust_score,
                    size_multiplier=prior.size_multiplier,
                    verdict=prior.verdict,
                    reasoning=prior.reasoning,
                    key_factors=list(prior.key_factors or []),
                    warnings=list(prior.warnings or []),
                    source_breakdown=list(prior.source_breakdown or []),
                    similar_market_summary=dict(prior.similar_market_summary or {}),
                    llm_query_id=None,
                )
                session.add(row)
                await session.flush()
                await session.commit()
                log.debug(
                    "signal_assessment.carried_forward",
                    market_id=market.id,
                    signal_id=signal.id,
                    prior_signal_id=str(prior.signal_id),
                    size_multiplier=prior.size_multiplier,
                    verdict=prior.verdict,
                )
                return _row_to_assessment(row)

        row = SignalAssessmentRow(
            signal_id=uuid.UUID(signal.id),
            trust_score=neutral.trust_score,
            size_multiplier=neutral.size_multiplier,
            verdict=neutral.verdict,
            reasoning=neutral.reasoning,
            key_factors=neutral.key_factors,
            warnings=neutral.warnings,
            source_breakdown=neutral.source_breakdown,
            similar_market_summary=neutral.similar_market_summary,
            llm_query_id=None,
        )
        session.add(row)
        await session.flush()
        await session.commit()
        return _row_to_assessment(row)

    prompt_payload = _build_prompt_payload(
        signal,
        market,
        strategy.config.name,
        source_breakdown,
        similar_market_summary,
        scale_min=strategy.config.assessment_scale_min,
        scale_max=strategy.config.assessment_scale_max,
    )
    prompt = json.dumps(prompt_payload, indent=2, sort_keys=True)
    llm_query_id: int | None = None

    try:
        llm_response = await llm_client.complete(
            prompt=prompt,
            model=judgment_model,
            query_type=_QUERY_TYPE,
            system=_SYSTEM_PROMPT,
            market_id=market.id,
            signal_id=signal.id,
            strategy=strategy.config.name,
            prompt_version=_PROMPT_VERSION,
            max_tokens=512,
        )
        llm_query_id = llm_response.llm_query_id
        parsed = _parse_assessment_response(llm_response.content)
        multiplier = _trust_score_to_multiplier(
            parsed["trust_score"],
            scale_min=strategy.config.assessment_scale_min,
            scale_max=strategy.config.assessment_scale_max,
        )
        multiplier = _clamp_multiplier(
            multiplier,
            scale_min=strategy.config.assessment_scale_min,
            scale_max=strategy.config.assessment_scale_max,
        )
        row = SignalAssessmentRow(
            signal_id=uuid.UUID(signal.id),
            trust_score=parsed["trust_score"],
            size_multiplier=multiplier,
            verdict=parsed["verdict"],
            reasoning=parsed["reasoning"],
            key_factors=parsed["key_factors"],
            warnings=parsed["warnings"],
            source_breakdown=source_breakdown,
            similar_market_summary=similar_market_summary,
            llm_query_id=llm_query_id,
        )
        session.add(row)
        await session.flush()
        await session.commit()
        return _row_to_assessment(row)
    except Exception as exc:
        log.warning(
            "signal_assessment.fallback_neutral",
            market_id=market.id,
            signal_id=signal.id,
            error=str(exc),
        )
        row = SignalAssessmentRow(
            signal_id=uuid.UUID(signal.id),
            trust_score=neutral.trust_score,
            size_multiplier=neutral.size_multiplier,
            verdict=neutral.verdict,
            reasoning="Assessment fell back to neutral after a judgment-model error.",
            key_factors=[],
            warnings=[str(exc)[:200]],
            source_breakdown=source_breakdown,
            similar_market_summary=similar_market_summary,
            llm_query_id=llm_query_id,
        )
        session.add(row)
        await session.flush()
        await session.commit()
        return _row_to_assessment(row)
