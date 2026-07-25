"""Signal assessment: source quality + similar-market trust."""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.metrics.calibration import MIN_EDGE_CALIBRATION_SAMPLES, edge_band
from freqpred.metrics.models import (
    EdgeCalibrationScoreRow,
    SignalAssessment,
    SignalAssessmentRow,
    SourceQualityScoreRow,
)
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.signal.models import SignalRow

if TYPE_CHECKING:
    from freqpred.ingestion.fetchers.factbase import FactbasePhraseData
    from freqpred.llm.client import LLMClient
    from freqpred.markets.models import Market
    from freqpred.signal.models import Signal
    from freqpred.strategy.base import IPredictionStrategy

log = structlog.get_logger(__name__)

_PROMPT_VERSION = "assessment-v8"
_QUERY_TYPE = "signal_assessment"
_LOOKBACK_DAYS = 90
_MAX_FACTORS = 5
_MAX_WARNINGS = 3
_MIN_EDGE_CALIBRATION_SAMPLES = MIN_EDGE_CALIBRATION_SAMPLES  # single source of truth
_MAX_HISTORY_POINTS = 10
_SYSTEM_PROMPT = """\
You are a risk and sizing judge for a prediction-market trading system.

The trade direction and base edge have already been decided upstream.
Do not re-predict the market outcome.
Do not change the trade direction.
Do not discuss exits or stop losses.
Your task is only to judge how much we should trust the base position sizing.

How to reach a score:
- Start from the reference class. Family-level statistics — family and \
exact-question Brier, and the strategy's win rate and mean PnL for this series — \
describe the population this signal belongs to. Within a single series they are \
frequently IDENTICAL for every signal, so on their own they cannot distinguish \
this signal from its siblings. Use them to set a starting point, and say what \
that starting point is. (Across DIFFERENT series they do discriminate, and there \
they are a primary signal — weight them fully when the comparison is between \
families.)
- Then move off that starting point using evidence specific to THIS signal: the \
trajectory in `market_reevaluation_history`, the source mix and \
weighted_delta_vs_overall in `source_quality_summary`, the exact-question subset \
where its sample is meaningful, days_to_close, and any genuine liquidity data. \
If nothing signal-specific distinguishes it, say so and stay at the starting point.
- A score that could be copied unchanged onto any other signal in the same series \
has not done its job. Two signals in the same family with different trajectories \
should not receive the same score.
- Use the full 0.0-1.0 range. A score above 0.5 is the correct output when the \
signal-specific evidence is favourable, even if the family baseline is weak. Do \
not compress every judgment into a narrow band.

Guidelines:
- Be conservative when sample sizes are small, mixed, or noisy.
- Prefer neutral output when the data is weak or conflicting.
- Exact-family history is more important than broad analogies.
- If source quality and similar-market history disagree, explain the conflict \
and stay closer to neutral unless one side clearly has stronger data.
- Treat this as a sizing-confidence judgment, not a market-prediction task.
- `edge_band_calibration` is the most important block, and \
`profit_edge_vs_price` is the figure that matters: hit_rate minus the average \
price actually paid. Positive means signals in this cell historically BEAT the \
price they traded at; negative means they lost money. Let that govern. Prefer \
`same_direction_only` wherever its sample is adequate, and consult \
`this_direction_all_bands` when the band-level cell is thin — trade direction is \
a strong and persistent discriminator in this data, and it is one of the few \
inputs that genuinely varies between signals.
- `avg_model_implied_p` is the model's own claim about itself. The gap between \
it and hit_rate measures the model's SELF-KNOWLEDGE, which is a different \
question from whether the trade earns money, and the two frequently disagree: a \
cell can be badly overconfident and still profitable, or look perfectly \
calibrated and still lose. Do NOT size down on an overconfidence gap alone. \
Check profit_edge_vs_price first; where the two conflict, profit wins. Mention \
the gap only if it changes your conclusion.
- The raw size of this signal's own edge is not itself evidence of anything. A \
large edge in a cell with a positive profit_edge_vs_price is not suspect.
- When `market_reevaluation_history` is present, read `sampled_history` for \
the actual trajectory: is the edge widening or narrowing, are the model's \
and the market's probabilities converging or diverging, and is the traded \
direction stable (see direction_change_count)? Judge the observed \
trajectory on its own terms — no single pattern is inherently suspect.
- Any field marked 'unavailable_at_signal_time' is UNKNOWN, not zero. Draw no \
inference from it in either direction.

Call the submit_assessment tool with your judgment.

trust_score is the only output that affects position sizing; reasoning, \
key_factors, and warnings exist solely so the decision can be audited later. \
Spend your output budget accordingly — do not restate the payload, do not \
recite figures that are already in the input, and do not explain your method. \
Emit trust_score as the FIRST field in the tool call, before any prose.

- trust_score: 0.0–1.0; 0.5 = neutral, < 0.5 = lower confidence, > 0.5 = higher confidence
- reasoning: at most 2 sentences. Give the reference-class starting point as a \
number, then what moved you off it and by roughly how much. Nothing else.
- key_factors: 1-3 short strings — routine observations and background context \
belong here
- warnings: 0-3 short strings recording concerns a reviewer should see. This is \
an audit annotation and has NO arithmetic relationship to trust_score. Noting a \
concern does not oblige you to lower the score, and a trust_score above 0.5 \
alongside a genuine warning is a legitimate, expected combination when the \
signal-specific evidence supports it. Record what concerned you, then set \
trust_score on the merits.
"""

_ASSESSMENT_TOOL: dict = {
    "name": "submit_assessment",
    "description": "Submit the sizing-confidence assessment for this signal.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "trust_score": {"type": "number"},
            "reasoning": {"type": "string"},
            "key_factors": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["trust_score", "reasoning", "key_factors", "warnings"],
        "additionalProperties": False,
    },
}


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
    signal: Signal,
    market: Market,
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
    market: Market,
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


def _edge_calibration_row_summary(row: EdgeCalibrationScoreRow | None) -> dict[str, Any]:
    if row is None:
        return {"n_signals": 0, "n_markets": 0}
    return {
        "n_signals": row.n_signals,
        "n_markets": row.n_markets,
        "hit_rate": round(row.hit_rate, 3),
        "avg_market_implied_p": round(row.avg_market_implied_p, 3),
        "avg_model_implied_p": round(row.avg_model_implied_p, 3),
        # v8: what the cell actually EARNED over the price paid. Every figure
        # needed for this was already here; what was missing was the subtraction
        # and the instruction to care about it. See _SYSTEM_PROMPT.
        "profit_edge_vs_price": round(row.hit_rate - row.avg_market_implied_p, 4),
    }


def _merge_edge_calibration_rows(
    same_direction: EdgeCalibrationScoreRow | None,
    other_direction: EdgeCalibrationScoreRow | None,
) -> dict[str, Any]:
    rows = [r for r in (same_direction, other_direction) if r is not None]
    return _merge_rows(rows)


def _merge_rows(rows: list[EdgeCalibrationScoreRow]) -> dict[str, Any]:
    """Sample-size-weighted merge of calibration rows."""
    rows = [r for r in rows if r is not None and r.n_signals > 0]
    if not rows:
        return {"n_signals": 0, "n_markets": 0}
    n_signals = sum(r.n_signals for r in rows)
    hit_rate = sum(r.hit_rate * r.n_signals for r in rows) / n_signals
    market_p = sum(r.avg_market_implied_p * r.n_signals for r in rows) / n_signals
    return {
        "n_signals": n_signals,
        "n_markets": sum(r.n_markets for r in rows),
        "hit_rate": round(hit_rate, 3),
        "avg_market_implied_p": round(market_p, 3),
        "avg_model_implied_p": round(
            sum(r.avg_model_implied_p * r.n_signals for r in rows) / n_signals, 3
        ),
        "profit_edge_vs_price": round(hit_rate - market_p, 4),
    }


async def _load_edge_band_calibration(
    session: AsyncSession,
    signal: Signal,
    market: Market,
) -> dict[str, Any] | None:
    """Latest edge_calibration_scores snapshot for this signal's own edge band.

    Prefers a series-scoped row (if it has >= _MIN_EDGE_CALIBRATION_SAMPLES
    signals) over the global row, per band+direction. The whole section is
    omitted when even the global row for this band+direction is too thin —
    the assessor should not be handed a near-zero-sample statistic.
    """
    band = edge_band(signal.edge * 100.0)
    other_direction = "NO" if signal.direction == "YES" else "YES"

    async def _latest(direction: str, series_ticker: str | None) -> EdgeCalibrationScoreRow | None:
        """Latest cell, preferring this signal's own prompt-version cohort.

        Measured performance is strongly version-dependent (KXTRUMPSAY NO-side
        profit edge: -0.240 on signal-v7, -0.067 on v4, +0.120 on v9, +0.133 on
        v11), so an all-versions pool describes a model production no longer
        runs. Falls back to the all-versions rollup (prompt_version IS NULL)
        when the cohort is missing or too thin — which is also the only shape
        that exists for rows written before migration 0059.
        """

        def _q(version_filter):
            return (
                select(EdgeCalibrationScoreRow)
                .where(
                    EdgeCalibrationScoreRow.edge_band == band,
                    EdgeCalibrationScoreRow.direction == direction,
                    EdgeCalibrationScoreRow.series_ticker == series_ticker
                    if series_ticker is not None
                    else EdgeCalibrationScoreRow.series_ticker.is_(None),
                    version_filter,
                )
                .order_by(EdgeCalibrationScoreRow.computed_at.desc())
                .limit(1)
            )

        cohort = (
            await session.execute(
                _q(EdgeCalibrationScoreRow.prompt_version == signal.prompt_version)
            )
        ).scalar_one_or_none()
        if cohort is not None and cohort.n_signals >= _MIN_EDGE_CALIBRATION_SAMPLES:
            return cohort
        return (
            await session.execute(
                _q(EdgeCalibrationScoreRow.prompt_version.is_(None))
            )
        ).scalar_one_or_none()

    global_same_direction = await _latest(signal.direction, None)
    if (
        global_same_direction is None
        or global_same_direction.n_signals < _MIN_EDGE_CALIBRATION_SAMPLES
    ):
        return None

    same_direction_row = global_same_direction
    other_direction_row = await _latest(other_direction, None)
    _series_scoped = False

    if market.series_ticker:
        series_same_direction = await _latest(signal.direction, market.series_ticker)
        if (
            series_same_direction is not None
            and series_same_direction.n_signals >= _MIN_EDGE_CALIBRATION_SAMPLES
        ):
            same_direction_row = series_same_direction
            other_direction_row = await _latest(other_direction, market.series_ticker)
            _series_scoped = True

    # v8: direction across ALL bands. Direction is the single most persistent
    # discriminator in the data (KXTRUMPSAY, 8,410 resolved signals: NO earns
    # +7.3pp over the price paid, YES loses 7.6pp, and YES is negative under
    # every signal prompt version), while the band-level cell is often thin.
    scope = market.series_ticker if _series_scoped else None
    # Must pin the SAME prompt-version scope as the chosen cell. Without this the
    # query returns both the version-scoped rows and the all-versions rollups,
    # which cover overlapping signals — the merge would double-count them.
    cohort_version = same_direction_row.prompt_version
    all_bands_stmt = select(EdgeCalibrationScoreRow).where(
        EdgeCalibrationScoreRow.direction == signal.direction,
        EdgeCalibrationScoreRow.series_ticker == scope
        if scope is not None
        else EdgeCalibrationScoreRow.series_ticker.is_(None),
        EdgeCalibrationScoreRow.prompt_version == cohort_version
        if cohort_version is not None
        else EdgeCalibrationScoreRow.prompt_version.is_(None),
        EdgeCalibrationScoreRow.computed_at == same_direction_row.computed_at,
    )
    all_bands_rows = list((await session.execute(all_bands_stmt)).scalars().all())

    return {
        "description": (
            "Empirical calibration for signals whose edge fell in the same band "
            "as this one. profit_edge_vs_price = hit_rate minus "
            "avg_market_implied_p is what signals in this cell actually EARNED "
            "relative to the price paid: positive means the cell historically "
            "beat its price, negative means it lost money. That is the figure "
            "that determines profitability. avg_model_implied_p is the model's "
            "own claim; the gap between it and hit_rate measures the model's "
            "self-knowledge, which is a DIFFERENT question and frequently points "
            "the other way. this_direction_all_bands aggregates every band for "
            "this signal's traded direction — consult it when the band-level "
            "cell is thin. All figures are drawn from the cohort named in "
            "cohort_prompt_version: this signal's own signal prompt version "
            "where that cohort is large enough, otherwise the all-versions "
            "rollup — measured performance is strongly version-dependent."
        ),
        "this_signal_edge_band": band,
        # Which history cohort these numbers came from: this signal's own signal
        # prompt version, or the all-versions rollup when that cohort was too thin.
        "cohort_prompt_version": cohort_version or "all_versions_fallback",
        "all_directions": _merge_edge_calibration_rows(same_direction_row, other_direction_row),
        "same_direction_only": _edge_calibration_row_summary(same_direction_row),
        "this_direction_all_bands": _merge_rows(all_bands_rows),
    }


_REEVALUATION_HISTORY_DESCRIPTION = (
    "History of this market's prior signals, oldest first. Field convention: "
    "'_event_yes' fields are always in terms of the event resolving YES, "
    "regardless of the side traded; '_traded_side' fields are converted to "
    "whichever side (YES/NO) that point actually traded — compare "
    "model_p_traded_side against market_cost_traded_side to read that "
    "point's edge. model_confidence is the model's self-assessed reliability, "
    "not an event probability. SKIP points are signals where the model "
    "analyzed but declined to trade (side-specific fields null). When history "
    "exceeds the point cap, points are the most recent signal of each even "
    "TIME bucket between the first and latest prior signals (both always "
    "shown verbatim); each bucketed point's bucket_summary carries raw counts "
    "covering ALL signals in its bucket, so direction oscillation cannot hide "
    "between points. Read the sequence for the actual trajectory: is the edge "
    "widening or narrowing, are model and market converging or diverging, is "
    "the direction stable — and compare the trajectory's endpoint against the "
    "current signal in trade_context."
)


def _direction_changes(directions: list[str]) -> int:
    return sum(1 for a, b in zip(directions, directions[1:], strict=False) if a != b)


def _history_point(row: Any, current_created_at: datetime) -> dict[str, Any]:
    """One rendered prior signal, per the issue #95 point schema: every field
    self-describing on model-vs-market, YES-space-vs-traded-side, and
    probability-vs-price. SKIP rows keep YES-space fields and null the
    side-specific ones (the stored SKIP edge is a YES-space audit value,
    never a tradeable edge)."""
    is_skip = row.direction == "SKIP"
    p_yes = float(row.estimated_probability)
    return {
        "hours_before_current_signal": round(
            (current_created_at - row.created_at).total_seconds() / 3600, 1
        ),
        "trigger": row.trigger,
        "traded_direction": row.direction,
        "model_p_event_yes": round(p_yes, 3),
        "model_p_traded_side": (
            None if is_skip else round(p_yes if row.direction == "YES" else 1.0 - p_yes, 3)
        ),
        "model_confidence": round(float(row.confidence), 2),
        "market_p_event_yes": round(float(row.market_mid_at_signal), 3),
        "market_cost_traded_side": (
            None
            if is_skip or row.market_ask_at_signal is None
            else round(float(row.market_ask_at_signal), 3)
        ),
        "edge_pct_traded_side": None if is_skip else round(float(row.edge) * 100.0, 1),
    }


def _bucketed_history_sample(
    rows: list[Any],
    current_created_at: datetime,
    max_points: int = _MAX_HISTORY_POINTS,
) -> list[dict[str, Any]]:
    """Anchored even-TIME bucket sampling per issue #95.

    <= max_points priors: every one rendered verbatim, no summaries.
    Otherwise: first and latest priors are verbatim anchors; the span between
    them splits into (max_points - 2) equal TIME buckets; each non-empty
    bucket renders its most recent signal plus a count-only bucket_summary
    covering ALL of that bucket's signals — never averaged probabilities, so
    a direction flip inside a bucket stays visible instead of washing out.
    """
    if len(rows) <= max_points:
        return [_history_point(r, current_created_at) for r in rows]
    first, last, middle = rows[0], rows[-1], rows[1:-1]
    n_buckets = max_points - 2
    span = (last.created_at - first.created_at).total_seconds()
    if span <= 0:  # degenerate: identical timestamps; render what fits
        keep = list(rows[: max_points - 1]) + [last]
        return [_history_point(r, current_created_at) for r in keep]
    buckets: dict[int, list[Any]] = defaultdict(list)
    for r in middle:
        frac = (r.created_at - first.created_at).total_seconds() / span
        buckets[min(int(frac * n_buckets), n_buckets - 1)].append(r)
    rendered = [_history_point(first, current_created_at)]
    for idx in sorted(buckets):
        brows = buckets[idx]
        point = _history_point(brows[-1], current_created_at)
        non_skip_edges = [float(r.edge) * 100.0 for r in brows if r.direction != "SKIP"]
        point["bucket_summary"] = {
            "signals_in_bucket": len(brows),
            "direction_changes_in_bucket": _direction_changes([r.direction for r in brows]),
            "edge_pct_traded_side_min": (
                round(min(non_skip_edges), 1) if non_skip_edges else None
            ),
            "edge_pct_traded_side_max": (
                round(max(non_skip_edges), 1) if non_skip_edges else None
            ),
        }
        rendered.append(point)
    rendered.append(_history_point(last, current_created_at))
    return rendered


async def _load_market_reevaluation_history(
    session: AsyncSession,
    signal: Signal,
) -> dict[str, Any]:
    """Verbatim trajectory of this market's prior signals (issue #95 shape).

    PIT-safe by construction: only signals created before this one exist to
    query in production (the current signal has already been written, but
    later ones cannot exist yet).
    """
    result = await session.execute(
        select(
            SignalRow.direction,
            SignalRow.edge,
            SignalRow.estimated_probability,
            SignalRow.market_mid_at_signal,
            SignalRow.market_ask_at_signal,
            SignalRow.confidence,
            SignalRow.created_at,
            SignalRow.trigger,
        )
        .where(
            SignalRow.market_id == signal.market_id,
            SignalRow.created_at < signal.created_at,
        )
        .order_by(SignalRow.created_at.asc())
    )
    prior = result.all()
    if not prior:
        return {"prior_signal_count": 0, "note": "First signal on this market."}

    return {
        "description": _REEVALUATION_HISTORY_DESCRIPTION,
        "prior_signal_count": len(prior),
        "direction_change_count": _direction_changes([r.direction for r in prior]),
        "history_span_hours": round(
            (signal.created_at - prior[0].created_at).total_seconds() / 3600, 1
        ),
        "sampled_history": _bucketed_history_sample(list(prior), signal.created_at),
    }


def _build_prompt_payload(
    signal: Signal,
    market: Market,
    strategy_name: str,
    source_breakdown: list[dict[str, Any]],
    similar_market_summary: dict[str, Any],
    scale_min: float = 0.80,
    scale_max: float = 1.20,
    phrase_data: FactbasePhraseData | None = None,
    edge_band_calibration: dict[str, Any] | None = None,
    market_reevaluation_history: dict[str, Any] | None = None,
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
        notes.append(
            "Weight family-level history more than the exact-question subset "
            "unless the exact sample is substantial."
        )

    now = datetime.now(tz=UTC)
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
        "task": (
            "Assess whether the trade should be sized down, left neutral, "
            "or sized up relative to the base Kelly target."
        ),
        "sizing_scale": {
            "description": (
                "Your trust_score maps to a position-size multiplier via linear "
                "interpolation. Use this table to calibrate your output."
            ),
            "score_to_multiplier_examples": score_examples,
            "neutral_dead_band": (
                "Scores between 0.45 and 0.55 will display as 'neutral' — reserve "
                "values outside this range for cases where you have meaningful signal."
            ),
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
            "note": (
                "Wide spread, very low volume_24h, or small book depth "
                "(yes_bid_size/yes_ask_size) suggests thin/illiquid pricing — "
                "a large edge in this context is likely artificial."
            ),
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
        **(
            {
                "phrase_frequency": {
                    "phrase": phrase_data.display_phrase,
                    "in_market_count": phrase_data.in_market_count,
                    "count_7d": phrase_data.count_7d,
                    "count_30d": phrase_data.count_30d,
                    "count_365d": phrase_data.count_365d,
                    "weekly_rate_30d": round(phrase_data.count_30d / 4.3, 1),
                    "top_quotes": phrase_data.top_quotes[:3],
                    "note": (
                        "in_market_count > 0 = confirmed YES occurrence in window. "
                        "Use weekly_rate_30d to judge if signal probability aligns with empirical cadence."
                    ),
                }
            }
            if phrase_data is not None
            else {}
        ),
        **(
            {"edge_band_calibration": edge_band_calibration}
            if edge_band_calibration is not None
            else {}
        ),
        **(
            {"market_reevaluation_history": market_reevaluation_history}
            if market_reevaluation_history is not None
            else {}
        ),
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
    signal: Signal,
    market: Market,
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
    signal: Signal,
    market: Market,
    strategy: IPredictionStrategy,
    llm_client: LLMClient,
    judgment_model: str,
    phrase_data: FactbasePhraseData | None = None,
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

    edge_band_calibration = await _load_edge_band_calibration(session, signal, market)
    market_reevaluation_history = await _load_market_reevaluation_history(session, signal)

    prompt_payload = _build_prompt_payload(
        signal,
        market,
        strategy.config.name,
        source_breakdown,
        similar_market_summary,
        scale_min=strategy.config.assessment_scale_min,
        scale_max=strategy.config.assessment_scale_max,
        phrase_data=phrase_data,
        edge_band_calibration=edge_band_calibration,
        market_reevaluation_history=market_reevaluation_history,
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
            # 1024 matches the audited v8 package. 768 was tuned for v6 and is
            # too tight for v8's profit-edge reasoning: in the first v8 draft the
            # challenger averaged 721 output tokens and truncated mid-sentence on
            # 2/9 calls. At 1024 on the 76-signal frozen set only 1/76 reached the
            # cap. trust_score is also emitted FIRST now, so a truncation costs
            # transparency but never the sizing decision (previously it could
            # drop trust_score entirely and fail open to neutral 1.0x).
            max_tokens=1024,
            json_tool=_ASSESSMENT_TOOL,
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
