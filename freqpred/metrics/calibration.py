"""Brier score and calibration curve calculation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.markets.models import MarketRow
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.signal.models import SignalRow


@dataclass
class CalibrationBucket:
    lower: float          # e.g. 0.60
    upper: float          # e.g. 0.70
    count: int
    mean_estimated_prob: float
    actual_resolution_rate: float   # fraction that resolved YES


@dataclass
class CalibrationReport:
    brier_score: float
    market_brier_score: float   # market's own score: using mid price at signal time
    n_samples: int
    lookback_days: int | None = None
    buckets: list[CalibrationBucket] = field(default_factory=list)
    market_buckets: list[CalibrationBucket] = field(default_factory=list)


async def compute_calibration(
    session: AsyncSession,
    mode: str = "paper",  # kept for API compat; signals are mode-agnostic
    lookback_days: int | None = None,
) -> CalibrationReport:
    """Compute Brier score over all signals for finalized markets.

    Each signal is scored independently (per-signal, not per-market), so
    multiple estimates for the same market all contribute — this measures
    prediction quality at each point in time rather than one score per market.

    Args:
        session: Async DB session.
        mode: Unused; kept for API compatibility with callers that pass mode.
        lookback_days: Only include signals created within the last N days.
                       None means all-time.
    """
    resolution_expr = case((MarketRow.result == "yes", 1), else_=0).label("resolution")

    where_clauses: list = [
        MarketRow.status == "finalized",
        MarketRow.result.is_not(None),
        SignalRow.model_used != "demo_harness",
        SignalRow.prompt_version != "demo",
    ]
    if lookback_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        where_clauses.append(SignalRow.created_at >= cutoff)

    stmt = (
        select(
            SignalRow.estimated_probability,
            SignalRow.market_mid_at_signal,
            resolution_expr,
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(and_(*where_clauses))
    )
    result = await session.execute(stmt)
    rows = result.all()

    n = len(rows)
    if n == 0:
        return CalibrationReport(
            brier_score=0.0,
            market_brier_score=0.0,
            n_samples=0,
            lookback_days=lookback_days,
            buckets=_empty_buckets(),
            market_buckets=_empty_buckets(),
        )

    brier_sum = 0.0
    naive_sum = 0.0
    # 10 equal-width bins: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bucket_prob_sums = [0.0] * 10
    bucket_resolution_sums = [0.0] * 10
    bucket_counts = [0] * 10
    # Market buckets: same structure but binned by market_mid
    market_prob_sums = [0.0] * 10
    market_resolution_sums = [0.0] * 10
    market_bucket_counts = [0] * 10

    for estimated_prob, market_mid, resolution in rows:
        y = float(resolution)
        p = float(estimated_prob)
        mid = float(market_mid)

        brier_sum += (p - y) ** 2
        naive_sum += (mid - y) ** 2

        # Model bucket: bin by estimated_prob
        idx = min(int(p * 10), 9)
        bucket_counts[idx] += 1
        bucket_prob_sums[idx] += p
        bucket_resolution_sums[idx] += y

        # Market bucket: bin by market_mid
        midx = min(int(mid * 10), 9)
        market_bucket_counts[midx] += 1
        market_prob_sums[midx] += mid
        market_resolution_sums[midx] += y

    buckets = []
    market_buckets = []
    for i in range(10):
        lower = i / 10.0
        upper = (i + 1) / 10.0

        count = bucket_counts[i]
        mean_p = bucket_prob_sums[i] / count if count > 0 else (lower + upper) / 2.0
        resolution_rate = bucket_resolution_sums[i] / count if count > 0 else 0.0
        buckets.append(
            CalibrationBucket(
                lower=lower,
                upper=upper,
                count=count,
                mean_estimated_prob=mean_p,
                actual_resolution_rate=resolution_rate,
            )
        )

        mcount = market_bucket_counts[i]
        mean_mid = market_prob_sums[i] / mcount if mcount > 0 else (lower + upper) / 2.0
        market_resolution_rate = market_resolution_sums[i] / mcount if mcount > 0 else 0.0
        market_buckets.append(
            CalibrationBucket(
                lower=lower,
                upper=upper,
                count=mcount,
                mean_estimated_prob=mean_mid,
                actual_resolution_rate=market_resolution_rate,
            )
        )

    return CalibrationReport(
        brier_score=brier_sum / n,
        market_brier_score=naive_sum / n,
        n_samples=n,
        lookback_days=lookback_days,
        buckets=buckets,
        market_buckets=market_buckets,
    )


@dataclass
class SourceBrierScore:
    source_name: str
    weighted_brier_score: float
    n_signals: int        # number of (signal, source_name) appearances
    total_doc_appearances: int  # total docs from this source used across all signals
    total_share: float    # sum of per-signal shares (the denominator)


async def compute_source_brier_scores(
    session: AsyncSession,
    lookback_days: int | None = None,
    min_docs: int = 0,
) -> list[SourceBrierScore]:
    """Compute a weighted Brier score for each document source name.

    For each resolved signal, the Brier loss is attributed to its evidence
    documents in proportion to each source name's share of that signal's
    document set.  Summing across signals and dividing by the total share
    gives a weighted-average Brier score per source.

    Only signals that have at least one linked document contribute.

    Args:
        session: Async DB session.
        lookback_days: Restrict to signals created within the last N days.
                       None means all-time.
        min_docs: Exclude sources whose total document appearances across all
                  qualifying signals is below this threshold.  Use to filter
                  out the long tail of low-volume sources whose scores are
                  statistically unreliable.

    Returns:
        List of SourceBrierScore, sorted ascending by weighted_brier_score
        (best source first).  Empty list when there are no qualifying signals.
    """
    where_clauses: list = [
        MarketRow.status == "finalized",
        MarketRow.result.is_not(None),
        SignalRow.model_used != "demo_harness",
        SignalRow.prompt_version != "demo",
        DocumentMarketLinkRow.signal_id.is_not(None),
    ]
    if lookback_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        where_clauses.append(SignalRow.created_at >= cutoff)

    resolution_expr = case((MarketRow.result == "yes", 1), else_=0)

    # Inner: per (signal, source_name) document count
    inner = (
        select(
            SignalRow.id.label("signal_id"),
            SignalRow.estimated_probability.label("estimated_probability"),
            resolution_expr.label("resolution"),
            DocumentRow.source_name.label("source_name"),
            func.count(DocumentRow.id).label("source_count"),
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .join(DocumentMarketLinkRow, DocumentMarketLinkRow.signal_id == SignalRow.id)
        .join(DocumentRow, DocumentRow.id == DocumentMarketLinkRow.document_id)
        .where(and_(*where_clauses))
        .group_by(
            SignalRow.id,
            SignalRow.estimated_probability,
            MarketRow.result,
            DocumentRow.source_name,
        )
        .subquery()
    )

    # Outer: add total doc count per signal via window function
    total_count_expr = func.sum(inner.c.source_count).over(
        partition_by=inner.c.signal_id
    )
    stmt = select(
        inner.c.estimated_probability,
        inner.c.resolution,
        inner.c.source_name,
        inner.c.source_count,
        total_count_expr.label("total_count"),
    )

    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return []

    # Accumulate per source_name
    error_pieces: dict[str, float] = {}
    share_sums: dict[str, float] = {}
    n_signals_counter: dict[str, int] = {}
    doc_appearances: dict[str, int] = {}

    for estimated_prob, resolution, source_name, source_count, total_count in rows:
        if total_count == 0:
            continue
        p = float(estimated_prob)
        y = float(resolution)
        brier_loss = (p - y) ** 2
        share = float(source_count) / float(total_count)
        error_piece = brier_loss * share

        error_pieces[source_name] = error_pieces.get(source_name, 0.0) + error_piece
        share_sums[source_name] = share_sums.get(source_name, 0.0) + share
        n_signals_counter[source_name] = n_signals_counter.get(source_name, 0) + 1
        doc_appearances[source_name] = doc_appearances.get(source_name, 0) + int(source_count)

    scores: list[SourceBrierScore] = []
    for source_name, total_err in error_pieces.items():
        total_appearances = doc_appearances[source_name]
        if min_docs > 0 and total_appearances < min_docs:
            continue
        total_sh = share_sums[source_name]
        weighted = total_err / total_sh if total_sh > 0 else 0.0
        scores.append(
            SourceBrierScore(
                source_name=source_name,
                weighted_brier_score=weighted,
                n_signals=n_signals_counter[source_name],
                total_doc_appearances=total_appearances,
                total_share=total_sh,
            )
        )

    scores.sort(key=lambda s: s.weighted_brier_score)
    return scores


def _empty_buckets() -> list[CalibrationBucket]:
    return [
        CalibrationBucket(
            lower=i / 10.0,
            upper=(i + 1) / 10.0,
            count=0,
            mean_estimated_prob=(i / 10.0 + (i + 1) / 10.0) / 2.0,
            actual_resolution_rate=0.0,
        )
        for i in range(10)
    ]
