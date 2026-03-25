"""Brier score and calibration curve calculation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, case, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.markets.models import MarketRow
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
        )

    brier_sum = 0.0
    naive_sum = 0.0
    # 10 equal-width bins: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bucket_prob_sums = [0.0] * 10
    bucket_resolution_sums = [0.0] * 10
    bucket_counts = [0] * 10

    for estimated_prob, market_mid, resolution in rows:
        y = float(resolution)
        p = float(estimated_prob)
        mid = float(market_mid)

        brier_sum += (p - y) ** 2
        naive_sum += (mid - y) ** 2

        # Assign to bucket: bin index = floor(p * 10), clamped to [0, 9]
        idx = min(int(p * 10), 9)
        bucket_counts[idx] += 1
        bucket_prob_sums[idx] += p
        bucket_resolution_sums[idx] += y

    buckets = []
    for i in range(10):
        lower = i / 10.0
        upper = (i + 1) / 10.0
        count = bucket_counts[i]
        if count > 0:
            mean_p = bucket_prob_sums[i] / count
            resolution_rate = bucket_resolution_sums[i] / count
        else:
            mean_p = (lower + upper) / 2.0
            resolution_rate = 0.0
        buckets.append(
            CalibrationBucket(
                lower=lower,
                upper=upper,
                count=count,
                mean_estimated_prob=mean_p,
                actual_resolution_rate=resolution_rate,
            )
        )

    return CalibrationReport(
        brier_score=brier_sum / n,
        market_brier_score=naive_sum / n,
        n_samples=n,
        lookback_days=lookback_days,
        buckets=buckets,
    )


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
