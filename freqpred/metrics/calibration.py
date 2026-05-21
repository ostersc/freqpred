"""Brier score and calibration curve calculation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.markets.models import MarketRow
from freqpred.metrics.models import SeriesOptionHistoryRow, SourceQualityScoreRow
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
    market_category: str | None = None,
    ticker_prefix: str | None = None,
    direction: str | None = None,
    model_used: str | None = None,
    prompt_version: str | None = None,
    series_ticker: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
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
        market_category: Optional MarketRow.category filter. None means all categories.
        ticker_prefix: Optional prefix filter on market ticker (MarketRow.id).
        direction: Optional SignalRow.direction filter ("YES" | "NO" | "SKIP").
        model_used: Optional SignalRow.model_used filter.
        prompt_version: Optional SignalRow.prompt_version filter.
        series_ticker: Optional MarketRow.series_ticker filter.
        min_confidence: Optional lower bound on SignalRow.estimated_probability.
        max_confidence: Optional upper bound on SignalRow.estimated_probability.
    """
    resolution_expr = case((MarketRow.result == "yes", 1), else_=0).label("resolution")

    where_clauses: list = [
        MarketRow.status == "finalized",
        MarketRow.result.is_not(None),
        SignalRow.model_used != "demo_harness",
        SignalRow.prompt_version != "demo",
        SignalRow.trigger != "price_moved",
    ]
    if lookback_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        where_clauses.append(SignalRow.created_at >= cutoff)
    if market_category is not None:
        where_clauses.append(MarketRow.category == market_category)
    if ticker_prefix is not None:
        where_clauses.append(MarketRow.id.ilike(f"{ticker_prefix}%"))
    if direction is not None:
        where_clauses.append(SignalRow.direction == direction)
    if model_used is not None:
        where_clauses.append(SignalRow.model_used == model_used)
    if prompt_version is not None:
        where_clauses.append(SignalRow.prompt_version == prompt_version)
    if series_ticker is not None:
        where_clauses.append(MarketRow.series_ticker == series_ticker)
    if min_confidence is not None:
        where_clauses.append(SignalRow.confidence >= min_confidence)
    if max_confidence is not None:
        where_clauses.append(SignalRow.confidence <= max_confidence)

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
    market_category: str | None = None,
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
        market_category: Optional MarketRow.category filter. None means all categories.

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
    if market_category is not None:
        where_clauses.append(MarketRow.category == market_category)

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


async def refresh_source_quality_scores(
    session: AsyncSession,
    lookback_days: int = 90,
) -> int:
    """Write one rolling snapshot row per source/category pair plus the global row.

    Caller owns commit/rollback.
    """
    distinct_categories_result = await session.execute(
        select(MarketRow.category)
        .join(SignalRow, SignalRow.market_id == MarketRow.id)
        .where(
            MarketRow.status == "finalized",
            MarketRow.result.is_not(None),
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
        .distinct()
        .order_by(MarketRow.category)
    )
    categories = [row[0] for row in distinct_categories_result.all()]

    rows_written = 0
    for market_category in [None, *categories]:
        calibration = await compute_calibration(
            session,
            lookback_days=lookback_days,
            market_category=market_category,
        )
        if calibration.n_samples == 0:
            continue

        scores = await compute_source_brier_scores(
            session,
            lookback_days=lookback_days,
            market_category=market_category,
        )
        for score in scores:
            session.add(
                SourceQualityScoreRow(
                    source_name=score.source_name,
                    market_category=market_category,
                    lookback_days=lookback_days,
                    weighted_brier=score.weighted_brier_score,
                    overall_brier=calibration.brier_score,
                    n_signals=score.n_signals,
                    total_doc_uses=score.total_doc_appearances,
                )
            )
            rows_written += 1

    await session.flush()
    return rows_written


@dataclass
class CalibrationTimeSeriesPoint:
    date: str
    brier_score: float | None
    market_brier_score: float | None
    n_samples: int


@dataclass
class CalibrationTimeSeries:
    points: list[CalibrationTimeSeriesPoint]


async def compute_calibration_time_series(
    session: AsyncSession,
    mode: str = "paper",
    lookback_days: int | None = None,
    market_category: str | None = None,
    ticker_prefix: str | None = None,
    direction: str | None = None,
    model_used: str | None = None,
    prompt_version: str | None = None,
    series_ticker: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> CalibrationTimeSeries:
    """Compute per-day Brier scores for the calibration over-time chart.

    Groups qualifying signals by the day they were created and computes a
    Brier score for each day.  Days with zero qualifying samples are omitted
    (sparse output).
    """
    resolution_expr = case((MarketRow.result == "yes", 1), else_=0).label("resolution")

    where_clauses: list = [
        MarketRow.status == "finalized",
        MarketRow.result.is_not(None),
        SignalRow.model_used != "demo_harness",
        SignalRow.prompt_version != "demo",
        SignalRow.trigger != "price_moved",
    ]
    if lookback_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        where_clauses.append(SignalRow.created_at >= cutoff)
    if market_category is not None:
        where_clauses.append(MarketRow.category == market_category)
    if ticker_prefix is not None:
        where_clauses.append(MarketRow.id.ilike(f"{ticker_prefix}%"))
    if direction is not None:
        where_clauses.append(SignalRow.direction == direction)
    if model_used is not None:
        where_clauses.append(SignalRow.model_used == model_used)
    if prompt_version is not None:
        where_clauses.append(SignalRow.prompt_version == prompt_version)
    if series_ticker is not None:
        where_clauses.append(MarketRow.series_ticker == series_ticker)
    if min_confidence is not None:
        where_clauses.append(SignalRow.confidence >= min_confidence)
    if max_confidence is not None:
        where_clauses.append(SignalRow.confidence <= max_confidence)

    # Keep only the latest signal per (market_id, day) — superseded estimates
    # on the same day for the same market are excluded.
    day_expr = func.date_trunc("day", SignalRow.created_at)
    subq = (
        select(
            day_expr.label("day"),
            SignalRow.estimated_probability,
            SignalRow.market_mid_at_signal,
            resolution_expr,
            func.row_number()
            .over(
                partition_by=[SignalRow.market_id, day_expr],
                order_by=SignalRow.created_at.desc(),
            )
            .label("rn"),
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(and_(*where_clauses))
        .subquery()
    )
    stmt = (
        select(subq.c.day, subq.c.estimated_probability, subq.c.market_mid_at_signal, subq.c.resolution)
        .where(subq.c.rn == 1)
        .order_by(subq.c.day)
    )
    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return CalibrationTimeSeries(points=[])

    day_accum: dict[str, list] = {}
    for day, estimated_prob, market_mid, resolution in rows:
        day_str = day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)[:10]
        if day_str not in day_accum:
            day_accum[day_str] = [0.0, 0.0, 0]
        p = float(estimated_prob)
        mid = float(market_mid)
        y = float(resolution)
        day_accum[day_str][0] += (p - y) ** 2
        day_accum[day_str][1] += (mid - y) ** 2
        day_accum[day_str][2] += 1

    points = [
        CalibrationTimeSeriesPoint(
            date=day_str,
            brier_score=acc[0] / acc[2] if acc[2] > 0 else None,
            market_brier_score=acc[1] / acc[2] if acc[2] > 0 else None,
            n_samples=acc[2],
        )
        for day_str, acc in sorted(day_accum.items())
    ]
    return CalibrationTimeSeries(points=points)


@dataclass
class CalibrationHeatmapCell:
    brier_score: float | None
    market_brier_score: float | None
    n_samples: int
    delta: float | None  # market_brier - model_brier; positive = model beats market


@dataclass
class CalibrationHeatmapRow:
    series_ticker: str   # "" for the synthetic "All Options" aggregate row
    option_code: str     # "All" for the aggregate row
    option_label: str
    cells: dict[str, CalibrationHeatmapCell]  # keyed by prompt_version or "All"


@dataclass
class CalibrationHeatmapReport:
    rows: list[CalibrationHeatmapRow]
    prompt_versions: list[str]  # sorted distinct prompt versions (excludes "All")


def _make_heatmap_cell(brier_sum: float, naive_sum: float, n: int) -> CalibrationHeatmapCell:
    if n == 0:
        return CalibrationHeatmapCell(
            brier_score=None, market_brier_score=None, n_samples=0, delta=None
        )
    bs = brier_sum / n
    mbs = naive_sum / n
    return CalibrationHeatmapCell(
        brier_score=bs, market_brier_score=mbs, n_samples=n, delta=mbs - bs
    )


async def compute_calibration_heatmap(
    session: AsyncSession,
    mode: str = "paper",
    lookback_days: int | None = None,
    market_category: str | None = None,
    ticker_prefix: str | None = None,
    direction: str | None = None,
    model_used: str | None = None,
    series_ticker: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> CalibrationHeatmapReport:
    """Compute Brier-score heatmap grouped by (series_ticker, option_code) × prompt_version.

    prompt_version is intentionally not a filter here — it is the column dimension of
    the heatmap.  Only markets with a non-null series_ticker are included.
    """
    resolution_expr = case((MarketRow.result == "yes", 1), else_=0).label("resolution")

    where_clauses: list = [
        MarketRow.status == "finalized",
        MarketRow.result.is_not(None),
        SignalRow.model_used != "demo_harness",
        SignalRow.prompt_version != "demo",
        SignalRow.trigger != "price_moved",
        MarketRow.series_ticker.is_not(None),
    ]
    if lookback_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        where_clauses.append(SignalRow.created_at >= cutoff)
    if market_category is not None:
        where_clauses.append(MarketRow.category == market_category)
    if ticker_prefix is not None:
        where_clauses.append(MarketRow.id.ilike(f"{ticker_prefix}%"))
    if direction is not None:
        where_clauses.append(SignalRow.direction == direction)
    if model_used is not None:
        where_clauses.append(SignalRow.model_used == model_used)
    if series_ticker is not None:
        where_clauses.append(MarketRow.series_ticker == series_ticker)
    if min_confidence is not None:
        where_clauses.append(SignalRow.confidence >= min_confidence)
    if max_confidence is not None:
        where_clauses.append(SignalRow.confidence <= max_confidence)

    stmt = (
        select(
            MarketRow.id.label("market_id"),
            MarketRow.series_ticker,
            SignalRow.prompt_version,
            SignalRow.estimated_probability,
            SignalRow.market_mid_at_signal,
            resolution_expr,
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(and_(*where_clauses))
    )
    result = await session.execute(stmt)
    rows = result.all()

    _empty_cell = CalibrationHeatmapCell(
        brier_score=None, market_brier_score=None, n_samples=0, delta=None
    )

    if not rows:
        return CalibrationHeatmapReport(
            rows=[
                CalibrationHeatmapRow(
                    series_ticker="",
                    option_code="All",
                    option_label="All Options",
                    cells={"All": _empty_cell},
                )
            ],
            prompt_versions=[],
        )

    # Accumulate: {(series_ticker, option_code, prompt_version): [brier_sum, naive_sum, n]}
    per_cell: dict[tuple[str, str, str], list] = {}
    for market_id, series_ticker_val, prompt_version, estimated_prob, market_mid, resolution in rows:
        if not series_ticker_val:
            continue
        option_code = market_id.rsplit("-", 1)[-1] if "-" in market_id else market_id
        p = float(estimated_prob)
        mid = float(market_mid)
        y = float(resolution)
        key = (series_ticker_val, option_code, prompt_version)
        if key not in per_cell:
            per_cell[key] = [0.0, 0.0, 0]
        per_cell[key][0] += (p - y) ** 2
        per_cell[key][1] += (mid - y) ** 2
        per_cell[key][2] += 1

    option_pairs: set[tuple[str, str]] = {(s, o) for s, o, _ in per_cell}
    prompt_versions = sorted({pv for _, _, pv in per_cell})

    # Fetch option labels
    option_label_map: dict[tuple[str, str], str] = {}
    if option_pairs:
        series_in = list({s for s, _ in option_pairs})
        label_result = await session.execute(
            select(
                SeriesOptionHistoryRow.series_ticker,
                SeriesOptionHistoryRow.option_code,
                SeriesOptionHistoryRow.option_label,
            ).where(SeriesOptionHistoryRow.series_ticker.in_(series_in))
        )
        for lrow in label_result.all():
            option_label_map[(lrow[0], lrow[1])] = lrow[2]

    # Build per-(series_ticker, option_code) row cells
    row_cells: dict[tuple[str, str], dict[str, CalibrationHeatmapCell]] = {}
    for s, o in sorted(option_pairs, key=lambda t: (t[0], t[1])):
        cells: dict[str, CalibrationHeatmapCell] = {}
        all_b, all_nb, all_n = 0.0, 0.0, 0
        for pv in prompt_versions:
            if (s, o, pv) in per_cell:
                b, nb, n = per_cell[(s, o, pv)]
                cells[pv] = _make_heatmap_cell(b, nb, n)
                all_b += b
                all_nb += nb
                all_n += n
            else:
                cells[pv] = _empty_cell
        cells["All"] = _make_heatmap_cell(all_b, all_nb, all_n)
        row_cells[(s, o)] = cells

    # Build "All Options" aggregate row
    all_row_cells: dict[str, CalibrationHeatmapCell] = {}
    total_b, total_nb, total_n = 0.0, 0.0, 0
    for pv in prompt_versions:
        pv_b, pv_nb, pv_n = 0.0, 0.0, 0
        for s, o in option_pairs:
            if (s, o, pv) in per_cell:
                b, nb, n = per_cell[(s, o, pv)]
                pv_b += b
                pv_nb += nb
                pv_n += n
        all_row_cells[pv] = _make_heatmap_cell(pv_b, pv_nb, pv_n)
        total_b += pv_b
        total_nb += pv_nb
        total_n += pv_n
    all_row_cells["All"] = _make_heatmap_cell(total_b, total_nb, total_n)

    result_rows: list[CalibrationHeatmapRow] = [
        CalibrationHeatmapRow(
            series_ticker="",
            option_code="All",
            option_label="All Options",
            cells=all_row_cells,
        )
    ]
    for s, o in sorted(option_pairs, key=lambda t: (t[0], t[1])):
        result_rows.append(
            CalibrationHeatmapRow(
                series_ticker=s,
                option_code=o,
                option_label=option_label_map.get((s, o), o),
                cells=row_cells[(s, o)],
            )
        )

    return CalibrationHeatmapReport(rows=result_rows, prompt_versions=prompt_versions)


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
