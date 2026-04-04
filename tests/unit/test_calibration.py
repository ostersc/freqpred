"""Unit tests for freqpred/metrics/calibration.py.

All DB interactions are mocked — no external dependencies.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.metrics.calibration import (
    CalibrationReport,
    SourceBrierScore,
    compute_calibration,
    compute_source_brier_scores,
)

# Ensure ORM relationships resolve
import freqpred.ingestion.models   # noqa: F401
import freqpred.llm.models         # noqa: F401
import freqpred.markets.models     # noqa: F401
import freqpred.rag.models         # noqa: F401
import freqpred.signal.models      # noqa: F401


def _make_session(rows: list[tuple[float, float, int]]) -> AsyncMock:
    """Return a mock AsyncSession whose execute() returns the given rows."""
    mock_result = MagicMock()
    mock_result.all.return_value = rows

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    return session


# ---------------------------------------------------------------------------
# Brier score formula tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perfect_calibration_brier_zero() -> None:
    """estimated=1.0, resolution=1 → Brier=0.0."""
    session = _make_session([(1.0, 0.5, 1)])
    report = await compute_calibration(session, mode="paper")
    assert report.brier_score == pytest.approx(0.0)
    assert report.n_samples == 1


@pytest.mark.asyncio
async def test_worst_calibration_brier_one() -> None:
    """estimated=1.0, resolution=0 → Brier=1.0."""
    session = _make_session([(1.0, 0.5, 0)])
    report = await compute_calibration(session, mode="paper")
    assert report.brier_score == pytest.approx(1.0)
    assert report.n_samples == 1


@pytest.mark.asyncio
async def test_brier_score_formula() -> None:
    """3 samples with known values → verify formula."""
    # (estimated_prob, mid, resolution)
    rows = [
        (0.8, 0.5, 1),   # (0.8 - 1)^2 = 0.04
        (0.3, 0.5, 0),   # (0.3 - 0)^2 = 0.09
        (0.6, 0.5, 1),   # (0.6 - 1)^2 = 0.16
    ]
    # mean = (0.04 + 0.09 + 0.16) / 3 = 0.29 / 3
    session = _make_session(rows)
    report = await compute_calibration(session, mode="paper")
    assert report.brier_score == pytest.approx(0.29 / 3, rel=1e-6)
    assert report.n_samples == 3


# ---------------------------------------------------------------------------
# Naive baseline tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naive_baseline_uses_market_mid() -> None:
    """Naive score is computed from mid_price, not estimated_prob."""
    rows = [
        (0.9, 0.4, 1),   # naive: (0.4 - 1)^2 = 0.36, model: (0.9-1)^2=0.01
        (0.9, 0.6, 0),   # naive: (0.6 - 0)^2 = 0.36, model: (0.9-0)^2=0.81
    ]
    session = _make_session(rows)
    report = await compute_calibration(session, mode="paper")
    expected_naive = (0.36 + 0.36) / 2
    expected_model = (0.01 + 0.81) / 2
    assert report.market_brier_score == pytest.approx(expected_naive, rel=1e-6)
    assert report.brier_score == pytest.approx(expected_model, rel=1e-6)


# ---------------------------------------------------------------------------
# Calibration bucket tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_buckets_cover_full_range() -> None:
    """All 10 buckets present, lower/upper correct."""
    session = _make_session([(0.5, 0.5, 1)])
    report = await compute_calibration(session, mode="paper")
    assert len(report.buckets) == 10
    for i, bucket in enumerate(report.buckets):
        assert bucket.lower == pytest.approx(i / 10.0)
        assert bucket.upper == pytest.approx((i + 1) / 10.0)


@pytest.mark.asyncio
async def test_buckets_count_samples_correctly() -> None:
    """Samples land in the correct bucket based on estimated_prob."""
    rows = [
        (0.65, 0.5, 1),   # bucket 6: [0.6, 0.7)
        (0.62, 0.5, 0),   # bucket 6: [0.6, 0.7)
        (0.71, 0.5, 1),   # bucket 7: [0.7, 0.8)
        (0.10, 0.5, 0),   # bucket 1: [0.1, 0.2)
    ]
    session = _make_session(rows)
    report = await compute_calibration(session, mode="paper")

    # bucket indices
    bucket_by_lower = {round(b.lower, 1): b for b in report.buckets}
    assert bucket_by_lower[0.6].count == 2
    assert bucket_by_lower[0.7].count == 1
    assert bucket_by_lower[0.1].count == 1
    # all others empty
    for lower, b in bucket_by_lower.items():
        if lower not in (0.6, 0.7, 0.1):
            assert b.count == 0


@pytest.mark.asyncio
async def test_bucket_resolution_rate() -> None:
    """Mean estimated prob and resolution rate computed correctly per bucket."""
    rows = [
        (0.65, 0.5, 1),   # bucket 6
        (0.67, 0.5, 0),   # bucket 6
    ]
    session = _make_session(rows)
    report = await compute_calibration(session, mode="paper")

    b = report.buckets[6]  # [0.6, 0.7)
    assert b.count == 2
    assert b.mean_estimated_prob == pytest.approx((0.65 + 0.67) / 2)
    assert b.actual_resolution_rate == pytest.approx(0.5)  # 1 YES out of 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_positions_returns_zero_samples() -> None:
    """Graceful handling with 0 resolved positions — n_samples=0."""
    session = _make_session([])
    report = await compute_calibration(session, mode="paper")
    assert report.n_samples == 0
    assert report.brier_score == pytest.approx(0.0)
    assert report.market_brier_score == pytest.approx(0.0)
    # Still returns 10 empty buckets
    assert len(report.buckets) == 10
    for b in report.buckets:
        assert b.count == 0


@pytest.mark.asyncio
async def test_prob_exactly_one_goes_to_last_bucket() -> None:
    """estimated_prob=1.0 should land in bucket 9 (not overflow)."""
    session = _make_session([(1.0, 0.5, 1)])
    report = await compute_calibration(session, mode="paper")
    assert report.buckets[9].count == 1


@pytest.mark.asyncio
async def test_two_signals_same_market_count_independently() -> None:
    """Each signal is scored independently — two signals for the same market
    produce n_samples=2, not 1."""
    rows = [
        (0.8, 0.5, 1),   # signal 1: (0.8 - 1)^2 = 0.04
        (0.6, 0.5, 1),   # signal 2: (0.6 - 1)^2 = 0.16
    ]
    session = _make_session(rows)
    report = await compute_calibration(session, mode="paper")

    assert report.n_samples == 2
    assert report.brier_score == pytest.approx((0.04 + 0.16) / 2, rel=1e-6)


@pytest.mark.asyncio
async def test_lookback_days_stored_in_report() -> None:
    """lookback_days is passed through and stored on the report."""
    session = _make_session([(0.7, 0.5, 1)])
    report = await compute_calibration(session, mode="paper", lookback_days=7)
    assert report.lookback_days == 7


@pytest.mark.asyncio
async def test_no_lookback_stored_as_none() -> None:
    """lookback_days=None (all-time) is stored on the report."""
    session = _make_session([(0.7, 0.5, 1)])
    report = await compute_calibration(session, mode="paper")
    assert report.lookback_days is None


@pytest.mark.asyncio
async def test_demo_harness_signals_excluded_from_query() -> None:
    """SQL query excludes demo_harness signals (model_used or prompt_version)."""
    captured_stmt = None

    async def _capture(stmt: object, *args: object, **kwargs: object) -> MagicMock:
        nonlocal captured_stmt
        captured_stmt = stmt
        result = MagicMock()
        result.all.return_value = []
        return result

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=_capture)
    await compute_calibration(session)

    assert captured_stmt is not None
    from sqlalchemy.dialects import postgresql

    sql = str(
        captured_stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "demo_harness" in sql
    assert "'demo'" in sql


# ---------------------------------------------------------------------------
# compute_source_brier_scores tests
# ---------------------------------------------------------------------------


def _make_source_session(
    rows: list[tuple[float, int, str, int, int]],
) -> AsyncMock:
    """Return a mock AsyncSession whose execute() returns the given rows.

    Row shape: (estimated_probability, resolution, source_type, source_count, total_count)
    """
    mock_result = MagicMock()
    mock_result.all.return_value = rows

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    return session


@pytest.mark.asyncio
async def test_source_brier_empty_returns_empty_list() -> None:
    """No qualifying signals → empty list."""
    session = _make_source_session([])
    scores = await compute_source_brier_scores(session)
    assert scores == []


@pytest.mark.asyncio
async def test_source_brier_single_source_single_signal() -> None:
    """Single signal, single source name (100% share) → weighted score = Brier loss."""
    # estimated=1.0, resolution=1 → brier_loss=0.0, share=1.0 → weighted=0.0
    session = _make_source_session([(1.0, 1, "Tavily", 3, 3)])
    scores = await compute_source_brier_scores(session)
    assert len(scores) == 1
    assert scores[0].source_name == "Tavily"
    assert scores[0].weighted_brier_score == pytest.approx(0.0)
    assert scores[0].n_signals == 1
    assert scores[0].total_share == pytest.approx(1.0)
    assert scores[0].total_doc_appearances == 3


@pytest.mark.asyncio
async def test_source_brier_matches_example_from_issue() -> None:
    """Verify the worked example from issue #56.

    Source A:
      Pred 1: brier=0.04, source_count=1, total=2 → share=0.50, piece=0.020
      Pred 2: brier=0.25, source_count=1, total=5 → share=0.20, piece=0.050
      Pred 3: brier=0.64, source_count=1, total=10 → share=0.10, piece=0.064
      weighted = (0.020 + 0.050 + 0.064) / (0.50 + 0.20 + 0.10) = 0.134 / 0.80 = 0.1675
    """
    # estimated_prob, resolution chosen to produce the stated Brier losses:
    # 0.04 → (0.8-1)^2, 0.25 → (0.5-1)^2, 0.64 → (0.2-1)^2
    rows = [
        (0.8, 1, "Tavily", 1, 2),   # piece 0.020
        (0.5, 1, "Tavily", 1, 5),   # piece 0.050
        (0.2, 1, "Tavily", 1, 10),  # piece 0.064
    ]
    session = _make_source_session(rows)
    scores = await compute_source_brier_scores(session)
    assert len(scores) == 1
    assert scores[0].source_name == "Tavily"
    assert scores[0].weighted_brier_score == pytest.approx(0.1675, rel=1e-4)
    assert scores[0].total_share == pytest.approx(0.80, rel=1e-6)
    assert scores[0].total_doc_appearances == 3


@pytest.mark.asyncio
async def test_source_brier_two_sources_sorted_ascending() -> None:
    """Results sorted by weighted Brier score ascending (best source first)."""
    # "r/politics" share=1.0, brier_loss=0.0 → score 0.0
    # "Tavily"     share=1.0, brier_loss=1.0 → score 1.0
    rows = [
        (1.0, 1, "r/politics", 5, 5),  # brier_loss=0.0
        (1.0, 0, "Tavily", 5, 5),      # brier_loss=1.0
    ]
    session = _make_source_session(rows)
    scores = await compute_source_brier_scores(session)
    assert len(scores) == 2
    assert scores[0].source_name == "r/politics"
    assert scores[1].source_name == "Tavily"
    assert scores[0].weighted_brier_score < scores[1].weighted_brier_score


@pytest.mark.asyncio
async def test_source_brier_mixed_sources_same_signal() -> None:
    """Signal with two source names: shares sum to 1, error split correctly.

    Signal: estimated=0.8, resolution=1, brier_loss=0.04
      Tavily: 2/5 = 0.40 share → piece=0.016
      r/politics: 3/5 = 0.60 share → piece=0.024
    """
    rows = [
        (0.8, 1, "Tavily", 2, 5),
        (0.8, 1, "r/politics", 3, 5),
    ]
    session = _make_source_session(rows)
    scores = await compute_source_brier_scores(session)
    by_name = {s.source_name: s for s in scores}
    assert by_name["Tavily"].weighted_brier_score == pytest.approx(0.016 / 0.40, rel=1e-6)
    assert by_name["r/politics"].weighted_brier_score == pytest.approx(0.024 / 0.60, rel=1e-6)
    # Both resolve to the same weighted score (0.04), since it's one signal
    assert by_name["Tavily"].weighted_brier_score == pytest.approx(
        by_name["r/politics"].weighted_brier_score, rel=1e-6
    )


@pytest.mark.asyncio
async def test_source_brier_min_docs_filters_low_volume() -> None:
    """Sources below min_docs threshold are excluded from results."""
    # Tavily: 2 doc appearances, r/politics: 10 doc appearances
    rows = [
        (0.8, 1, "Tavily", 2, 12),
        (0.8, 1, "r/politics", 10, 12),
    ]
    session = _make_source_session(rows)
    scores = await compute_source_brier_scores(session, min_docs=5)
    assert len(scores) == 1
    assert scores[0].source_name == "r/politics"


@pytest.mark.asyncio
async def test_source_brier_min_docs_zero_includes_all() -> None:
    """min_docs=0 disables filtering — all sources appear."""
    rows = [
        (0.8, 1, "Tavily", 1, 2),
        (0.8, 1, "r/politics", 1, 2),
    ]
    session = _make_source_session(rows)
    scores = await compute_source_brier_scores(session, min_docs=0)
    assert len(scores) == 2
