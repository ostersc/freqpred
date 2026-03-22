"""Unit tests for freqpred/metrics/calibration.py.

All DB interactions are mocked — no external dependencies.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from freqpred.metrics.calibration import (
    CalibrationReport,
    compute_calibration,
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
    assert report.naive_brier_score == pytest.approx(expected_naive, rel=1e-6)
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
    assert report.naive_brier_score == pytest.approx(0.0)
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
