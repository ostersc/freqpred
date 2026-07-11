from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.scheduler import run_cycle
from freqpred.metrics.scheduler import (
    refresh_edge_calibration_scores_if_due,
    run_source_quality_scheduler,
)
from freqpred.runtime.telemetry import (
    SERVICE_EDGE_CALIBRATION,
    SERVICE_INGESTION_SCHEDULER,
    SERVICE_SOURCE_QUALITY_SCHEDULER,
)


@pytest.mark.asyncio
async def test_ingestion_scheduler_updates_heartbeat_on_success() -> None:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.commit = AsyncMock()
    session_factory = MagicMock(return_value=session)
    telemetry = AsyncMock()

    with patch(
        "freqpred.ingestion.scheduler._load_active_market_queries",
        new=AsyncMock(return_value=[]),
    ):
        stats = await run_cycle(
            session_factory=session_factory,
            embedder=MagicMock(),
            telemetry=telemetry,
        )

    assert stats["markets_processed"] == 0
    telemetry.mark_success.assert_awaited_once_with(
        SERVICE_INGESTION_SCHEDULER,
        details=stats,
    )


@pytest.mark.asyncio
async def test_source_quality_scheduler_updates_heartbeat_on_success() -> None:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session_factory = MagicMock(return_value=session)
    telemetry = AsyncMock()

    async def _cancel_sleep(_: float) -> None:
        raise asyncio.CancelledError

    with patch(
        "freqpred.metrics.scheduler.refresh_source_quality_scores_if_due",
        new=AsyncMock(return_value=4),
    ), patch(
        "freqpred.metrics.scheduler.refresh_edge_calibration_scores_if_due",
        new=AsyncMock(return_value=2),
    ), patch("asyncio.sleep", new=_cancel_sleep):
        with pytest.raises(asyncio.CancelledError):
            await run_source_quality_scheduler(
                session_factory=session_factory,
                telemetry=telemetry,
            )

    telemetry.mark_success.assert_any_await(
        SERVICE_SOURCE_QUALITY_SCHEDULER,
        details={"reason": "startup", "rows_written": 4},
    )
    telemetry.mark_success.assert_any_await(
        SERVICE_EDGE_CALIBRATION,
        details={"reason": "startup", "rows_written": 2},
    )
    assert telemetry.mark_success.await_count == 2


@pytest.mark.asyncio
async def test_edge_calibration_failure_does_not_block_source_quality_heartbeat() -> None:
    """Edge-calibration and source-quality report independent heartbeats — one
    job failing must never suppress or corrupt the other's heartbeat."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session_factory = MagicMock(return_value=session)
    telemetry = AsyncMock()

    async def _cancel_sleep(_: float) -> None:
        raise asyncio.CancelledError

    with patch(
        "freqpred.metrics.scheduler.refresh_source_quality_scores_if_due",
        new=AsyncMock(return_value=4),
    ), patch(
        "freqpred.metrics.scheduler.refresh_edge_calibration_scores_if_due",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ), patch("asyncio.sleep", new=_cancel_sleep):
        with pytest.raises(asyncio.CancelledError):
            await run_source_quality_scheduler(
                session_factory=session_factory,
                telemetry=telemetry,
            )

    telemetry.mark_success.assert_any_await(
        SERVICE_SOURCE_QUALITY_SCHEDULER,
        details={"reason": "startup", "rows_written": 4},
    )
    telemetry.mark_error.assert_any_await(SERVICE_EDGE_CALIBRATION, "boom")


@pytest.mark.asyncio
async def test_refresh_edge_calibration_scores_if_due_skips_when_fresh() -> None:
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    latest_result = MagicMock()
    latest_result.scalar_one.return_value = datetime(2026, 7, 11, 7, 0, tzinfo=UTC)
    session = AsyncMock()
    session.execute = AsyncMock(return_value=latest_result)

    with patch(
        "freqpred.metrics.scheduler.refresh_edge_calibration_scores",
        new_callable=AsyncMock,
    ) as mock_refresh:
        rows_written = await refresh_edge_calibration_scores_if_due(session, now=now)

    assert rows_written == 0
    mock_refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_edge_calibration_scores_if_due_runs_when_stale() -> None:
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    latest_result = MagicMock()
    latest_result.scalar_one.return_value = datetime(2026, 7, 10, 7, 0, tzinfo=UTC)
    session = AsyncMock()
    session.execute = AsyncMock(return_value=latest_result)

    with patch(
        "freqpred.metrics.scheduler.refresh_edge_calibration_scores",
        new_callable=AsyncMock,
        return_value=7,
    ) as mock_refresh:
        rows_written = await refresh_edge_calibration_scores_if_due(session, now=now)

    assert rows_written == 7
    mock_refresh.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_refresh_edge_calibration_scores_if_due_runs_when_no_prior_snapshot() -> None:
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    latest_result = MagicMock()
    latest_result.scalar_one.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=latest_result)

    with patch(
        "freqpred.metrics.scheduler.refresh_edge_calibration_scores",
        new_callable=AsyncMock,
        return_value=3,
    ) as mock_refresh:
        rows_written = await refresh_edge_calibration_scores_if_due(session, now=now)

    assert rows_written == 3
    mock_refresh.assert_awaited_once_with(session)
