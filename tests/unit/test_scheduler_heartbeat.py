from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.scheduler import run_cycle
from freqpred.metrics.scheduler import run_source_quality_scheduler
from freqpred.runtime.telemetry import (
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
    ), patch("asyncio.sleep", new=_cancel_sleep):
        with pytest.raises(asyncio.CancelledError):
            await run_source_quality_scheduler(
                session_factory=session_factory,
                telemetry=telemetry,
            )

    telemetry.mark_success.assert_awaited_once_with(
        SERVICE_SOURCE_QUALITY_SCHEDULER,
        details={"reason": "startup", "rows_written": 4},
    )
