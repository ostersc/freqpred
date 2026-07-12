"""Unit tests for mode-scoped run_state risk windows (T90).

Uses an in-memory fake session that serves and mutates a real ``RunStateRow``
instance — no DB. The point of these tests is the mode partitioning itself:
a baseline or acknowledgement written in one trading mode must never be
visible from the other.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

# Ensure ORM relationships resolve
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.alerts.models import RunStateRow
from freqpred.alerts.run_state import (
    get_daily_loss_ack_at,
    get_drawdown_window,
    reset_drawdown,
    set_run_state,
)


class _FakeSession:
    """Simulates the run_state singleton table: one row, served on every select."""

    def __init__(self, row: RunStateRow | None = None) -> None:
        self.row = row
        self.commit_count = 0

    async def execute(self, stmt: object) -> MagicMock:
        result = MagicMock()
        result.scalar_one_or_none.return_value = self.row
        return result

    def add(self, obj: RunStateRow) -> None:
        self.row = obj

    async def commit(self) -> None:
        self.commit_count += 1


def _row(**overrides: object) -> RunStateRow:
    defaults: dict[str, object] = {
        "id": 1,
        "state": "running",
        "updated_at": datetime(2026, 7, 1, tzinfo=UTC),
        "drawdown_reset_at_paper": None,
        "drawdown_reset_bankroll_paper": None,
        "drawdown_reset_at_live": None,
        "drawdown_reset_bankroll_live": None,
        "daily_loss_ack_at_paper": None,
        "daily_loss_ack_at_live": None,
    }
    defaults.update(overrides)
    return RunStateRow(**defaults)


# ---------------------------------------------------------------------------
# Drawdown window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("reset_mode", "other_mode"), [("paper", "live"), ("live", "paper")])
async def test_drawdown_window_is_mode_scoped(reset_mode: str, other_mode: str) -> None:
    """A reset in one mode leaves the other mode's window untouched (None, None)."""
    session = _FakeSession()

    reset_at = await reset_drawdown(session, reset_mode, 374.92)

    own_at, own_bankroll = await get_drawdown_window(session, reset_mode)
    assert own_at == reset_at
    assert own_bankroll == 374.92

    assert await get_drawdown_window(session, other_mode) == (None, None)


@pytest.mark.asyncio
async def test_drawdown_reset_writes_only_target_mode_columns() -> None:
    """reset_drawdown(mode='live') must not touch the _paper columns."""
    paper_at = datetime(2026, 6, 1, tzinfo=UTC)
    session = _FakeSession(
        _row(drawdown_reset_at_paper=paper_at, drawdown_reset_bankroll_paper=374.92)
    )

    await reset_drawdown(session, "live", 10.0)

    assert session.row.drawdown_reset_at_paper == paper_at
    assert session.row.drawdown_reset_bankroll_paper == 374.92
    assert session.row.drawdown_reset_at_live is not None
    assert session.row.drawdown_reset_bankroll_live == 10.0
    assert session.commit_count == 1


@pytest.mark.asyncio
async def test_drawdown_window_no_row_returns_none() -> None:
    session = _FakeSession()
    assert await get_drawdown_window(session, "paper") == (None, None)
    assert await get_drawdown_window(session, "live") == (None, None)


@pytest.mark.asyncio
async def test_drawdown_window_signal_only_has_no_window() -> None:
    """signal-only places no orders — both modes' baselines are invisible to it."""
    session = _FakeSession(
        _row(
            drawdown_reset_at_paper=datetime(2026, 6, 1, tzinfo=UTC),
            drawdown_reset_bankroll_paper=374.92,
            drawdown_reset_at_live=datetime(2026, 7, 1, tzinfo=UTC),
            drawdown_reset_bankroll_live=10.0,
        )
    )
    assert await get_drawdown_window(session, "signal-only") == (None, None)


@pytest.mark.asyncio
async def test_reset_drawdown_rejects_non_trading_modes() -> None:
    session = _FakeSession()
    with pytest.raises(ValueError, match="signal-only"):
        await reset_drawdown(session, "signal-only", 100.0)
    assert session.row is None
    assert session.commit_count == 0


@pytest.mark.asyncio
async def test_unknown_mode_raises_value_error() -> None:
    """A typo'd mode must fail loudly, not silently disable the risk window."""
    session = _FakeSession(_row())
    with pytest.raises(ValueError, match="unknown trading mode"):
        await get_drawdown_window(session, "lvie")
    with pytest.raises(ValueError, match="unknown trading mode"):
        await get_daily_loss_ack_at(session, "papr")
    with pytest.raises(ValueError, match="unknown trading mode"):
        await reset_drawdown(session, "LIVE", 100.0)


# ---------------------------------------------------------------------------
# Daily-loss acknowledgement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("ack_mode", "other_mode"), [("paper", "live"), ("live", "paper")])
async def test_daily_loss_ack_is_mode_scoped(ack_mode: str, other_mode: str) -> None:
    """/start in one mode must not acknowledge the other mode's breaker."""
    session = _FakeSession()

    await set_run_state(session, "running", mode=ack_mode)

    assert await get_daily_loss_ack_at(session, ack_mode) is not None
    assert await get_daily_loss_ack_at(session, other_mode) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "signal-only"])
async def test_set_run_state_without_trading_mode_stamps_neither_ack(
    mode: str | None,
) -> None:
    session = _FakeSession()

    await set_run_state(session, "running", mode=mode)

    assert session.row.state == "running"
    assert session.row.daily_loss_ack_at_paper is None
    assert session.row.daily_loss_ack_at_live is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_set_run_state_non_running_does_not_stamp_ack(mode: str) -> None:
    """Only the transition to 'running' acknowledges the breaker."""
    session = _FakeSession(_row())

    await set_run_state(session, "paused", mode=mode)

    assert session.row.state == "paused"
    assert session.row.daily_loss_ack_at_paper is None
    assert session.row.daily_loss_ack_at_live is None
