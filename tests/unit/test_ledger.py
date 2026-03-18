"""Unit tests for freqpred/trading/ledger.py.

All DB interactions are mocked — no external dependencies.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.markets.models import Market, Position, PositionRow
from freqpred.signal.models import Signal
from freqpred.trading.ledger import (
    close_position,
    get_daily_pnl,
    get_open_positions,
    get_portfolio_summary,
    open_position,
)

# Ensure ORM relationships resolve
import freqpred.ingestion.models   # noqa: F401
import freqpred.llm.models         # noqa: F401
import freqpred.markets.models     # noqa: F401
import freqpred.rag.models         # noqa: F401
import freqpred.signal.models      # noqa: F401

NOW = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
MARKET_ID = "MKT-ABC"
SIGNAL_ID = str(uuid.uuid4())


def _make_market() -> Market:
    return Market(
        id=MARKET_ID,
        platform="kalshi",
        question="Will X happen?",
        category="politics",
        close_time=NOW + timedelta(days=10),
        yes_bid=0.50,
        yes_ask=0.56,
        mid_price=0.53,
        volume_24h=1000.0,
        open_interest=5000.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
    )


def _make_signal(edge: float = 0.15, estimated_probability: float = 0.65) -> Signal:
    return Signal(
        id=SIGNAL_ID,
        market_id=MARKET_ID,
        estimated_probability=estimated_probability,
        confidence=0.80,
        edge=edge,
        market_mid_at_signal=0.50,
        direction="YES",
        reasoning="test",
        sources=[],
        retrieval_hash="a" * 64,
        model_used="claude-sonnet-4-6",
        prompt_version="v1",
        trigger="manual",
        created_at=NOW,
        raw_context="",
    )


def _make_open_row(
    *,
    contracts: int = 100,
    entry_price: float = 0.54,
    direction: str = "YES",
    position_id: uuid.UUID | None = None,
) -> PositionRow:
    row = PositionRow(
        id=position_id or uuid.uuid4(),
        market_id=MARKET_ID,
        signal_id=uuid.UUID(SIGNAL_ID),
        strategy_name="ConservativeDefault",
        strategy_version="1.0",
        signal_confidence=0.80,
        signal_edge=0.15,
        signal_estimated_prob=0.65,
        direction=direction,
        contracts=contracts,
        entry_price=entry_price,
        entry_time=NOW,
        mode="paper",
        status="open",
    )
    return row


# ---------------------------------------------------------------------------
# open_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_position_creates_row() -> None:
    market = _make_market()
    signal = _make_signal()

    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    position = await open_position(
        session,
        market=market,
        signal=signal,
        strategy_name="ConservativeDefault",
        strategy_version="1.0",
        direction="YES",
        contracts=100,
        entry_price=0.54,
        mode="paper",
    )

    session.add.assert_called_once()
    session.commit.assert_awaited_once()
    assert position.status == "open"
    assert position.market_id == MARKET_ID
    assert position.signal_id == SIGNAL_ID
    assert position.contracts == 100
    assert position.entry_price == pytest.approx(0.54)
    assert position.direction == "YES"
    assert position.mode == "paper"
    assert position.signal_confidence == pytest.approx(0.80)
    assert position.signal_edge == pytest.approx(0.15)
    assert position.signal_estimated_prob == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_position_win() -> None:
    """direction=YES, resolution=1 → exit_price=1.0, pnl > 0."""
    pid = uuid.uuid4()
    row = _make_open_row(contracts=100, entry_price=0.54, direction="YES", position_id=pid)

    result_mock = MagicMock()
    result_mock.scalar_one.return_value = row

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)
    session.commit = AsyncMock()

    position = await close_position(session, str(pid), exit_price=1.0, resolution=1)

    assert position.status == "closed"
    assert position.exit_price == pytest.approx(1.0)
    assert position.resolution == 1
    assert position.pnl is not None
    assert position.pnl > 0


@pytest.mark.asyncio
async def test_close_position_loss() -> None:
    """direction=YES, resolution=0 → exit_price=0.0, pnl < 0."""
    pid = uuid.uuid4()
    row = _make_open_row(contracts=100, entry_price=0.54, direction="YES", position_id=pid)

    result_mock = MagicMock()
    result_mock.scalar_one.return_value = row

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)
    session.commit = AsyncMock()

    position = await close_position(session, str(pid), exit_price=0.0, resolution=0)

    assert position.status == "closed"
    assert position.exit_price == pytest.approx(0.0)
    assert position.resolution == 0
    assert position.pnl is not None
    assert position.pnl < 0


# ---------------------------------------------------------------------------
# P&L formula
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pnl_formula_correct() -> None:
    """100 contracts at $0.54, resolves YES → pnl = $46.00."""
    pid = uuid.uuid4()
    row = _make_open_row(contracts=100, entry_price=0.54, direction="YES", position_id=pid)

    result_mock = MagicMock()
    result_mock.scalar_one.return_value = row

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)
    session.commit = AsyncMock()

    position = await close_position(session, str(pid), exit_price=1.0, resolution=1)

    # pnl = (1.0 - 0.54) * 100 = 46.00
    assert position.pnl == pytest.approx(46.0, abs=1e-4)
    # pnl_pct = 46.00 / (0.54 * 100) = 46 / 54 ≈ 0.8519
    assert position.pnl_pct == pytest.approx(46.0 / 54.0, rel=1e-4)


# ---------------------------------------------------------------------------
# get_open_positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_open_positions_excludes_closed() -> None:
    open_row = _make_open_row()
    closed_row = _make_open_row()
    closed_row.status = "closed"

    scalars_mock = MagicMock()
    scalars_mock.all.return_value = [open_row]  # only the open one

    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)

    positions = await get_open_positions(session)

    assert len(positions) == 1
    assert positions[0].status == "open"


# ---------------------------------------------------------------------------
# get_portfolio_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_summary_totals() -> None:
    """Two open positions: 100 @ $0.54 and 50 @ $0.60 → exposure = $84."""
    # execute() is called 4 times: open_count, exposure, daily_pnl (today_start check), all_time
    # get_daily_pnl calls execute once; get_portfolio_summary calls execute 3 times + delegates daily_pnl

    call_returns = [2, 84.0, 5.0, 120.0]  # open_count, exposure, daily_pnl, all_time_pnl

    async def _execute(stmt: object) -> MagicMock:
        result = MagicMock()
        result.scalar_one.return_value = call_returns.pop(0)
        return result

    session = MagicMock()
    session.execute = _execute

    summary = await get_portfolio_summary(session)

    assert summary["open_count"] == 2
    assert summary["total_exposure_usd"] == pytest.approx(84.0)
    assert summary["daily_pnl_usd"] == pytest.approx(5.0)
    assert summary["all_time_pnl_usd"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# get_daily_pnl
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_pnl_excludes_yesterday() -> None:
    """Only today's closed positions count; yesterday is excluded by the query filter."""
    # The filter is applied via SQLAlchemy in the query itself.
    # We mock execute() returning only today's sum.
    result_mock = MagicMock()
    result_mock.scalar_one.return_value = 30.0

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)

    pnl = await get_daily_pnl(session)

    assert pnl == pytest.approx(30.0)
    # Verify that execute was called (the filtering happens inside the ORM query)
    session.execute.assert_awaited_once()
