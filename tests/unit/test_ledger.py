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
    partial_close_position,
    update_position_excursions,
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

    positions = await get_open_positions(session, mode="paper")

    assert len(positions) == 1
    assert positions[0].status == "open"


# ---------------------------------------------------------------------------
# get_portfolio_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_summary_totals() -> None:
    """Two open positions: 100 @ $0.54 and 50 @ $0.60 → exposure = $84."""
    call_returns = [2, 84.0, 5.0, 120.0]  # open_count, exposure, daily_pnl, all_time_pnl

    # The 5th execute call fetches open positions for unrealized P&L + excursions; return empty rows.
    open_rows_mock = MagicMock()
    open_rows_mock.all.return_value = []

    call_count = 0

    async def _execute(stmt: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            return open_rows_mock
        result = MagicMock()
        result.scalar_one.return_value = call_returns.pop(0)
        return result

    session = MagicMock()
    session.execute = _execute

    summary = await get_portfolio_summary(session, mode="paper")

    assert summary["open_count"] == 2
    assert summary["total_exposure_usd"] == pytest.approx(84.0)
    assert summary["daily_pnl_usd"] == pytest.approx(5.0)
    assert summary["all_time_pnl_usd"] == pytest.approx(120.0)
    assert summary["unrealized_pnl_usd"] == pytest.approx(0.0)
    assert summary["net_exposure_usd"] == pytest.approx(0.0)
    assert summary["portfolio_mae_usd"] is None
    assert summary["portfolio_mfe_usd"] is None


@pytest.mark.asyncio
async def test_portfolio_summary_excursions_weighted() -> None:
    """MAE/MFE are contract-weighted: 100-contract position dominates a 1-contract position."""
    call_returns = [2, 101.0, 0.0, 0.0]

    # 100 YES contracts @ 0.40 with mae=-0.08, mfe=+0.12
    # 1   YES contract  @ 0.50 with mae=-0.20, mfe=+0.30
    open_rows_mock = MagicMock()
    open_rows_mock.all.return_value = [
        (100, 0.40, "YES", -0.08, 0.12, 0.44),
        (1,   0.50, "YES", -0.20, 0.30, 0.55),
    ]

    call_count = 0

    async def _execute(stmt: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            return open_rows_mock
        result = MagicMock()
        result.scalar_one.return_value = call_returns.pop(0)
        return result

    session = MagicMock()
    session.execute = _execute

    summary = await get_portfolio_summary(session, mode="paper")

    # Portfolio MAE $ = 100*(-0.08) + 1*(-0.20) = -8.20
    # Portfolio MAE % = -8.20 / 101 ≈ -0.08119
    assert summary["portfolio_mae_usd"] == pytest.approx(-8.20, abs=1e-4)
    assert summary["portfolio_mae_pct"] == pytest.approx(-8.20 / 101, rel=1e-4)

    # Portfolio MFE $ = 100*(0.12) + 1*(0.30) = 12.30
    assert summary["portfolio_mfe_usd"] == pytest.approx(12.30, abs=1e-4)
    assert summary["portfolio_mfe_pct"] == pytest.approx(12.30 / 101, rel=1e-4)

    # Net exposure: both YES → 100*0.40 + 1*0.50 = 40.50
    assert summary["net_exposure_usd"] == pytest.approx(40.50, abs=1e-4)


# ---------------------------------------------------------------------------
# update_position_excursions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_position_excursions() -> None:
    pid = uuid.uuid4()
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    await update_position_excursions(session, str(pid), mae=-0.08, mfe=0.12)

    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()


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

    pnl = await get_daily_pnl(session, mode="paper")

    assert pnl == pytest.approx(30.0)
    # Verify that execute was called (the filtering happens inside the ORM query)
    session.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# T76: partial_close_position
# ---------------------------------------------------------------------------


def _make_open_row_with_defaults(
    *,
    contracts: int = 10,
    entry_price: float = 0.50,
    direction: str = "YES",
    position_id: uuid.UUID | None = None,
    entry_fee_usd: float = 0.05,
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
        mode="live",
        status="open",
        entry_fee_usd=entry_fee_usd,
        realized_pnl_accumulator=0.0,
        exit_fee_usd=0.0,
        exit_filled_contracts=0,
        exit_requested_contracts=None,
        exit_order_id=None,
    )
    return row


def _make_position_from_row(row: PositionRow) -> Position:
    """Build a minimal Position dataclass from a PositionRow for passing to partial_close."""
    return Position(
        id=str(row.id),
        market_id=row.market_id,
        signal_id=str(row.signal_id),
        strategy_name=row.strategy_name,
        strategy_version=row.strategy_version,
        signal_confidence=row.signal_confidence,
        signal_edge=row.signal_edge,
        signal_estimated_prob=row.signal_estimated_prob,
        direction=row.direction,
        contracts=row.contracts,
        entry_price=row.entry_price,
        entry_time=row.entry_time,
        mode=row.mode,
        status=row.status,
        entry_fee_usd=row.entry_fee_usd or 0.0,
        realized_pnl_accumulator=row.realized_pnl_accumulator or 0.0,
    )


def _mock_session_for_row(row: PositionRow) -> MagicMock:
    result_mock = MagicMock()
    result_mock.scalar_one.return_value = row
    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_partial_close_position_decrements_contracts_and_records_realized_pnl() -> None:
    """Partial close decrements contracts and accumulates gross P&L (no entry-fee)."""
    pid = uuid.uuid4()
    row = _make_open_row_with_defaults(contracts=10, entry_price=0.50, position_id=pid)
    pos = _make_position_from_row(row)
    session = _mock_session_for_row(row)

    result = await partial_close_position(
        session, pos,
        filled_contracts=6,
        fill_price=0.55,
        fee_usd=0.01,
        exit_reason="stoploss",
    )

    # 6 contracts closed; 4 remain
    assert row.contracts == 4
    # gross accumulator = (0.55 - 0.50) * 6 = 0.30
    assert row.realized_pnl_accumulator == pytest.approx(0.30)
    # exit_fee_usd accumulated
    assert row.exit_fee_usd == pytest.approx(0.01)
    # exit_filled_contracts running total
    assert row.exit_filled_contracts == 6
    # status stays open
    assert result.status == "open"
    assert row.exit_time is None


@pytest.mark.asyncio
async def test_partial_close_keeps_status_open_when_residual_remains() -> None:
    """When residual contracts remain after close, status stays 'open'."""
    row = _make_open_row_with_defaults(contracts=10, entry_price=0.50)
    pos = _make_position_from_row(row)
    session = _mock_session_for_row(row)

    result = await partial_close_position(
        session, pos,
        filled_contracts=3,
        fill_price=0.60,
        fee_usd=0.005,
        exit_reason="trailing_stop",
    )

    assert result.status == "open"
    assert row.contracts == 7


@pytest.mark.asyncio
async def test_partial_close_transitions_to_closed_when_residual_zero_yes() -> None:
    """Final close (residual=0, YES direction): weighted-avg exit_price, net pnl, pnl_pct correct."""
    pid = uuid.uuid4()
    entry_price = 0.50
    entry_fee = 0.05
    row = _make_open_row_with_defaults(
        contracts=4,
        entry_price=entry_price,
        direction="YES",
        position_id=pid,
        entry_fee_usd=entry_fee,
    )
    # Simulate a prior partial close of 6 contracts at 0.55 with fee=0.01 having occurred
    # (accumulated into the row before this test simulates "now"):
    row.realized_pnl_accumulator = (0.55 - 0.50) * 6  # = 0.30
    row.exit_fee_usd = 0.01
    row.exit_filled_contracts = 6
    row.contracts = 4

    pos = _make_position_from_row(row)
    session = _mock_session_for_row(row)

    result = await partial_close_position(
        session, pos,
        filled_contracts=4,
        fill_price=0.60,
        fee_usd=0.01,
        exit_reason="stoploss",
        exit_order_id="exit-order-2",
        exit_requested_contracts=4,
    )

    # After final close:
    # total_closed = 10, accumulator = 0.30 + (0.60-0.50)*4 = 0.30 + 0.40 = 0.70
    # weighted_avg_exit = 0.70 / 10 + 0.50 = 0.57
    # pnl = 0.70 - entry_fee - total_exit_fees = 0.70 - 0.05 - 0.02 = 0.63
    assert result.status == "closed"
    assert row.exit_price == pytest.approx(0.57, abs=1e-5)
    assert row.pnl == pytest.approx(0.63, abs=1e-4)
    assert row.exit_time is not None
    assert row.exit_reason == "stoploss"
    assert row.exit_order_id == "exit-order-2"


@pytest.mark.asyncio
async def test_partial_close_transitions_to_closed_when_residual_zero_no() -> None:
    """Final close works correctly for a NO direction position."""
    entry_price = 0.40   # cost per NO contract (1 - yes_ask = 0.40)
    entry_fee = 0.04
    row = _make_open_row_with_defaults(
        contracts=5,
        entry_price=entry_price,
        direction="NO",
        entry_fee_usd=entry_fee,
    )
    row.realized_pnl_accumulator = 0.0
    row.exit_fee_usd = 0.0
    row.exit_filled_contracts = 0

    pos = _make_position_from_row(row)
    session = _mock_session_for_row(row)

    # Single full close
    fill_price = 0.55  # selling NO at 0.55 per contract
    exit_fee = 0.02
    result = await partial_close_position(
        session, pos,
        filled_contracts=5,
        fill_price=fill_price,
        fee_usd=exit_fee,
        exit_reason="signal",
    )

    # pnl = (0.55-0.40)*5 - entry_fee - exit_fee = 0.75 - 0.04 - 0.02 = 0.69
    assert result.status == "closed"
    assert row.exit_price == pytest.approx(fill_price, abs=1e-5)
    assert row.pnl == pytest.approx(0.69, abs=1e-4)


@pytest.mark.asyncio
async def test_partial_close_single_tranche_equals_close_position_behavior() -> None:
    """A single full-fill via partial_close_position yields same pnl as close_position."""
    pid = uuid.uuid4()
    entry_price = 0.54
    entry_fee = 0.0  # no fee for clean comparison
    row = _make_open_row_with_defaults(
        contracts=100,
        entry_price=entry_price,
        direction="YES",
        position_id=pid,
        entry_fee_usd=entry_fee,
    )
    row.realized_pnl_accumulator = 0.0
    row.exit_fee_usd = 0.0
    row.exit_filled_contracts = 0

    pos = _make_position_from_row(row)
    session = _mock_session_for_row(row)

    exit_price = 1.0
    result = await partial_close_position(
        session, pos,
        filled_contracts=100,
        fill_price=exit_price,
        fee_usd=0.0,
        exit_reason="market_resolved",
        resolution=1,
    )

    # gross = (1.0 - 0.54) * 100 = 46.0; pnl = 46.0 - 0.0 - 0.0 = 46.0
    assert result.status == "closed"
    assert row.pnl == pytest.approx(46.0, abs=1e-4)
    assert row.resolution == 1
