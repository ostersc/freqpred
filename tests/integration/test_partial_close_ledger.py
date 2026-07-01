"""Integration tests for T76: ledger.partial_close_position.

Requires running Postgres (docker-compose up -d db) and DATABASE_URL pointing
at freqpred_test. All DB writes/reads use real SQLAlchemy + Postgres.
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select, text

pytestmark = pytest.mark.skipif(
    "freqpred_test" not in os.environ.get("DATABASE_URL", ""),
    reason="Integration tests require DATABASE_URL pointing to freqpred_test",
)

import freqpred.alerts.models  # noqa: F401
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.metrics.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.runtime.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
import freqpred.strategy.models  # noqa: F401
from freqpred.db import Base, make_engine, make_session_factory
from freqpred.markets.models import MarketRow, Position, PositionRow
from freqpred.signal.models import SignalRow
from freqpred.trading.ledger import _row_to_position, partial_close_position

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test",
)
NOW = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def engine():
    eng = make_engine(DATABASE_URL)
    async with eng.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return make_session_factory(engine)


async def _seed_position(
    session_factory,
    *,
    contracts: int = 10,
    entry_price: float = 0.50,
    direction: str = "YES",
    entry_fee_usd: float = 0.05,
) -> tuple[uuid.UUID, Position]:
    mkt_id = f"MKT-PARTIAL-{uuid.uuid4().hex[:8]}"
    pos_id = uuid.uuid4()

    async with session_factory() as session:
        market = MarketRow(
            id=mkt_id,
            platform="kalshi",
            question="partial close test market",
            category="other",
            close_time=NOW + timedelta(days=5),
            yes_bid=0.49,
            yes_ask=0.51,
            mid_price=0.50,
            last_price=0.50,
            volume_24h=100.0,
            volume_total=200.0,
            open_interest=50.0,
            yes_bid_size=10.0,
            yes_ask_size=10.0,
            last_fetched_at=NOW,
            price_updated_at=NOW,
            metadata_fetched_at=NOW,
            metadata_={},
        )
        signal = SignalRow(
            id=uuid.uuid4(),
            market_id=mkt_id,
            estimated_probability=0.60,
            confidence=0.80,
            edge=0.10,
            market_mid_at_signal=0.50,
            direction=direction,
            reasoning="integration seed",
            sources=[],
            retrieval_hash="x" * 64,
            model_used="seed",
            prompt_version="v1",
            trigger="manual",
            created_at=NOW,
            raw_context="",
        )
        session.add(market)
        await session.flush()
        session.add(signal)
        await session.flush()

        row = PositionRow(
            id=pos_id,
            market_id=mkt_id,
            signal_id=signal.id,
            strategy_name="IntegrationStrategy",
            strategy_version="1.0",
            signal_confidence=0.80,
            signal_edge=0.10,
            signal_estimated_prob=0.60,
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
        )
        session.add(row)
        await session.commit()

        pos = _row_to_position(row)
    return pos_id, pos


@pytest.mark.asyncio
async def test_partial_close_round_trip_against_real_db(session_factory) -> None:
    """Insert open position; partial close; verify contracts/accumulator; second close → closed."""
    _pos_id, pos = await _seed_position(
        session_factory, contracts=10, entry_price=0.50, entry_fee_usd=0.05
    )

    # First partial close: 6 of 10 at 0.55, fee=0.01
    async with session_factory() as session:
        result = await partial_close_position(
            session, pos,
            filled_contracts=6,
            fill_price=0.55,
            fee_usd=0.01,
            exit_reason="stoploss",
            exit_order_id="exit-1",
            exit_requested_contracts=10,
        )

    assert result.status == "open"
    assert result.contracts == 4

    # Reload from DB and verify accumulator state
    async with session_factory() as session:
        db_row = (
            await session.execute(
                select(PositionRow).where(PositionRow.id == _pos_id)
            )
        ).scalar_one()
        assert db_row.contracts == 4
        # gross = (0.55 - 0.50) * 6 = 0.30
        assert abs(db_row.realized_pnl_accumulator - 0.30) < 1e-5
        assert abs(db_row.exit_fee_usd - 0.01) < 1e-5
        assert db_row.exit_filled_contracts == 6
        assert db_row.exit_order_id == "exit-1"
        assert db_row.status == "open"

    # Reload position from DB (contracts = 4)
    async with session_factory() as session:
        db_row2 = (
            await session.execute(
                select(PositionRow).where(PositionRow.id == _pos_id)
            )
        ).scalar_one()
        pos2 = _row_to_position(db_row2)

    # Final partial close: 4 of 4 at 0.60, fee=0.01 → position closes
    async with session_factory() as session:
        final = await partial_close_position(
            session, pos2,
            filled_contracts=4,
            fill_price=0.60,
            fee_usd=0.01,
            exit_reason="stoploss",
            exit_order_id="exit-2",
            exit_requested_contracts=4,
        )

    assert final.status == "closed"
    assert final.contracts == 0

    # Verify DB row
    async with session_factory() as session:
        closed_row = (
            await session.execute(
                select(PositionRow).where(PositionRow.id == _pos_id)
            )
        ).scalar_one()

    # accumulator = 0.30 + (0.60 - 0.50) * 4 = 0.30 + 0.40 = 0.70
    # weighted_avg_exit = (0.70 / 10) + 0.50 = 0.57
    # pnl = 0.70 - 0.05 (entry) - 0.02 (exit fees) = 0.63
    assert closed_row.status == "closed"
    assert abs(closed_row.exit_price - 0.57) < 1e-5
    assert abs(closed_row.pnl - 0.63) < 1e-4
    assert closed_row.exit_filled_contracts == 10
    assert closed_row.exit_time is not None


@pytest.mark.asyncio
async def test_partial_fill_then_market_resolution_combines_pnl_correctly(
    session_factory,
) -> None:
    """Partial IOC fill at $0.55 for 6/10; residual 4 settle at $1.00; weighted-avg P&L correct."""
    _pos_id, pos = await _seed_position(
        session_factory, contracts=10, entry_price=0.50, entry_fee_usd=0.05
    )

    # Leg 1: IOC partial fill — 6 contracts at $0.55
    async with session_factory() as session:
        after_partial = await partial_close_position(
            session, pos,
            filled_contracts=6,
            fill_price=0.55,
            fee_usd=0.01,
            exit_reason="stoploss",
        )

    assert after_partial.status == "open"
    assert after_partial.contracts == 4

    async with session_factory() as session:
        db_row = (
            await session.execute(
                select(PositionRow).where(PositionRow.id == _pos_id)
            )
        ).scalar_one()
        pos2 = _row_to_position(db_row)

    # Leg 2: market resolves YES at $1.00 — the 4 residual settle at $1.00
    async with session_factory() as session:
        final = await partial_close_position(
            session, pos2,
            filled_contracts=4,
            fill_price=1.0,
            fee_usd=0.0,
            exit_reason="market_resolved",
            resolution=1,
        )

    assert final.status == "closed"

    async with session_factory() as session:
        closed_row = (
            await session.execute(
                select(PositionRow).where(PositionRow.id == _pos_id)
            )
        ).scalar_one()

    # accumulator = (0.55-0.50)*6 + (1.0-0.50)*4 = 0.30 + 2.00 = 2.30
    # weighted_avg_exit = (2.30 / 10) + 0.50 = 0.73
    # pnl = 2.30 - 0.05 (entry) - 0.01 (exit fee leg 1) = 2.24
    assert abs(closed_row.exit_price - 0.73) < 1e-5
    assert abs(closed_row.pnl - 2.24) < 1e-4
    assert closed_row.resolution == 1


@pytest.mark.asyncio
async def test_force_exit_residual_after_partial_fill_against_real_session(
    session_factory,
) -> None:
    """Partial fill leaves residual open; force_exit on the residual fully closes it."""
    from unittest.mock import AsyncMock, MagicMock

    from freqpred.markets.models import Order
    from freqpred.trading.order_manager import OrderManager
    from freqpred.trading.risk import RiskEngine

    _pos_id, pos = await _seed_position(
        session_factory, contracts=10, entry_price=0.50, direction="YES",
        entry_fee_usd=0.05
    )

    # Simulate first partial exit (6 of 10 at 0.55) via ledger directly
    async with session_factory() as session:
        after_partial = await partial_close_position(
            session, pos,
            filled_contracts=6,
            fill_price=0.55,
            fee_usd=0.01,
            exit_reason="stoploss",
            exit_order_id="exit-order-1",
            exit_requested_contracts=10,
        )

    assert after_partial.status == "open"
    assert after_partial.contracts == 4

    # Now operator calls force_exit — it should sell the residual 4 contracts
    filled_terminal = Order(
        market_id=pos.market_id,
        direction="YES",
        contracts=4,
        price=0.60,
        mode="live",
        status="executed",
        exchange_order_id="exit-order-2",
        filled_yes_count=4,
        requested_count=4,
        fee_usd=0.01,
    )

    kalshi_client = MagicMock()
    kalshi_client.place_order = AsyncMock(return_value=filled_terminal)
    kalshi_client.get_positions = AsyncMock(return_value=[
        MagicMock(market_id=pos.market_id, direction="YES", contracts=4),
    ])

    import os as _os
    _os.environ["LIVE_TRADING_ENABLED"] = "true"
    try:
        risk = MagicMock(spec=RiskEngine)
        om = OrderManager(
            risk=risk,
            session_factory=session_factory,
            bankroll=1000.0,
            mode="live",
            kalshi_client=kalshi_client,
        )

        result = await om.force_exit(str(_pos_id))
    finally:
        _os.environ.pop("LIVE_TRADING_ENABLED", None)

    # Either position is fully closed (all 4 contracts filled) or still open (partial)
    # In this test the mock always fills fully → should be closed
    assert result.status == "closed"

    # Verify DB final state
    async with session_factory() as session:
        closed_row = (
            await session.execute(
                select(PositionRow).where(PositionRow.id == _pos_id)
            )
        ).scalar_one()

    # accumulator from prior partial: (0.55-0.50)*6 = 0.30
    # then force_exit adds (0.60-0.50)*4 = 0.40 → total = 0.70
    # weighted_avg_exit = (0.70/10) + 0.50 = 0.57
    assert closed_row.status == "closed"
    assert abs(closed_row.exit_price - 0.57) < 1e-5
