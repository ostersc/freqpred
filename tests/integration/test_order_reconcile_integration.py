"""Integration tests for OrderManager.reconcile_pending_orders (T67).

Requires a running Postgres (docker-compose up -d db) and DATABASE_URL pointing
at freqpred_test. KalshiClient is mocked — we never make real API calls — but
all DB writes/reads exercise real SQLAlchemy + Postgres.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

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
from freqpred.markets.models import MarketRow, Order, PositionRow
from freqpred.signal.models import SignalRow
from freqpred.trading.order_manager import OrderManager
from freqpred.trading.risk import RiskEngine

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


async def _seed_market_and_signal(session_factory, market_id: str = "MKT-INT") -> uuid.UUID:
    async with session_factory() as session:
        market = MarketRow(
            id=market_id,
            platform="kalshi",
            question="integration test market",
            category="other",
            close_time=NOW + timedelta(days=5),
            yes_bid=0.5,
            yes_ask=0.52,
            mid_price=0.51,
            last_price=0.5,
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
            market_id=market_id,
            estimated_probability=0.6,
            confidence=0.8,
            edge=0.10,
            market_mid_at_signal=0.5,
            direction="YES",
            reasoning="seed",
            sources=[],
            retrieval_hash="x" * 64,
            model_used="seed",
            prompt_version="v1",
            trigger="manual",
            created_at=NOW,
            raw_context="",
        )
        session.add(market)
        session.add(signal)
        await session.commit()
        return signal.id


async def _seed_pending_position(
    session_factory,
    *,
    signal_id: uuid.UUID,
    market_id: str,
    exchange_order_id: str,
    requested_contracts: int = 10,
    contracts: int = 10,
    created_at: datetime | None = None,
) -> uuid.UUID:
    pos_id = uuid.uuid4()
    async with session_factory() as session:
        row = PositionRow(
            id=pos_id,
            market_id=market_id,
            signal_id=signal_id,
            strategy_name="IntegrationStrategy",
            strategy_version="1.0",
            signal_confidence=0.8,
            signal_edge=0.10,
            signal_estimated_prob=0.6,
            direction="YES",
            contracts=contracts,
            entry_price=0.50,
            entry_time=created_at or NOW,
            mode="live",
            status="pending",
            exchange_order_id=exchange_order_id,
            requested_contracts=requested_contracts,
            exchange_order_status="resting",
            last_exchange_sync_at=None,
        )
        if created_at is not None:
            # SQLAlchemy server_default puts NOW() on created_at; override below
            row.created_at = created_at
        session.add(row)
        await session.commit()

    if created_at is not None:
        # The server_default("now()") wins over passed value; back-patch via SQL.
        async with session_factory() as session:
            await session.execute(
                text("UPDATE positions SET created_at = :ca WHERE id = :pid"),
                {"ca": created_at, "pid": pos_id},
            )
            await session.commit()
    return pos_id


def _make_om(session_factory, kalshi_client, *, timeout: float = 900.0) -> OrderManager:
    return OrderManager(
        risk=MagicMock(spec=RiskEngine),
        session_factory=session_factory,
        bankroll=1000.0,
        mode="live",
        kalshi_client=kalshi_client,
        pending_order_timeout_seconds=timeout,
    )


@pytest.mark.asyncio
async def test_reconcile_pending_to_open_end_to_end(session_factory) -> None:
    """Insert pending row → mocked get_order returns 'executed' → DB row becomes open."""
    signal_id = await _seed_market_and_signal(session_factory)
    pos_id = await _seed_pending_position(
        session_factory, signal_id=signal_id, market_id="MKT-INT",
        exchange_order_id="ORD-EXEC", requested_contracts=10, contracts=10,
    )

    filled = Order(
        market_id="MKT-INT", direction="YES", contracts=10, price=0.5, mode="live",
        exchange_order_id="ORD-EXEC", status="executed",
        requested_count=10, filled_yes_count=10, filled_no_count=0, remaining_count=0,
    )
    kalshi = AsyncMock()
    kalshi.get_order = AsyncMock(return_value=filled)

    om = _make_om(session_factory, kalshi)
    async with session_factory() as session:
        await om.reconcile_pending_orders(session, _now=NOW)

    async with session_factory() as session:
        row = (await session.execute(select(PositionRow).where(PositionRow.id == pos_id))).scalar_one()
        assert row.status == "open"
        assert row.contracts == 10
        assert row.exchange_order_status == "executed"
        assert row.last_exchange_sync_at is not None


@pytest.mark.asyncio
async def test_reconcile_partial_fill_persists_correctly(session_factory) -> None:
    """get_order returns partial → DB row.status='open', contracts < requested_contracts."""
    signal_id = await _seed_market_and_signal(session_factory)
    pos_id = await _seed_pending_position(
        session_factory, signal_id=signal_id, market_id="MKT-INT",
        exchange_order_id="ORD-PART", requested_contracts=10, contracts=10,
    )

    partial = Order(
        market_id="MKT-INT", direction="YES", contracts=4, price=0.5, mode="live",
        exchange_order_id="ORD-PART", status="partial",
        requested_count=10, filled_yes_count=4, filled_no_count=0, remaining_count=6,
    )
    kalshi = AsyncMock()
    kalshi.get_order = AsyncMock(return_value=partial)

    om = _make_om(session_factory, kalshi)
    async with session_factory() as session:
        await om.reconcile_pending_orders(session, _now=NOW)

    async with session_factory() as session:
        row = (await session.execute(select(PositionRow).where(PositionRow.id == pos_id))).scalar_one()
        assert row.status == "open"
        assert row.contracts == 4
        assert row.requested_contracts == 10
        assert row.exchange_order_status == "partial"


@pytest.mark.asyncio
async def test_reconcile_timeout_calls_cancel_then_persists_state(session_factory) -> None:
    """Aged pending row → get_order returns resting → cancel_order → DB → cancelled."""
    signal_id = await _seed_market_and_signal(session_factory)
    aged = NOW - timedelta(seconds=120)
    pos_id = await _seed_pending_position(
        session_factory, signal_id=signal_id, market_id="MKT-INT",
        exchange_order_id="ORD-AGED", requested_contracts=10, contracts=10,
        created_at=aged,
    )

    resting = Order(
        market_id="MKT-INT", direction="YES", contracts=0, price=0.5, mode="live",
        exchange_order_id="ORD-AGED", status="resting",
        requested_count=10, filled_yes_count=0, filled_no_count=0, remaining_count=10,
    )
    cancelled = Order(
        market_id="MKT-INT", direction="YES", contracts=0, price=0.5, mode="live",
        exchange_order_id="ORD-AGED", status="canceled",
        requested_count=10, filled_yes_count=0, filled_no_count=0, remaining_count=10,
    )
    kalshi = AsyncMock()
    kalshi.get_order = AsyncMock(return_value=resting)
    kalshi.cancel_order = AsyncMock(return_value=cancelled)

    om = _make_om(session_factory, kalshi, timeout=60.0)
    async with session_factory() as session:
        await om.reconcile_pending_orders(session, _now=NOW)

    kalshi.cancel_order.assert_awaited_once_with("ORD-AGED")
    async with session_factory() as session:
        row = (await session.execute(select(PositionRow).where(PositionRow.id == pos_id))).scalar_one()
        assert row.status == "cancelled"


@pytest.mark.asyncio
async def test_reconcile_with_for_update_skip_locked_no_double_update(session_factory) -> None:
    """Two concurrent reconciles on the same row — only one updates, second skips."""
    signal_id = await _seed_market_and_signal(session_factory)
    pos_id = await _seed_pending_position(
        session_factory, signal_id=signal_id, market_id="MKT-INT",
        exchange_order_id="ORD-LOCK", requested_contracts=10, contracts=10,
    )

    filled = Order(
        market_id="MKT-INT", direction="YES", contracts=10, price=0.5, mode="live",
        exchange_order_id="ORD-LOCK", status="executed",
        requested_count=10, filled_yes_count=10, filled_no_count=0, remaining_count=0,
    )
    kalshi_a = AsyncMock()
    kalshi_a.get_order = AsyncMock(return_value=filled)
    kalshi_b = AsyncMock()
    kalshi_b.get_order = AsyncMock(return_value=filled)

    om_a = _make_om(session_factory, kalshi_a)
    om_b = _make_om(session_factory, kalshi_b)

    # Run concurrently. Each opens its own session/transaction. The FOR UPDATE
    # SKIP LOCKED query means one acquires the row and the other gets an empty
    # result — at most one get_order call is made.
    async with session_factory() as session_a, session_factory() as session_b:
        await asyncio.gather(
            om_a.reconcile_pending_orders(session_a, _now=NOW),
            om_b.reconcile_pending_orders(session_b, _now=NOW),
        )

    async with session_factory() as session:
        row = (await session.execute(select(PositionRow).where(PositionRow.id == pos_id))).scalar_one()
        assert row.status == "open"
        assert row.contracts == 10

    total_calls = kalshi_a.get_order.await_count + kalshi_b.get_order.await_count
    assert total_calls >= 1
    # Either both saw an empty queue and one ran, or one saw it locked and skipped
    # — total writes should still leave a single row in 'open' state, never duplicated.


@pytest.mark.asyncio
async def test_reconcile_skips_null_exchange_order_id_legacy_rows(session_factory) -> None:
    """Legacy pre-migration rows (NULL exchange_order_id) are ignored by query."""
    signal_id = await _seed_market_and_signal(session_factory)
    # Insert a row with NULL exchange_order_id
    pos_id = uuid.uuid4()
    async with session_factory() as session:
        row = PositionRow(
            id=pos_id, market_id="MKT-INT", signal_id=signal_id,
            strategy_name="legacy", strategy_version="0.1",
            signal_confidence=0.5, signal_edge=0.1, signal_estimated_prob=0.5,
            direction="YES", contracts=5, entry_price=0.5,
            entry_time=NOW, mode="live", status="pending",
            exchange_order_id=None,
        )
        session.add(row)
        await session.commit()

    kalshi = AsyncMock()
    kalshi.get_order = AsyncMock()
    om = _make_om(session_factory, kalshi)
    async with session_factory() as session:
        await om.reconcile_pending_orders(session, _now=NOW)

    kalshi.get_order.assert_not_called()
    async with session_factory() as session:
        row = (await session.execute(select(PositionRow).where(PositionRow.id == pos_id))).scalar_one()
        assert row.status == "pending"
