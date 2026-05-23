"""Integration tests for the risk engine treating pending orders as committed exposure (T67).

Requires a running Postgres (docker-compose up -d db) and DATABASE_URL pointing
at freqpred_test.
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    "freqpred_test" not in os.environ.get("DATABASE_URL", ""),
    reason="Integration tests require DATABASE_URL pointing to freqpred_test",
)

from freqpred.config import RiskConfig
from freqpred.db import Base, make_engine, make_session_factory
from freqpred.markets.models import MarketRow, PositionRow
from freqpred.signal.models import Signal, SignalRow
from freqpred.trading.risk import RiskEngine

import freqpred.alerts.models      # noqa: F401
import freqpred.ingestion.models   # noqa: F401
import freqpred.llm.models         # noqa: F401
import freqpred.markets.models     # noqa: F401
import freqpred.metrics.models     # noqa: F401
import freqpred.rag.models         # noqa: F401
import freqpred.runtime.models     # noqa: F401
import freqpred.signal.models      # noqa: F401
import freqpred.strategy.models    # noqa: F401

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


def _make_signal(market_id: str = "MKT-INT") -> Signal:
    return Signal(
        id=str(uuid.uuid4()),
        market_id=market_id,
        estimated_probability=0.6,
        confidence=0.8,
        edge=0.20,
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


async def _seed_market_and_signal(session_factory, market_id: str = "MKT-INT") -> uuid.UUID:
    sig_id = uuid.uuid4()
    async with session_factory() as session:
        session.add(MarketRow(
            id=market_id, platform="kalshi", question="risk integration",
            category="other", close_time=NOW + timedelta(days=5),
            yes_bid=0.5, yes_ask=0.52, mid_price=0.51, last_price=0.5,
            volume_24h=100.0, volume_total=200.0, open_interest=50.0,
            yes_bid_size=10.0, yes_ask_size=10.0,
            last_fetched_at=NOW, price_updated_at=NOW, metadata_fetched_at=NOW,
            metadata_={},
        ))
        session.add(SignalRow(
            id=sig_id, market_id=market_id,
            estimated_probability=0.6, confidence=0.8, edge=0.2,
            market_mid_at_signal=0.5, direction="YES",
            reasoning="seed", sources=[], retrieval_hash="x" * 64,
            model_used="seed", prompt_version="v1", trigger="manual",
            created_at=NOW, raw_context="",
        ))
        await session.commit()
    return sig_id


async def _seed_position(
    session_factory, *, signal_id: uuid.UUID, market_id: str = "MKT-INT",
    status: str = "open", contracts: int = 10,
    requested_contracts: int | None = None, entry_price: float = 0.5,
) -> uuid.UUID:
    pos_id = uuid.uuid4()
    async with session_factory() as session:
        session.add(PositionRow(
            id=pos_id, market_id=market_id, signal_id=signal_id,
            strategy_name="IntTest", strategy_version="1.0",
            signal_confidence=0.8, signal_edge=0.2, signal_estimated_prob=0.6,
            direction="YES", contracts=contracts, entry_price=entry_price,
            entry_time=NOW, mode="live", status=status,
            exchange_order_id=f"ORD-{pos_id.hex[:8]}",
            requested_contracts=requested_contracts,
        ))
        await session.commit()
    return pos_id


@pytest.mark.asyncio
async def test_risk_engine_blocks_when_pending_plus_open_exceeds_global_cap(
    session_factory,
) -> None:
    """Seeded mix of open + pending rows hitting the cap → next entry blocked."""
    sig_id = await _seed_market_and_signal(session_factory, "MKT-A")
    # Need additional markets for distinct positions
    for i in range(4):
        await _seed_market_and_signal(session_factory, f"MKT-EXTRA-{i}")
    # 3 open + 2 pending = 5 active; max=5 → blocked
    await _seed_position(session_factory, signal_id=sig_id, market_id="MKT-A", status="open")
    await _seed_position(session_factory, signal_id=sig_id, market_id="MKT-EXTRA-0", status="open")
    await _seed_position(session_factory, signal_id=sig_id, market_id="MKT-EXTRA-1", status="open")
    await _seed_position(
        session_factory, signal_id=sig_id, market_id="MKT-EXTRA-2",
        status="pending", contracts=10, requested_contracts=10,
    )
    await _seed_position(
        session_factory, signal_id=sig_id, market_id="MKT-EXTRA-3",
        status="pending", contracts=10, requested_contracts=10,
    )

    engine = RiskEngine(RiskConfig(
        max_position_pct=0.10, max_daily_loss_pct=0.15,
        max_total_exposure_pct=0.80, min_edge_floor=0.05,
        max_open_positions=5,
    ))
    signal = _make_signal("MKT-A")

    async with session_factory() as session:
        decision = await engine.check_position(
            session, signal, requested_size=10.0, bankroll=1000.0,
            market_id="MKT-A", max_market_exposure=500.0, mode="live",
        )

    assert decision.allowed is False
    assert "active positions 5" in decision.reason
    assert "pending=2" in decision.reason


@pytest.mark.asyncio
async def test_per_market_cap_blocks_when_pending_present(session_factory) -> None:
    """A pending position on the same market consumes the per-market cap."""
    sig_id = await _seed_market_and_signal(session_factory, "MKT-INT")
    # Pending 10 contracts at $0.50 = $5 committed; per-market cap = $5
    await _seed_position(
        session_factory, signal_id=sig_id, market_id="MKT-INT",
        status="pending", contracts=10, requested_contracts=10, entry_price=0.5,
    )

    engine = RiskEngine(RiskConfig(
        max_position_pct=0.50, max_daily_loss_pct=0.15,
        max_total_exposure_pct=0.80, min_edge_floor=0.05,
        max_open_positions=20,
    ))
    signal = _make_signal("MKT-INT")

    async with session_factory() as session:
        decision = await engine.check_position(
            session, signal, requested_size=10.0, bankroll=1000.0,
            market_id="MKT-INT", max_market_exposure=5.0, mode="live",
        )

    assert decision.allowed is False
    assert "MKT-INT" in decision.reason
    assert "pending=5" in decision.reason
