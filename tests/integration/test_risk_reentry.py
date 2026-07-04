"""Integration tests for the loss re-entry guard's actual DB query.

Unit tests in tests/unit/test_risk.py mock session.execute() to return a
canned count, so they can't catch a regression in the WHERE clause itself
(e.g. an exit_reason missing from an allowlist). These tests run the real
query against Postgres to prove the guard blocks on pnl < 0 regardless of
exit_reason.

Requires a running Postgres (docker-compose up -d db) and DATABASE_URL
pointing at freqpred_test.
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

import freqpred.alerts.models  # noqa: F401
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.metrics.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.runtime.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
import freqpred.strategy.models  # noqa: F401
from freqpred.config import RiskConfig
from freqpred.db import Base, make_engine, make_session_factory
from freqpred.markets.models import Market, MarketRow, PositionRow
from freqpred.signal.models import Signal, SignalRow
from freqpred.trading.risk import RiskEngine

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test",
)
NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
MARKET_ID = "MKT-REENTRY"


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


def _make_signal() -> Signal:
    return Signal(
        id=str(uuid.uuid4()), market_id=MARKET_ID,
        estimated_probability=0.6, confidence=0.8, edge=0.20,
        market_mid_at_signal=0.5, direction="YES", reasoning="seed",
        sources=[], retrieval_hash="x" * 64, model_used="seed",
        prompt_version="v1", trigger="manual", created_at=NOW, raw_context="",
    )


def _make_market() -> Market:
    return Market(
        id=MARKET_ID, platform="kalshi", question="reentry integration",
        category="other", close_time=NOW + timedelta(days=5),
        yes_bid=0.5, yes_ask=0.52, mid_price=0.51, last_price=0.5,
        volume_24h=100.0, volume_total=200.0, open_interest=50.0,
        yes_bid_size=10.0, yes_ask_size=10.0,
        last_fetched_at=NOW, price_updated_at=NOW, metadata_fetched_at=NOW,
        metadata={},
    )


async def _seed_market_and_signal(session_factory) -> uuid.UUID:
    sig_id = uuid.uuid4()
    async with session_factory() as session:
        session.add(MarketRow(
            id=MARKET_ID, platform="kalshi", question="reentry integration",
            category="other", close_time=NOW + timedelta(days=5),
            yes_bid=0.5, yes_ask=0.52, mid_price=0.51, last_price=0.5,
            volume_24h=100.0, volume_total=200.0, open_interest=50.0,
            yes_bid_size=10.0, yes_ask_size=10.0,
            last_fetched_at=NOW, price_updated_at=NOW, metadata_fetched_at=NOW,
            metadata_={},
        ))
        session.add(SignalRow(
            id=sig_id, market_id=MARKET_ID,
            estimated_probability=0.6, confidence=0.8, edge=0.2,
            market_mid_at_signal=0.5, direction="YES",
            reasoning="seed", sources=[], retrieval_hash="x" * 64,
            model_used="seed", prompt_version="v1", trigger="manual",
            created_at=NOW, raw_context="",
        ))
        await session.commit()
    return sig_id


async def _seed_closed_position(
    session_factory, *, signal_id: uuid.UUID, exit_reason: str, pnl: float,
) -> None:
    async with session_factory() as session:
        session.add(PositionRow(
            id=uuid.uuid4(), market_id=MARKET_ID, signal_id=signal_id,
            strategy_name="IntTest", strategy_version="1.0",
            signal_confidence=0.8, signal_edge=0.2, signal_estimated_prob=0.6,
            direction="YES", contracts=0, entry_price=0.5,
            entry_time=NOW - timedelta(hours=1), mode="live", status="closed",
            exit_time=NOW - timedelta(minutes=30),
            exit_price=0.5 + pnl / 10.0, exit_reason=exit_reason, pnl=pnl,
        ))
        await session.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_reason",
    [
        "force_exit:algo_exit",
        "market_resolved",
        "reconcile_auto_close",
        "reconcile_drift_manual",
        "custom_exit:some_tag",
    ],
)
async def test_reentry_blocked_for_any_loss_exit_reason(
    session_factory, exit_reason: str,
) -> None:
    """A losing exit blocks re-entry regardless of exit_reason — not just
    'stoploss'/'trailing_stop'/'signal'/'force_exit:manual' as before."""
    sig_id = await _seed_market_and_signal(session_factory)
    await _seed_closed_position(
        session_factory, signal_id=sig_id, exit_reason=exit_reason, pnl=-5.0,
    )

    engine = RiskEngine(RiskConfig(
        max_position_pct=0.10, max_daily_loss_pct=0.15,
        max_total_exposure_pct=0.80, min_edge_floor=0.05,
        max_open_positions=20,
    ))

    async with session_factory() as session:
        decision = await engine.check_position(
            session, _make_signal(), requested_size=10.0, bankroll=1000.0,
            market_id=MARKET_ID, max_market_exposure=500.0, mode="live",
            block_reentry_after_stoploss=True,
        )
        blocked, reason = await engine.pre_signal_gate(
            session, _make_market(), mode="live",
            effective_max_spread=0.10,
            block_reentry_after_stoploss=True,
            stoploss_cooldown_hours=0.0,
        )

    assert decision.allowed is False
    assert MARKET_ID in decision.reason
    assert blocked is True
    assert MARKET_ID in reason


@pytest.mark.asyncio
async def test_reentry_allowed_after_profitable_exit_any_reason(
    session_factory,
) -> None:
    """A profitable exit must not trigger the guard, whatever exit_reason."""
    sig_id = await _seed_market_and_signal(session_factory)
    await _seed_closed_position(
        session_factory, signal_id=sig_id,
        exit_reason="force_exit:algo_exit", pnl=5.0,
    )

    engine = RiskEngine(RiskConfig(
        max_position_pct=0.10, max_daily_loss_pct=0.15,
        max_total_exposure_pct=0.80, min_edge_floor=0.05,
        max_open_positions=20,
    ))

    async with session_factory() as session:
        decision = await engine.check_position(
            session, _make_signal(), requested_size=10.0, bankroll=1000.0,
            market_id=MARKET_ID, max_market_exposure=500.0, mode="live",
            block_reentry_after_stoploss=True,
        )

    assert decision.allowed is True
