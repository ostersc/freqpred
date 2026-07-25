"""Integration tests for prompt-version cohorts in edge-band calibration.

`refresh_edge_calibration_scores` had no test coverage at all before migration
0059 changed its grouping, and the reader/writer contract here is easy to get
subtly wrong in ways unit tests with mocked sessions cannot catch:

  * the writer now emits four cells per record ({global, series} x {all-versions,
    this-version}), so the all-versions rollups must come out numerically
    IDENTICAL to the pre-0059 behaviour — a regression there silently changes
    every assessment;
  * the reader must prefer the signal's own cohort but fall back to the
    all-versions rollup, including for rows written before 0059 existed;
  * `this_direction_all_bands` merges every band for one direction, and if it
    does not pin the same prompt-version scope as the chosen cell it sums
    version rows *and* rollups, which cover overlapping signals — double
    counting the same history.

Requires a running Postgres (docker-compose up -d db) and DATABASE_URL
pointing at freqpred_test.
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
from freqpred.markets.models import Market, MarketRow
from freqpred.metrics.assessment import _load_edge_band_calibration
from freqpred.metrics.calibration import refresh_edge_calibration_scores
from freqpred.metrics.models import EdgeCalibrationScoreRow
from freqpred.signal.models import Signal, SignalRow

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test"
)
NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)
SERIES = "KXCOHORT"
CURRENT_V = "signal-v11"
OLD_V = "signal-v4"


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


def _market_row(mid: str, result: str) -> MarketRow:
    return MarketRow(
        id=mid, platform="kalshi", question=f"q {mid}", category="politics",
        close_time=NOW - timedelta(days=1), yes_bid=0.4, yes_ask=0.6, mid_price=0.5,
        volume_24h=100.0, open_interest=100.0, last_fetched_at=NOW,
        price_updated_at=NOW, metadata_fetched_at=NOW, status="finalized",
        result=result, series_ticker=SERIES,
    )


def _signal_row(mid: str, version: str, direction: str) -> SignalRow:
    # edge 0.20 -> the "15-40" band for every seeded signal.
    return SignalRow(
        id=uuid.uuid4(), market_id=mid, estimated_probability=0.75, confidence=0.8,
        edge=0.20, market_mid_at_signal=0.5, market_ask_at_signal=0.60,
        direction=direction, reasoning="seed", sources=[], retrieval_hash="x" * 64,
        model_used="seed-model", prompt_version=version, trigger="scheduled",
        created_at=NOW - timedelta(days=2), raw_context="",
    )


async def _seed(session, *, n_current: int, n_old: int, current_wins: int) -> None:
    """Seed resolved YES signals: `n_current` on CURRENT_V (of which
    `current_wins` resolve yes) and `n_old` on OLD_V (all losing)."""
    for i in range(n_current):
        mid = f"MKT-CUR-{i}"
        session.add(_market_row(mid, "yes" if i < current_wins else "no"))
        session.add(_signal_row(mid, CURRENT_V, "YES"))
    for i in range(n_old):
        mid = f"MKT-OLD-{i}"
        session.add(_market_row(mid, "no"))
        session.add(_signal_row(mid, OLD_V, "YES"))
    await session.flush()


class TestWriterEmitsCohorts:
    @pytest.mark.asyncio
    async def test_rollup_is_unchanged_and_cohorts_are_added(self, session_factory) -> None:
        """The all-versions rollup must still describe EVERY signal, and version
        cells must partition it — not duplicate or replace it."""
        async with session_factory() as s:
            await _seed(s, n_current=20, n_old=20, current_wins=20)
            await refresh_edge_calibration_scores(s)
            rows = (await s.execute(select(EdgeCalibrationScoreRow))).scalars().all()

        rollup = [r for r in rows if r.prompt_version is None and r.series_ticker == SERIES]
        assert len(rollup) == 1
        # 40 signals total, 20 winners -> the pre-0059 answer, unchanged.
        assert rollup[0].n_signals == 40
        assert rollup[0].hit_rate == pytest.approx(0.5)

        cohorts = {
            r.prompt_version: r
            for r in rows
            if r.prompt_version is not None and r.series_ticker == SERIES
        }
        assert set(cohorts) == {CURRENT_V, OLD_V}
        # The cohorts separate what the rollup averages away.
        assert cohorts[CURRENT_V].hit_rate == pytest.approx(1.0)
        assert cohorts[OLD_V].hit_rate == pytest.approx(0.0)
        assert cohorts[CURRENT_V].n_signals + cohorts[OLD_V].n_signals == rollup[0].n_signals

    @pytest.mark.asyncio
    async def test_thin_cohort_is_not_written(self, session_factory) -> None:
        """A cohort below the render gate would never be shown to the assessor;
        writing it would multiply this table by every version ever run."""
        async with session_factory() as s:
            await _seed(s, n_current=20, n_old=3, current_wins=10)
            await refresh_edge_calibration_scores(s)
            rows = (await s.execute(select(EdgeCalibrationScoreRow))).scalars().all()

        versions = {r.prompt_version for r in rows if r.prompt_version is not None}
        assert CURRENT_V in versions
        assert OLD_V not in versions
        # ...but the thin cohort's signals still count toward the rollup.
        rollup = next(r for r in rows if r.prompt_version is None and r.series_ticker == SERIES)
        assert rollup.n_signals == 23


class TestReaderPrefersCohort:
    def _signal(self, version: str, direction: str = "YES") -> Signal:
        return Signal(
            id=str(uuid.uuid4()), market_id="MKT-CUR-0", estimated_probability=0.75,
            confidence=0.8, edge=0.20, market_mid_at_signal=0.5, direction=direction,
            reasoning="", sources=[], retrieval_hash="x" * 64, model_used="m",
            prompt_version=version, trigger="scheduled", created_at=NOW, raw_context="",
        )

    def _market(self) -> Market:
        return Market(
            id="MKT-CUR-0", platform="kalshi", question="q", category="politics",
            close_time=NOW + timedelta(days=2), yes_bid=0.4, yes_ask=0.6, mid_price=0.5,
            volume_24h=100.0, open_interest=100.0, last_fetched_at=NOW,
            price_updated_at=NOW, metadata_fetched_at=NOW, series_ticker=SERIES,
        )

    @pytest.mark.asyncio
    async def test_uses_own_cohort_not_the_pooled_rollup(self, session_factory) -> None:
        """The whole point of 0059: a v11 signal must be judged on v11 history,
        not on a pool half-composed of a version production no longer runs."""
        async with session_factory() as s:
            await _seed(s, n_current=20, n_old=20, current_wins=20)
            await refresh_edge_calibration_scores(s)
            block = await _load_edge_band_calibration(s, self._signal(CURRENT_V), self._market())

        assert block is not None
        assert block["cohort_prompt_version"] == CURRENT_V
        # v11 cohort hit 1.0; the pooled rollup would have said 0.5.
        assert block["same_direction_only"]["hit_rate"] == pytest.approx(1.0)
        assert block["same_direction_only"]["profit_edge_vs_price"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_falls_back_to_rollup_for_an_unknown_version(self, session_factory) -> None:
        """A freshly shipped signal prompt has no cohort yet; it must still get
        the pooled history rather than an empty block."""
        async with session_factory() as s:
            await _seed(s, n_current=20, n_old=20, current_wins=20)
            await refresh_edge_calibration_scores(s)
            block = await _load_edge_band_calibration(s, self._signal("signal-v99"), self._market())

        assert block is not None
        assert block["cohort_prompt_version"] == "all_versions_fallback"
        assert block["same_direction_only"]["hit_rate"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_direction_all_bands_does_not_double_count(self, session_factory) -> None:
        """If the all-bands merge fails to pin the cohort scope it sums the
        version rows AND the rollups, which cover the same signals twice."""
        async with session_factory() as s:
            await _seed(s, n_current=20, n_old=20, current_wins=20)
            await refresh_edge_calibration_scores(s)
            block = await _load_edge_band_calibration(s, self._signal(CURRENT_V), self._market())

        # Every seeded signal is YES in the 15-40 band, so this direction's
        # all-bands total is exactly the v11 cohort: 20, never 40 or 60.
        assert block["this_direction_all_bands"]["n_signals"] == 20

    @pytest.mark.asyncio
    async def test_pre_0059_rows_still_resolve(self, session_factory) -> None:
        """Rows written before the column existed carry prompt_version NULL. They
        must remain readable as the all-versions rollup rather than being skipped."""
        async with session_factory() as s:
            await _seed(s, n_current=20, n_old=20, current_wins=20)
            await refresh_edge_calibration_scores(s)
            # Simulate a pre-migration table: only NULL-version rows survive.
            await s.execute(
                text("DELETE FROM edge_calibration_scores WHERE prompt_version IS NOT NULL")
            )
            block = await _load_edge_band_calibration(s, self._signal(CURRENT_V), self._market())

        assert block is not None
        assert block["cohort_prompt_version"] == "all_versions_fallback"
        assert block["same_direction_only"]["n_signals"] == 40
