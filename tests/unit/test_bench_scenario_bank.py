"""Unit tests for freqpred/bench/scenarios.py — scenario assembly + contamination.

All timestamps are frozen scenario data compared against frozen cutoffs —
no wall-clock dependence.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import freqpred.ingestion.models  # noqa: F401 — registers mappers
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.bench.scenarios import (
    Scenario,
    apply_extraction,
    build_fixture_scenarios,
    filter_contaminated,
    scenario_from_fixture,
    scenario_from_signal,
)
from freqpred.replay.engine import compute_expectations, render_prompt_from_inputs
from freqpred.replay.fixtures import (
    FixtureDocument,
    FixtureInputs,
    FixtureMarket,
    ReplayFixture,
)

FROZEN_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 3, 1, tzinfo=UTC)


def _llm_response(direction: str = "YES", probability: float = 0.7) -> str:
    return json.dumps(
        {
            "prior": 0.6,
            "prior_basis": "basis",
            "updates_applied": [],
            "posterior": probability,
            "probability": probability,
            "confidence": 0.8,
            "direction": direction,
            "reasoning": "reasoning",
        }
    )


def _fake_signal(
    direction: str = "YES",
    probability: float = 0.7,
    mid: float = 0.5,
    ask: float | None = 0.52,
    edge: float = 0.18,
    raw_context: str = "THE EXACT STORED PROMPT",
):
    return SimpleNamespace(
        id="sig-1",
        market_id="MKT-1",
        direction=direction,
        estimated_probability=probability,
        edge=edge,
        market_mid_at_signal=mid,
        market_ask_at_signal=ask,
        raw_context=raw_context,
        created_at=FROZEN_NOW,
    )


def _fake_market(result: str | None = "yes", close_time: datetime | None = None):
    return SimpleNamespace(
        id="MKT-1",
        question="Will the event happen?",
        close_time=close_time or FROZEN_NOW,
        result=result,
    )


def _fake_llm_row(direction: str = "YES", probability: float = 0.7):
    return SimpleNamespace(
        model_used="claude-sonnet-4-6",
        response=_llm_response(direction, probability),
        tokens_input=1000,
        tokens_output=200,
        cost_usd=0.01,
        latency_ms=1500,
    )


# ---------------------------------------------------------------------------
# scenario_from_signal — model mode
# ---------------------------------------------------------------------------


def test_model_mode_replays_raw_context_verbatim() -> None:
    """The candidate must receive byte-for-byte what the incumbent saw."""
    signal = _fake_signal(raw_context="THE EXACT STORED PROMPT — bytes matter")
    scenario = scenario_from_signal(signal, _fake_market(), _fake_llm_row())
    assert scenario is not None
    assert scenario.prompt == "THE EXACT STORED PROMPT — bytes matter"
    assert scenario.incumbent.model == "claude-sonnet-4-6"
    assert scenario.outcome == 1.0


def test_scenario_outcome_mapping_and_scalar_rejection() -> None:
    assert scenario_from_signal(_fake_signal(), _fake_market("yes"), _fake_llm_row()).outcome == 1.0
    no = scenario_from_signal(
        _fake_signal("NO", 0.3, ask=0.5, edge=0.2), _fake_market("no"),
        _fake_llm_row("NO", 0.3),
    )
    assert no.outcome == 0.0
    # Scalar / unknown results are rejected, never miscategorized as NO.
    assert scenario_from_signal(_fake_signal(), _fake_market(None), _fake_llm_row()) is None
    assert scenario_from_signal(_fake_signal(), _fake_market("scalar"), _fake_llm_row()) is None


def test_price_reconstruction_yes_direction() -> None:
    # YES: yes_ask = stored ask; yes_bid = 2*mid - ask
    scenario = scenario_from_signal(
        _fake_signal("YES", mid=0.50, ask=0.52), _fake_market(), _fake_llm_row()
    )
    assert scenario.yes_ask == 0.52
    assert scenario.yes_bid == 0.48


def test_price_reconstruction_no_direction() -> None:
    # NO: stored ask is the NO-side ask (1 - yes_bid); yes_ask = 2*mid - yes_bid
    scenario = scenario_from_signal(
        _fake_signal("NO", probability=0.3, mid=0.50, ask=0.52, edge=0.18),
        _fake_market("no"),
        _fake_llm_row("NO", 0.3),
    )
    assert scenario.yes_bid == 0.48
    assert scenario.yes_ask == 0.52


def test_unparseable_incumbent_response_returns_none() -> None:
    llm_row = _fake_llm_row()
    llm_row.response = "not json at all"
    assert scenario_from_signal(_fake_signal(), _fake_market(), llm_row) is None


# ---------------------------------------------------------------------------
# scenario_from_fixture — prompt mode
# ---------------------------------------------------------------------------


def _make_fixture() -> ReplayFixture:
    inputs = FixtureInputs(
        now=FROZEN_NOW,
        market=FixtureMarket(
            id="FIX-MKT",
            question="Will the fixture event happen?",
            category="test",
            close_time=FROZEN_NOW + timedelta(days=3),
            open_time=FROZEN_NOW - timedelta(days=4),
            yes_bid=0.40,
            yes_ask=0.44,
            mid_price=0.42,
        ),
        documents=[
            FixtureDocument(
                id="11111111-1111-1111-1111-111111111111",
                source_url="https://example.com/a",
                title="Doc A",
                body="Body about the fixture event.",
                source_type="news",
                source_name="Reuters",
                published_at=FROZEN_NOW - timedelta(days=1),
                fetched_at=FROZEN_NOW,
            )
        ],
        llm_response=_llm_response("YES", 0.7),
    )
    return ReplayFixture(
        name="fixture_scenario",
        inputs=inputs,
        expectations=compute_expectations(inputs, strategy=None),
    )


def test_prompt_mode_rerenders_from_fixture_inputs() -> None:
    """Prompt mode sends the CURRENT template's render, not any stored string."""
    fixture = _make_fixture()
    scenario = scenario_from_fixture(
        fixture,
        outcome=1.0,
        close_time=fixture.inputs.market.close_time,
        incumbent_model="claude-sonnet-4-6",
    )
    assert scenario is not None
    assert scenario.prompt == render_prompt_from_inputs(fixture.inputs)
    assert "Will the fixture event happen?" in scenario.prompt
    assert "Current Date (UTC): 2026-06-01 12:00" in scenario.prompt
    # Recorded under the current template → no drift note.
    assert scenario.notes == []
    # Frozen prices come straight from the fixture market snapshot.
    assert (scenario.yes_bid, scenario.yes_ask) == (0.40, 0.44)


def test_prompt_mode_flags_template_drift() -> None:
    """A fixture recorded under an older template gets an explanatory note."""
    fixture = _make_fixture()
    fixture.expectations.rendered_prompt = "RENDERED UNDER AN OLDER TEMPLATE"
    fixture.expectations.prompt_version = "signal-v0"
    scenario = scenario_from_fixture(
        fixture, outcome=1.0, close_time=fixture.inputs.market.close_time
    )
    assert scenario.notes and "signal-v0" in scenario.notes[0]


# ---------------------------------------------------------------------------
# Contamination filter
# ---------------------------------------------------------------------------


def _scenario_closing(close_time: datetime) -> Scenario:
    signal = _fake_signal()
    return scenario_from_signal(signal, _fake_market("yes", close_time), _fake_llm_row())


def test_excludes_scenarios_at_or_before_training_cutoff() -> None:
    before = _scenario_closing(CUTOFF - timedelta(days=10))
    at = _scenario_closing(CUTOFF)  # boundary counts as contaminated
    after = _scenario_closing(CUTOFF + timedelta(days=10))

    kept, excluded = filter_contaminated([before, at, after], CUTOFF)
    assert kept == [after]
    assert excluded == [before, at]
    assert not after.contaminated


def test_include_contaminated_flags_rows() -> None:
    before = _scenario_closing(CUTOFF - timedelta(days=10))
    after = _scenario_closing(CUTOFF + timedelta(days=10))

    kept, excluded = filter_contaminated(
        [before, after], CUTOFF, include_contaminated=True
    )
    assert excluded == []
    assert kept == [before, after]
    assert before.contaminated is True
    assert after.contaminated is False


async def test_fixture_bank_collapses_same_clock_duplicates(tmp_path) -> None:
    """Two fixtures with the same market AND frozen clock are the same decision
    point — one scenario survives, with a skip note. Distinct signal times on
    one market are kept (sampling/clustering handles them downstream)."""
    from datetime import timedelta
    from unittest.mock import AsyncMock, MagicMock

    from freqpred.replay import save_fixture

    older = _make_fixture()
    older.name = "older"
    older.recorded_at = FROZEN_NOW - timedelta(days=2)
    newer = _make_fixture()
    newer.name = "newer"
    newer.recorded_at = FROZEN_NOW
    save_fixture(older, tmp_path / "older.json")
    save_fixture(newer, tmp_path / "newer.json")

    market_row = SimpleNamespace(
        id="FIX-MKT", status="finalized", result="yes",
        close_time=FROZEN_NOW, question="q",
    )
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value = [market_row]
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    scenarios, skipped, _ = await build_fixture_scenarios(session, tmp_path)
    assert [s.id for s in scenarios] == ["newer"]
    assert any("duplicate of newer" in reason for reason in skipped)


# ---------------------------------------------------------------------------
# Benchmark-time sampling — markets and per-market signals
# ---------------------------------------------------------------------------


def _scen_at(market_id: str, hour: int):
    from datetime import timedelta

    from freqpred.bench.scenarios import ModelOutput, Scenario

    return Scenario(
        id=f"{market_id}-h{hour}",
        source="fixture",
        market_id=market_id,
        market_question="q",
        close_time=FROZEN_NOW,
        outcome=1.0,
        prompt="p",
        incumbent=ModelOutput(
            model="m", prior=0.5, posterior=0.6, confidence=0.7,
            direction="YES", updates_count=0,
        ),
        yes_bid=0.48,
        yes_ask=0.50,
        mid_price=0.49,
        signal_time=FROZEN_NOW + timedelta(hours=hour),
    )


def test_sample_per_market_last_all_and_spread() -> None:
    from freqpred.bench.scenarios import sample_per_market

    series = [_scen_at("M1", h) for h in range(10)] + [_scen_at("M2", h) for h in range(2)]

    last = sample_per_market(series, "last")
    assert sorted(s.id for s in last) == ["M1-h9", "M2-h1"]

    first = sample_per_market(series, "first")
    assert sorted(s.id for s in first) == ["M1-h0", "M2-h0"]

    assert len(sample_per_market(series, "all")) == 12

    spread = sample_per_market(series, "spread:3")
    # M1: first, middle (index 4.5 → banker's round → 4), last; M2 (<=3) all kept.
    assert [s.id for s in spread if s.market_id == "M1"] == ["M1-h0", "M1-h4", "M1-h9"]
    assert [s.id for s in spread if s.market_id == "M2"] == ["M2-h0", "M2-h1"]

    assert [s.id for s in sample_per_market(series, "spread:1") if s.market_id == "M1"] == [
        "M1-h9"
    ]

    import pytest as _pytest

    with _pytest.raises(ValueError):
        sample_per_market(series, "bogus")
    with _pytest.raises(ValueError):
        sample_per_market(series, "spread:0")


def test_sample_markets_random_seeded() -> None:
    from freqpred.bench.scenarios import sample_markets

    scenarios = [_scen_at(f"M{i}", h) for i in range(20) for h in range(3)]

    kept, n_markets = sample_markets(scenarios, 5, seed=7)
    assert n_markets == 5
    kept_markets = {s.market_id for s in kept}
    assert len(kept_markets) == 5
    # Whole markets kept: all 3 signals of each sampled market survive.
    assert len(kept) == 15
    # Deterministic under the same seed; different under another.
    again, _ = sample_markets(scenarios, 5, seed=7)
    assert {s.id for s in again} == {s.id for s in kept}
    other, _ = sample_markets(scenarios, 5, seed=8)
    assert {s.market_id for s in other} != kept_markets

    # No limit (or limit >= markets) keeps everything.
    all_kept, n_all = sample_markets(scenarios, None)
    assert len(all_kept) == 60
    assert n_all == 20


# ---------------------------------------------------------------------------
# T101 — prompt mode must actually extract, or it benchmarks a null change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_mode_renders_extracts_when_a_client_is_supplied(tmp_path) -> None:
    """The wiring that makes a T101 benchmark mean anything.

    A bank that carries full bodies but a harness that never extracts renders
    the raw cut and reports "no change" however good the extractor is — the
    same trap as the 500-char frozen bodies, one layer up.
    """
    from unittest.mock import AsyncMock, MagicMock

    from freqpred.replay import save_fixture
    from freqpred.signal.extractor import DocumentExtract

    fixture = _make_fixture()
    doc = fixture.inputs.documents[0]
    doc.full_body = "A long body with the relevant passage buried well inside it."
    save_fixture(fixture, tmp_path / "f.json")

    market_row = SimpleNamespace(
        id="FIX-MKT", status="finalized", result="yes",
        close_time=FROZEN_NOW, question="q",
    )
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value = [market_row]
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    extractor = AsyncMock(
        return_value={
            doc.id: DocumentExtract(
                document_id=doc.id,
                relevance="direct",
                extract="THE BURIED PASSAGE",
                model_used="claude-haiku-4-5-20251001",
                prompt_version="extract-v1",
            )
        }
    )

    scenarios, _, fixtures_by_id = await build_fixture_scenarios(session, tmp_path)
    # The build itself must never extract — it runs over the whole bank.
    assert "Body about the fixture event" in scenarios[0].prompt

    with patch("freqpred.bench.scenarios.extract_for_documents", new=extractor):
        changed = await apply_extraction(
            session, MagicMock(), scenarios, fixtures_by_id,
            model="claude-haiku-4-5-20251001",
        )

    assert changed == 1
    assert "THE BURIED PASSAGE" in scenarios[0].prompt
    assert "Body about the fixture event" not in scenarios[0].prompt

    # Extraction must read full_body, not the frozen excerpt — otherwise the
    # extractor is handed exactly the text T101 replaces.
    passed_docs = extractor.await_args.args[3]
    assert passed_docs[0].body == doc.full_body


@pytest.mark.asyncio
async def test_prompt_mode_without_a_client_renders_the_raw_cut(tmp_path) -> None:
    """The control side of the experiment, and the default for every other change."""
    from unittest.mock import AsyncMock, MagicMock

    from freqpred.replay import save_fixture

    fixture = _make_fixture()
    fixture.inputs.documents[0].full_body = "Never rendered without a client."
    save_fixture(fixture, tmp_path / "f.json")

    market_row = SimpleNamespace(
        id="FIX-MKT", status="finalized", result="yes",
        close_time=FROZEN_NOW, question="q",
    )
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value = [market_row]
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    extractor = AsyncMock()
    with patch("freqpred.bench.scenarios.extract_for_documents", new=extractor):
        scenarios, _, _ = await build_fixture_scenarios(session, tmp_path)

    extractor.assert_not_awaited()
    assert "Body about the fixture event" in scenarios[0].prompt


@pytest.mark.asyncio
async def test_extraction_is_scoped_to_the_scenarios_it_is_given(tmp_path) -> None:
    """Regression: extraction must bill the sample, not the bank.

    The first cut extracted inside build_fixture_scenarios, before
    sample_markets ran — so `--limit 50` still paid to extract all 311 markets
    in the bank. It burned $3.82 on 2026-08-11 before being caught by watching
    the row count outrun the limit. apply_extraction takes the already-sampled
    list, so the only defence needed is that it touches nothing else.
    """
    from unittest.mock import AsyncMock, MagicMock

    from freqpred.replay import save_fixture

    for i in range(5):
        f = _make_fixture()
        f.name = f"fixture_{i}"
        f.inputs.now = FROZEN_NOW + timedelta(hours=i)
        f.inputs.documents[0].full_body = "Long body " * 80
        f.expectations = compute_expectations(f.inputs, strategy=None)
        save_fixture(f, tmp_path / f"{i}.json")

    market_row = SimpleNamespace(
        id="FIX-MKT", status="finalized", result="yes",
        close_time=FROZEN_NOW, question="q",
    )
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value = [market_row]
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    scenarios, _, fixtures_by_id = await build_fixture_scenarios(session, tmp_path)
    assert len(scenarios) == 5

    # Sampling happens here in the real flow; extraction must follow it.
    sampled = scenarios[:2]

    extractor = AsyncMock(return_value={})
    with patch("freqpred.bench.scenarios.extract_for_documents", new=extractor):
        await apply_extraction(
            session, MagicMock(), sampled, fixtures_by_id,
            model="claude-haiku-4-5-20251001",
        )

    assert extractor.await_count == 2, (
        f"extracted {extractor.await_count} scenarios for a sample of 2 — "
        "extraction is escaping the sample and billing the whole bank"
    )
