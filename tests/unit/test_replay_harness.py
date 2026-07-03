"""Unit tests for the deterministic replay harness (T66).

Two layers of coverage:

1. Synthetic fixtures built in-test exercise the engine mechanics — YES/NO/SKIP
   decisions, risk caps, regression detection, the PROMPT_VERSION guard — with
   no dependence on what happens to be recorded under tests/fixtures/replay/.
2. ``test_recorded_fixtures_replay_green`` replays every checked-in recorded
   fixture: this is the actual regression gate for prompt/pipeline changes.
   Set FREQPRED_UPDATE_FIXTURES=1 to regenerate expectations after an
   intentional change (the diff then shows up in code review).

All fixture timestamps are *frozen inputs*, not wall-clock assertions: the
harness pins the clock to each fixture's ``now`` (via ``_now`` injection and
freezegun), so these tests stay correct as calendar time advances.
"""
from __future__ import annotations

import json
import os
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import freqpred.ingestion.models  # noqa: F401 — registers mappers
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.replay import (
    compute_expectations,
    load_fixture,
    replay_fixture,
    save_fixture,
)
from freqpred.replay.fixtures import (
    FixtureDecisionContext,
    FixtureDocument,
    FixtureInputs,
    FixtureMarket,
    FixturePortfolio,
    FixturePriorScheduledSignal,
    ReplayFixture,
)
from freqpred.signal.llm import PROMPT_VERSION
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig

RECORDED_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "replay"

# Frozen decision time — an input to the harness, never compared to the wall
# clock (the engine pins all time math to it).
FROZEN_NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


class _PermissiveStrategy(IPredictionStrategy):
    """Market-order strategy with wide-open filters for synthetic fixtures."""

    config = StrategyConfig(
        name="replay_test_strategy",
        min_edge=0.05,
        min_confidence=0.60,
        max_exposure_per_market=0.20,
        kelly_fraction=0.50,
        categories=[],
        min_volume_24h=0.0,
        max_days_to_close=365.0,
        min_days_to_close=0.0,
        min_mid_price=None,
        max_mid_price=None,
        max_spread=0.10,
    )


def _llm_response(
    direction: str = "YES",
    probability: float = 0.70,
    confidence: float = 0.80,
    prior: float = 0.65,
    updates: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "prior": prior,
            "prior_basis": "test prior basis",
            "updates_applied": updates or [],
            "posterior": probability,
            "probability": probability,
            "confidence": confidence,
            "direction": direction,
            "reasoning": "test reasoning",
        }
    )


def _make_inputs(
    *,
    direction: str = "YES",
    probability: float = 0.70,
    confidence: float = 0.80,
    yes_bid: float = 0.48,
    yes_ask: float = 0.50,
    bankroll: float = 1000.0,
    portfolio: FixturePortfolio | None = None,
    risk_config: dict | None = None,
    prior_scheduled: FixturePriorScheduledSignal | None = None,
) -> FixtureInputs:
    docs = [
        FixtureDocument(
            id="11111111-1111-1111-1111-111111111111",
            source_url="https://example.com/a",
            title="Doc A",
            body="Body of document A about the event.",
            source_type="news",
            source_name="Reuters",
            published_at=FROZEN_NOW - timedelta(days=1),
            fetched_at=FROZEN_NOW,
        ),
        FixtureDocument(
            id="22222222-2222-2222-2222-222222222222",
            source_url="https://example.com/b",
            title="Doc B",
            body="Body of document B about the event.",
            source_type="news",
            source_name="AP",
            published_at=FROZEN_NOW - timedelta(days=2),
            fetched_at=FROZEN_NOW,
        ),
    ]
    return FixtureInputs(
        now=FROZEN_NOW,
        trigger="scheduled",
        market=FixtureMarket(
            id="TEST-MKT",
            question="Will the test event happen?",
            category="test",
            close_time=FROZEN_NOW + timedelta(days=3),
            open_time=FROZEN_NOW - timedelta(days=4),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            mid_price=round((yes_bid + yes_ask) / 2, 4),
            volume_24h=5000.0,
        ),
        documents=docs,
        llm_response=_llm_response(
            direction=direction, probability=probability, confidence=confidence
        ),
        prior_scheduled_signal=prior_scheduled,
        decision_context=FixtureDecisionContext(
            strategy="replay_test_strategy",
            bankroll=bankroll,
            portfolio=portfolio or FixturePortfolio(),
            risk_config=risk_config or {"min_edge_floor": 0.05},
        ),
    )


def _make_fixture(name: str = "synthetic", **kwargs) -> ReplayFixture:
    inputs = _make_inputs(**kwargs)
    strategy = _PermissiveStrategy()
    return ReplayFixture(
        name=name,
        inputs=inputs,
        expectations=compute_expectations(inputs, strategy=strategy),
    )


# ---------------------------------------------------------------------------
# Decision outcomes — YES / NO / SKIP
# ---------------------------------------------------------------------------


def test_replay_fixture_yes_trade() -> None:
    fixture = _make_fixture(direction="YES", probability=0.70, yes_bid=0.48, yes_ask=0.50)

    exp = fixture.expectations
    assert exp.parsed.direction == "YES"
    assert exp.edge == pytest.approx(0.70 - 0.50)
    assert exp.market_ask_at_signal == pytest.approx(0.50)
    assert exp.entry.would_trade is True
    assert exp.entry.decline_reason == ""
    assert exp.entry.entry_price == pytest.approx(0.50)  # market order at yes_ask
    assert exp.entry.contracts >= 1

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert result.passed, [c.detail for c in result.failures]


def test_replay_fixture_no_trade() -> None:
    # NO at probability 0.30: no_ask = 1 - yes_bid = 0.52; edge = 0.70 - 0.52 = 0.18
    fixture = _make_fixture(direction="NO", probability=0.30, yes_bid=0.48, yes_ask=0.50)

    exp = fixture.expectations
    assert exp.parsed.direction == "NO"
    assert exp.market_ask_at_signal == pytest.approx(0.52)  # NO-side ask = 1 - yes_bid
    assert exp.edge == pytest.approx((1.0 - 0.30) - 0.52)
    assert exp.entry.would_trade is True
    assert exp.entry.entry_price == pytest.approx(0.52)  # market order at NO-side ask
    assert exp.entry.contracts >= 1

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert result.passed, [c.detail for c in result.failures]


def test_replay_fixture_low_confidence_skip() -> None:
    fixture = _make_fixture(direction="YES", probability=0.70, confidence=0.40)

    exp = fixture.expectations
    assert exp.entry.would_trade is False
    assert exp.entry.decline_reason == "strategy_declined"

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert result.passed, [c.detail for c in result.failures]


def test_replay_fixture_skip_direction() -> None:
    fixture = _make_fixture(direction="SKIP", probability=0.50)

    exp = fixture.expectations
    assert exp.market_ask_at_signal is None
    assert exp.entry.would_trade is False
    assert exp.entry.decline_reason == "skip_direction"

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert result.passed, [c.detail for c in result.failures]


# ---------------------------------------------------------------------------
# Decision layer through risk caps — both directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "probability"),
    [("YES", 0.70), ("NO", 0.30)],
    ids=["yes", "no"],
)
def test_replay_decision_layer_risk_blocked_both_directions(
    direction: str, probability: float
) -> None:
    """A full portfolio blocks the trade through the real risk-cap arithmetic."""
    fixture = _make_fixture(
        direction=direction,
        probability=probability,
        portfolio=FixturePortfolio(open_count=20),  # at RiskConfig.max_open_positions
    )

    exp = fixture.expectations
    assert exp.entry.would_trade is False
    assert exp.entry.decline_reason == "risk_blocked"
    assert exp.entry.risk_allowed is False
    assert "active positions 20" in exp.entry.risk_reason

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert result.passed, [c.detail for c in result.failures]


@pytest.mark.parametrize(
    ("direction", "probability"),
    [("YES", 0.70), ("NO", 0.30)],
    ids=["yes", "no"],
)
def test_replay_decision_layer_size_capped_both_directions(
    direction: str, probability: float
) -> None:
    """max_position_pct caps the Kelly size on the way through risk."""
    fixture = _make_fixture(
        direction=direction,
        probability=probability,
        bankroll=1000.0,
        risk_config={"min_edge_floor": 0.05, "max_position_pct": 0.01},
    )

    exp = fixture.expectations
    assert exp.entry.risk_allowed is True
    assert exp.entry.risk_capped_size <= 10.0 + 1e-9  # 1% of bankroll
    assert exp.entry.risk_capped_size <= exp.entry.position_size_raw


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def test_replay_detects_retrieval_hash_regression() -> None:
    """Changing the document set fails the retrieval-hash check."""
    fixture = _make_fixture()
    fixture.inputs.documents = fixture.inputs.documents[:1]  # a doc disappeared

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert not result.passed
    assert "retrieval_hash" in {c.name for c in result.failures}


def test_replay_detects_decision_regression() -> None:
    """A changed trade/no-trade outcome under frozen inputs fails entry_decision."""
    fixture = _make_fixture(direction="YES", probability=0.70)
    assert fixture.expectations.entry.would_trade is True
    fixture.expectations.entry.would_trade = False  # as if the recorded world differed
    fixture.expectations.entry.decline_reason = "strategy_declined"

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert not result.passed
    assert "entry_decision" in {c.name for c in result.failures}


def test_replay_prompt_change_requires_prompt_version_bump() -> None:
    """Prompt drift with an unchanged PROMPT_VERSION names the guard explicitly."""
    fixture = _make_fixture()
    fixture.expectations.rendered_prompt += "\nEXTRA LINE THE CODE NO LONGER PRODUCES"

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert not result.passed
    failure = next(c for c in result.failures if c.name == "rendered_prompt")
    assert "bump PROMPT_VERSION" in failure.detail


def test_replay_stale_fixture_after_prompt_version_bump() -> None:
    """A fixture recorded under an older PROMPT_VERSION demands regeneration."""
    fixture = _make_fixture()
    fixture.expectations.prompt_version = "signal-v0-obsolete"

    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert not result.passed
    failure = next(c for c in result.failures if c.name == "prompt_version")
    assert "regenerate" in failure.detail
    assert PROMPT_VERSION in failure.detail


def test_rendered_prompt_is_deterministic_for_frozen_now() -> None:
    """Two renders from identical inputs are byte-identical (clock is pinned)."""
    inputs = _make_inputs()
    a = compute_expectations(inputs, strategy=_PermissiveStrategy())
    b = compute_expectations(inputs, strategy=_PermissiveStrategy())
    assert a.rendered_prompt == b.rendered_prompt
    assert f"Current Date (UTC): {FROZEN_NOW.strftime('%Y-%m-%d %H:%M')}" in a.rendered_prompt


# ---------------------------------------------------------------------------
# Scheduled skip / cooldown decisions
# ---------------------------------------------------------------------------


def test_replay_scheduled_skip_and_cooldown_decisions() -> None:
    inputs = _make_inputs(
        prior_scheduled=FixturePriorScheduledSignal(
            retrieval_hash="",  # filled below with the computed hash
            created_at=FROZEN_NOW - timedelta(hours=2),
            confidence=0.55,  # low-confidence tier → 4h cooldown
        )
    )
    # Same doc set as the prior signal → hash matches → scheduled skip.
    baseline = compute_expectations(inputs, strategy=_PermissiveStrategy())
    inputs.prior_scheduled_signal.retrieval_hash = baseline.retrieval_hash

    exp = compute_expectations(inputs, strategy=_PermissiveStrategy())
    assert exp.skip_decisions is not None
    assert exp.skip_decisions.scheduled_skip is True
    assert exp.skip_decisions.cooldown_hours_remaining == pytest.approx(2.0)

    # New evidence (different hash) → no skip, cooldown unchanged.
    inputs.prior_scheduled_signal.retrieval_hash = "0" * 64
    exp = compute_expectations(inputs, strategy=_PermissiveStrategy())
    assert exp.skip_decisions.scheduled_skip is False


# ---------------------------------------------------------------------------
# Determinism boundary: no network, no LLM
# ---------------------------------------------------------------------------


def test_replay_uses_no_live_network_or_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay of a real recorded fixture succeeds with sockets disabled."""

    def _no_network(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("replay attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _no_network)

    fixture = _make_fixture()
    result = replay_fixture(fixture, strategy=_PermissiveStrategy())
    assert result.passed, [c.detail for c in result.failures]

    recorded = sorted(RECORDED_FIXTURE_DIR.glob("*.json"))
    if recorded:  # also prove it on a real recorded fixture
        result = replay_fixture(load_fixture(recorded[0]))
        assert result.passed, [c.detail for c in result.failures]


# ---------------------------------------------------------------------------
# Recorded fixtures — the actual regression gate
# ---------------------------------------------------------------------------


def _recorded_fixture_paths() -> list[Path]:
    return sorted(RECORDED_FIXTURE_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "path", _recorded_fixture_paths(), ids=lambda p: p.stem
)
def test_recorded_fixtures_replay_green(path: Path) -> None:
    """Every checked-in fixture must replay clean against the current code.

    After an *intentional* prompt/decision change, regenerate expectations with
    FREQPRED_UPDATE_FIXTURES=1 (or `freqpred fixtures replay --update`) and
    review the fixture diff.
    """
    fixture = load_fixture(path)

    if os.environ.get("FREQPRED_UPDATE_FIXTURES") == "1":
        fixture.expectations = compute_expectations(
            fixture.inputs, fixture_name=fixture.name
        )
        save_fixture(fixture, path)
        return

    result = replay_fixture(fixture)
    assert result.passed, f"{fixture.name}: " + "; ".join(
        f"{c.name}: {c.detail}" for c in result.failures
    )


def test_recorded_fixtures_cover_both_directions() -> None:
    """The checked-in corpus must include both YES and NO scenarios (repo rule)."""
    directions = {
        load_fixture(p).expectations.parsed.direction for p in _recorded_fixture_paths()
    }
    assert {"YES", "NO"} <= directions
