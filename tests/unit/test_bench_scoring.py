"""Unit tests for freqpred/bench/scoring.py — paired stats + trade decisions."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from freqpred.bench.scenarios import ModelOutput, Scenario
from freqpred.bench.scoring import (
    aggregate,
    bootstrap_mean_ci,
    brier,
    direction_correct,
    log_loss,
    score_pair,
    sign_test_p,
    trade_decision,
)

FROZEN_CLOSE = datetime(2026, 6, 1, tzinfo=UTC)


def _output(
    direction: str = "YES",
    posterior: float = 0.70,
    confidence: float = 0.80,
    model: str = "incumbent-model",
) -> ModelOutput:
    return ModelOutput(
        model=model,
        prior=0.6,
        posterior=posterior,
        confidence=confidence,
        direction=direction,
        updates_count=0,
    )


def _scenario(
    outcome: float = 1.0,
    yes_bid: float = 0.48,
    yes_ask: float = 0.50,
    incumbent: ModelOutput | None = None,
    scenario_id: str = "s1",
) -> Scenario:
    return Scenario(
        id=scenario_id,
        source="db",
        market_id=f"MKT-{scenario_id}",
        market_question="Will it happen?",
        close_time=FROZEN_CLOSE,
        outcome=outcome,
        prompt="prompt",
        incumbent=incumbent or _output(),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        mid_price=round((yes_bid + yes_ask) / 2, 4),
    )


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


def test_brier_and_log_loss_basics() -> None:
    assert brier(0.7, 1.0) == pytest.approx(0.09)
    assert brier(0.7, 0.0) == pytest.approx(0.49)
    assert log_loss(1.0, 1.0) < log_loss(0.5, 1.0)  # certainty when right is cheap
    assert log_loss(0.0, 1.0) > 10  # clamped, not infinite


def test_direction_correct_grades_yes_no_and_skips_skip() -> None:
    assert direction_correct("YES", 1.0) is True
    assert direction_correct("YES", 0.0) is False
    assert direction_correct("NO", 0.0) is True
    assert direction_correct("NO", 1.0) is False
    assert direction_correct("SKIP", 1.0) is None


# ---------------------------------------------------------------------------
# Trade decisions — both directions (repo rule)
# ---------------------------------------------------------------------------


def test_would_trade_edge_yes_direction() -> None:
    scenario = _scenario(yes_bid=0.48, yes_ask=0.50)
    # YES edge = posterior - yes_ask = 0.70 - 0.50 = 0.20
    decision = trade_decision(
        _output("YES", 0.70), scenario, min_edge=0.10, min_confidence=0.60
    )
    assert decision.would_trade is True
    assert decision.edge == pytest.approx(0.20)
    assert decision.side_ask == pytest.approx(0.50)

    weak = trade_decision(
        _output("YES", 0.55), scenario, min_edge=0.10, min_confidence=0.60
    )
    assert weak.would_trade is False  # edge 0.05 below threshold
    assert weak.ev is None


def test_would_trade_edge_no_direction() -> None:
    scenario = _scenario(yes_bid=0.48, yes_ask=0.50)
    # NO edge = (1 - posterior) - (1 - yes_bid) = 0.70 - 0.52 = 0.18
    decision = trade_decision(
        _output("NO", 0.30), scenario, min_edge=0.10, min_confidence=0.60
    )
    assert decision.would_trade is True
    assert decision.edge == pytest.approx(0.18)
    assert decision.side_ask == pytest.approx(0.52)


def test_would_trade_confidence_gate() -> None:
    """The confidence gate is what exposes a better-calibrated-but-timid model."""
    scenario = _scenario()
    timid = trade_decision(
        _output("YES", 0.70, confidence=0.50), scenario, min_edge=0.10, min_confidence=0.60
    )
    assert timid.would_trade is False
    skip = trade_decision(
        _output("SKIP", 0.70), scenario, min_edge=0.10, min_confidence=0.60
    )
    assert skip.would_trade is False


def test_per_trade_ev_yes_and_no() -> None:
    # YES trade, resolved YES: EV = outcome - yes_ask = 1.0 - 0.50 = +0.50
    won_yes = trade_decision(
        _output("YES", 0.70), _scenario(outcome=1.0), min_edge=0.10, min_confidence=0.60
    )
    assert won_yes.ev == pytest.approx(0.50)
    # YES trade, resolved NO: EV = 0.0 - 0.50 = -0.50
    lost_yes = trade_decision(
        _output("YES", 0.70), _scenario(outcome=0.0), min_edge=0.10, min_confidence=0.60
    )
    assert lost_yes.ev == pytest.approx(-0.50)
    # NO trade, resolved NO: EV = (1-0) - (1-yes_bid) = 1.0 - 0.52 = +0.48
    won_no = trade_decision(
        _output("NO", 0.30), _scenario(outcome=0.0), min_edge=0.10, min_confidence=0.60
    )
    assert won_no.ev == pytest.approx(0.48)
    # NO trade, resolved YES: EV = 0.0 - 0.52 = -0.52
    lost_no = trade_decision(
        _output("NO", 0.30), _scenario(outcome=1.0), min_edge=0.10, min_confidence=0.60
    )
    assert lost_no.ev == pytest.approx(-0.52)


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def test_paired_brier_bootstrap_ci_and_sign_test() -> None:
    # Candidate posterior closer to the outcome on every scenario → all deltas
    # negative, CI must exclude zero, sign test small.
    scores = [
        score_pair(
            _scenario(outcome=1.0, incumbent=_output("YES", 0.60), scenario_id=f"s{i}"),
            _output("YES", 0.80, model="candidate"),
            min_edge=0.10,
            min_confidence=0.60,
        )
        for i in range(8)
    ]
    summary = aggregate(scores)
    assert summary["candidate_wins"] == 8
    assert summary["incumbent_wins"] == 0
    assert summary["sign_test_p"] == pytest.approx(2 * 0.5**8)
    lo, hi = summary["brier_delta_ci95"]
    assert hi < 0
    assert summary["brier_delta_significant"] is True


def test_sign_test_exact_values() -> None:
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == pytest.approx(1.0)
    assert sign_test_p(8, 0) == pytest.approx(2 * 0.5**8)
    assert sign_test_p(0, 8) == pytest.approx(2 * 0.5**8)  # symmetric


def test_bootstrap_ci_is_deterministic_and_ordered() -> None:
    values = [-0.05, -0.02, -0.08, -0.01, -0.04, -0.03]
    a = bootstrap_mean_ci(values)
    b = bootstrap_mean_ci(values)
    assert a == b  # seeded
    assert a[0] <= a[1]
    assert bootstrap_mean_ci([0.5]) == (0.5, 0.5)
    assert bootstrap_mean_ci([]) == (0.0, 0.0)


def test_disagreement_table_one_model_trades() -> None:
    # Incumbent trades (edge 0.20, conf 0.8); candidate abstains (SKIP).
    disagree = score_pair(
        _scenario(outcome=1.0, incumbent=_output("YES", 0.70), scenario_id="d1"),
        _output("SKIP", 0.50, model="candidate"),
        min_edge=0.10,
        min_confidence=0.60,
    )
    # Both trade — must NOT appear in the table.
    both = score_pair(
        _scenario(outcome=1.0, incumbent=_output("YES", 0.70), scenario_id="d2"),
        _output("YES", 0.75, model="candidate"),
        min_edge=0.10,
        min_confidence=0.60,
    )
    summary = aggregate([disagree, both])
    trades = summary["trade_decisions"]
    assert trades["incumbent_would_trade"] == 2
    assert trades["candidate_would_trade"] == 1
    assert len(trades["disagreements"]) == 1
    row = trades["disagreements"][0]
    assert row["scenario_id"] == "d1"
    assert row["incumbent_trades"] is True
    assert row["candidate_trades"] is False
    assert row["trade_ev"] == pytest.approx(1.0 - 0.50)  # the trading side's EV


def test_aggregate_empty() -> None:
    assert aggregate([]) == {"n_scenarios": 0}


# ---------------------------------------------------------------------------
# Regime segmentation — favorites vs upsets
# ---------------------------------------------------------------------------


def test_classify_regime() -> None:
    from freqpred.bench.scoring import classify_regime

    assert classify_regime(0.9, 1.0) == "favorite"   # priced YES, resolved YES
    assert classify_regime(0.1, 0.0) == "favorite"   # priced NO, resolved NO
    assert classify_regime(0.2, 1.0) == "upset"      # priced NO, resolved YES
    assert classify_regime(0.8, 0.0) == "upset"      # priced YES, resolved NO
    assert classify_regime(0.5, 1.0) == "coin_flip"
    assert classify_regime(0.42, 0.0) == "coin_flip"


def test_aggregate_segments_by_regime() -> None:
    """A hedging candidate must show as losing on the favorite and winning on
    the upset — the exact structural split the breakdown exists to surface."""
    # Favorite: market at 0.9, resolved YES. Incumbent extreme (0.97) beats
    # the hedging candidate (0.85).
    favorite = _scenario(
        outcome=1.0, yes_bid=0.89, yes_ask=0.91,
        incumbent=_output("YES", 0.97), scenario_id="fav",
    )
    fav_score = score_pair(
        favorite, _output("YES", 0.85, model="candidate"),
        min_edge=0.10, min_confidence=0.60,
    )
    # Upset: market at 0.9 but resolved NO. The incumbent's extreme YES gets
    # demolished; the hedging candidate loses far less.
    upset = _scenario(
        outcome=0.0, yes_bid=0.89, yes_ask=0.91,
        incumbent=_output("YES", 0.97), scenario_id="ups",
    )
    ups_score = score_pair(
        upset, _output("YES", 0.85, model="candidate"),
        min_edge=0.10, min_confidence=0.60,
    )

    assert fav_score.regime == "favorite"
    assert ups_score.regime == "upset"

    summary = aggregate([fav_score, ups_score])
    regimes = summary["regimes"]
    assert regimes["favorite"]["n"] == 1
    assert regimes["favorite"]["incumbent_wins"] == 1  # extreme wins when market is right
    assert regimes["upset"]["n"] == 1
    assert regimes["upset"]["candidate_wins"] == 1     # hedging wins when market is wrong
    assert "coin_flip" not in regimes                  # empty segments omitted
    # Per-upset delta dwarfs the per-favorite delta (the 10x asymmetry).
    assert abs(regimes["upset"]["brier_delta_mean"]) > 5 * abs(
        regimes["favorite"]["brier_delta_mean"]
    )
