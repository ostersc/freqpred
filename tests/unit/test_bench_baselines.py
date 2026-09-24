"""Unit tests for freqpred/bench/baselines.py — the free comparators.

These exist because the benchmark harness only ever compared the candidate
model against the incumbent model, which cannot answer whether either clears an
estimator that costs nothing (``docs/POSTMORTEM.md`` §3.1, §6).
"""
from __future__ import annotations

import math
import re
from datetime import UTC, datetime

import pytest

from freqpred.bench.baselines import (
    constant_estimator,
    market_mid_estimate,
    parse_factbase,
    poisson_30d_estimate,
    poisson_365d_estimate,
    poisson_probability,
    score_all_baselines,
    score_baseline,
)
from freqpred.bench.scenarios import ModelOutput, Scenario

FROZEN_CLOSE = datetime(2026, 6, 1, tzinfo=UTC)
FROZEN_SIGNAL = datetime(2026, 5, 25, tzinfo=UTC)


def _factbase_block(
    *,
    in_market: int = 0,
    count_7d: int = 0,
    count_30d: int = 6,
    count_365d: int = 178,
    days: float = 6.2,
) -> str:
    """Render a FactBase block the way freqpred.signal.llm does.

    Mirrors the real emitter's formatting, including printing the Poisson
    figures at one decimal, so the parser is exercised against the shape it
    actually meets in the bank.
    """
    p30 = poisson_probability(count_30d, 30.0, days)
    p365 = poisson_probability(count_365d, 365.0, days)
    return (
        "=== PHRASE FREQUENCY DATA (FactBase) ===\n"
        'Phrase: "Antifa"\n'
        "Speaker: Donald Trump\n"
        "\n"
        "Occurrence counts (Trump statements archive):\n"
        f"  Since market opened : {in_market}\n"
        f"  Last 7 days         : {count_7d}\n"
        f"  Last 30 days        : {count_30d}\n"
        f"  Last 365 days       : {count_365d}\n"
        "\n"
        "Derived rates:\n"
        f"  Annual rate (365d)   : {count_365d / 365.0:.4f}/day\n"
        f"  Recent rate (30d)    : {count_30d / 30.0:.4f}/day\n"
        "\n"
        f"Poisson baseline P(≥1 occurrence in remaining {days:.1f} days):\n"
        f"  Using 365d rate      : {p365:.1%}\n"
        f"  Using 30d rate       : {p30:.1%}\n"
    )


def _scenario(
    *,
    market_id: str = "M1",
    outcome: float = 1.0,
    posterior: float = 0.70,
    mid_price: float = 0.50,
    prompt: str | None = None,
    scenario_id: str = "s1",
) -> Scenario:
    return Scenario(
        id=scenario_id,
        source="fixture",
        market_id=market_id,
        market_question="Will Trump say Antifa?",
        close_time=FROZEN_CLOSE,
        outcome=outcome,
        prompt=prompt if prompt is not None else _factbase_block(),
        incumbent=ModelOutput(
            model="incumbent",
            prior=0.4,
            posterior=posterior,
            confidence=0.7,
            direction="YES" if posterior >= 0.5 else "NO",
            updates_count=1,
            reasoning="",
        ),
        yes_bid=mid_price - 0.01,
        yes_ask=mid_price + 0.01,
        mid_price=mid_price,
        signal_time=FROZEN_SIGNAL,
    )


def _model(scenario: Scenario) -> float:
    return scenario.incumbent.posterior


class TestPoissonProbability:
    def test_matches_the_closed_form(self):
        # 178 occurrences over 365 days, 6.2 days remaining.
        expected = 1.0 - math.exp(-(178 / 365.0) * 6.2)
        assert poisson_probability(178, 365.0, 6.2) == pytest.approx(expected)

    def test_zero_count_is_a_hard_zero_not_a_smoothed_rate(self):
        # This mirrors _build_factbase_block exactly. A smoothed value would
        # score an estimator the model was never shown.
        assert poisson_probability(0, 30.0, 6.2) == 0.0

    def test_longer_window_never_lowers_the_probability(self):
        assert poisson_probability(6, 30.0, 1.0) < poisson_probability(6, 30.0, 10.0)


class TestParseFactbase:
    def test_round_trips_counts_and_window(self):
        numbers = parse_factbase(_factbase_block(count_30d=6, count_365d=178, days=6.2))
        assert numbers is not None
        assert (numbers.count_30d, numbers.count_365d) == (6, 178)
        assert numbers.days_to_close == pytest.approx(6.2)
        assert numbers.in_market_count == 0

    def test_prefers_the_printed_value_over_recomputation(self):
        # The prompt prints days_to_close to one decimal, so recomputing from
        # counts drifts at short horizons. The printed figure is what the model
        # was shown and is therefore what gets scored.
        numbers = parse_factbase(_factbase_block(count_365d=119, days=0.7))
        assert numbers is not None
        assert numbers.poisson_365d == numbers.printed_poisson_365d

    def test_self_check_passes_on_a_faithfully_rendered_block(self):
        for days in (0.7, 1.0, 6.2, 30.0):
            numbers = parse_factbase(_factbase_block(days=days))
            assert numbers is not None
            assert numbers.printed_mismatch() is None, days

    def test_self_check_flags_counts_that_contradict_the_printed_value(self):
        block = _factbase_block(count_30d=6, days=6.2)
        # Contradict the printed probability without touching the counts.
        block = re.sub(r"Using 30d rate\s*:.*", "Using 30d rate      : 5.0%", block)
        numbers = parse_factbase(block)
        assert numbers is not None
        assert numbers.printed_mismatch() is not None

    def test_returns_none_when_there_is_no_factbase_block(self):
        # A market outside the phrase-frequency allowlist has no Poisson
        # baseline at all. Scoring it as 0.0 would invent one.
        assert parse_factbase("=== HISTORICAL BASE RATE ===\nSeries overall: 5 YES") is None

    def test_returns_none_when_the_block_is_truncated(self):
        block = _factbase_block()
        assert parse_factbase(block.split("Last 365 days")[0]) is None


class TestEstimators:
    def test_poisson_estimators_read_the_prompt(self):
        scenario = _scenario(prompt=_factbase_block(count_30d=6, count_365d=178, days=6.2))
        assert poisson_30d_estimate(scenario) == pytest.approx(0.711, abs=0.001)
        assert poisson_365d_estimate(scenario) == pytest.approx(0.951, abs=0.001)

    def test_estimators_return_none_rather_than_zero_without_a_block(self):
        scenario = _scenario(prompt="no factbase here")
        assert poisson_30d_estimate(scenario) is None
        assert poisson_365d_estimate(scenario) is None

    def test_market_mid_is_the_frozen_price(self):
        assert market_mid_estimate(_scenario(mid_price=0.37)) == 0.37


class TestConstantEstimator:
    def test_pooled_constant_is_the_per_market_base_rate(self):
        scenarios = [
            _scenario(market_id="A", outcome=1.0),
            _scenario(market_id="B", outcome=0.0),
            _scenario(market_id="C", outcome=0.0),
            _scenario(market_id="D", outcome=0.0),
        ]
        estimate = constant_estimator(scenarios)
        assert estimate(scenarios[0]) == pytest.approx(0.25)

    def test_pooled_constant_weights_markets_not_signals(self):
        # Market A carries three correlated snapshots of one outcome; it must
        # not get three votes in the base rate.
        scenarios = [
            _scenario(market_id="A", outcome=1.0, scenario_id=f"a{i}") for i in range(3)
        ] + [_scenario(market_id="B", outcome=0.0)]
        assert constant_estimator(scenarios)(scenarios[0]) == pytest.approx(0.5)

    def test_leave_one_market_out_never_sees_its_own_outcome(self):
        # Two markets, opposite outcomes: each must be estimated purely from
        # the other, so a peeking implementation would return 0.5 here.
        yes = _scenario(market_id="A", outcome=1.0)
        no = _scenario(market_id="B", outcome=0.0)
        estimate = constant_estimator([yes, no], leave_one_market_out=True)
        assert estimate(yes) == 0.0
        assert estimate(no) == 1.0

    def test_handles_an_empty_sample(self):
        assert constant_estimator([])(_scenario()) is None


class TestScoreBaseline:
    @pytest.mark.parametrize("outcome", [1.0, 0.0])
    def test_scores_both_resolution_directions(self, outcome):
        # P(YES) is scored without a direction flip, matching
        # freqpred.metrics.calibration, so a YES-resolving and a NO-resolving
        # market must both grade correctly.
        scenarios = [
            _scenario(market_id=f"M{i}", outcome=outcome, posterior=0.6, mid_price=0.5,
                      scenario_id=f"s{i}")
            for i in range(4)
        ]
        score = score_baseline("market_mid", market_mid_estimate, scenarios, _model)
        assert score is not None
        assert score.baseline_mean_brier == pytest.approx((0.5 - outcome) ** 2)
        assert score.model_mean_brier == pytest.approx((0.6 - outcome) ** 2)

    def test_delta_sign_favours_the_model_when_it_is_closer(self):
        # Model 0.9 vs market 0.5 on a YES: the model is better, so the
        # delta (model - baseline) must be negative.
        scenarios = [
            _scenario(market_id=f"M{i}", outcome=1.0, posterior=0.9, mid_price=0.5,
                      scenario_id=f"s{i}")
            for i in range(5)
        ]
        score = score_baseline("market_mid", market_mid_estimate, scenarios, _model)
        assert score is not None
        assert score.brier_delta_mean < 0

    def test_counts_skips_instead_of_scoring_absent_baselines(self):
        scenarios = [
            _scenario(market_id="A", prompt=_factbase_block()),
            _scenario(market_id="B", prompt="no factbase", scenario_id="s2"),
        ]
        score = score_baseline("poisson_30d", poisson_30d_estimate, scenarios, _model)
        assert score is not None
        assert (score.n_scenarios, score.n_skipped) == (1, 1)

    def test_returns_none_when_no_scenario_has_the_baseline(self):
        scenarios = [_scenario(prompt="no factbase")]
        assert score_baseline("poisson_30d", poisson_30d_estimate, scenarios, _model) is None

    def test_clusters_by_market_so_one_market_gets_one_vote(self):
        scenarios = [
            _scenario(market_id="A", outcome=1.0, scenario_id=f"a{i}") for i in range(9)
        ] + [_scenario(market_id="B", outcome=1.0, scenario_id="b0")]
        score = score_baseline("market_mid", market_mid_estimate, scenarios, _model)
        assert score is not None
        assert score.n_markets == 2
        assert score.n_scenarios == 10


class TestScoreAllBaselines:
    def test_reports_every_baseline_that_applies(self):
        scenarios = [
            _scenario(market_id=f"M{i}", outcome=float(i % 2), scenario_id=f"s{i}")
            for i in range(6)
        ]
        results = score_all_baselines(scenarios, _model)
        assert set(results) == {"poisson_30d", "poisson_365d", "market_mid", "constant"}
        for name, result in results.items():
            assert result["n_markets"] == 6, name
            assert result["parse_warnings"] == [], name

    def test_empty_input_is_safe(self):
        assert score_all_baselines([], _model) == {}

    def test_omits_a_baseline_no_scenario_supports(self):
        scenarios = [
            _scenario(market_id=f"M{i}", prompt="no factbase", scenario_id=f"s{i}")
            for i in range(3)
        ]
        results = score_all_baselines(scenarios, _model)
        assert "poisson_30d" not in results
        assert "market_mid" in results
