"""Unit tests for freqpred/bench/scoring.py — paired stats + trade decisions."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from freqpred.bench import scoring
from freqpred.bench.scenarios import ModelOutput, Scenario
from freqpred.bench.scoring import (
    aggregate,
    bootstrap_mean_ci,
    brier,
    direction_correct,
    log_loss,
    sign_test_p,
)
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig

FROZEN_CLOSE = datetime(2026, 6, 1, tzinfo=UTC)


class _UnitKellyStrategy(IPredictionStrategy):
    """Base sizing with unit constants: at bankroll=1.0 the stake equals the
    raw confidence-blended Kelly fraction — hand-checkable numbers."""

    config = StrategyConfig(
        name="unit-kelly",
        min_confidence=0.0,
        max_exposure_per_market=1.0,
        kelly_fraction=1.0,
        categories=[],
        min_volume_24h=0.0,
        max_days_to_close=365.0,
        min_days_to_close=0.0,
    )


_STRATEGY = _UnitKellyStrategy()


def trade_decision(output, scenario, **kwargs):
    kwargs.setdefault("min_edge", 0.10)
    kwargs.setdefault("min_confidence", 0.60)
    kwargs.setdefault("strategy", _STRATEGY)
    kwargs.setdefault("bankroll", 1.0)
    return scoring.trade_decision(output, scenario, **kwargs)


def score_pair(scenario, candidate, **kwargs):
    kwargs.setdefault("min_edge", 0.10)
    kwargs.setdefault("min_confidence", 0.60)
    kwargs.setdefault("strategy", _STRATEGY)
    kwargs.setdefault("bankroll", 1.0)
    return scoring.score_pair(scenario, candidate, **kwargs)


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
# Stake-weighted trade metrics — confidence scales position size in
# production, so overconfidence-when-wrong must cost proportionally more here.
# ---------------------------------------------------------------------------


def test_gate_applies_strategy_entry_filters_yes_and_no() -> None:
    """max_edge and the side-price band from the strategy config must block
    exactly the trades production should_trade blocks — a huge edge or a
    longshot entry price is a no-trade even when min_edge/min_confidence pass."""

    class _Filtered(IPredictionStrategy):
        config = StrategyConfig(
            name="filtered",
            min_confidence=0.60,
            max_exposure_per_market=1.0,
            kelly_fraction=1.0,
            categories=[],
            min_volume_24h=0.0,
            max_days_to_close=365.0,
            min_days_to_close=0.0,
            max_edge=0.40,
            min_mid_price=0.10,
            max_mid_price=0.95,
        )

    strategy = _Filtered()

    # YES longshot: market at 7c — edge 0.89 > max_edge AND side price < 0.10.
    longshot_yes = trade_decision(
        _output("YES", 0.97, confidence=0.95),
        _scenario(yes_bid=0.06, yes_ask=0.08),
        strategy=strategy,
    )
    assert longshot_yes.would_trade is False

    # NO longshot: market at 0.97 — NO side costs 0.03 < 0.10, edge in band.
    longshot_no = trade_decision(
        _output("NO", 0.80, confidence=0.80),
        _scenario(yes_bid=0.96, yes_ask=0.98),
        strategy=strategy,
    )
    assert 0.15 <= longshot_no.edge <= 0.40  # blocked by price, not by edge
    assert longshot_no.would_trade is False

    # max_edge alone: side price fine (0.30), edge 0.58 too large.
    overconfident = trade_decision(
        _output("YES", 0.90, confidence=0.90),
        _scenario(yes_bid=0.28, yes_ask=0.32),
        strategy=strategy,
    )
    assert overconfident.edge > 0.40
    assert overconfident.would_trade is False

    # Within all bands → trades (both directions).
    ok_yes = trade_decision(
        _output("YES", 0.70), _scenario(yes_bid=0.48, yes_ask=0.50), strategy=strategy
    )
    ok_no = trade_decision(
        _output("NO", 0.30), _scenario(yes_bid=0.48, yes_ask=0.50), strategy=strategy
    )
    assert ok_yes.would_trade is True
    assert ok_no.would_trade is True


def test_stake_is_production_kelly_hand_computed_yes_and_no() -> None:
    """The stake is the strategy's real position_size — at unit constants the
    numbers are hand-checkable Kelly: b=(1-ask)/ask, p_adj=c*p+(1-c)*ask,
    f*=(b*p_adj-(1-p_adj))/b."""
    # YES: ask=0.50, p=0.70, c=0.80 → b=1, p_adj=0.66, f*=0.32
    yes = trade_decision(_output("YES", 0.70, confidence=0.80), _scenario())
    assert yes.stake == pytest.approx(0.32)
    # NO: no_ask=0.52, p_est=0.70, c=0.80 → b=0.48/0.52, p_adj=0.664, f*=0.30
    no = trade_decision(_output("NO", 0.30, confidence=0.80), _scenario())
    assert no.stake == pytest.approx(0.30, abs=1e-4)


def test_stake_honors_strategy_position_size_override() -> None:
    """A custom strategy's overridden sizing must flow through — the whole
    point of sizing via the strategy instead of a copied formula."""

    class _FlatFraction(IPredictionStrategy):
        config = _UnitKellyStrategy.config

        def position_size(
            self, signal, bankroll, existing_market_exposure=0.0, assessment=None
        ):
            return 0.042 * bankroll

    for direction, posterior in [("YES", 0.70), ("NO", 0.30)]:
        decision = trade_decision(
            _output(direction, posterior),
            _scenario(outcome=1.0),
            strategy=_FlatFraction(),
            bankroll=100.0,
        )
        assert decision.would_trade is True
        assert decision.stake == pytest.approx(4.2)


def test_higher_confidence_stakes_more_both_directions() -> None:
    scenario = _scenario()
    for direction, posterior in [("YES", 0.70), ("NO", 0.30)]:
        low = trade_decision(
            _output(direction, posterior, confidence=0.65),
            scenario, min_edge=0.10, min_confidence=0.60,
        )
        high = trade_decision(
            _output(direction, posterior, confidence=0.95),
            scenario, min_edge=0.10, min_confidence=0.60,
        )
        assert low.would_trade and high.would_trade
        assert high.stake > low.stake > 0.0


def test_stake_weighted_pnl_yes_and_no() -> None:
    # pnl = (stake / side_ask) * ev — the stake buys stake/side_ask contracts.
    won_yes = trade_decision(
        _output("YES", 0.70), _scenario(outcome=1.0), min_edge=0.10, min_confidence=0.60
    )
    assert won_yes.pnl == pytest.approx(won_yes.stake / 0.50 * 0.50)
    lost_yes = trade_decision(
        _output("YES", 0.70), _scenario(outcome=0.0), min_edge=0.10, min_confidence=0.60
    )
    assert lost_yes.pnl == pytest.approx(-lost_yes.stake)  # YES loss = full stake
    won_no = trade_decision(
        _output("NO", 0.30), _scenario(outcome=0.0), min_edge=0.10, min_confidence=0.60
    )
    assert won_no.pnl == pytest.approx(won_no.stake / 0.52 * 0.48)
    lost_no = trade_decision(
        _output("NO", 0.30), _scenario(outcome=1.0), min_edge=0.10, min_confidence=0.60
    )
    assert lost_no.pnl == pytest.approx(-lost_no.stake)  # NO loss = full stake
    # No trade → no stake, no pnl.
    skip = trade_decision(
        _output("SKIP", 0.50), _scenario(), min_edge=0.10, min_confidence=0.60
    )
    assert skip.stake is None and skip.pnl is None


def test_aggregate_stake_totals_and_common_trades() -> None:
    # Both trade YES on s1 (candidate more confident → larger stake); only the
    # incumbent trades s2 (candidate skips).
    both = score_pair(
        _scenario(outcome=1.0, incumbent=_output("YES", 0.70, confidence=0.65), scenario_id="s1"),
        _output("YES", 0.70, confidence=0.95, model="candidate"),
        min_edge=0.10, min_confidence=0.60,
    )
    only_inc = score_pair(
        _scenario(outcome=0.0, incumbent=_output("YES", 0.70), scenario_id="s2"),
        _output("SKIP", 0.50, model="candidate"),
        min_edge=0.10, min_confidence=0.60,
    )
    trades = aggregate([both, only_inc])["trade_decisions"]

    assert trades["incumbent_total_stake"] == pytest.approx(
        both.incumbent_trade.stake + only_inc.incumbent_trade.stake
    )
    assert trades["candidate_total_stake"] == pytest.approx(both.candidate_trade.stake)
    # Incumbent won s1 and lost s2; candidate only won s1.
    assert trades["incumbent_stake_weighted_pnl"] == pytest.approx(
        both.incumbent_trade.pnl + only_inc.incumbent_trade.pnl
    )
    assert trades["candidate_stake_weighted_pnl"] == pytest.approx(both.candidate_trade.pnl)
    assert trades["candidate_pnl_per_dollar_staked"] == pytest.approx(
        both.candidate_trade.pnl / both.candidate_trade.stake
    )
    common = trades["common_trades"]
    assert common["n"] == 1
    assert common["candidate_mean_stake"] > common["incumbent_mean_stake"]
    # Disagreement rows carry the trading side's stake.
    assert trades["disagreements"][0]["trade_stake"] == pytest.approx(
        only_inc.incumbent_trade.stake
    )


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


def test_disagreement_table_one_model_enters() -> None:
    # Incumbent enters d1 (edge 0.20, conf 0.8); candidate abstains (SKIP).
    disagree = score_pair(
        _scenario(outcome=1.0, incumbent=_output("YES", 0.70), scenario_id="d1"),
        _output("SKIP", 0.50, model="candidate"),
        min_edge=0.10,
        min_confidence=0.60,
    )
    # Both enter — must NOT appear in the table. (Different market.)
    both = _scenario(outcome=1.0, incumbent=_output("YES", 0.70), scenario_id="d2")
    both.market_id = "MKT-d2"
    both_score = score_pair(
        both, _output("YES", 0.75, model="candidate"), min_edge=0.10, min_confidence=0.60
    )
    summary = aggregate([disagree, both_score])
    trades = summary["trade_decisions"]
    assert trades["incumbent_would_trade"] == 2  # markets entered
    assert trades["candidate_would_trade"] == 1
    assert len(trades["disagreements"]) == 1
    row = trades["disagreements"][0]
    assert row["scenario_id"] == "d1"
    assert row["entered_by"] == "incumbent"
    assert row["trade_ev"] == pytest.approx(1.0 - 0.50)  # the entering side's EV


def test_entry_faithful_first_clearing_signal_per_market() -> None:
    """One market, three signals over time: incumbent clears the gate at t1,
    candidate not until t2 — each side's entry is its own first clearing
    signal (different times, different prices), and later clearing signals do
    not add trades. Mirrors production: enter once, then exposure blocks."""
    from datetime import timedelta

    def scen(i, yes_bid, yes_ask, inc):
        s = _scenario(outcome=1.0, yes_bid=yes_bid, yes_ask=yes_ask,
                      incumbent=inc, scenario_id=f"t{i}")
        s.market_id = "MKT-SERIES"
        s.signal_time = FROZEN_CLOSE + timedelta(hours=i)
        return s

    # t1: incumbent trades (edge 0.20), candidate below min_edge.
    s1 = score_pair(scen(1, 0.48, 0.50, _output("YES", 0.70)),
                    _output("YES", 0.58, model="cand"), min_edge=0.10, min_confidence=0.60)
    # t2: both would clear; must be candidate's entry, NOT a second incumbent entry.
    s2 = score_pair(scen(2, 0.38, 0.40, _output("YES", 0.70)),
                    _output("YES", 0.60, model="cand"), min_edge=0.10, min_confidence=0.60)
    # t3: both clear again — must add nothing.
    s3 = score_pair(scen(3, 0.28, 0.30, _output("YES", 0.70)),
                    _output("YES", 0.55, model="cand"), min_edge=0.10, min_confidence=0.60)

    trades = aggregate([s3, s1, s2])["trade_decisions"]  # order-independent
    assert trades["incumbent_would_trade"] == 1
    assert trades["candidate_would_trade"] == 1
    # Incumbent entered at t1 prices (stake computed at ask 0.50); candidate
    # at t2 prices (ask 0.40) — one market in common, different entry scores.
    assert trades["common_trades"]["n"] == 1
    assert trades["incumbent_total_stake"] == pytest.approx(
        s1.incumbent_trade.stake
    )
    assert trades["candidate_total_stake"] == pytest.approx(
        s2.candidate_trade.stake
    )
    assert trades["disagreements"] == []


def test_cluster_bootstrap_and_market_sign_test() -> None:
    """10 correlated scenarios on ONE market must not manufacture
    significance: one cluster → degenerate CI, one sign-test vote."""
    from freqpred.bench.scoring import cluster_bootstrap_mean_ci

    one_market = []
    for i in range(10):
        scenario = _scenario(outcome=1.0, incumbent=_output("YES", 0.60), scenario_id=f"c{i}")
        scenario.market_id = "MKT-ONE"
        one_market.append(
            score_pair(
                scenario, _output("YES", 0.80, model="cand"),
                min_edge=0.10, min_confidence=0.60,
            )
        )
    summary = aggregate(one_market)
    assert summary["n_scenarios"] == 10
    assert summary["n_markets"] == 1
    assert summary["candidate_wins"] == 1          # one market, one vote
    assert summary["sign_test_p"] == 1.0           # n=1 proves nothing
    lo, hi = summary["brier_delta_ci95"]
    assert lo == pytest.approx(hi)                 # single cluster: no width

    # Same deltas spread over 10 markets → real evidence again.
    ten_markets = []
    for i, s in enumerate(one_market):
        s.market_id = f"MKT-{i}"
        ten_markets.append(s)
    summary10 = aggregate(ten_markets)
    assert summary10["n_markets"] == 10
    assert summary10["candidate_wins"] == 10
    assert summary10["sign_test_p"] < 0.01
    lo10, hi10 = summary10["brier_delta_ci95"]
    assert hi10 < 0

    # Direct: cluster CI degenerates to plain bootstrap at 1 scenario/cluster.
    assert cluster_bootstrap_mean_ci({"a": [0.5]}) == (0.5, 0.5)


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
