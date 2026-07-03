"""Scoring for the benchmark harness (T93).

Calibration metrics (Brier, log loss, direction correctness) score each
model's posterior against the actual market outcome. Paired statistics
(bootstrap CI on the mean Brier delta, exact sign test) are the adopt/reject
gate: a raw mean improvement on a small noisy sample is not evidence.

Trade-decision metrics capture what Brier misses — a better-calibrated but
systematically less confident model trades less. They are decision-quality
signals at frozen side-specific prices, deliberately **not** a P&L
simulation: no sizing, no bankroll, no exits, no portfolio path.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, median

from freqpred.bench.scenarios import ModelOutput, Scenario
from freqpred.signal.pipeline import compute_signal_edge

_EPS = 1e-6  # log-loss clamp

# Regime thresholds on the frozen mid price at signal time. A hedging model
# loses to an extreme model on favorites and beats it on upsets — segmenting
# the paired scores by regime makes that structural trade-off visible instead
# of letting a lucky no-upset sample decide the aggregate.
_FAVORITE_THRESHOLD = 0.6  # mid beyond this on the outcome side = market called it
_UPSET_THRESHOLD = 0.4     # mid beyond this AGAINST the outcome = market was wrong


def classify_regime(mid_price: float, outcome: float) -> str:
    """Classify a resolved scenario by how the market had priced it.

    - ``favorite``: the market side that won was priced > 0.6 — extreme
      posteriors get rewarded here.
    - ``upset``: the market priced the winning side < 0.4 — extreme posteriors
      on the market's side get punished hard here.
    - ``coin_flip``: mid in [0.4, 0.6] — the market itself was uncertain.
    """
    winning_side_price = mid_price if outcome == 1.0 else 1.0 - mid_price
    if winning_side_price > _FAVORITE_THRESHOLD:
        return "favorite"
    if winning_side_price < _UPSET_THRESHOLD:
        return "upset"
    return "coin_flip"


def brier(posterior: float, outcome: float) -> float:
    """Squared error of P(YES) vs the outcome — matches freqpred.metrics.calibration:
    ``estimated_probability`` is always P(YES) regardless of direction, so no
    direction-based flip is applied."""
    return (posterior - outcome) ** 2


def log_loss(posterior: float, outcome: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, posterior))
    return -(outcome * math.log(p) + (1.0 - outcome) * math.log(1.0 - p))


def direction_correct(direction: str, outcome: float) -> bool | None:
    """True/False for a graded YES/NO call; None for SKIP (nothing to grade)."""
    if direction == "YES":
        return outcome == 1.0
    if direction == "NO":
        return outcome == 0.0
    return None


@dataclass
class TradeDecision:
    would_trade: bool
    edge: float
    side_ask: float | None  # entry cost for the chosen side; None for SKIP
    ev: float | None        # settlement value minus entry cost, when trading


def trade_decision(
    output: ModelOutput,
    scenario: Scenario,
    *,
    min_edge: float,
    min_confidence: float,
) -> TradeDecision:
    """Would this output pull the trigger at the frozen prices — and at what EV?

    Trades iff direction != SKIP, side-specific edge >= min_edge, and
    confidence >= min_confidence (the confidence gate is what surfaces a
    "better Brier but too timid to trade" candidate). Edge uses the same
    side-specific ask arithmetic as production (`compute_signal_edge`).

    EV per contract at settlement: YES pays ``outcome - yes_ask``; NO pays
    ``(1 - outcome) - no_ask``.
    """
    edge, side_ask = compute_signal_edge(
        output.direction, output.posterior, scenario.yes_bid, scenario.yes_ask
    )
    would = (
        output.direction in ("YES", "NO")
        and edge >= min_edge
        and output.confidence >= min_confidence
    )
    ev: float | None = None
    if would:
        if output.direction == "YES":
            ev = scenario.outcome - scenario.yes_ask
        else:
            no_ask = round(1.0 - scenario.yes_bid, 4)
            ev = (1.0 - scenario.outcome) - no_ask
    return TradeDecision(would_trade=would, edge=edge, side_ask=side_ask, ev=ev)


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of *values* (deterministic seed)."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test p-value; ties are excluded upstream."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5**n
    return min(1.0, 2.0 * tail)


@dataclass
class PairScore:
    """Per-scenario comparison of incumbent vs candidate point estimate."""

    scenario_id: str
    market_id: str
    contaminated: bool
    outcome: float
    regime: str  # "favorite" | "upset" | "coin_flip" — see classify_regime()
    incumbent_brier: float
    candidate_brier: float
    brier_delta: float          # candidate - incumbent; negative = candidate better
    incumbent_log_loss: float
    candidate_log_loss: float
    incumbent_direction_correct: bool | None
    candidate_direction_correct: bool | None
    posterior_delta: float
    confidence_delta: float
    incumbent_trade: TradeDecision
    candidate_trade: TradeDecision


def score_pair(
    scenario: Scenario,
    candidate: ModelOutput,
    *,
    min_edge: float,
    min_confidence: float,
) -> PairScore:
    inc = scenario.incumbent
    return PairScore(
        scenario_id=scenario.id,
        market_id=scenario.market_id,
        contaminated=scenario.contaminated,
        outcome=scenario.outcome,
        regime=classify_regime(scenario.mid_price, scenario.outcome),
        incumbent_brier=brier(inc.posterior, scenario.outcome),
        candidate_brier=brier(candidate.posterior, scenario.outcome),
        brier_delta=brier(candidate.posterior, scenario.outcome)
        - brier(inc.posterior, scenario.outcome),
        incumbent_log_loss=log_loss(inc.posterior, scenario.outcome),
        candidate_log_loss=log_loss(candidate.posterior, scenario.outcome),
        incumbent_direction_correct=direction_correct(inc.direction, scenario.outcome),
        candidate_direction_correct=direction_correct(candidate.direction, scenario.outcome),
        posterior_delta=candidate.posterior - inc.posterior,
        confidence_delta=candidate.confidence - inc.confidence,
        incumbent_trade=trade_decision(
            inc, scenario, min_edge=min_edge, min_confidence=min_confidence
        ),
        candidate_trade=trade_decision(
            candidate, scenario, min_edge=min_edge, min_confidence=min_confidence
        ),
    )


def aggregate(scores: list[PairScore]) -> dict:
    """Aggregate paired statistics across scenarios.

    ``brier_delta_ci95`` excluding zero, or ``sign_test_p`` < 0.05, is the
    primary adopt/reject evidence; the trade-decision block is the
    degradation guard.
    """
    if not scores:
        return {"n_scenarios": 0}

    deltas = [s.brier_delta for s in scores]
    candidate_wins = sum(1 for d in deltas if d < 0)
    incumbent_wins = sum(1 for d in deltas if d > 0)

    def _accuracy(graded: list[bool | None]) -> tuple[int, int]:
        graded_only = [g for g in graded if g is not None]
        return sum(1 for g in graded_only if g), len(graded_only)

    inc_correct, inc_graded = _accuracy([s.incumbent_direction_correct for s in scores])
    cand_correct, cand_graded = _accuracy([s.candidate_direction_correct for s in scores])

    inc_trades = [s for s in scores if s.incumbent_trade.would_trade]
    cand_trades = [s for s in scores if s.candidate_trade.would_trade]
    disagreements = [
        {
            "scenario_id": s.scenario_id,
            "market_id": s.market_id,
            "outcome": s.outcome,
            "incumbent_trades": s.incumbent_trade.would_trade,
            "candidate_trades": s.candidate_trade.would_trade,
            "incumbent_edge": round(s.incumbent_trade.edge, 4),
            "candidate_edge": round(s.candidate_trade.edge, 4),
            "trade_ev": (
                s.incumbent_trade.ev
                if s.incumbent_trade.would_trade
                else s.candidate_trade.ev
            ),
        }
        for s in scores
        if s.incumbent_trade.would_trade != s.candidate_trade.would_trade
    ]

    # Regime breakdown: a hedging candidate loses on favorites and wins on
    # upsets — the split shows whether an aggregate result is a structural
    # difference or an artifact of how many upsets the sample happened to have.
    regimes: dict[str, dict] = {}
    for regime in ("favorite", "upset", "coin_flip"):
        segment = [s for s in scores if s.regime == regime]
        if not segment:
            continue
        segment_deltas = [s.brier_delta for s in segment]
        regimes[regime] = {
            "n": len(segment),
            "incumbent_mean_brier": mean(s.incumbent_brier for s in segment),
            "candidate_mean_brier": mean(s.candidate_brier for s in segment),
            "brier_delta_mean": mean(segment_deltas),
            "candidate_wins": sum(1 for d in segment_deltas if d < 0),
            "incumbent_wins": sum(1 for d in segment_deltas if d > 0),
        }

    lo, hi = bootstrap_mean_ci(deltas)
    return {
        "n_scenarios": len(scores),
        "n_contaminated": sum(1 for s in scores if s.contaminated),
        "incumbent_mean_brier": mean(s.incumbent_brier for s in scores),
        "candidate_mean_brier": mean(s.candidate_brier for s in scores),
        "brier_delta_mean": mean(deltas),
        "brier_delta_median": median(deltas),
        "brier_delta_ci95": [lo, hi],
        "brier_delta_significant": hi < 0 or lo > 0,
        "candidate_wins": candidate_wins,
        "incumbent_wins": incumbent_wins,
        "ties": len(deltas) - candidate_wins - incumbent_wins,
        "sign_test_p": sign_test_p(candidate_wins, incumbent_wins),
        "incumbent_mean_log_loss": mean(s.incumbent_log_loss for s in scores),
        "candidate_mean_log_loss": mean(s.candidate_log_loss for s in scores),
        "incumbent_direction_accuracy": [inc_correct, inc_graded],
        "candidate_direction_accuracy": [cand_correct, cand_graded],
        "regimes": regimes,
        "trade_decisions": {
            "incumbent_would_trade": len(inc_trades),
            "candidate_would_trade": len(cand_trades),
            "incumbent_mean_ev_per_trade": (
                mean(s.incumbent_trade.ev for s in inc_trades) if inc_trades else None
            ),
            "candidate_mean_ev_per_trade": (
                mean(s.candidate_trade.ev for s in cand_trades) if cand_trades else None
            ),
            "disagreements": disagreements,
        },
    }
