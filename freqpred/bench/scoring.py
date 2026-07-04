"""Scoring for the benchmark harness (T93).

Calibration metrics (Brier, log loss, direction correctness) score each
model's posterior against the actual market outcome. Paired statistics
(bootstrap CI on the mean Brier delta, exact sign test) are the adopt/reject
gate: a raw mean improvement on a small noisy sample is not evidence. Both
are clustered by market — signals within one market share its outcome, so
the bootstrap resamples markets and the sign test votes once per market.

Trade-decision metrics capture what Brier misses — a better-calibrated but
systematically less confident model trades less, and confidence also scales
position size (the Kelly blend in ``IPredictionStrategy.position_size``), so
an overconfident model has more dollars on the line when it is wrong. Each
would-trade therefore carries a ``stake`` — the run strategy's own
``position_size()`` called with a Signal built from that model's posterior
and confidence — and a stake-weighted settlement P&L. Custom strategies with
overridden sizing are honored automatically. These are decision-quality
signals at frozen side-specific prices, deliberately **not** a portfolio
simulation: no risk caps, no exits, no bankroll path — each stake assumes
zero existing exposure and no assessment, independent across scenarios.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from typing import TYPE_CHECKING

from freqpred.bench.scenarios import ModelOutput, Scenario
from freqpred.signal.models import Signal
from freqpred.signal.pipeline import compute_signal_edge

if TYPE_CHECKING:
    from freqpred.strategy.base import IPredictionStrategy

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


def _signal_for_output(output: ModelOutput, scenario: Scenario) -> Signal:
    """Materialize the Signal DTO that production sizing consumes, priced with
    the same side-specific edge arithmetic as the live pipeline. Provenance
    fields are placeholders — ``position_size`` reads only direction,
    estimated_probability, edge, and confidence."""
    edge, side_ask = compute_signal_edge(
        output.direction, output.posterior, scenario.yes_bid, scenario.yes_ask
    )
    return Signal(
        id=scenario.id,
        market_id=scenario.market_id,
        estimated_probability=output.posterior,
        confidence=output.confidence,
        edge=edge,
        market_mid_at_signal=scenario.mid_price,
        direction=output.direction,
        reasoning=output.reasoning,
        sources=[],
        retrieval_hash="",
        model_used=output.model,
        prompt_version="",
        trigger="manual",
        created_at=scenario.close_time,
        raw_context="",
        market_ask_at_signal=side_ask,
    )


@dataclass
class TradeDecision:
    would_trade: bool
    edge: float
    side_ask: float | None  # entry cost for the chosen side; None for SKIP
    ev: float | None        # settlement value minus entry cost, when trading
    stake: float | None     # strategy.position_size() dollars at the run bankroll
    pnl: float | None       # stake-weighted settlement P&L: (stake / side_ask) * ev


def trade_decision(
    output: ModelOutput,
    scenario: Scenario,
    *,
    min_edge: float,
    min_confidence: float,
    strategy: IPredictionStrategy,
    bankroll: float,
) -> TradeDecision:
    """Would this output pull the trigger at the frozen prices — and at what EV?

    Trades iff direction != SKIP, side-specific edge >= min_edge, and
    confidence >= min_confidence (the confidence gate is what surfaces a
    "better Brier but too timid to trade" candidate). Edge uses the same
    side-specific ask arithmetic as production (`compute_signal_edge`).

    Beyond min_edge/min_confidence, the gate applies the signal-level entry
    filters from the strategy's config, mirroring ``should_trade``:
    ``max_edge`` (a huge edge usually means the market is right and the model
    is wrong) and ``min_mid_price``/``max_mid_price`` on the entry side's own
    cost (longshot/decided-market filter). Market-selection filters (volume,
    category, days-to-close) and the spread gate are not replayable from a
    frozen scenario and are identical for both sides of a comparison.

    EV per contract at settlement: YES pays ``outcome - yes_ask``; NO pays
    ``(1 - outcome) - no_ask``. ``stake`` is the *strategy's* own
    ``position_size()`` for a Signal built from this output's posterior and
    confidence (zero existing exposure, no assessment), and ``pnl`` is the
    settlement P&L of that stake (stake dollars buy stake/side_ask contracts)
    — overconfidence when wrong loses proportionally more. Two deliberate
    simplifications vs live: entry fills at the frozen ask (production posts
    resting limits at estimated_probability - min_edge), and positions ride
    to settlement (production exits via stoploss/signal/force_exit).
    """
    edge, side_ask = compute_signal_edge(
        output.direction, output.posterior, scenario.yes_bid, scenario.yes_ask
    )
    cfg = strategy.config
    side_price = (
        1.0 - scenario.mid_price if output.direction == "NO" else scenario.mid_price
    )
    would = (
        output.direction in ("YES", "NO")
        and edge >= min_edge
        and output.confidence >= min_confidence
        and (cfg.max_edge is None or edge <= cfg.max_edge)
        and (cfg.min_mid_price is None or side_price >= cfg.min_mid_price)
        and (cfg.max_mid_price is None or side_price <= cfg.max_mid_price)
    )
    ev: float | None = None
    stake: float | None = None
    pnl: float | None = None
    if would:
        if output.direction == "YES":
            ev = scenario.outcome - scenario.yes_ask
        else:
            no_ask = round(1.0 - scenario.yes_bid, 4)
            ev = (1.0 - scenario.outcome) - no_ask
        stake = strategy.position_size(_signal_for_output(output, scenario), bankroll)
        pnl = (stake / side_ask) * ev if side_ask else 0.0
    return TradeDecision(
        would_trade=would, edge=edge, side_ask=side_ask, ev=ev, stake=stake, pnl=pnl
    )


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


def cluster_bootstrap_mean_ci(
    values_by_cluster: dict[str, list[float]],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling whole clusters.

    Multiple signals on one market share the market's outcome, so their
    deltas are correlated — resampling scenarios independently would fake
    n-fold more evidence than exists. Resampling markets (clusters) with
    replacement keeps the CI honest; with one scenario per cluster this
    degenerates to the plain bootstrap.
    """
    clusters = sorted(values_by_cluster)
    if not clusters:
        return (0.0, 0.0)
    if len(clusters) == 1:
        m = mean(values_by_cluster[clusters[0]])
        return (m, m)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        resampled: list[float] = []
        for _ in range(len(clusters)):
            resampled.extend(values_by_cluster[rng.choice(clusters)])
        means.append(mean(resampled))
    means.sort()
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
    signal_time: datetime | None
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
    strategy: IPredictionStrategy,
    bankroll: float,
) -> PairScore:
    inc = scenario.incumbent
    return PairScore(
        scenario_id=scenario.id,
        market_id=scenario.market_id,
        signal_time=scenario.signal_time,
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
            inc, scenario, min_edge=min_edge, min_confidence=min_confidence,
            strategy=strategy, bankroll=bankroll,
        ),
        candidate_trade=trade_decision(
            candidate, scenario, min_edge=min_edge, min_confidence=min_confidence,
            strategy=strategy, bankroll=bankroll,
        ),
    )


def _entry_scores(
    by_market: dict[str, list[PairScore]], side: str
) -> dict[str, PairScore]:
    """Each model's entry per market: the FIRST gate-clearing signal, matching
    production, where entry happens on the first signal that passes
    should_trade and existing exposure then blocks re-entry. Later signals on
    the same market are exit/monitoring inputs, not additional trades."""
    entries: dict[str, PairScore] = {}
    for market_id, series in by_market.items():
        for score in series:  # series is signal_time-sorted
            trade = score.incumbent_trade if side == "incumbent" else score.candidate_trade
            if trade.would_trade:
                entries[market_id] = score
                break
    return entries


def aggregate(scores: list[PairScore]) -> dict:
    """Aggregate paired statistics, clustered by market.

    Signals within one market share its outcome, so scenarios are not
    independent: the bootstrap CI resamples markets, and the sign test gets
    one vote per market (the sign of its mean Brier delta).
    ``brier_delta_ci95`` excluding zero, or ``sign_test_p`` < 0.05, is the
    primary adopt/reject evidence; the trade-decision block is the
    degradation guard.
    """
    if not scores:
        return {"n_scenarios": 0}

    by_market: dict[str, list[PairScore]] = {}
    for score in scores:
        by_market.setdefault(score.market_id, []).append(score)
    for series in by_market.values():
        series.sort(key=lambda s: (s.signal_time is None, s.signal_time))

    deltas = [s.brier_delta for s in scores]
    deltas_by_market = {
        market_id: [s.brier_delta for s in series]
        for market_id, series in by_market.items()
    }
    market_mean_deltas = {m: mean(v) for m, v in deltas_by_market.items()}
    candidate_wins = sum(1 for d in market_mean_deltas.values() if d < 0)
    incumbent_wins = sum(1 for d in market_mean_deltas.values() if d > 0)

    def _accuracy(graded: list[bool | None]) -> tuple[int, int]:
        graded_only = [g for g in graded if g is not None]
        return sum(1 for g in graded_only if g), len(graded_only)

    inc_correct, inc_graded = _accuracy([s.incumbent_direction_correct for s in scores])
    cand_correct, cand_graded = _accuracy([s.candidate_direction_correct for s in scores])

    # Entry-faithful trades: one entry per market per model, at that model's
    # own first gate-clearing signal — models may enter the same market at
    # different times and prices, exactly like live.
    inc_entries = _entry_scores(by_market, "incumbent")
    cand_entries = _entry_scores(by_market, "candidate")
    common_markets = sorted(set(inc_entries) & set(cand_entries))
    disagreements = []
    for market_id in sorted(set(inc_entries) ^ set(cand_entries)):
        entry = inc_entries.get(market_id) or cand_entries[market_id]
        side = "incumbent" if market_id in inc_entries else "candidate"
        trade = entry.incumbent_trade if side == "incumbent" else entry.candidate_trade
        disagreements.append(
            {
                "scenario_id": entry.scenario_id,
                "market_id": market_id,
                "outcome": entry.outcome,
                "entered_by": side,
                "entry_signal_time": (
                    entry.signal_time.isoformat() if entry.signal_time else None
                ),
                "entry_edge": round(trade.edge, 4),
                "trade_ev": trade.ev,
                "trade_stake": trade.stake,
            }
        )

    # Stake-weighted view: confidence scales position size in production, so
    # even when both models trade the same market, the more confident one has
    # more dollars on the line — its being wrong must cost more here too.
    inc_total_stake = sum(s.incumbent_trade.stake or 0.0 for s in inc_entries.values())
    cand_total_stake = sum(s.candidate_trade.stake or 0.0 for s in cand_entries.values())
    inc_total_pnl = sum(s.incumbent_trade.pnl or 0.0 for s in inc_entries.values())
    cand_total_pnl = sum(s.candidate_trade.pnl or 0.0 for s in cand_entries.values())

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

    lo, hi = cluster_bootstrap_mean_ci(deltas_by_market)
    return {
        "n_scenarios": len(scores),
        "n_markets": len(by_market),
        "n_contaminated": sum(1 for s in scores if s.contaminated),
        "incumbent_mean_brier": mean(s.incumbent_brier for s in scores),
        "candidate_mean_brier": mean(s.candidate_brier for s in scores),
        "brier_delta_mean": mean(deltas),
        "brier_delta_median": median(deltas),
        "brier_delta_ci95": [lo, hi],
        "brier_delta_significant": hi < 0 or lo > 0,
        "candidate_wins": candidate_wins,   # markets, by sign of mean delta
        "incumbent_wins": incumbent_wins,
        "ties": len(by_market) - candidate_wins - incumbent_wins,
        "sign_test_p": sign_test_p(candidate_wins, incumbent_wins),
        "incumbent_mean_log_loss": mean(s.incumbent_log_loss for s in scores),
        "candidate_mean_log_loss": mean(s.candidate_log_loss for s in scores),
        "incumbent_direction_accuracy": [inc_correct, inc_graded],
        "candidate_direction_accuracy": [cand_correct, cand_graded],
        "regimes": regimes,
        "trade_decisions": {
            "incumbent_would_trade": len(inc_entries),   # markets entered
            "candidate_would_trade": len(cand_entries),
            "incumbent_mean_ev_per_trade": (
                mean(s.incumbent_trade.ev for s in inc_entries.values())
                if inc_entries else None
            ),
            "candidate_mean_ev_per_trade": (
                mean(s.candidate_trade.ev for s in cand_entries.values())
                if cand_entries else None
            ),
            "incumbent_total_stake": inc_total_stake,
            "candidate_total_stake": cand_total_stake,
            "incumbent_stake_weighted_pnl": inc_total_pnl,
            "candidate_stake_weighted_pnl": cand_total_pnl,
            "incumbent_pnl_per_dollar_staked": (
                inc_total_pnl / inc_total_stake if inc_total_stake > 0 else None
            ),
            "candidate_pnl_per_dollar_staked": (
                cand_total_pnl / cand_total_stake if cand_total_stake > 0 else None
            ),
            "common_trades": {
                "n": len(common_markets),
                "incumbent_mean_stake": (
                    mean(inc_entries[m].incumbent_trade.stake for m in common_markets)
                    if common_markets else None
                ),
                "candidate_mean_stake": (
                    mean(cand_entries[m].candidate_trade.stake for m in common_markets)
                    if common_markets else None
                ),
            },
            "disagreements": disagreements,
        },
    }
