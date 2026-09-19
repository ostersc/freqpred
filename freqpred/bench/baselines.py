"""Free baselines the signal estimator has to clear.

The benchmark harness compares a candidate model against the *incumbent model*.
That answers "is this prompt/model better than the last one" and cannot answer
the question the project actually rested on: does any of it beat an estimator
that costs nothing? Twelve signal-prompt versions were screened without that
comparator, and when it was finally run the model lost to a Poisson process fit
on the same data it had been handed (Brier 0.2407 vs 0.2093) and to a constant
(0.2349). See ``docs/POSTMORTEM.md`` §3.1 and §6.

This module makes those baselines first-class so the comparison runs on every
benchmark instead of being reconstructed by hand.

Point-in-time correctness
-------------------------
The Poisson rates are parsed out of the **frozen rendered prompt**, never from
``factbase_phrase_frequency``. That table holds one row per market refreshed
until close, so it carries *final pre-close* counts rather than counts as-of
signal time; scoring a baseline off it hands that baseline look-ahead over the
model it is being compared against, manufacturing the very result under test
(``docs/weekly-review/reports/2026-07-28.md``). The prompt is the only
per-signal record of what the model was actually shown, and it prints the
counts and the remaining window verbatim.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean

from freqpred.bench.scenarios import Scenario
from freqpred.bench.scoring import (
    brier,
    cluster_bootstrap_mean_ci,
    log_loss,
    sign_test_p,
)

# ``_build_factbase_block`` prints ``days_to_close`` to one decimal, so the
# remaining window recovered from a prompt carries up to +/-0.05 days of error.
# At short horizons that dominates: at 0.7 printed days the true window can be
# 0.65-0.75, a 7% swing in lambda*t. The self-check tolerance is therefore
# derived per scenario rather than fixed -- see ``_recompute_tolerance``.
_DAYS_PRINT_HALF_WIDTH = 0.05
_PROB_PRINT_HALF_WIDTH = 0.0005

_COUNT_RE = {
    "in_market_count": re.compile(r"^\s*Since market opened\s*:\s*(\d+)\s*$", re.M),
    "count_7d": re.compile(r"^\s*Last 7 days\s*:\s*(\d+)\s*$", re.M),
    "count_30d": re.compile(r"^\s*Last 30 days\s*:\s*(\d+)\s*$", re.M),
    "count_365d": re.compile(r"^\s*Last 365 days\s*:\s*(\d+)\s*$", re.M),
}
# The remaining window is taken from the Poisson header rather than the window
# line above it: this is the exact value _build_factbase_block passed into the
# formula, so parsing it here keeps the re-computation self-consistent.
_DAYS_RE = re.compile(r"Poisson baseline P\(≥?1 occurrence in remaining ([0-9.]+) days\)")
_PRINTED_365_RE = re.compile(r"^\s*Using 365d rate\s*:\s*([0-9.]+)%\s*$", re.M)
_PRINTED_30_RE = re.compile(r"^\s*Using 30d rate\s*:\s*([0-9.]+)%\s*$", re.M)


@dataclass(frozen=True)
class FactbaseNumbers:
    """The point-in-time FactBase figures recovered from a rendered prompt."""

    in_market_count: int
    count_7d: int
    count_30d: int
    count_365d: int
    days_to_close: float
    printed_poisson_30d: float | None
    printed_poisson_365d: float | None

    @property
    def poisson_30d(self) -> float:
        """The 30d Poisson baseline, preferring the value the model was shown.

        The printed figure is authoritative: it is what the prompt put in front
        of the model, and it is free of the rounding error that recomputing
        from the one-decimal ``days_to_close`` introduces at short horizons.
        """
        if self.printed_poisson_30d is not None:
            return self.printed_poisson_30d
        return self.recomputed_poisson_30d

    @property
    def poisson_365d(self) -> float:
        if self.printed_poisson_365d is not None:
            return self.printed_poisson_365d
        return self.recomputed_poisson_365d

    @property
    def recomputed_poisson_30d(self) -> float:
        return poisson_probability(self.count_30d, 30.0, self.days_to_close)

    @property
    def recomputed_poisson_365d(self) -> float:
        return poisson_probability(self.count_365d, 365.0, self.days_to_close)

    def printed_mismatch(self) -> str | None:
        """Flag a genuine disagreement between the counts and the printed value.

        Recomputing from the counts and the one-decimal ``days_to_close`` cannot
        land exactly on the printed probability, so the tolerance is the width
        that rounding alone can explain. Anything wider means the parser latched
        onto the wrong numbers, which would silently score a baseline the model
        was never shown.
        """
        for label, count, window, recomputed, printed in (
            ("30d", self.count_30d, 30.0, self.recomputed_poisson_30d, self.printed_poisson_30d),
            ("365d", self.count_365d, 365.0, self.recomputed_poisson_365d, self.printed_poisson_365d),
        ):
            if printed is None:
                continue
            if abs(recomputed - printed) > _recompute_tolerance(count, window, recomputed):
                return (
                    f"{label} Poisson recomputed {recomputed:.4f} but prompt printed "
                    f"{printed:.4f}"
                )
        return None


def poisson_probability(count: int, window_days: float, days_to_close: float) -> float:
    """P(at least one occurrence in the remaining window) under a Poisson fit.

    Mirrors ``freqpred.signal.llm._build_factbase_block`` exactly, including its
    treatment of a zero count as a hard 0.0 rather than a smoothed rate — that
    is the estimator the model was shown and therefore the one to score.
    """
    daily_rate = count / window_days
    if daily_rate <= 0:
        return 0.0
    return 1.0 - math.exp(-daily_rate * days_to_close)


def _recompute_tolerance(count: int, window_days: float, probability: float) -> float:
    """How far recomputation can drift from the printed value on rounding alone.

    ``p = 1 - exp(-lambda*t)``, so ``dp/dt = lambda*(1-p)``; with ``t`` known
    only to +/-0.05 days the probability is uncertain by that much again, plus
    the half-width of the printed figure itself.
    """
    daily_rate = count / window_days
    return daily_rate * _DAYS_PRINT_HALF_WIDTH * (1.0 - probability) + _PROB_PRINT_HALF_WIDTH


def parse_factbase(prompt: str) -> FactbaseNumbers | None:
    """Recover the FactBase figures from a rendered prompt.

    Returns ``None`` when the prompt carries no FactBase block — a market
    outside the phrase-frequency allowlist has no Poisson baseline, and scoring
    it as 0.0 would invent an estimator that was never shown.
    """
    days_match = _DAYS_RE.search(prompt)
    if days_match is None:
        return None

    counts: dict[str, int] = {}
    for field, pattern in _COUNT_RE.items():
        match = pattern.search(prompt)
        if match is None:
            return None
        counts[field] = int(match.group(1))

    def _printed(pattern: re.Pattern[str]) -> float | None:
        match = pattern.search(prompt)
        return float(match.group(1)) / 100.0 if match else None

    return FactbaseNumbers(
        days_to_close=float(days_match.group(1)),
        printed_poisson_30d=_printed(_PRINTED_30_RE),
        printed_poisson_365d=_printed(_PRINTED_365_RE),
        **counts,
    )


# --- Estimators ------------------------------------------------------------
# Each returns P(YES) or None when it does not apply to that scenario. None is
# never coerced to a number: a baseline that cannot speak is dropped from its
# own comparison rather than scored as a confident 0.0.


def poisson_30d_estimate(scenario: Scenario) -> float | None:
    numbers = parse_factbase(scenario.prompt)
    return None if numbers is None else numbers.poisson_30d


def poisson_365d_estimate(scenario: Scenario) -> float | None:
    numbers = parse_factbase(scenario.prompt)
    return None if numbers is None else numbers.poisson_365d


def market_mid_estimate(scenario: Scenario) -> float | None:
    """The market's own price — the estimator an edge has to be measured against."""
    return scenario.mid_price


def constant_estimator(
    scenarios: Sequence[Scenario], *, leave_one_market_out: bool = False
) -> Callable[[Scenario], float | None]:
    """A fixed probability, ignoring every input.

    ``leave_one_market_out`` fits each market's constant on every *other*
    market, so the baseline never sees the outcome it is scored against. The
    pooled default reproduces the published comparison and is deliberately
    generous to the baseline — it already beat the model with that advantage.
    """
    outcome_by_market = {s.market_id: s.outcome for s in scenarios}
    if not outcome_by_market:
        return lambda _scenario: None

    total = sum(outcome_by_market.values())
    n = len(outcome_by_market)

    if not leave_one_market_out:
        pooled = total / n
        return lambda _scenario: pooled

    def _estimate(scenario: Scenario) -> float | None:
        if n <= 1:
            return None
        own = outcome_by_market.get(scenario.market_id, 0.0)
        return (total - own) / (n - 1)

    return _estimate


BASELINES: dict[str, Callable[[Scenario], float | None]] = {
    "poisson_30d": poisson_30d_estimate,
    "poisson_365d": poisson_365d_estimate,
    "market_mid": market_mid_estimate,
}


# --- Scoring ---------------------------------------------------------------


@dataclass
class BaselineScore:
    """How one free baseline did, and how the model did on the same scenarios."""

    name: str
    n_scenarios: int
    n_markets: int
    n_skipped: int
    baseline_mean_brier: float
    model_mean_brier: float
    baseline_mean_log_loss: float
    model_mean_log_loss: float
    brier_delta_mean: float          # model - baseline; negative = model better
    brier_delta_ci95: tuple[float, float]
    model_beats_baseline: bool
    sign_test_p: float
    parse_warnings: list[str]

    def as_dict(self) -> dict:
        return {
            "n_scenarios": self.n_scenarios,
            "n_markets": self.n_markets,
            "n_skipped": self.n_skipped,
            "baseline_mean_brier": self.baseline_mean_brier,
            "model_mean_brier": self.model_mean_brier,
            "baseline_mean_log_loss": self.baseline_mean_log_loss,
            "model_mean_log_loss": self.model_mean_log_loss,
            "brier_delta_mean": self.brier_delta_mean,
            "brier_delta_ci95": list(self.brier_delta_ci95),
            "model_beats_baseline": self.model_beats_baseline,
            "sign_test_p": self.sign_test_p,
            "parse_warnings": self.parse_warnings[:10],
        }


def score_baseline(
    name: str,
    estimator: Callable[[Scenario], float | None],
    scenarios: Sequence[Scenario],
    model_estimate: Callable[[Scenario], float],
) -> BaselineScore | None:
    """Score one baseline against the model over the scenarios it applies to.

    The comparison is paired and clustered by market: signals on one market
    share its outcome, so they are not independent observations.
    """
    paired: list[tuple[str, float, float, float]] = []
    warnings: list[str] = []
    skipped = 0
    is_poisson = name.startswith("poisson")

    for scenario in scenarios:
        estimate = estimator(scenario)
        if estimate is None:
            skipped += 1
            continue
        if is_poisson:
            numbers = parse_factbase(scenario.prompt)
            mismatch = numbers.printed_mismatch() if numbers else None
            if mismatch is not None:
                warnings.append(f"{scenario.id}: {mismatch}")
        paired.append(
            (scenario.market_id, estimate, model_estimate(scenario), scenario.outcome)
        )

    if not paired:
        return None

    by_market: dict[str, list[float]] = {}
    baseline_briers: list[float] = []
    model_briers: list[float] = []
    baseline_lls: list[float] = []
    model_lls: list[float] = []

    for market_id, estimate, model_p, outcome in paired:
        b_brier = brier(estimate, outcome)
        m_brier = brier(model_p, outcome)
        baseline_briers.append(b_brier)
        model_briers.append(m_brier)
        baseline_lls.append(log_loss(estimate, outcome))
        model_lls.append(log_loss(model_p, outcome))
        by_market.setdefault(market_id, []).append(m_brier - b_brier)

    lo, hi = cluster_bootstrap_mean_ci(by_market)
    market_means = {m: mean(v) for m, v in by_market.items()}
    model_wins = sum(1 for d in market_means.values() if d < 0)
    baseline_wins = sum(1 for d in market_means.values() if d > 0)

    return BaselineScore(
        name=name,
        n_scenarios=len(paired),
        n_markets=len(by_market),
        n_skipped=skipped,
        baseline_mean_brier=mean(baseline_briers),
        model_mean_brier=mean(model_briers),
        baseline_mean_log_loss=mean(baseline_lls),
        model_mean_log_loss=mean(model_lls),
        brier_delta_mean=mean(m - b for m, b in zip(model_briers, baseline_briers, strict=True)),
        brier_delta_ci95=(lo, hi),
        model_beats_baseline=hi < 0,
        sign_test_p=sign_test_p(model_wins, baseline_wins),
        parse_warnings=warnings,
    )




def score_all_baselines(
    scenarios: Sequence[Scenario],
    model_estimate: Callable[[Scenario], float],
    *,
    leave_one_market_out_constant: bool = False,
) -> dict[str, dict]:
    """Score every free baseline against the model. Safe to call with no data."""
    if not scenarios:
        return {}

    estimators = dict(BASELINES)
    estimators["constant"] = constant_estimator(
        scenarios, leave_one_market_out=leave_one_market_out_constant
    )

    results: dict[str, dict] = {}
    for name, estimator in estimators.items():
        score = score_baseline(name, estimator, scenarios, model_estimate)
        if score is not None:
            results[name] = score.as_dict()
    return results
