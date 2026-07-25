"""Weekly profitability review — deterministic counterfactual analysis, no LLM calls.

This is the data layer behind the `weekly-review` skill
(`.claude/skills/weekly-review/`), which runs once a week after markets resolve.
Everything here is pure computation over rows already in the DB, so a review
costs nothing, is reproducible, and can be re-run over any window.

What it answers
---------------
1. **Ledger** — what actually happened in the window, split by mode.
2. **Exit effectiveness** — for every early exit on a market that has since
   resolved, what P&L would holding to settlement have produced? A stoploss that
   fires on a market we ultimately win is a pure loss.
3. **Stoploss sweep** — using each position's recorded MAE (worst price
   excursion from entry), what would total P&L have been at other stoploss
   thresholds?
4. **Entry gates** — for each `StrategyConfig` gate, what is the realised profit
   edge of the signals it *blocks* versus the ones it *admits*? A gate whose
   blocked set is profitable is costing money.
5. **Signal accuracy** — hit rate and profit-vs-price by week, prompt version,
   and correlate slices (direction, edge band, confidence, days to close).
6. **Assessor** — capital tilt (mean size multiplier on winners minus losers),
   verdict mix, neutral-fallback rate, and realised sizing P&L delta.
7. **Sources** — realised profit edge of signals that retrieved each document
   source, versus the pool.

Conventions that matter
-----------------------
* **Prices are in the traded side's own space.** `Position.entry_price` /
  `exit_price` and `Signal.market_ask_at_signal` are what that side cost, so a
  NO position's entry_price is the NO price. Settlement is therefore 1.0 when
  the position's side won and 0.0 when it lost, for both directions — no
  inversion is needed anywhere in this module, which is exactly why every test
  must still cover both directions.
* **Profit edge vs price** = ``hit_rate - mean(ask_paid)``. It is the per-contract
  expected profit in dollars, and it is the metric the sizing assessor is also
  judged on (`freqpred/metrics/CLAUDE.md`). It is *not* the same as calibration:
  a model can be badly calibrated and still profitable, and the reverse.
* **Signals cluster by market.** Re-evaluation produces many signals per market
  (every 30 min), so signal-level n is not an independent sample size. Every
  table reports `n_markets` alongside `n_signals`; treat the market count as the
  real one when judging whether a result is meaningful.
* **Paper and live are different populations.** Paper fills are frictionless and
  its P&L is not evidence about live execution. Nothing here pools them.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Positions closed for these reasons never had a "hold to resolution"
# alternative — the market resolving IS the exit, or the position never opened.
_NON_DISCRETIONARY_EXITS = frozenset({"market_resolved", "cancelled"})

# Gates in StrategyConfig that cannot be evaluated point-in-time.
# markets.volume_24h holds the CURRENT value, not the value at signal time, so a
# min_volume_24h counterfactual would score today's liquidity against a decision
# made weeks ago. Reported as a known blind spot rather than guessed at.
UNEVALUABLE_GATES = ("min_volume_24h",)

_DEFAULT_SEED = 20260725


# --------------------------------------------------------------------------
# Row types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """Analysis window. `end` is exclusive."""

    start: datetime
    end: datetime
    weeks: int

    @property
    def label(self) -> str:
        return f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d} ({self.weeks}w)"


@dataclass(frozen=True)
class ClosedPosition:
    position_id: str
    market_id: str
    series_ticker: str | None
    direction: str
    mode: str
    strategy_name: str
    contracts: int
    entry_price: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    pnl: float
    entry_fee_usd: float
    exit_fee_usd: float
    mae: float | None
    result: str | None
    signal_edge: float
    signal_confidence: float
    size_multiplier: float | None = None
    verdict: str | None = None
    trust_score: float | None = None
    assessment_version: str | None = None


@dataclass(frozen=True)
class ResolvedSignal:
    signal_id: str
    market_id: str
    series_ticker: str | None
    direction: str
    created_at: datetime
    close_time: datetime
    edge: float
    confidence: float
    estimated_probability: float
    market_mid_at_signal: float
    market_ask_at_signal: float
    prompt_version: str
    trigger: str
    result: str
    traded: bool
    sources: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return signal_hit(self.direction, self.result)

    @property
    def days_to_close(self) -> float:
        return (self.close_time - self.created_at).total_seconds() / 86400.0


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitStat:
    exit_reason: str
    direction: str
    n: int
    n_unresolved: int
    actual_pnl: float
    held_pnl: float
    saved: float
    saved_ci: tuple[float, float]
    n_helped: int
    n_hurt: int


@dataclass(frozen=True)
class StoplossPoint:
    threshold: float
    n_stopped: int
    n_censored: int
    total_pnl: float
    delta_vs_actual: float


@dataclass(frozen=True)
class GroupStat:
    """Realised performance of a set of signals, deduped to one per market."""

    key: str
    n_signals: int
    n_markets: int
    hit_rate: float
    avg_ask: float
    profit_edge: float
    profit_edge_ci: tuple[float, float]

    @property
    def per_100_contracts(self) -> float:
        return self.profit_edge * 100.0


@dataclass(frozen=True)
class GateStat:
    gate: str
    setting: str
    admitted: GroupStat
    blocked: GroupStat


@dataclass(frozen=True)
class SweepPoint:
    gate: str
    value: float
    n_markets: int
    hit_rate: float
    profit_edge: float
    total_expected_pnl_per_contract: float


@dataclass(frozen=True)
class AssessorStat:
    version: str
    n_signals: int
    capital_tilt: float
    capital_tilt_ci: tuple[float, float]
    verdict_mix: dict[str, int]
    mean_multiplier: float
    realized_sizing_delta_usd: float | None
    n_positions: int


@dataclass
class WeeklyReview:
    window: Window
    generated_at: datetime
    ledger: dict[str, dict[str, float]]
    exit_stats: list[ExitStat]
    exit_stats_window: list[ExitStat]
    stoploss_sweep: list[StoplossPoint]
    stoploss_actual_pnl: float
    stoploss_sweep_uncensored: list[StoplossPoint]
    stoploss_uncensored_pnl: float
    gate_stats: list[GateStat]
    gate_sweeps: list[SweepPoint]
    accuracy_by_week: list[GroupStat]
    accuracy_by_version: list[GroupStat]
    accuracy_slices: dict[str, list[GroupStat]]
    assessor_stats: list[AssessorStat]
    assessor_neutral_fallback_rate: float | None
    source_stats: list[GroupStat]
    source_pool: GroupStat | None
    scope: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def signal_hit(direction: str, result: str | None) -> bool:
    """True when the traded side won. Works identically for YES and NO."""
    return (direction == "YES" and result == "yes") or (
        direction == "NO" and result == "no"
    )


def settle_price(direction: str, result: str | None) -> float | None:
    """Settlement value in the traded side's own price space, or None if unresolved.

    Kalshi pays $1 per winning contract. Because entry/exit prices are already
    stored in the traded side's space, a winning NO settles at 1.0 exactly as a
    winning YES does.
    """
    if result not in ("yes", "no"):
        return None
    return 1.0 if signal_hit(direction, result) else 0.0


def hold_to_resolution_pnl(position: ClosedPosition) -> float | None:
    """P&L had this position been held to settlement instead of exited early.

    Settlement charges no exit fee on Kalshi, so only the entry fee is deducted —
    which is also why the comparison is not symmetric: the actual P&L already
    paid an exit fee that the counterfactual never incurs.
    """
    settle = settle_price(position.direction, position.result)
    if settle is None:
        return None
    return (settle - position.entry_price) * position.contracts - position.entry_fee_usd


def reconstruct_spread(
    direction: str, mid: float, side_ask: float
) -> float | None:
    """Recover the yes_bid/yes_ask spread from what a signal stored.

    Only the traded side's price is stored directly; the other side is derived by
    reflecting around the mid. That is sound only when mid and ask came from the
    same book snapshot. When they did not, the reflection produces bid > ask —
    a reconstruction failure, not a market state — and this returns None rather
    than a negative spread. Measured at ~10% of signals (2026-07-24).
    """
    if direction == "YES":
        spread = 2.0 * (side_ask - mid)
    elif direction == "NO":
        spread = 2.0 * (mid + side_ask - 1.0)
    else:
        return None
    return spread if spread >= 0.0 else None


def entry_side_cost(direction: str, mid: float) -> float:
    """The mid-implied cost of the entry side — what min/max_mid_price gates."""
    return mid if direction == "YES" else 1.0 - mid


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. (0.0, 0.0) on an empty sample."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    )
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return (lo, hi)


def first_per_market(signals: Iterable[ResolvedSignal]) -> list[ResolvedSignal]:
    """Keep the earliest signal per market.

    Entry is a once-per-market decision — the strategy does not re-enter a market
    it already holds — so counterfactual entry analysis must not count the same
    market once per re-evaluation. Ties break on signal_id for determinism.
    """
    best: dict[str, ResolvedSignal] = {}
    for sig in sorted(signals, key=lambda s: (s.created_at, s.signal_id)):
        best.setdefault(sig.market_id, sig)
    return list(best.values())


def group_stat(
    key: str,
    signals: Sequence[ResolvedSignal],
    *,
    seed: int = _DEFAULT_SEED,
    bootstrap: bool = True,
) -> GroupStat:
    """Hit rate and per-contract profit edge for a set of signals.

    profit_edge = mean(hit) - mean(ask paid) = expected dollars per contract.
    The CI is bootstrapped over per-signal profit (1-ask on a hit, -ask on a
    miss), so it reflects price dispersion as well as outcome variance.
    """
    if not signals:
        return GroupStat(key, 0, 0, 0.0, 0.0, 0.0, (0.0, 0.0))
    per_signal = [
        (1.0 - s.market_ask_at_signal) if s.hit else -s.market_ask_at_signal
        for s in signals
    ]
    hit_rate = sum(1 for s in signals if s.hit) / len(signals)
    avg_ask = sum(s.market_ask_at_signal for s in signals) / len(signals)
    ci = bootstrap_mean_ci(per_signal, seed=seed) if bootstrap else (0.0, 0.0)
    return GroupStat(
        key=key,
        n_signals=len(signals),
        n_markets=len({s.market_id for s in signals}),
        hit_rate=hit_rate,
        avg_ask=avg_ask,
        profit_edge=sum(per_signal) / len(per_signal),
        profit_edge_ci=ci,
    )


# --------------------------------------------------------------------------
# 2. Exit effectiveness
# --------------------------------------------------------------------------


def exit_effectiveness(
    positions: Sequence[ClosedPosition],
    *,
    by_direction: bool = True,
    seed: int = _DEFAULT_SEED,
) -> list[ExitStat]:
    """Per exit_reason: actual P&L vs holding to settlement.

    `saved` is actual minus held, so a positive number means exiting early was
    the right call and a negative number is the dollar cost of exiting. Only
    discretionary exits appear — `market_resolved` IS holding to settlement, so
    comparing it against itself is meaningless.
    """
    buckets: dict[tuple[str, str], list[ClosedPosition]] = defaultdict(list)
    for pos in positions:
        if pos.exit_reason in _NON_DISCRETIONARY_EXITS:
            continue
        buckets[(pos.exit_reason, "ALL")].append(pos)
        if by_direction:
            buckets[(pos.exit_reason, pos.direction)].append(pos)

    stats: list[ExitStat] = []
    for (reason, direction), group in buckets.items():
        deltas: list[float] = []
        actual = held = 0.0
        unresolved = 0
        for pos in group:
            hold = hold_to_resolution_pnl(pos)
            if hold is None:
                unresolved += 1
                continue
            deltas.append(pos.pnl - hold)
            actual += pos.pnl
            held += hold
        stats.append(
            ExitStat(
                exit_reason=reason,
                direction=direction,
                n=len(deltas),
                n_unresolved=unresolved,
                actual_pnl=actual,
                held_pnl=held,
                saved=actual - held,
                saved_ci=bootstrap_mean_ci(deltas, seed=seed),
                n_helped=sum(1 for d in deltas if d > 0),
                n_hurt=sum(1 for d in deltas if d < 0),
            )
        )
    stats.sort(key=lambda s: (s.saved, s.exit_reason, s.direction))
    return stats


# --------------------------------------------------------------------------
# 3. Stoploss threshold sweep
# --------------------------------------------------------------------------

DEFAULT_STOPLOSS_GRID: tuple[float, ...] = (
    -0.05, -0.08, -0.10, -0.12, -0.15, -0.20, -0.25, -0.30, -0.40, -1.00,
)


def stoploss_sweep(
    positions: Sequence[ClosedPosition],
    thresholds: Sequence[float] = DEFAULT_STOPLOSS_GRID,
    *,
    exit_fee_per_contract: float = 0.0,
    uncensored_only: bool = False,
) -> tuple[list[StoplossPoint], float]:
    """Total P&L across candidate stoploss thresholds, plus the actual baseline.

    A position would have stopped at threshold `t` iff its recorded MAE (worst
    per-contract excursion from entry, in the traded side's price space) reached
    `t`; otherwise it rides to settlement. `-1.00` is the no-stoploss arm.

    **MAE is censored at the actual exit.** A position that really stopped out at
    -0.15 stopped being monitored there, so its MAE cannot tell us whether it
    would also have breached -0.30 later. Those positions are scored as "held to
    settlement" at any wider threshold, which understates how often a wider stop
    would still have fired. `n_censored` counts them per threshold — a row with a
    large `n_censored` is the least trustworthy row in the table, and
    `uncensored_only=True` restricts the whole sweep to positions that ran to
    resolution, where MAE covers the full life of the trade.

    Three further approximations, all flattering to stopping:
      * MAE is sampled at the monitor's poll cadence, so a real stop could fill
        below `t` on a fast move.
      * It assumes a clean fill exactly at `t`. The tightest thresholds are the
        least believable for this reason — a -0.05 stop is inside the typical
        spread, so in practice it would churn on noise rather than capping losses
        at five cents.
      * It ignores the capital an early exit frees and any re-entry the stoploss
        cooldown would then have blocked.

    Positions with no MAE or no resolution are excluded, so the returned baseline
    is over that same subset — never compare it to the ledger's headline P&L.
    """
    usable = [
        p
        for p in positions
        if p.mae is not None and settle_price(p.direction, p.result) is not None
    ]
    if uncensored_only:
        usable = [p for p in usable if p.exit_reason in _NON_DISCRETIONARY_EXITS]
    actual_pnl = sum(p.pnl for p in usable)

    points: list[StoplossPoint] = []
    for threshold in thresholds:
        total = 0.0
        n_stopped = 0
        n_censored = 0
        for pos in usable:
            assert pos.mae is not None  # filtered above
            if pos.mae <= threshold:
                n_stopped += 1
                total += (
                    threshold * pos.contracts
                    - pos.entry_fee_usd
                    - exit_fee_per_contract * pos.contracts
                )
            else:
                if pos.exit_reason not in _NON_DISCRETIONARY_EXITS:
                    # Assumed to ride to settlement, but its MAE stopped updating
                    # when it was exited early — this arm may be optimistic.
                    n_censored += 1
                hold = hold_to_resolution_pnl(pos)
                total += hold if hold is not None else 0.0
        points.append(
            StoplossPoint(
                threshold=threshold,
                n_stopped=n_stopped,
                n_censored=n_censored,
                total_pnl=total,
                delta_vs_actual=total - actual_pnl,
            )
        )
    return points, actual_pnl


# --------------------------------------------------------------------------
# 4. Entry gates
# --------------------------------------------------------------------------


def _gate_predicates(config: object) -> dict[str, tuple[str, Callable[[ResolvedSignal], bool]]]:
    """Map gate name -> (human setting, predicate returning True when ADMITTED).

    Only gates whose inputs are recoverable point-in-time from the signal row
    appear here; see UNEVALUABLE_GATES for the rest. A signal whose book cannot
    be reconstructed is admitted by the spread gate rather than blocked, so a
    reconstruction artefact never manufactures a "the gate cost us money" finding.
    """
    min_edge = float(getattr(config, "min_edge", 0.0))
    max_edge = getattr(config, "max_edge", None)
    min_conf = float(getattr(config, "min_confidence", 0.0))
    min_mid = getattr(config, "min_mid_price", None)
    max_mid = getattr(config, "max_mid_price", None)
    max_spread = getattr(config, "max_spread", None)
    if max_spread is None:
        max_spread = min_edge / 2.0  # documented auto-compute
    min_days = float(getattr(config, "min_days_to_close", 0.0))
    max_days = float(getattr(config, "max_days_to_close", 1e9))

    def spread_ok(sig: ResolvedSignal) -> bool:
        spread = reconstruct_spread(
            sig.direction, sig.market_mid_at_signal, sig.market_ask_at_signal
        )
        return True if spread is None else spread <= max_spread

    gates: dict[str, tuple[str, Callable[[ResolvedSignal], bool]]] = {
        "min_edge": (f"{min_edge:.3f}", lambda s: s.edge >= min_edge),
        "min_confidence": (f"{min_conf:.2f}", lambda s: s.confidence >= min_conf),
        "max_spread": (f"{max_spread:.3f}", spread_ok),
        "days_to_close": (
            f"[{min_days:g}, {max_days:g}]",
            lambda s: min_days <= s.days_to_close <= max_days,
        ),
    }
    if max_edge is not None:
        cap = float(max_edge)
        gates["max_edge"] = (f"{cap:.3f}", lambda s: s.edge <= cap)
    if min_mid is not None or max_mid is not None:
        lo = float(min_mid) if min_mid is not None else 0.0
        hi = float(max_mid) if max_mid is not None else 1.0
        gates["mid_price_band"] = (
            f"[{lo:.2f}, {hi:.2f}]",
            lambda s: lo <= entry_side_cost(s.direction, s.market_mid_at_signal) <= hi,
        )
    return gates


def entry_gate_analysis(
    signals: Sequence[ResolvedSignal],
    config: object,
    *,
    seed: int = _DEFAULT_SEED,
) -> list[GateStat]:
    """Marginal effect of each entry gate: what it admits vs what it blocks.

    For gate G, every other gate is held at its configured value, so the split is
    the decision G alone is responsible for — a gate evaluated in isolation would
    be credited for rejections another gate already made. Signals are deduped to
    the earliest per market first, because entry is a once-per-market decision.

    Read it as: if `blocked.profit_edge` is solidly positive with enough markets
    behind it, that gate is refusing profitable trades.
    """
    gates = _gate_predicates(config)
    stats: list[GateStat] = []
    for name, (setting, predicate) in gates.items():
        others = [p for gname, (_, p) in gates.items() if gname != name]
        candidates = [s for s in signals if all(p(s) for p in others)]
        eligible = first_per_market(candidates)
        admitted = [s for s in eligible if predicate(s)]
        blocked = [s for s in eligible if not predicate(s)]
        stats.append(
            GateStat(
                gate=name,
                setting=setting,
                admitted=group_stat(f"{name} admitted", admitted, seed=seed),
                blocked=group_stat(f"{name} blocked", blocked, seed=seed),
            )
        )
    stats.sort(key=lambda g: -g.blocked.profit_edge * g.blocked.n_markets)
    return stats


def gate_threshold_sweep(
    signals: Sequence[ResolvedSignal],
    config: object,
    *,
    grids: dict[str, Sequence[float]] | None = None,
) -> list[SweepPoint]:
    """Sweep one numeric gate at a time, holding the others at their config value.

    `total_expected_pnl_per_contract` is profit_edge x n_markets — the volume-aware
    figure. A tighter threshold usually raises profit_edge while cutting n; the
    product is what actually pays, subject to the exposure caps this ignores.
    """
    grids = grids or {
        "min_edge": (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        "min_confidence": (0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8),
        "max_spread": (0.02, 0.03, 0.05, 0.08, 0.12, 1.0),
    }
    base = _gate_predicates(config)
    out: list[SweepPoint] = []
    for gate, values in grids.items():
        if gate not in base:
            continue
        others = [p for gname, (_, p) in base.items() if gname != gate]
        candidates = first_per_market([s for s in signals if all(p(s) for p in others)])
        for value in values:
            if gate == "min_edge":
                kept = [s for s in candidates if s.edge >= value]
            elif gate == "min_confidence":
                kept = [s for s in candidates if s.confidence >= value]
            else:
                kept = [
                    s
                    for s in candidates
                    if (
                        sp := reconstruct_spread(
                            s.direction, s.market_mid_at_signal, s.market_ask_at_signal
                        )
                    )
                    is None
                    or sp <= value
                ]
            stat = group_stat(f"{gate}={value}", kept, bootstrap=False)
            out.append(
                SweepPoint(
                    gate=gate,
                    value=value,
                    n_markets=stat.n_markets,
                    hit_rate=stat.hit_rate,
                    profit_edge=stat.profit_edge,
                    total_expected_pnl_per_contract=stat.profit_edge * stat.n_markets,
                )
            )
    return out


# --------------------------------------------------------------------------
# 5. Signal accuracy
# --------------------------------------------------------------------------


def _edge_band(edge_pct: float) -> str:
    if edge_pct < 0:
        return "<0"
    if edge_pct < 15:
        return "0-15"
    if edge_pct < 40:
        return "15-40"
    return ">40"


def accuracy_by(
    signals: Sequence[ResolvedSignal],
    key_fn: Callable[[ResolvedSignal], str],
    *,
    min_signals: int = 1,
    seed: int = _DEFAULT_SEED,
) -> list[GroupStat]:
    buckets: dict[str, list[ResolvedSignal]] = defaultdict(list)
    for sig in signals:
        buckets[key_fn(sig)].append(sig)
    stats = [
        group_stat(key, group, seed=seed)
        for key, group in buckets.items()
        if len(group) >= min_signals
    ]
    stats.sort(key=lambda s: s.key)
    return stats


def accuracy_slices(
    signals: Sequence[ResolvedSignal],
    *,
    min_signals: int = 20,
    seed: int = _DEFAULT_SEED,
) -> dict[str, list[GroupStat]]:
    """Correlate slices — where signal quality concentrates.

    These are observational cuts, not experiments, and there are enough of them
    that the best-looking slice is partly selection. Treat a slice as a lead to
    confirm on the next window, not as a finding on its own.
    """

    def conf_band(sig: ResolvedSignal) -> str:
        return f"conf {min(int(sig.confidence * 10) / 10, 0.9):.1f}+"

    def dtc_band(sig: ResolvedSignal) -> str:
        days = sig.days_to_close
        if days < 1:
            return "<1d to close"
        if days < 3:
            return "1-3d to close"
        if days < 7:
            return "3-7d to close"
        return ">7d to close"

    return {
        "direction": accuracy_by(signals, lambda s: s.direction, min_signals=min_signals, seed=seed),
        "edge_band": accuracy_by(signals, lambda s: _edge_band(s.edge * 100.0), min_signals=min_signals, seed=seed),
        "confidence": accuracy_by(signals, conf_band, min_signals=min_signals, seed=seed),
        "days_to_close": accuracy_by(signals, dtc_band, min_signals=min_signals, seed=seed),
        "trigger": accuracy_by(signals, lambda s: s.trigger, min_signals=min_signals, seed=seed),
    }


# --------------------------------------------------------------------------
# 6. Assessor effectiveness
# --------------------------------------------------------------------------


def capital_tilt(
    pairs: Sequence[tuple[float, bool]],
    *,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, tuple[float, float]]:
    """Mean multiplier on winners minus mean multiplier on losers.

    The direct expression of "put more size on winners". Positive is good; ~0
    means the assessor is a flat tax rather than a discriminator. The CI is
    bootstrapped over the two groups jointly so it reflects both sample sizes.
    """
    winners = [m for m, hit in pairs if hit]
    losers = [m for m, hit in pairs if not hit]
    if not winners or not losers:
        return (0.0, (0.0, 0.0))
    tilt = sum(winners) / len(winners) - sum(losers) / len(losers)

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(2000):
        w = sum(winners[rng.randrange(len(winners))] for _ in winners) / len(winners)
        losing = sum(losers[rng.randrange(len(losers))] for _ in losers) / len(losers)
        samples.append(w - losing)
    samples.sort()
    return tilt, (samples[50], samples[1949])


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


_POSITIONS_SQL = text(
    """
    SELECT p.id::text          AS position_id,
           p.market_id,
           m.series_ticker,
           p.direction,
           p.mode,
           p.strategy_name,
           COALESCE(NULLIF(p.exit_filled_contracts, 0), p.contracts) AS contracts,
           p.entry_price,
           p.entry_time,
           p.exit_time,
           p.exit_reason,
           p.pnl,
           p.entry_fee_usd,
           p.exit_fee_usd,
           p.mae,
           m.result,
           p.signal_edge,
           p.signal_confidence,
           sa.size_multiplier,
           sa.verdict,
           sa.trust_score,
           lq.prompt_version AS assessment_version
    FROM positions p
    JOIN markets m ON m.id = p.market_id
    LEFT JOIN LATERAL (
        SELECT * FROM signal_assessments x
        WHERE x.signal_id = p.signal_id
        ORDER BY x.created_at ASC LIMIT 1
    ) sa ON TRUE
    LEFT JOIN llm_queries lq ON lq.id = sa.llm_query_id
    WHERE p.status = 'closed'
      AND p.exit_time IS NOT NULL
      AND p.exit_time >= :start
      AND p.exit_time < :end
      AND (CAST(:mode AS text) IS NULL OR p.mode = CAST(:mode AS text))
    -- Deterministic order is load-bearing: the bootstrap resamples by index, so
    -- an unordered result set makes every CI differ between identical runs.
    ORDER BY p.id
    """
)


async def load_closed_positions(
    session: AsyncSession,
    *,
    start: datetime,
    end: datetime,
    mode: str | None = None,
) -> list[ClosedPosition]:
    """Closed positions by EXIT time — the week's realised results.

    The entry-side assessment is the FIRST assessment on the entry signal, which
    is the one sizing actually used; later re-assessments of the same signal did
    not move the contract count.
    """
    rows = (
        await session.execute(
            _POSITIONS_SQL, {"start": start, "end": end, "mode": mode}
        )
    ).mappings().all()
    return [
        ClosedPosition(
            position_id=r["position_id"],
            market_id=r["market_id"],
            series_ticker=r["series_ticker"],
            direction=r["direction"],
            mode=r["mode"],
            strategy_name=r["strategy_name"],
            contracts=int(r["contracts"] or 0),
            entry_price=float(r["entry_price"]),
            entry_time=r["entry_time"],
            exit_time=r["exit_time"],
            exit_reason=r["exit_reason"] or "unknown",
            pnl=float(r["pnl"] or 0.0),
            entry_fee_usd=float(r["entry_fee_usd"] or 0.0),
            exit_fee_usd=float(r["exit_fee_usd"] or 0.0),
            mae=float(r["mae"]) if r["mae"] is not None else None,
            result=r["result"],
            signal_edge=float(r["signal_edge"] or 0.0),
            signal_confidence=float(r["signal_confidence"] or 0.0),
            size_multiplier=float(r["size_multiplier"]) if r["size_multiplier"] is not None else None,
            verdict=r["verdict"],
            trust_score=float(r["trust_score"]) if r["trust_score"] is not None else None,
            assessment_version=r["assessment_version"],
        )
        for r in rows
    ]


_SIGNALS_SQL = text(
    """
    SELECT s.id::text AS signal_id,
           s.market_id,
           m.series_ticker,
           s.direction,
           s.created_at,
           m.close_time,
           s.edge,
           s.confidence,
           s.estimated_probability,
           s.market_mid_at_signal,
           s.market_ask_at_signal,
           s.prompt_version,
           s.trigger,
           m.result,
           EXISTS (SELECT 1 FROM positions p WHERE p.signal_id = s.id) AS traded,
           COALESCE(
               (SELECT array_agg(DISTINCT d.source_name)
                FROM document_market_links dml
                JOIN documents d ON d.id = dml.document_id
                WHERE dml.signal_id = s.id),
               ARRAY[]::text[]
           ) AS sources
    FROM signals s
    JOIN markets m ON m.id = s.market_id
    WHERE m.status = 'finalized'
      AND m.result IN ('yes', 'no')
      AND s.direction IN ('YES', 'NO')
      AND s.market_ask_at_signal IS NOT NULL
      AND s.market_ask_at_signal > 0
      AND s.market_ask_at_signal < 1
      AND s.model_used <> 'demo_harness'
      AND s.prompt_version <> 'demo'
      AND (CAST(:since AS timestamptz) IS NULL OR s.created_at >= CAST(:since AS timestamptz))
      AND (CAST(:until AS timestamptz) IS NULL OR s.created_at < CAST(:until AS timestamptz))
      AND (CAST(:prompt_version AS text) IS NULL OR s.prompt_version = CAST(:prompt_version AS text))
    ORDER BY s.id
    """
)


async def load_resolved_signals(
    session: AsyncSession,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    prompt_version: str | None = None,
) -> list[ResolvedSignal]:
    """Every resolved, priced, non-demo signal — traded or not.

    Untraded signals are the entire point of the gate analysis: they are the
    counterfactual population, the trades the config declined to make.

    `until` caps signal creation at the window end so an `--as-of` re-run does
    not analyse signals from the future. Outcome knowledge is deliberately NOT
    rewound: `result` is always as known today, so re-running an old week
    legitimately sees markets that had not yet settled back then, and will not
    reproduce that week's report verbatim.
    """
    rows = (
        await session.execute(
            _SIGNALS_SQL,
            {"since": since, "until": until, "prompt_version": prompt_version},
        )
    ).mappings().all()
    return [
        ResolvedSignal(
            signal_id=r["signal_id"],
            market_id=r["market_id"],
            series_ticker=r["series_ticker"],
            direction=r["direction"],
            created_at=r["created_at"],
            close_time=r["close_time"],
            edge=float(r["edge"]),
            confidence=float(r["confidence"]),
            estimated_probability=float(r["estimated_probability"]),
            market_mid_at_signal=float(r["market_mid_at_signal"]),
            market_ask_at_signal=float(r["market_ask_at_signal"]),
            prompt_version=r["prompt_version"],
            trigger=r["trigger"],
            result=r["result"],
            traded=bool(r["traded"]),
            sources=tuple(sorted(r["sources"] or ())),
        )
        for r in rows
    ]


async def latest_signal_prompt_version(
    session: AsyncSession, *, as_of: datetime | None = None
) -> str | None:
    """The prompt version producing signals at `as_of` (default: now).

    Signal cohorts are not exchangeable — profit edge shifts materially between
    prompt versions — so accuracy and gate analysis default to the live cohort
    rather than pooling every version ever run.

    `as_of` matters for backfills: reviewing May 2026 with today's cohort would
    select a version that did not exist yet and silently analyse zero signals.
    """
    row = (
        await session.execute(
            text(
                "SELECT prompt_version FROM signals "
                "WHERE prompt_version <> 'demo' "
                "  AND (CAST(:as_of AS timestamptz) IS NULL "
                "       OR created_at < CAST(:as_of AS timestamptz)) "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"as_of": as_of},
        )
    ).first()
    return row[0] if row else None


async def assessor_neutral_fallback_rate(
    session: AsyncSession, *, since: datetime | None = None
) -> float | None:
    """Share of assessments with no LLM call behind them.

    A neutral fallback sizes at 1.0x, so a rising rate means the assessor is
    silently not running — the failure mode that looks like "the assessor stopped
    helping" while actually being an outage.
    """
    row = (
        await session.execute(
            text(
                "SELECT count(*) AS n, "
                "count(*) FILTER (WHERE llm_query_id IS NULL) AS fallback "
                "FROM signal_assessments "
                "WHERE (CAST(:since AS timestamptz) IS NULL "
                "OR created_at >= CAST(:since AS timestamptz))"
            ),
            {"since": since},
        )
    ).mappings().first()
    if not row or not row["n"]:
        return None
    return float(row["fallback"]) / float(row["n"])


async def load_ledger(
    session: AsyncSession, *, start: datetime, end: datetime
) -> dict[str, dict[str, float]]:
    """Deterministic activity counts for the window, split by mode."""
    positions = (
        await session.execute(
            text(
                """
                SELECT mode,
                       count(*) FILTER (WHERE closed_in_window) AS closed,
                       count(*) FILTER (
                           WHERE entry_time >= :start AND entry_time < :end
                       ) AS entered,
                       COALESCE(sum(pnl) FILTER (WHERE closed_in_window), 0) AS pnl,
                       count(*) FILTER (WHERE closed_in_window AND pnl > 0) AS wins,
                       count(*) FILTER (WHERE open_at_end) AS open_now
                FROM (
                    SELECT *,
                           status = 'closed'
                             AND exit_time >= :start
                             AND exit_time < :end AS closed_in_window,
                           -- Open AS OF the window end, not as of today, so an
                           -- --as-of re-run reports that week rather than now.
                           -- A NULL exit_time alone is not "open": pending and
                           -- cancelled rows have one too and never held risk.
                           entry_time < :end
                             AND (
                               status = 'open'
                               OR (exit_time IS NOT NULL AND exit_time >= :end)
                             ) AS open_at_end
                    FROM positions
                ) p
                GROUP BY mode
                """
            ),
            {"start": start, "end": end},
        )
    ).mappings().all()

    signals = (
        await session.execute(
            text(
                "SELECT count(*) AS n, count(DISTINCT market_id) AS markets "
                "FROM signals WHERE created_at >= :start AND created_at < :end "
                "AND direction IN ('YES','NO')"
            ),
            {"start": start, "end": end},
        )
    ).mappings().first()

    spend = (
        await session.execute(
            text(
                "SELECT COALESCE(sum(cost_usd), 0) AS usd, count(*) AS calls "
                "FROM llm_queries WHERE created_at >= :start AND created_at < :end"
            ),
            {"start": start, "end": end},
        )
    ).mappings().first()

    ledger: dict[str, dict[str, float]] = {
        r["mode"]: {
            "closed": float(r["closed"]),
            "entered": float(r["entered"]),
            "pnl": float(r["pnl"]),
            "wins": float(r["wins"]),
            "open_now": float(r["open_now"]),
        }
        for r in positions
    }
    ledger["_signals"] = {
        "n": float(signals["n"] if signals else 0),
        "markets": float(signals["markets"] if signals else 0),
    }
    ledger["_llm"] = {
        "usd": float(spend["usd"] if spend else 0.0),
        "calls": float(spend["calls"] if spend else 0),
    }
    return ledger


async def mean_exit_fee_per_contract(
    session: AsyncSession, *, mode: str | None = None
) -> float:
    """Observed exit fee per contract on discretionary exits (0.0 in paper mode)."""
    row = (
        await session.execute(
            text(
                """
                SELECT COALESCE(sum(exit_fee_usd), 0) AS fees,
                       COALESCE(sum(COALESCE(NULLIF(exit_filled_contracts,0), contracts)), 0) AS ct
                FROM positions
                WHERE status='closed' AND exit_reason NOT IN ('market_resolved','cancelled')
                  AND (CAST(:mode AS text) IS NULL OR mode = CAST(:mode AS text))
                """
            ),
            {"mode": mode},
        )
    ).mappings().first()
    if not row or not row["ct"]:
        return 0.0
    return float(row["fees"]) / float(row["ct"])


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def source_analysis(
    signals: Sequence[ResolvedSignal],
    *,
    min_signals: int = 30,
    seed: int = _DEFAULT_SEED,
) -> tuple[list[GroupStat], GroupStat | None]:
    """Realised profit edge of signals that retrieved each source, vs the pool.

    Attribution is by presence, not contribution: a signal retrieving six sources
    is counted once under each, so these are overlapping populations and the
    numbers do not decompose additively. A source that looks bad may simply
    co-occur with hard markets. Use it to nominate a source for a controlled
    check (`freqpred metrics source-calibration`), never to drop one outright.
    """
    with_docs = [s for s in signals if s.sources]
    if not with_docs:
        return [], None
    pool = group_stat("all signals with docs", with_docs, seed=seed)

    buckets: dict[str, list[ResolvedSignal]] = defaultdict(list)
    for sig in with_docs:
        for source in sig.sources:
            buckets[source].append(sig)

    stats = [
        group_stat(source, group, seed=seed)
        for source, group in buckets.items()
        if len(group) >= min_signals
    ]
    stats.sort(key=lambda s: -s.profit_edge)
    return stats, pool


def assessor_analysis(
    positions: Sequence[ClosedPosition],
    assessed: Sequence[tuple[str, float, str, bool]],
    *,
    seed: int = _DEFAULT_SEED,
) -> list[AssessorStat]:
    """Capital tilt and verdict mix per assessment prompt version.

    `assessed` is (version, size_multiplier, verdict, hit) per resolved assessed
    signal. Realised sizing delta is computed only from actual positions, where
    the multiplier really moved contracts; the signal-level tilt covers the much
    larger assessed-but-untraded population.
    """
    by_version: dict[str, list[tuple[str, float, str, bool]]] = defaultdict(list)
    for version, multiplier, verdict, hit in assessed:
        by_version[version or "unknown"].append((version, multiplier, verdict, hit))

    pos_by_version: dict[str, list[ClosedPosition]] = defaultdict(list)
    for pos in positions:
        if pos.size_multiplier is not None:
            pos_by_version[pos.assessment_version or "unknown"].append(pos)

    stats: list[AssessorStat] = []
    for version, rows in by_version.items():
        tilt, ci = capital_tilt([(m, hit) for _, m, _, hit in rows], seed=seed)
        mix: dict[str, int] = defaultdict(int)
        for _, _, verdict, _ in rows:
            mix[verdict] += 1

        group = pos_by_version.get(version, [])
        realized: float | None = None
        if group:
            # What the same trades would have earned at neutral 1.0x sizing.
            # P&L scales linearly with contracts, so dividing out the multiplier
            # recovers the un-sized P&L without needing the base Kelly count.
            realized = sum(
                p.pnl - (p.pnl / p.size_multiplier if p.size_multiplier else p.pnl)
                for p in group
            )
        stats.append(
            AssessorStat(
                version=version,
                n_signals=len(rows),
                capital_tilt=tilt,
                capital_tilt_ci=ci,
                verdict_mix=dict(mix),
                mean_multiplier=sum(m for _, m, _, _ in rows) / len(rows),
                realized_sizing_delta_usd=realized,
                n_positions=len(group),
            )
        )
    stats.sort(key=lambda s: s.version)
    return stats


_ASSESSED_SQL = text(
    """
    SELECT COALESCE(lq.prompt_version, 'neutral-fallback') AS version,
           sa.size_multiplier,
           sa.verdict,
           CASE WHEN (s.direction='YES' AND m.result='yes')
                  OR (s.direction='NO'  AND m.result='no') THEN TRUE ELSE FALSE END AS hit
    FROM signal_assessments sa
    JOIN signals s ON s.id = sa.signal_id
    JOIN markets m ON m.id = s.market_id
    LEFT JOIN llm_queries lq ON lq.id = sa.llm_query_id
    WHERE m.result IN ('yes','no')
      AND s.direction IN ('YES','NO')
      AND (CAST(:since AS timestamptz) IS NULL OR sa.created_at >= CAST(:since AS timestamptz))
    ORDER BY sa.id
    """
)


async def build_weekly_review(
    session: AsyncSession,
    *,
    weeks: int = 1,
    mode: str | None = None,
    config: object,
    history_days: int | None = 90,
    signal_prompt_version: str | None = None,
    all_versions: bool = False,
    _now: datetime | None = None,
) -> WeeklyReview:
    """Assemble the full review. Pure DB reads — no LLM calls, no writes."""
    now = _now or datetime.now(UTC)
    window = Window(start=now - timedelta(weeks=weeks), end=now, weeks=weeks)
    history_start = now - timedelta(days=history_days) if history_days else None

    warnings: list[str] = []

    version = signal_prompt_version
    if version is None and not all_versions:
        version = await latest_signal_prompt_version(session, as_of=window.end)

    ledger = await load_ledger(session, start=window.start, end=window.end)
    window_positions = await load_closed_positions(
        session, start=window.start, end=window.end, mode=mode
    )
    history_positions = await load_closed_positions(
        session,
        start=history_start or datetime(1970, 1, 1, tzinfo=UTC),
        end=window.end,
        mode=mode,
    )
    signals = await load_resolved_signals(
        session, since=history_start, until=window.end, prompt_version=version
    )
    if len(first_per_market(signals)) < 20:
        warnings.append(
            f"Only {len(first_per_market(signals))} resolved markets under "
            f"prompt_version={version} in the history window — entry-gate and "
            "accuracy findings are underpowered. Re-run with --all-versions to "
            "see the pooled view, but read cohort differences before acting."
        )

    exit_fee = await mean_exit_fee_per_contract(session, mode=mode)
    sweep, sweep_baseline = stoploss_sweep(
        history_positions, exit_fee_per_contract=exit_fee
    )
    sweep_clean, sweep_clean_baseline = stoploss_sweep(
        history_positions, exit_fee_per_contract=exit_fee, uncensored_only=True
    )

    assessed_rows = (
        await session.execute(_ASSESSED_SQL, {"since": history_start})
    ).mappings().all()
    assessed = [
        (r["version"], float(r["size_multiplier"]), r["verdict"], bool(r["hit"]))
        for r in assessed_rows
    ]

    sources, source_pool = source_analysis(signals)

    if mode is None and len({p.mode for p in history_positions}) > 1:
        warnings.append(
            "Paper and live positions are both in scope. Paper fills are "
            "frictionless — never read a pooled P&L number as evidence about "
            "live execution. Re-run with --mode live to isolate."
        )

    return WeeklyReview(
        window=window,
        generated_at=now,
        ledger=ledger,
        exit_stats=exit_effectiveness(history_positions),
        exit_stats_window=exit_effectiveness(window_positions, by_direction=False),
        stoploss_sweep=sweep,
        stoploss_actual_pnl=sweep_baseline,
        stoploss_sweep_uncensored=sweep_clean,
        stoploss_uncensored_pnl=sweep_clean_baseline,
        gate_stats=entry_gate_analysis(signals, config),
        gate_sweeps=gate_threshold_sweep(signals, config),
        accuracy_by_week=accuracy_by(
            signals, lambda s: f"{s.created_at:%Y-W%V}", min_signals=5
        ),
        accuracy_by_version=accuracy_by(
            await load_resolved_signals(
                session, since=history_start, until=window.end
            ),
            lambda s: s.prompt_version,
            min_signals=20,
        ),
        accuracy_slices=accuracy_slices(signals),
        assessor_stats=assessor_analysis(history_positions, assessed),
        assessor_neutral_fallback_rate=await assessor_neutral_fallback_rate(
            session, since=history_start
        ),
        source_stats=sources,
        source_pool=source_pool,
        scope={
            "mode": mode or "all",
            "signal_prompt_version": version or "all",
            "history_days": history_days,
            "strategy": getattr(config, "name", "unknown"),
            "exit_fee_per_contract": exit_fee,
            "unevaluable_gates": list(UNEVALUABLE_GATES),
        },
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _ci(interval: tuple[float, float], *, unit: str = "") -> str:
    return f"[{interval[0]:+.3f}, {interval[1]:+.3f}]{unit}"


def _significant(interval: tuple[float, float]) -> bool:
    """Whether a bootstrap CI excludes zero — the bar for calling something real."""
    return interval[0] > 0.0 or interval[1] < 0.0


def _group_line(stat: GroupStat, width: int = 26) -> str:
    marker = " *" if _significant(stat.profit_edge_ci) else "  "
    return (
        f"  {stat.key:<{width}.{width}} n={stat.n_signals:>5} "
        f"mkts={stat.n_markets:>4} hit={stat.hit_rate:>6.1%} "
        f"ask={stat.avg_ask:>5.3f} edge={stat.profit_edge:>+7.3f} "
        f"{_ci(stat.profit_edge_ci)}{marker}"
    )


def render_text(review: WeeklyReview) -> str:  # noqa: C901 — a report is a long list of sections
    """Human- and agent-readable report. Sections mirror the skill's reading order."""
    out: list[str] = []
    add = out.append

    add("=" * 96)
    add(f"WEEKLY PROFITABILITY REVIEW   {review.window.label}")
    add(f"generated {review.generated_at:%Y-%m-%d %H:%M} UTC   scope: {review.scope}")
    add("=" * 96)

    for warning in review.warnings:
        add(f"! {warning}")
    if review.warnings:
        add("")

    # --- 1. ledger -------------------------------------------------------
    add("1. WINDOW LEDGER  (always all modes; the analysis sections below respect --mode)")
    for mode, row in sorted(review.ledger.items()):
        if mode.startswith("_"):
            continue
        closed = int(row["closed"])
        win_rate = row["wins"] / closed if closed else 0.0
        add(
            f"  {mode:<6} entered={int(row['entered']):>4} closed={closed:>4} "
            f"open={int(row['open_now']):>3} pnl=${row['pnl']:>+9.2f} "
            f"win_rate={win_rate:>6.1%}"
        )
    sig = review.ledger.get("_signals", {})
    llm = review.ledger.get("_llm", {})
    add(
        f"  signals={int(sig.get('n', 0))} over {int(sig.get('markets', 0))} markets   "
        f"llm=${llm.get('usd', 0.0):.2f} in {int(llm.get('calls', 0))} calls"
    )
    add("")

    # --- 2. exits --------------------------------------------------------
    add("2. EXIT EFFECTIVENESS  (actual minus hold-to-settlement; negative = the exit cost money)")
    add(f"  {'exit_reason':<24} {'dir':<4} {'n':>4} {'actual':>10} {'held':>10} {'saved':>10}  per-trade 95% CI")
    for stat in review.exit_stats:
        marker = " *" if _significant(stat.saved_ci) else ""
        add(
            f"  {stat.exit_reason:<24} {stat.direction:<4} {stat.n:>4} "
            f"${stat.actual_pnl:>+9.2f} ${stat.held_pnl:>+9.2f} ${stat.saved:>+9.2f}  "
            f"{_ci(stat.saved_ci)} helped={stat.n_helped}/hurt={stat.n_hurt}{marker}"
        )
    if review.exit_stats_window:
        add("  -- this window only --")
        for stat in review.exit_stats_window:
            add(
                f"  {stat.exit_reason:<24} {'ALL':<4} {stat.n:>4} "
                f"${stat.actual_pnl:>+9.2f} ${stat.held_pnl:>+9.2f} ${stat.saved:>+9.2f}"
            )
    add("")

    # --- 3. stoploss sweep -----------------------------------------------
    add("3. STOPLOSS SWEEP  (MAE-based; -1.00 = no stoploss)")
    add("   Assumes a clean fill at the threshold, so tight rows are the least believable —")
    add("   a stop inside the typical spread churns on noise rather than capping the loss.")
    add("   NEITHER arm below is unbiased, and they are biased in opposite directions:")
    add("     * all positions — right population, but MAE stops updating at an early exit, so")
    add("       wide thresholds under-count stops that would still have fired (`cens` = how many).")
    add("     * uncensored only — full-life MAE, but the population is conditioned on NOT having")
    add("       stopped out under the live rule, which flatters the no-stoploss arm by construction.")
    add("   The clean read on the CURRENT threshold is section 2's stoploss row: for positions that")
    add("   really did stop, actual-vs-held needs no MAE and no counterfactual population.")
    add(f"  all positions (actual P&L over the same set: ${review.stoploss_actual_pnl:+.2f})")
    add(f"  {'threshold':>10} {'stopped':>8} {'cens':>6} {'total P&L':>12} {'vs actual':>12}")
    for point in review.stoploss_sweep:
        add(
            f"  {point.threshold:>10.2f} {point.n_stopped:>8} {point.n_censored:>6} "
            f"${point.total_pnl:>+11.2f} ${point.delta_vs_actual:>+11.2f}"
        )
    add(
        f"  uncensored only — held to resolution, full-life MAE "
        f"(actual: ${review.stoploss_uncensored_pnl:+.2f})"
    )
    for point in review.stoploss_sweep_uncensored:
        add(
            f"  {point.threshold:>10.2f} {point.n_stopped:>8} {point.n_censored:>6} "
            f"${point.total_pnl:>+11.2f} ${point.delta_vs_actual:>+11.2f}"
        )
    add("")

    # --- 4. entry gates --------------------------------------------------
    add("4. ENTRY GATES  (marginal: every OTHER gate held at its configured value)")
    add("   A profitable 'blocked' row means the gate is refusing trades that pay. * = CI excludes zero.")
    for gate in review.gate_stats:
        add(f"  {gate.gate} = {gate.setting}")
        add(_group_line(gate.admitted))
        add(_group_line(gate.blocked))
    add(f"  not evaluable point-in-time: {', '.join(UNEVALUABLE_GATES)}")
    add("")

    add("   threshold sweeps (profit_edge x n_markets = volume-aware total, ignores exposure caps)")
    current_gate = ""
    for point in review.gate_sweeps:
        if point.gate != current_gate:
            current_gate = point.gate
            add(f"   {current_gate}:")
        add(
            f"     {point.value:>6.2f}  mkts={point.n_markets:>4} hit={point.hit_rate:>6.1%} "
            f"edge={point.profit_edge:>+7.3f}  total={point.total_expected_pnl_per_contract:>+8.2f}"
        )
    add("")

    # --- 5. signal accuracy ----------------------------------------------
    add("5. SIGNAL ACCURACY  (edge = per-contract expected profit vs the price paid)")
    add("   The most recent weeks are incomplete: only markets that have already settled")
    add("   appear, so short-dated markets are over-represented there. Read the trend from")
    add("   the fully-resolved weeks, not the last row.")
    add("   by week (current prompt cohort):")
    for stat in review.accuracy_by_week:
        add(_group_line(stat))
    add("   by prompt version (all cohorts — this is where step changes show up):")
    for stat in review.accuracy_by_version:
        add(_group_line(stat))
    for name, stats in review.accuracy_slices.items():
        if not stats:
            continue
        add(f"   by {name}:")
        for stat in stats:
            add(_group_line(stat))
    add("")

    # --- 6. assessor -----------------------------------------------------
    add("6. SIZING ASSESSOR  (capital tilt = mean multiplier on winners minus losers; ~0 = flat tax)")
    if review.assessor_neutral_fallback_rate is not None:
        add(f"  neutral-fallback rate (no LLM call behind the assessment): "
            f"{review.assessor_neutral_fallback_rate:.1%}")
    for stat in review.assessor_stats:
        realized = (
            f"${stat.realized_sizing_delta_usd:+.2f} over {stat.n_positions} positions"
            if stat.realized_sizing_delta_usd is not None
            else "no positions"
        )
        marker = " *" if _significant(stat.capital_tilt_ci) else ""
        add(
            f"  {stat.version:<24} n={stat.n_signals:>5} tilt={stat.capital_tilt:>+6.4f} "
            f"{_ci(stat.capital_tilt_ci)} mean_mult={stat.mean_multiplier:.3f}{marker}"
        )
        add(f"    verdicts={dict(sorted(stat.verdict_mix.items()))}  realised sizing delta: {realized}")
    add("")

    # --- 7. sources ------------------------------------------------------
    add("7. DOCUMENT SOURCES  (overlapping populations — presence, not contribution)")
    if review.source_pool:
        add(_group_line(review.source_pool))
    for stat in review.source_stats:
        add(_group_line(stat))
    add("")
    add("=" * 96)
    return "\n".join(out)


def _sweep_point(point: StoplossPoint) -> dict:
    return {
        "threshold": point.threshold,
        "n_stopped": point.n_stopped,
        "n_censored": point.n_censored,
        "total_pnl": round(point.total_pnl, 2),
        "delta_vs_actual": round(point.delta_vs_actual, 2),
    }


def to_dict(review: WeeklyReview) -> dict:
    """JSON-serialisable form for diffing one week's review against another."""

    def group(stat: GroupStat | None) -> dict | None:
        if stat is None:
            return None
        return {
            "key": stat.key,
            "n_signals": stat.n_signals,
            "n_markets": stat.n_markets,
            "hit_rate": round(stat.hit_rate, 4),
            "avg_ask": round(stat.avg_ask, 4),
            "profit_edge": round(stat.profit_edge, 4),
            "profit_edge_ci": [round(v, 4) for v in stat.profit_edge_ci],
            "significant": _significant(stat.profit_edge_ci),
        }

    return {
        "window": {
            "start": review.window.start.isoformat(),
            "end": review.window.end.isoformat(),
            "weeks": review.window.weeks,
        },
        "generated_at": review.generated_at.isoformat(),
        "scope": review.scope,
        "warnings": review.warnings,
        "ledger": review.ledger,
        "exit_stats": [
            {
                "exit_reason": s.exit_reason,
                "direction": s.direction,
                "n": s.n,
                "n_unresolved": s.n_unresolved,
                "actual_pnl": round(s.actual_pnl, 2),
                "held_pnl": round(s.held_pnl, 2),
                "saved": round(s.saved, 2),
                "saved_ci": [round(v, 4) for v in s.saved_ci],
                "significant": _significant(s.saved_ci),
                "n_helped": s.n_helped,
                "n_hurt": s.n_hurt,
            }
            for s in review.exit_stats
        ],
        "stoploss_actual_pnl": round(review.stoploss_actual_pnl, 2),
        "stoploss_uncensored_pnl": round(review.stoploss_uncensored_pnl, 2),
        "stoploss_sweep": [_sweep_point(p) for p in review.stoploss_sweep],
        "stoploss_sweep_uncensored": [
            _sweep_point(p) for p in review.stoploss_sweep_uncensored
        ],
        "gate_stats": [
            {
                "gate": g.gate,
                "setting": g.setting,
                "admitted": group(g.admitted),
                "blocked": group(g.blocked),
            }
            for g in review.gate_stats
        ],
        "gate_sweeps": [
            {
                "gate": p.gate,
                "value": p.value,
                "n_markets": p.n_markets,
                "hit_rate": round(p.hit_rate, 4),
                "profit_edge": round(p.profit_edge, 4),
                "total": round(p.total_expected_pnl_per_contract, 3),
            }
            for p in review.gate_sweeps
        ],
        "accuracy_by_week": [group(s) for s in review.accuracy_by_week],
        "accuracy_by_version": [group(s) for s in review.accuracy_by_version],
        "accuracy_slices": {
            name: [group(s) for s in stats]
            for name, stats in review.accuracy_slices.items()
        },
        "assessor_stats": [
            {
                "version": s.version,
                "n_signals": s.n_signals,
                "capital_tilt": round(s.capital_tilt, 4),
                "capital_tilt_ci": [round(v, 4) for v in s.capital_tilt_ci],
                "significant": _significant(s.capital_tilt_ci),
                "verdict_mix": s.verdict_mix,
                "mean_multiplier": round(s.mean_multiplier, 4),
                "realized_sizing_delta_usd": (
                    round(s.realized_sizing_delta_usd, 2)
                    if s.realized_sizing_delta_usd is not None
                    else None
                ),
                "n_positions": s.n_positions,
            }
            for s in review.assessor_stats
        ],
        "assessor_neutral_fallback_rate": review.assessor_neutral_fallback_rate,
        "source_pool": group(review.source_pool),
        "source_stats": [group(s) for s in review.source_stats],
    }
