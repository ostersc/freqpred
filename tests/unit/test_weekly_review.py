"""Unit tests for the weekly profitability review's pure analysis layer.

No DB, no LLM. Every counterfactual is exercised for both YES and NO, because
the module's central claim is that prices are already stored in the traded
side's own space and therefore need no direction-specific inversion — a claim
that is only worth anything if it is tested from both sides.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from freqpred.metrics.weekly_review import (
    ClosedPosition,
    ResolvedSignal,
    accuracy_by,
    bootstrap_mean_ci,
    capital_tilt,
    entry_gate_analysis,
    entry_side_cost,
    exit_effectiveness,
    first_per_market,
    gate_threshold_sweep,
    group_stat,
    hold_to_resolution_pnl,
    reconstruct_spread,
    render_text,
    settle_price,
    signal_hit,
    source_analysis,
    stoploss_sweep,
    to_dict,
)

# Fixed reference instant. Nothing in this module reads the wall clock — every
# time-dependent value is derived from timestamps on the fixtures themselves —
# so this constant cannot rot the way a hardcoded date beside a datetime.now()
# call would.
_NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


def _position(
    *,
    direction: str = "YES",
    entry_price: float = 0.40,
    contracts: int = 10,
    pnl: float = 0.0,
    exit_reason: str = "stoploss",
    result: str | None = "yes",
    mae: float | None = -0.20,
    entry_fee: float = 0.0,
    market_id: str = "MKT-1",
    mode: str = "live",
    size_multiplier: float | None = None,
    assessment_version: str | None = None,
) -> ClosedPosition:
    return ClosedPosition(
        position_id=f"pos-{market_id}-{direction}-{exit_reason}",
        market_id=market_id,
        series_ticker="KXTEST",
        direction=direction,
        mode=mode,
        strategy_name="TestStrategy",
        contracts=contracts,
        entry_price=entry_price,
        entry_time=_NOW - timedelta(days=3),
        exit_time=_NOW - timedelta(days=1),
        exit_reason=exit_reason,
        pnl=pnl,
        entry_fee_usd=entry_fee,
        exit_fee_usd=0.0,
        mae=mae,
        result=result,
        signal_edge=0.20,
        signal_confidence=0.70,
        size_multiplier=size_multiplier,
        assessment_version=assessment_version,
    )


def _signal(
    *,
    direction: str = "YES",
    result: str = "yes",
    edge: float = 0.20,
    confidence: float = 0.70,
    ask: float = 0.40,
    mid: float = 0.38,
    market_id: str = "MKT-1",
    created_offset_h: float = 0.0,
    days_to_close: float = 2.0,
    prompt_version: str = "signal-v11",
    trigger: str = "scheduled",
    sources: tuple[str, ...] = (),
    signal_id: str | None = None,
) -> ResolvedSignal:
    created = _NOW - timedelta(days=10) + timedelta(hours=created_offset_h)
    return ResolvedSignal(
        signal_id=signal_id or f"sig-{market_id}-{created_offset_h}-{direction}",
        market_id=market_id,
        series_ticker="KXTEST",
        direction=direction,
        created_at=created,
        close_time=created + timedelta(days=days_to_close),
        edge=edge,
        confidence=confidence,
        estimated_probability=0.60,
        market_mid_at_signal=mid,
        market_ask_at_signal=ask,
        prompt_version=prompt_version,
        trigger=trigger,
        result=result,
        traded=False,
        sources=sources,
    )


class _Config:
    """Minimal StrategyConfig stand-in — only the gated fields are read."""

    name = "TestStrategy"
    min_edge = 0.10
    max_edge = 0.40
    min_confidence = 0.60
    min_mid_price = 0.05
    max_mid_price = 0.95
    max_spread = 0.05
    min_days_to_close = 0.25
    max_days_to_close = 7.0


# --------------------------------------------------------------------------
# Settlement / hit direction symmetry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "result", "expected"),
    [
        ("YES", "yes", True),
        ("YES", "no", False),
        ("NO", "no", True),
        ("NO", "yes", False),
    ],
)
def test_signal_hit_both_directions(direction: str, result: str, expected: bool) -> None:
    assert signal_hit(direction, result) is expected


@pytest.mark.parametrize(
    ("direction", "result", "expected"),
    [
        ("YES", "yes", 1.0),
        ("YES", "no", 0.0),
        ("NO", "no", 1.0),
        ("NO", "yes", 0.0),
        ("YES", None, None),
        ("NO", "void", None),
    ],
)
def test_settle_price(direction: str, result: str | None, expected: float | None) -> None:
    """A winning NO settles at 1.0 exactly as a winning YES does."""
    assert settle_price(direction, result) == expected


def test_hold_to_resolution_pnl_winning_yes_and_no_are_symmetric() -> None:
    """Same entry price and contracts on a winning side => same counterfactual P&L."""
    yes = _position(direction="YES", result="yes", entry_price=0.40, contracts=10)
    no = _position(direction="NO", result="no", entry_price=0.40, contracts=10)
    assert hold_to_resolution_pnl(yes) == pytest.approx(6.0)
    assert hold_to_resolution_pnl(no) == pytest.approx(6.0)


def test_hold_to_resolution_pnl_losing_side_both_directions() -> None:
    yes = _position(direction="YES", result="no", entry_price=0.40, contracts=10)
    no = _position(direction="NO", result="yes", entry_price=0.40, contracts=10)
    assert hold_to_resolution_pnl(yes) == pytest.approx(-4.0)
    assert hold_to_resolution_pnl(no) == pytest.approx(-4.0)


def test_hold_to_resolution_pnl_deducts_entry_fee_but_not_exit_fee() -> None:
    """Settlement is free on Kalshi — only the entry fee survives the counterfactual."""
    pos = _position(direction="NO", result="no", entry_price=0.30, contracts=20, entry_fee=1.5)
    assert hold_to_resolution_pnl(pos) == pytest.approx(20 * 0.70 - 1.5)


def test_hold_to_resolution_pnl_none_when_unresolved() -> None:
    assert hold_to_resolution_pnl(_position(result=None)) is None


# --------------------------------------------------------------------------
# Book reconstruction
# --------------------------------------------------------------------------


def test_reconstruct_spread_yes_and_no() -> None:
    # Book: yes_bid 0.30, yes_ask 0.40 => mid 0.35, spread 0.10.
    assert reconstruct_spread("YES", 0.35, 0.40) == pytest.approx(0.10)
    # Same book seen from the NO side: no_ask = 1 - yes_bid = 0.70.
    assert reconstruct_spread("NO", 0.35, 0.70) == pytest.approx(0.10)


def test_reconstruct_spread_returns_none_on_crossed_book() -> None:
    """A YES mid above its own ask is a reconstruction failure, not a market state."""
    assert reconstruct_spread("YES", 0.35, 0.33) is None
    assert reconstruct_spread("NO", 0.35, 0.60) is None


def test_reconstruct_spread_none_for_skip() -> None:
    assert reconstruct_spread("SKIP", 0.5, 0.5) is None


@pytest.mark.parametrize(
    ("direction", "mid", "expected"), [("YES", 0.93, 0.93), ("NO", 0.93, 0.07)]
)
def test_entry_side_cost(direction: str, mid: float, expected: float) -> None:
    """A NO entry on a market at 0.93 costs 0.07 — the longshot the band rejects."""
    assert entry_side_cost(direction, mid) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Exit effectiveness
# --------------------------------------------------------------------------


def test_exit_effectiveness_flags_a_stoploss_that_cost_money() -> None:
    """Stopped at -0.10/contract on a market that ultimately won."""
    pos = _position(
        direction="YES", result="yes", entry_price=0.40, contracts=10, pnl=-1.0
    )
    stats = {(s.exit_reason, s.direction): s for s in exit_effectiveness([pos])}
    stat = stats[("stoploss", "ALL")]
    assert stat.n == 1
    assert stat.held_pnl == pytest.approx(6.0)
    assert stat.saved == pytest.approx(-7.0)  # exiting cost $7
    assert stat.n_hurt == 1


def test_exit_effectiveness_credits_a_stoploss_that_saved_money_no_side() -> None:
    pos = _position(
        direction="NO", result="yes", entry_price=0.40, contracts=10, pnl=-1.0
    )
    stat = next(
        s for s in exit_effectiveness([pos]) if s.direction == "NO"
    )
    assert stat.held_pnl == pytest.approx(-4.0)
    assert stat.saved == pytest.approx(3.0)
    assert stat.n_helped == 1


def test_exit_effectiveness_excludes_non_discretionary_exits() -> None:
    """market_resolved IS holding to settlement — comparing it to itself is noise."""
    positions = [
        _position(exit_reason="market_resolved"),
        _position(exit_reason="cancelled"),
    ]
    assert exit_effectiveness(positions) == []


def test_exit_effectiveness_counts_unresolved_separately() -> None:
    stats = exit_effectiveness([_position(result=None, exit_reason="signal")])
    stat = next(s for s in stats if s.direction == "ALL")
    assert stat.n == 0
    assert stat.n_unresolved == 1


def test_exit_effectiveness_splits_by_direction() -> None:
    positions = [
        _position(direction="YES", result="yes", market_id="M1", pnl=-1.0),
        _position(direction="NO", result="no", market_id="M2", pnl=-1.0),
    ]
    keys = {(s.exit_reason, s.direction) for s in exit_effectiveness(positions)}
    assert ("stoploss", "YES") in keys
    assert ("stoploss", "NO") in keys
    assert ("stoploss", "ALL") in keys


# --------------------------------------------------------------------------
# Stoploss sweep
# --------------------------------------------------------------------------


def test_stoploss_sweep_stops_only_when_mae_breaches_threshold() -> None:
    pos = _position(
        direction="YES", result="yes", entry_price=0.40, contracts=10,
        pnl=-1.5, mae=-0.15, exit_reason="market_resolved",
    )
    points, actual = stoploss_sweep([pos], thresholds=(-0.10, -0.30))
    assert actual == pytest.approx(-1.5)
    tight, wide = points
    assert tight.n_stopped == 1
    assert tight.total_pnl == pytest.approx(-1.0)  # -0.10 * 10 contracts
    assert wide.n_stopped == 0
    assert wide.total_pnl == pytest.approx(6.0)  # rode to a winning settlement


def test_stoploss_sweep_no_side_uses_the_same_arithmetic() -> None:
    pos = _position(
        direction="NO", result="no", entry_price=0.40, contracts=10,
        pnl=-1.5, mae=-0.15, exit_reason="market_resolved",
    )
    points, _ = stoploss_sweep([pos], thresholds=(-0.10, -0.30))
    assert points[0].total_pnl == pytest.approx(-1.0)
    assert points[1].total_pnl == pytest.approx(6.0)


def test_stoploss_sweep_counts_censored_positions() -> None:
    """A position exited early stopped updating MAE — flag it, don't hide it."""
    pos = _position(mae=-0.05, exit_reason="stoploss", result="yes")
    points, _ = stoploss_sweep([pos], thresholds=(-0.30,))
    assert points[0].n_stopped == 0
    assert points[0].n_censored == 1


def test_stoploss_sweep_uncensored_only_drops_early_exits() -> None:
    positions = [
        _position(market_id="M1", exit_reason="stoploss", mae=-0.05, result="yes"),
        _position(market_id="M2", exit_reason="market_resolved", mae=-0.05, result="yes"),
    ]
    points, _ = stoploss_sweep(positions, thresholds=(-0.30,), uncensored_only=True)
    assert points[0].n_censored == 0
    assert points[0].total_pnl == pytest.approx(6.0)  # only M2 survives the filter


def test_stoploss_sweep_charges_the_exit_fee_on_stopped_positions() -> None:
    pos = _position(
        contracts=10, mae=-0.50, exit_reason="market_resolved", result="yes"
    )
    points, _ = stoploss_sweep(
        [pos], thresholds=(-0.10,), exit_fee_per_contract=0.01
    )
    assert points[0].total_pnl == pytest.approx(-1.0 - 0.10)


def test_stoploss_sweep_skips_positions_without_mae_or_resolution() -> None:
    positions = [_position(mae=None), _position(result=None)]
    points, actual = stoploss_sweep(positions, thresholds=(-0.10,))
    assert actual == 0.0
    assert points[0].n_stopped == 0


# --------------------------------------------------------------------------
# Signal grouping
# --------------------------------------------------------------------------


def test_group_stat_profit_edge_is_hit_rate_minus_price_paid() -> None:
    signals = [
        _signal(market_id="M1", ask=0.40, result="yes"),
        _signal(market_id="M2", ask=0.60, result="no"),
    ]
    stat = group_stat("t", signals)
    assert stat.hit_rate == pytest.approx(0.5)
    assert stat.avg_ask == pytest.approx(0.5)
    assert stat.profit_edge == pytest.approx(0.0)
    assert stat.n_markets == 2


def test_group_stat_no_direction_profits_when_no_wins() -> None:
    signals = [_signal(direction="NO", result="no", ask=0.30, market_id=f"M{i}") for i in range(4)]
    stat = group_stat("no-only", signals)
    assert stat.hit_rate == pytest.approx(1.0)
    assert stat.profit_edge == pytest.approx(0.70)
    assert stat.per_100_contracts == pytest.approx(70.0)


def test_group_stat_empty() -> None:
    stat = group_stat("empty", [])
    assert stat.n_signals == 0
    assert stat.profit_edge == 0.0


def test_first_per_market_keeps_the_earliest_signal() -> None:
    late = _signal(market_id="M1", created_offset_h=5, signal_id="late")
    early = _signal(market_id="M1", created_offset_h=1, signal_id="early")
    other = _signal(market_id="M2", created_offset_h=3, signal_id="other")
    picked = {s.signal_id for s in first_per_market([late, early, other])}
    assert picked == {"early", "other"}


def test_accuracy_by_respects_min_signals() -> None:
    signals = [
        _signal(direction="YES", market_id="M1"),
        _signal(direction="NO", market_id="M2", result="no"),
        _signal(direction="NO", market_id="M3", result="no"),
    ]
    keys = {s.key for s in accuracy_by(signals, lambda s: s.direction, min_signals=2)}
    assert keys == {"NO"}


# --------------------------------------------------------------------------
# Entry gates
# --------------------------------------------------------------------------


def test_entry_gate_analysis_splits_admitted_from_blocked() -> None:
    """min_edge is the only gate distinguishing these two, and it blocks the winner."""
    signals = [
        _signal(market_id="PASS", edge=0.20, result="yes", ask=0.40),
        _signal(market_id="BLOCK", edge=0.02, result="yes", ask=0.40),
    ]
    stats = {g.gate: g for g in entry_gate_analysis(signals, _Config())}
    gate = stats["min_edge"]
    assert gate.admitted.n_markets == 1
    assert gate.blocked.n_markets == 1
    assert gate.blocked.profit_edge == pytest.approx(0.60)


def test_entry_gate_analysis_holds_other_gates_fixed() -> None:
    """A signal another gate already rejects belongs to neither gate's split.

    The marginal design credits a gate only for the rejections it alone is
    responsible for, so BOTH — which fails edge and confidence — must not be
    counted as either gate's cost.
    """
    signals = [
        _signal(market_id="OK", edge=0.20, confidence=0.70),
        _signal(market_id="LOWEDGE", edge=0.02, confidence=0.70),
        _signal(market_id="LOWCONF", edge=0.20, confidence=0.10),
        _signal(market_id="BOTH", edge=0.02, confidence=0.10),
    ]
    stats = {g.gate: g for g in entry_gate_analysis(signals, _Config())}
    assert {s.market_id for s in signals} == {"OK", "LOWEDGE", "LOWCONF", "BOTH"}
    assert stats["min_edge"].blocked.n_markets == 1  # LOWEDGE only
    assert stats["min_confidence"].blocked.n_markets == 1  # LOWCONF only
    assert stats["min_edge"].admitted.n_markets == 1  # OK only
    assert stats["min_confidence"].admitted.n_markets == 1


def test_entry_gate_analysis_admits_unreconstructible_books() -> None:
    """A crossed book must never manufacture a 'the spread gate cost us money' finding."""
    crossed = _signal(market_id="CROSSED", mid=0.35, ask=0.33, edge=0.20)
    assert reconstruct_spread("YES", 0.35, 0.33) is None
    stats = {g.gate: g for g in entry_gate_analysis([crossed], _Config())}
    assert stats["max_spread"].blocked.n_markets == 0
    assert stats["max_spread"].admitted.n_markets == 1


def test_entry_gate_analysis_mid_band_uses_no_side_own_cost() -> None:
    """A NO entry on a market trading at 0.97 costs 0.03 and must be blocked."""
    config = _Config()
    longshot = _signal(direction="NO", market_id="LONGSHOT", mid=0.97, ask=0.04, result="no")
    stats = {g.gate: g for g in entry_gate_analysis([longshot], config)}
    assert stats["mid_price_band"].blocked.n_markets == 1


def test_entry_gate_analysis_days_to_close_band() -> None:
    signals = [
        _signal(market_id="SOON", days_to_close=0.1),
        _signal(market_id="FAR", days_to_close=30.0),
        _signal(market_id="OK", days_to_close=2.0),
    ]
    stats = {g.gate: g for g in entry_gate_analysis(signals, _Config())}
    assert stats["days_to_close"].blocked.n_markets == 2
    assert stats["days_to_close"].admitted.n_markets == 1


def test_gate_threshold_sweep_is_monotone_in_admitted_count() -> None:
    signals = [
        _signal(market_id=f"M{i}", edge=i / 100.0, result="yes") for i in range(1, 40)
    ]
    points = [p for p in gate_threshold_sweep(signals, _Config()) if p.gate == "min_edge"]
    counts = [p.n_markets for p in points]
    assert counts == sorted(counts, reverse=True)


# --------------------------------------------------------------------------
# Assessor
# --------------------------------------------------------------------------


def test_capital_tilt_positive_when_winners_are_sized_up() -> None:
    tilt, ci = capital_tilt([(1.2, True)] * 20 + [(0.8, False)] * 20)
    assert tilt == pytest.approx(0.4)
    assert ci[0] > 0.0


def test_capital_tilt_zero_for_a_flat_tax() -> None:
    """An assessor that shrinks everything equally discriminates nothing."""
    tilt, _ = capital_tilt([(0.9, True)] * 10 + [(0.9, False)] * 10)
    assert tilt == pytest.approx(0.0)


def test_capital_tilt_needs_both_outcome_classes() -> None:
    assert capital_tilt([(1.2, True), (1.1, True)]) == (0.0, (0.0, 0.0))


def test_bootstrap_mean_ci_brackets_the_mean_and_is_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    first = bootstrap_mean_ci(values)
    assert first == bootstrap_mean_ci(values)
    assert first[0] < 3.0 < first[1]


def test_bootstrap_mean_ci_degenerate_inputs() -> None:
    assert bootstrap_mean_ci([]) == (0.0, 0.0)
    assert bootstrap_mean_ci([2.5]) == (2.5, 2.5)


# --------------------------------------------------------------------------
# Backfill correctness
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latest_signal_prompt_version_respects_as_of() -> None:
    """A backfilled week must use the cohort live THEN, not the one live now.

    Without the as_of bound a review of May would select today's prompt version,
    which did not exist yet, and silently analyse zero signals.
    """
    from freqpred.metrics.weekly_review import latest_signal_prompt_version

    captured: dict = {}

    class _Result:
        def first(self):
            return ("signal-v7",)

    class _Session:
        async def execute(self, stmt, params=None):
            captured["params"] = params
            captured["sql"] = str(stmt)
            return _Result()

    version = await latest_signal_prompt_version(_Session(), as_of=_NOW)
    assert version == "signal-v7"
    assert captured["params"] == {"as_of": _NOW}
    assert "created_at <" in captured["sql"]


def test_loader_sql_is_deterministically_ordered() -> None:
    """Every loader must ORDER BY — the bootstrap resamples by index.

    Without a stable order the same data returns a different CI on each run,
    which silently breaks week-over-week comparison of the committed snapshots.
    """
    from freqpred.metrics import weekly_review as wr

    for name in ("_POSITIONS_SQL", "_SIGNALS_SQL", "_ASSESSED_SQL"):
        assert "ORDER BY" in str(getattr(wr, name)), f"{name} has no ORDER BY"


def test_bootstrap_mean_ci_is_order_sensitive() -> None:
    """Why the ORDER BY matters: the same multiset in another order shifts the CI."""
    values = [float(i) for i in range(40)]
    assert bootstrap_mean_ci(values) != bootstrap_mean_ci(list(reversed(values)))


@pytest.mark.asyncio
async def test_latest_signal_prompt_version_unbounded_by_default() -> None:
    from freqpred.metrics.weekly_review import latest_signal_prompt_version

    captured: dict = {}

    class _Result:
        def first(self):
            return None

    class _Session:
        async def execute(self, stmt, params=None):
            captured["params"] = params
            return _Result()

    assert await latest_signal_prompt_version(_Session()) is None
    assert captured["params"] == {"as_of": None}


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def test_source_analysis_counts_a_signal_under_every_source_it_retrieved() -> None:
    signals = [
        _signal(market_id=f"M{i}", sources=("Guardian", "Tavily"), result="yes")
        for i in range(3)
    ]
    stats, pool = source_analysis(signals, min_signals=1)
    assert pool is not None
    assert pool.n_signals == 3
    assert {s.key for s in stats} == {"Guardian", "Tavily"}
    # Overlapping populations: both sources see all three signals.
    assert all(s.n_signals == 3 for s in stats)


def test_source_analysis_ignores_signals_with_no_documents() -> None:
    stats, pool = source_analysis([_signal(sources=())], min_signals=1)
    assert stats == []
    assert pool is None


def test_source_analysis_applies_min_signals() -> None:
    signals = [_signal(market_id="M1", sources=("Rare",))] + [
        _signal(market_id=f"M{i}", sources=("Common",)) for i in range(2, 8)
    ]
    stats, _ = source_analysis(signals, min_signals=5)
    assert {s.key for s in stats} == {"Common"}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _review():
    from freqpred.metrics.weekly_review import WeeklyReview, Window

    signals = [
        _signal(market_id="M1", result="yes"),
        _signal(market_id="M2", direction="NO", result="no"),
    ]
    sweep, baseline = stoploss_sweep(
        [_position(exit_reason="market_resolved")], thresholds=(-0.10,)
    )
    sources, pool = source_analysis(
        [_signal(market_id="M1", sources=("Guardian",))], min_signals=1
    )
    return WeeklyReview(
        window=Window(start=_NOW - timedelta(weeks=1), end=_NOW, weeks=1),
        generated_at=_NOW,
        ledger={"live": {"closed": 2, "entered": 3, "pnl": 1.5, "wins": 1, "open_now": 0}},
        exit_stats=exit_effectiveness([_position(pnl=-1.0)]),
        exit_stats_window=[],
        stoploss_sweep=sweep,
        stoploss_actual_pnl=baseline,
        stoploss_sweep_uncensored=sweep,
        stoploss_uncensored_pnl=baseline,
        stoploss_sweep_candles=sweep,
        stoploss_candles_pnl=baseline,
        stoploss_candles_n=1,
        candle_coverage={
            "positions_resolved": 1,
            "positions_with_path": 1,
            "periods_no_bid": 0,
            "periods_total": 4,
        },
        gate_stats=entry_gate_analysis(signals, _Config()),
        gate_sweeps=gate_threshold_sweep(signals, _Config()),
        accuracy_by_week=accuracy_by(signals, lambda s: "2026-W30"),
        accuracy_by_version=accuracy_by(signals, lambda s: s.prompt_version),
        accuracy_slices={"direction": accuracy_by(signals, lambda s: s.direction)},
        assessor_stats=[],
        assessor_neutral_fallback_rate=0.12,
        source_stats=sources,
        source_pool=pool,
        scope={"mode": "live"},
        warnings=["test warning"],
    )


def test_render_text_covers_every_section() -> None:
    text = render_text(_review())
    for heading in (
        "1. WINDOW LEDGER",
        "2. EXIT EFFECTIVENESS",
        "3. STOPLOSS SWEEP",
        "4. ENTRY GATES",
        "5. SIGNAL ACCURACY",
        "6. SIZING ASSESSOR",
        "7. DOCUMENT SOURCES",
    ):
        assert heading in text
    assert "test warning" in text


def test_to_dict_is_json_serialisable() -> None:
    import json

    payload = to_dict(_review())
    assert json.loads(json.dumps(payload))["scope"] == {"mode": "live"}
    assert payload["stoploss_sweep"][0]["n_censored"] == 0
    assert payload["candle_coverage"]["positions_with_path"] == 1


def test_render_text_prompts_for_a_backfill_when_candle_coverage_is_missing() -> None:
    """A silent empty section would read as "no stoploss would have helped"."""
    review = _review()
    review.stoploss_candles_n = 0
    review.stoploss_sweep_candles = []
    assert "candles backfill" in render_text(review)
