"""Unit tests for AlgoExampleStrategy — exit signals and the liquidity gate.

This strategy had no test coverage at all, which is how its `populate_exit_trend`
came to reference a `choppiness` column that `populate_indicators` never creates.
Because logger arguments are evaluated before the call, that KeyError fired on
every invocation at any log level, `IAlgoStrategy.force_exit` swallowed it, and
the strategy's exit could never fire. `test_populate_exit_trend_does_not_raise`
is the regression guard.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from freqpred.markets.models import Market, Position
from freqpred.strategy.defaults.algo_example import (
    _PRICE_CEILING,
    AlgoExampleStrategy,
)

# min_edge = 0.10, max_spread = None → effective_max_spread = 0.05
_MAX_SPREAD = 0.05
_TIGHT = 0.03
_WIDE = 0.19

# safe zone for these = [min(0.60, 0.70) - 0.10, max(0.60, 0.70) + 0.10] = [0.50, 0.80]
_ENTRY = 0.60
_P_EST = 0.70

_BASE_TS = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)


def _df(*, close: float, spread: float, n: int = 3, choppy: bool = True):
    """n identical candles at a given close and book width.

    choppy=True → range 0.08 (> 0.05 threshold) with body 0.01 (< range/2).
    choppy=False → a clean trending candle: body == range.
    """
    import pandas as pd

    if choppy:
        rows = {"open": close + 0.01, "high": close + 0.07, "low": close - 0.01}
    else:
        rows = {"open": close - 0.08, "high": close, "low": close - 0.08}
    return pd.DataFrame(
        {
            "open": [rows["open"]] * n,
            "high": [rows["high"]] * n,
            "low": [rows["low"]] * n,
            "close": [close] * n,
            "spread": [spread] * n,
        }
    )


def _fires(*, close: float, spread: float, choppy: bool = True) -> bool:
    strategy = AlgoExampleStrategy()
    meta = {"market_id": "MKT-1", "entry_price": _ENTRY, "p_est": _P_EST}
    df = strategy.populate_indicators(_df(close=close, spread=spread, choppy=choppy), meta)
    df = strategy.populate_exit_trend(df, meta)
    return bool(df["exit_long"].iloc[-1])


def test_populate_exit_trend_does_not_raise() -> None:
    """Regression: a stale logger reference made every call raise KeyError."""
    strategy = AlgoExampleStrategy()
    meta = {"market_id": "MKT-1", "entry_price": _ENTRY, "p_est": _P_EST}
    df = strategy.populate_indicators(_df(close=0.45, spread=_TIGHT), meta)
    out = strategy.populate_exit_trend(df, meta)  # must not raise
    assert "exit_long" in out.columns


def test_displacement_exit_fires_on_a_liquid_book() -> None:
    assert _fires(close=0.45, spread=_TIGHT) is True


def test_displacement_exit_suppressed_on_a_wide_book() -> None:
    """The change under test: chop quoted across a hollow book is not a signal."""
    assert _fires(close=0.45, spread=_WIDE) is False


def test_displacement_exit_needs_chop_not_just_displacement() -> None:
    """Control: a tight book alone does not trigger an exit."""
    assert _fires(close=0.45, spread=_TIGHT, choppy=False) is False


@pytest.mark.parametrize("spread", [_TIGHT, _WIDE])
def test_ceiling_exit_fires_regardless_of_spread(spread: float) -> None:
    """Signal 1 is structural — a statement about price level, not fluctuation.

    A hollow book is no reason to keep capped upside against full binary
    downside; the framework-level guard in PositionMonitor._execute_exit still
    stops the resulting order from crossing a wide spread.
    """
    assert _fires(close=_PRICE_CEILING + 0.05, spread=spread, choppy=False) is True


@pytest.mark.parametrize(
    ("spread", "expected"),
    [(0.0495, True), (0.05, True), (0.0505, False)],
)
def test_liquidity_threshold_boundary(spread: float, expected: bool) -> None:
    assert AlgoExampleStrategy.config.effective_max_spread == pytest.approx(_MAX_SPREAD)
    assert _fires(close=0.45, spread=spread) is expected


def test_one_hollow_candle_breaks_the_run() -> None:
    """Liquidity is per-candle, so a hole resets persistence rather than averaging out."""
    strategy = AlgoExampleStrategy()
    meta = {"market_id": "MKT-1", "entry_price": _ENTRY, "p_est": _P_EST}
    df = _df(close=0.45, spread=_TIGHT, n=3)
    df.loc[1, "spread"] = _WIDE
    df = strategy.populate_indicators(df, meta)
    df = strategy.populate_exit_trend(df, meta)
    assert bool(df["exit_long"].iloc[-1]) is False


# ---------------------------------------------------------------------------
# End-to-end through force_exit(), both directions.
# timeframe is 1min and min_candles is 25, so this needs 26+ minutes of ticks.
# ---------------------------------------------------------------------------

def _make_market() -> Market:
    return Market(
        id="MKT-1",
        platform="kalshi",
        question="Will X happen?",
        category="politics",
        close_time=_BASE_TS + timedelta(days=5),
        yes_bid=0.44,
        yes_ask=0.46,
        mid_price=0.45,
        volume_24h=1000.0,
        open_interest=500.0,
        last_fetched_at=_BASE_TS,
        price_updated_at=_BASE_TS,
        metadata_fetched_at=_BASE_TS,
    )


def _make_position(direction: str) -> Position:
    p_est_yes = _P_EST if direction == "YES" else round(1.0 - _P_EST, 4)
    return Position(
        id=str(uuid.uuid4()),
        market_id="MKT-1",
        signal_id=str(uuid.uuid4()),
        strategy_name="AlgoExampleStrategy",
        strategy_version="1.0",
        signal_confidence=0.8,
        signal_edge=0.15,
        signal_estimated_prob=p_est_yes,
        direction=direction,
        contracts=10,
        entry_price=_ENTRY,
        entry_time=_BASE_TS,
        mode="live",
        status="open",
    )


def _feed(strategy: AlgoExampleStrategy, spread: float, *, direction: str) -> None:
    """27 one-minute buckets of choppy, below-safe-zone ticks (last is partial)."""
    contract_mids = [0.46, 0.52, 0.44, 0.45]
    for minute in range(27):
        for i, cv in enumerate(contract_mids):
            yes_mid = cv if direction == "YES" else round(1.0 - cv, 4)
            strategy.ingest_tick(
                "MKT-1",
                round(yes_mid - spread / 2, 4),
                round(yes_mid + spread / 2, 4),
                _BASE_TS + timedelta(minutes=minute, seconds=15 * i),
            )


@pytest.mark.parametrize("direction", ["YES", "NO"])
def test_force_exit_fires_on_liquid_book_both_directions(direction: str) -> None:
    strategy = AlgoExampleStrategy()
    _feed(strategy, _TIGHT, direction=direction)
    assert strategy.force_exit(_make_position(direction), _make_market()) == "algo_exit"


@pytest.mark.parametrize("direction", ["YES", "NO"])
def test_force_exit_suppressed_on_wide_book_both_directions(direction: str) -> None:
    """Spread is direction-invariant, so the gate must hold for NO as well as YES."""
    strategy = AlgoExampleStrategy()
    _feed(strategy, _WIDE, direction=direction)
    assert strategy.force_exit(_make_position(direction), _make_market()) is None
