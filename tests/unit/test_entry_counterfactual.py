"""Unit tests for replaying signals through real entry and exit rules.

Every case runs both YES and NO where the arithmetic could differ. The NO side
is where this is easy to get wrong: it *buys* at `1 - yes_bid` and *sells* into
`1 - yes_ask`, so a naive implementation that reuses the YES legs is wrong on
both ends and still produces plausible-looking numbers.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from freqpred.metrics.entry_counterfactual import (
    Candle,
    LaterSignal,
    limit_entry_price,
    simulate_trade,
)

_T0 = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _candle(hours: float, *, bid_lo=0.30, bid_hi=0.35, ask_lo=0.40, ask_hi=0.45) -> Candle:
    return Candle(
        end_ts=_T0 + timedelta(hours=hours),
        yes_bid_low=bid_lo,
        yes_bid_high=bid_hi,
        yes_ask_low=ask_lo,
        yes_ask_high=ask_hi,
    )


def _sim(**kw):
    base = {
        "market_id": "M1",
        "signal_id": "S1",
        "direction": "YES",
        "signal_time": _T0,
        "estimated_probability": 0.60,
        "result": "yes",
        "candles": [_candle(1), _candle(2), _candle(3)],
        "later_signals": [],
        "min_edge": 0.15,
        "min_confidence": 0.60,
        "stoploss": -1.0,
        "entry_mode": "limit",
        "limit_timeout_hours": 2.0,
    }
    base.update(kw)
    return simulate_trade(**base)


# --------------------------------------------------------------------------
# Limit price
# --------------------------------------------------------------------------


def test_limit_entry_price_matches_order_manager_for_both_sides() -> None:
    # YES posts at p_est - min_edge.
    assert limit_entry_price("YES", 0.60, 0.15) == pytest.approx(0.45)
    # NO posts at (1 - p_est) - min_edge, in the NO side's own price space.
    assert limit_entry_price("NO", 0.60, 0.15) == pytest.approx(0.25)


def test_limit_below_zero_is_unplaceable() -> None:
    """A model estimate under min_edge yields a limit at or below zero."""
    trade = _sim(estimated_probability=0.10)
    assert trade.filled is False
    assert trade.exit_reason == "limit_unplaceable"
    assert trade.pnl_per_contract == 0.0


# --------------------------------------------------------------------------
# Fill behaviour
# --------------------------------------------------------------------------


def test_yes_fills_when_the_ask_reaches_the_limit() -> None:
    # limit = 0.60 - 0.15 = 0.45; the ask low of 0.40 crosses it.
    trade = _sim(candles=[_candle(1, ask_lo=0.40)])
    assert trade.filled is True
    assert trade.entry_price == pytest.approx(0.45)


def test_yes_does_not_fill_when_the_ask_stays_above_the_limit() -> None:
    trade = _sim(candles=[_candle(1, ask_lo=0.50)])
    assert trade.filled is False
    assert trade.exit_reason == "unfilled"
    assert trade.pnl_per_contract == 0.0


def test_no_fills_off_the_yes_bid_not_the_yes_ask() -> None:
    """NO buys at 1 - yes_bid. limit = 0.25, so it needs yes_bid_high >= 0.75."""
    filled = _sim(
        direction="NO", result="no", candles=[_candle(1, bid_hi=0.80)]
    )
    assert filled.filled is True
    assert filled.entry_price == pytest.approx(0.25)

    unfilled = _sim(
        direction="NO", result="no", candles=[_candle(1, bid_hi=0.70)]
    )
    assert unfilled.filled is False


def test_fill_respects_the_timeout() -> None:
    """A crossing after the window is not a fill — the order was cancelled."""
    late = _sim(candles=[_candle(1, ask_lo=0.50), _candle(5, ask_lo=0.30)])
    assert late.filled is False

    in_time = _sim(candles=[_candle(1, ask_lo=0.50), _candle(1.5, ask_lo=0.30)])
    assert in_time.filled is True


def test_candles_at_or_before_the_signal_cannot_fill_it() -> None:
    """The order does not exist yet — a prior crossing is not a fill."""
    trade = _sim(candles=[_candle(-1, ask_lo=0.10), _candle(1, ask_lo=0.90)])
    assert trade.filled is False


def test_market_entry_mode_fills_at_the_ask_immediately() -> None:
    trade = _sim(entry_mode="market", market_ask_at_signal=0.55)
    assert trade.filled is True
    assert trade.entry_price == pytest.approx(0.55)


# --------------------------------------------------------------------------
# Exits
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "result", "won"), [("YES", "yes", True), ("NO", "no", True),
                                     ("YES", "no", False), ("NO", "yes", False)]
)
def test_resolution_settles_at_one_or_zero_in_side_space(direction, result, won) -> None:
    p_est = 0.60 if direction == "YES" else 0.40  # limit = 0.45 either way
    fill = {"bid_hi": 0.80} if direction == "NO" else {"ask_lo": 0.40}
    trade = _sim(
        direction=direction,
        result=result,
        estimated_probability=p_est,
        candles=[_candle(1, **fill), _candle(2, **fill)],
    )
    assert trade.filled is True
    assert trade.exit_reason == "resolution"
    expected_entry = limit_entry_price(direction, p_est, 0.15)
    assert trade.pnl_per_contract == pytest.approx(
        (1.0 if won else 0.0) - expected_entry
    )


def test_stoploss_fires_when_the_exit_price_breaches_it() -> None:
    # Fill at 0.45; stoploss -0.10 => exit at 0.35 when the bid reaches it.
    trade = _sim(
        stoploss=-0.10,
        candles=[_candle(1, ask_lo=0.40, bid_lo=0.44), _candle(2, bid_lo=0.30)],
    )
    assert trade.exit_reason == "stoploss"
    assert trade.exit_price == pytest.approx(0.35)
    assert trade.pnl_per_contract == pytest.approx(-0.10)


def test_stoploss_at_minus_one_is_inert() -> None:
    """The live config: -1.0 is unreachable on a 0-1 price scale."""
    trade = _sim(
        stoploss=-1.0,
        candles=[_candle(1, ask_lo=0.40), _candle(2, bid_lo=0.01)],
    )
    assert trade.exit_reason == "resolution"


def test_stoploss_ignores_an_empty_book() -> None:
    """A 0.0 bid is no bid — it must not trigger a stop we could not have filled."""
    trade = _sim(
        stoploss=-0.10,
        candles=[_candle(1, ask_lo=0.40), _candle(2, bid_lo=0.0)],
    )
    assert trade.exit_reason == "resolution"


def test_signal_exit_fires_on_a_confident_flip_with_enough_edge_decay() -> None:
    later = LaterSignal(
        created_at=_T0 + timedelta(hours=1, minutes=30),
        direction="NO",
        confidence=0.70,
        estimated_probability=0.40,  # drop of 0.20 > min_edge 0.15
    )
    trade = _sim(
        candles=[_candle(1, ask_lo=0.40), _candle(2, bid_lo=0.33)],
        later_signals=[later],
    )
    assert trade.exit_reason == "signal"
    assert trade.exit_price == pytest.approx(0.33)


def test_signal_exit_ignores_a_flip_that_does_not_clear_min_edge() -> None:
    """Crossing 0.5 by a hair flips the label without moving the thesis."""
    later = LaterSignal(
        created_at=_T0 + timedelta(hours=1, minutes=30),
        direction="NO",
        confidence=0.70,
        estimated_probability=0.49,  # drop of 0.11 < min_edge
    )
    trade = _sim(
        candles=[_candle(1, ask_lo=0.40), _candle(2, bid_lo=0.33)],
        later_signals=[later],
    )
    assert trade.exit_reason == "resolution"


def test_signal_exit_ignores_low_confidence_and_same_direction_and_skip() -> None:
    for later in (
        LaterSignal(_T0 + timedelta(hours=1.5), "NO", 0.10, 0.40),   # unconfident
        LaterSignal(_T0 + timedelta(hours=1.5), "YES", 0.90, 0.40),  # same side
        LaterSignal(_T0 + timedelta(hours=1.5), "SKIP", 0.90, 0.40),
    ):
        trade = _sim(
            candles=[_candle(1, ask_lo=0.40), _candle(2, bid_lo=0.33)],
            later_signals=[later],
        )
        assert trade.exit_reason == "resolution"


def test_no_side_signal_exit_uses_the_inverted_probability_drop() -> None:
    """For a NO position the held side's probability is 1 - p, so the sign flips."""
    later = LaterSignal(
        created_at=_T0 + timedelta(hours=1.5),
        direction="YES",
        confidence=0.70,
        estimated_probability=0.80,  # NO thesis fell from 0.60 to 0.20
    )
    trade = _sim(
        direction="NO",
        result="no",
        estimated_probability=0.40,
        candles=[_candle(1, bid_hi=0.80), _candle(2, ask_hi=0.70)],
        later_signals=[later],
    )
    assert trade.exit_reason == "signal"
    assert trade.exit_price == pytest.approx(0.30)  # 1 - 0.70


def test_stoploss_takes_priority_over_a_signal_exit_in_the_same_period() -> None:
    later = LaterSignal(_T0 + timedelta(hours=1.5), "NO", 0.90, 0.40)
    trade = _sim(
        stoploss=-0.10,
        candles=[_candle(1, ask_lo=0.40), _candle(2, bid_lo=0.20)],
        later_signals=[later],
    )
    assert trade.exit_reason == "stoploss"


def test_unfilled_orders_carry_no_pnl_but_keep_the_outcome() -> None:
    """Needed so a gate blocking never-filling signals scores as costing nothing."""
    trade = _sim(candles=[_candle(1, ask_lo=0.99)], result="yes")
    assert trade.filled is False
    assert trade.pnl_per_contract == 0.0
    assert trade.hit is True
