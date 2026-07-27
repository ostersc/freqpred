"""Unit tests for Kalshi price-grid alignment (freqpred/markets/ticks.py)."""
from __future__ import annotations

import pytest

from freqpred.markets.ticks import round_to_tick


@pytest.mark.parametrize(
    "price,expected",
    [
        (0.615, 0.61),   # computed from prob 0.735 - min_edge 0.12
        (0.685, 0.68),
        (0.5633, 0.56),
        (0.7999, 0.79),
        (0.56, 0.56),    # already on grid — unchanged
        (0.01, 0.01),
    ],
)
def test_buy_rounds_down(price: float, expected: float) -> None:
    """A buy never pays more than the price the edge was derived from."""
    assert round_to_tick(price, action="buy") == pytest.approx(expected)


@pytest.mark.parametrize(
    "price,expected",
    [
        (0.615, 0.62),
        (0.685, 0.69),
        (0.5633, 0.57),
        (0.56, 0.56),    # already on grid — unchanged
    ],
)
def test_sell_rounds_up(price: float, expected: float) -> None:
    """A sell never accepts less than the price asked for."""
    assert round_to_tick(price, action="sell") == pytest.approx(expected)


def test_buy_and_sell_straddle_an_off_grid_price() -> None:
    """The two directions bracket the input rather than both landing the same side."""
    price = 0.615
    assert round_to_tick(price, action="buy") < price < round_to_tick(price, action="sell")


@pytest.mark.parametrize("action", ["buy", "sell"])
def test_result_is_always_a_whole_cent(action: str) -> None:
    """Whole cents are valid under every Kalshi price_level_structure."""
    for raw in (0.0137, 0.2222, 0.4999, 0.735, 0.9184):
        snapped = round_to_tick(raw, action=action)
        assert round(snapped * 100) == pytest.approx(snapped * 100, abs=1e-9)


@pytest.mark.parametrize(
    "price,action",
    [(0.004, "buy"), (0.0, "buy"), (-0.05, "buy"), (-0.05, "sell")],
)
def test_non_tradeable_prices_collapse_to_zero(price: float, action: str) -> None:
    """Prices at or below one tick return 0.0 for the caller to reject."""
    assert round_to_tick(price, action=action) == 0.0


def test_prices_at_or_above_one_clamp_to_one() -> None:
    assert round_to_tick(1.0, action="buy") == 1.0
    assert round_to_tick(0.996, action="sell") == 1.0


def test_yes_and_no_prices_stay_complementary() -> None:
    """A NO price snapped on the NO scale still maps to a whole-cent YES leg.

    The Kalshi client submits NO orders as ``1 - price`` on the YES leg, so
    the complement of a snapped NO price must itself be on the grid.
    """
    for no_price in (0.615, 0.2233, 0.885):
        snapped_no = round_to_tick(no_price, action="buy")
        yes_leg = 1.0 - snapped_no
        assert round(yes_leg * 100) == pytest.approx(yes_leg * 100, abs=1e-9)


def test_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="must be 'buy' or 'sell'"):
        round_to_tick(0.5, action="hold")
