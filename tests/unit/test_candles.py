"""Unit tests for candle fetching, parsing, and the candle-based stoploss sweep.

The two things most likely to be silently wrong here are the direction mapping
(a NO position exits into `1 - yes_ask`, not the yes_bid) and the empty-book
guard (a 0.0 bid is "no bid", not a price of zero — treating it as tradeable
stops out every position on illiquidity). Both are covered from both sides.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from freqpred.markets.candles import (
    MAX_CANDLES_PER_REQUEST,
    BackfillResult,
    candle_to_row,
    chunk_ranges,
)
from freqpred.metrics.weekly_review import (
    CandlePath,
    ClosedPosition,
    candle_stoploss_sweep,
)

_NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# Candle parsing
# --------------------------------------------------------------------------


def _raw_candle(**overrides) -> dict:
    base = {
        "end_period_ts": 1784563200,
        "price": {
            "open_dollars": "0.8500",
            "high_dollars": "0.8700",
            "low_dollars": "0.8500",
            "close_dollars": "0.8700",
            "mean_dollars": "0.8664",
        },
        "yes_bid": {
            "open_dollars": "0.7000",
            "high_dollars": "0.7800",
            "low_dollars": "0.6200",
            "close_dollars": "0.7800",
        },
        "yes_ask": {
            "open_dollars": "0.8400",
            "high_dollars": "0.9000",
            "low_dollars": "0.8000",
            "close_dollars": "0.8700",
        },
        "volume_fp": "9.71",
        "open_interest_fp": "132.91",
    }
    base.update(overrides)
    return base


def _row(**overrides):
    return candle_to_row(
        _raw_candle(**overrides),
        market_id="KXTEST-1",
        series_ticker="KXTEST",
        period_interval=60,
        fetched_at=_NOW,
    )


def test_candle_to_row_parses_decimal_strings() -> None:
    """Kalshi returns prices as strings — floats must come out the other side."""
    row = _row()
    assert row is not None
    assert row["yes_bid_low"] == pytest.approx(0.62)
    assert row["yes_ask_high"] == pytest.approx(0.90)
    assert row["price_close"] == pytest.approx(0.87)
    assert row["volume"] == pytest.approx(9.71)
    assert row["open_interest"] == pytest.approx(132.91)
    assert row["end_period_ts"] == datetime.fromtimestamp(1784563200, tz=UTC)


def test_candle_to_row_tolerates_a_period_with_no_trades() -> None:
    """No trades means no price block. That is information, not an error."""
    row = _row(price={})
    assert row is not None
    assert row["price_close"] is None
    assert row["price_mean"] is None
    # The book is still quoted even when nothing trades.
    assert row["yes_bid_low"] == pytest.approx(0.62)


def test_candle_to_row_requires_a_timestamp() -> None:
    assert _row(end_period_ts=None) is None


def test_candle_to_row_survives_unparseable_numbers() -> None:
    row = _row(volume_fp="not-a-number", yes_bid={"low_dollars": "oops"})
    assert row is not None
    assert row["volume"] == 0.0
    assert row["yes_bid_low"] is None


def test_candle_to_row_preserves_a_zero_bid_rather_than_dropping_it() -> None:
    """Storage keeps 0.0; interpreting it as "no bid" is the reader's job."""
    row = _row(yes_bid={"low_dollars": "0.0000", "close_dollars": "0.0000"})
    assert row is not None
    assert row["yes_bid_low"] == 0.0


# --------------------------------------------------------------------------
# Request chunking
# --------------------------------------------------------------------------


def test_chunk_ranges_single_request_when_under_the_cap() -> None:
    start, end = 1_000_000, 1_000_000 + 60 * 100  # 100 one-minute candles
    assert chunk_ranges(start, end, 1) == [(start, end)]


def test_chunk_ranges_splits_a_long_one_minute_span() -> None:
    """A week of 1m candles is ~10k and the server rejects it outright."""
    start = 1_000_000
    end = start + 7 * 24 * 3600
    chunks = chunk_ranges(start, end, 1)
    assert len(chunks) > 1
    span = MAX_CANDLES_PER_REQUEST * 60
    assert all(hi - lo <= span for lo, hi in chunks)
    # Contiguous and complete — a gap would be a silent hole in the price path.
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for (_, prev_hi), (next_lo, _) in zip(chunks, chunks[1:], strict=False):
        assert next_lo == prev_hi


def test_chunk_ranges_hourly_covers_a_long_window_in_one_request() -> None:
    start = 1_000_000
    end = start + 30 * 24 * 3600
    assert len(chunk_ranges(start, end, 60)) == 1


def test_chunk_ranges_empty_for_inverted_or_zero_range() -> None:
    assert chunk_ranges(1000, 1000, 60) == []
    assert chunk_ranges(2000, 1000, 60) == []


def test_backfill_result_summary_flags_budget_exhaustion() -> None:
    result = BackfillResult(markets_fetched=3, candles_written=90, requests_made=5)
    assert "BUDGET EXHAUSTED" not in result.summary()
    result.budget_exhausted = True
    assert "BUDGET EXHAUSTED" in result.summary()


# --------------------------------------------------------------------------
# CandlePath / candle-based sweep
# --------------------------------------------------------------------------


def _position(
    *,
    position_id: str = "p1",
    direction: str = "YES",
    entry_price: float = 0.40,
    contracts: int = 10,
    pnl: float = 0.0,
    result: str | None = "yes",
    exit_reason: str = "stoploss",
) -> ClosedPosition:
    return ClosedPosition(
        position_id=position_id,
        market_id="KXTEST-1",
        series_ticker="KXTEST",
        direction=direction,
        mode="live",
        strategy_name="T",
        contracts=contracts,
        entry_price=entry_price,
        entry_time=_NOW,
        exit_time=_NOW,
        exit_reason=exit_reason,
        pnl=pnl,
        entry_fee_usd=0.0,
        exit_fee_usd=0.0,
        mae=None,
        result=result,
        signal_edge=0.2,
        signal_confidence=0.7,
    )


def test_first_touch_detects_a_breach() -> None:
    path = CandlePath("p1", lows=(0.38, 0.31, 0.36), n_periods=3, n_no_bid=0)
    assert path.first_touch(0.40, -0.05) is True   # 0.31 <= 0.35
    assert path.first_touch(0.40, -0.15) is False  # never reached 0.25


def test_first_touch_on_an_empty_path_never_triggers() -> None:
    """All periods were unquoted — we could not have exited, so no stop fires."""
    path = CandlePath("p1", lows=(), n_periods=5, n_no_bid=5)
    assert path.first_touch(0.40, -0.01) is False


@pytest.mark.parametrize("direction,result", [("YES", "yes"), ("NO", "no")])
def test_candle_sweep_holds_a_winner_that_never_breached(direction, result) -> None:
    """Identical arithmetic for both sides — entry/exit are in the side's own space."""
    pos = _position(direction=direction, result=result, entry_price=0.40, contracts=10)
    paths = {"p1": CandlePath("p1", lows=(0.39, 0.45), n_periods=2, n_no_bid=0)}
    points, actual, n = candle_sweep_one(pos, paths, -0.10)
    assert n == 1
    assert actual == pytest.approx(0.0)
    assert points.n_stopped == 0
    assert points.total_pnl == pytest.approx(6.0)  # settles at 1.0


@pytest.mark.parametrize("direction,result", [("YES", "no"), ("NO", "yes")])
def test_candle_sweep_stops_a_loser_that_breached(direction, result) -> None:
    pos = _position(direction=direction, result=result, entry_price=0.40, contracts=10)
    paths = {"p1": CandlePath("p1", lows=(0.38, 0.25), n_periods=2, n_no_bid=0)}
    point, _, _ = candle_sweep_one(pos, paths, -0.10)
    assert point.n_stopped == 1
    assert point.total_pnl == pytest.approx(-1.0)  # -0.10 * 10


def candle_sweep_one(pos, paths, threshold):
    """Helper: run the sweep at a single threshold and return that point."""
    points, actual, n = candle_stoploss_sweep([pos], paths, thresholds=(threshold,))
    return points[0], actual, n


def test_candle_sweep_never_reports_censoring() -> None:
    """The path continues past the exit, so the censoring caveat cannot apply."""
    pos = _position(exit_reason="stoploss")
    paths = {"p1": CandlePath("p1", lows=(0.39,), n_periods=1, n_no_bid=0)}
    points, _, _ = candle_stoploss_sweep([pos], paths)
    assert all(p.n_censored == 0 for p in points)


def test_candle_sweep_ignores_positions_without_a_path() -> None:
    """Uncovered markets must be excluded, not silently treated as never-stopped."""
    covered = _position(position_id="p1")
    uncovered = _position(position_id="p2")
    paths = {"p1": CandlePath("p1", lows=(0.39,), n_periods=1, n_no_bid=0)}
    _, _, n = candle_stoploss_sweep([covered, uncovered], paths)
    assert n == 1


def test_candle_sweep_ignores_unresolved_positions() -> None:
    pos = _position(result=None)
    paths = {"p1": CandlePath("p1", lows=(0.10,), n_periods=1, n_no_bid=0)}
    _, _, n = candle_stoploss_sweep([pos], paths)
    assert n == 0


def test_candle_sweep_charges_fees_on_a_stopped_position() -> None:
    pos = _position(direction="NO", result="yes", entry_price=0.40, contracts=10)
    paths = {"p1": CandlePath("p1", lows=(0.10,), n_periods=1, n_no_bid=0)}
    points, _, _ = candle_stoploss_sweep(
        [pos], paths, thresholds=(-0.10,), exit_fee_per_contract=0.01
    )
    assert points[0].total_pnl == pytest.approx(-1.0 - 0.10)
