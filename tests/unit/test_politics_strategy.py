"""Unit tests for PoliticsEdgeStrategy — factbase phrase cache gate."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from freqpred.ingestion.fetchers.factbase import FactbasePhraseCache
from freqpred.markets.models import Market
from freqpred.strategy.defaults.politics import PoliticsEdgeStrategy


def _make_market(
    market_id: str = "KXTRUMPSAY-001",
    series_ticker: str | None = "KXTRUMPSAY",
    question: str = "Will Trump say 'Communist' before May 20?",
    mid_price: float = 0.50,
    volume_24h: float = 5000.0,
    days_to_close: float = 3.0,
) -> Market:
    now = datetime.now(tz=UTC)
    return Market(
        id=market_id,
        platform="kalshi",
        question=question,
        category="Politics",
        status="open",
        result=None,
        close_time=now + timedelta(days=days_to_close),
        open_time=now - timedelta(days=1),
        yes_bid=mid_price - 0.01,
        yes_ask=mid_price + 0.01,
        mid_price=mid_price,
        volume_24h=volume_24h,
        open_interest=10_000.0,
        last_fetched_at=now,
        price_updated_at=now,
        metadata_fetched_at=now,
        series_ticker=series_ticker,
    )


def test_gates_market_when_cache_not_ready() -> None:
    cache = FactbasePhraseCache()
    strategy = PoliticsEdgeStrategy(phrase_cache=cache)
    market = _make_market()
    # cache has no entry for this market — is_market_interesting must return False
    assert strategy.is_market_interesting(market) is False


def test_passes_market_when_cache_ready() -> None:
    cache = FactbasePhraseCache()
    cache.mark_ready("KXTRUMPSAY-001")
    strategy = PoliticsEdgeStrategy(phrase_cache=cache)
    market = _make_market()
    # Cache ready — gate passes, allowlist matches, super() passes on price/volume.
    assert strategy.is_market_interesting(market) is True


def test_blocks_non_allowlist_series() -> None:
    cache = FactbasePhraseCache()
    cache.mark_ready("KXPRES-001")
    strategy = PoliticsEdgeStrategy(phrase_cache=cache)
    # KXPRES is not in factbase_series_allowlist — blocked regardless of cache state
    market = _make_market(series_ticker="KXPRES", question="Will Trump win the 2028 election?")
    assert strategy.is_market_interesting(market) is False


def test_no_gate_when_cache_is_none() -> None:
    strategy = PoliticsEdgeStrategy(phrase_cache=None)
    market = _make_market()
    # No cache injected → factbase gate is bypassed, allowlist still applies
    assert strategy.is_market_interesting(market) is True


def test_blocks_when_series_ticker_is_none() -> None:
    cache = FactbasePhraseCache()
    strategy = PoliticsEdgeStrategy(phrase_cache=cache)
    market = _make_market(series_ticker=None)
    # No series_ticker — cannot be in allowlist, always blocked
    assert strategy.is_market_interesting(market) is False


# ---------------------------------------------------------------------------
# Exit liquidity gate
#
# Regression cover for KXTRUMPSAY-26AUG03-TIKT (2026-07-28). The exit fired on a
# 0.38/0.57 book — a 19c spread against a 7.5c effective_max_spread — where the
# "choppiness" was one lot crossing a chasm between a stale bid and a stale ask
# rather than anyone repricing the question. The market closed the day at 0.88.
# ---------------------------------------------------------------------------

import uuid

import pytest

from freqpred.markets.models import Position

# PoliticsEdgeStrategy.min_edge = 0.15 → effective_max_spread = 0.075
_TIGHT_SPREAD = 0.03
_TIKT_SPREAD = 0.19

_ENTRY = 0.6575
_P_EST = 0.97
# safe_low = min(entry, p_est) - min_edge = 0.6575 - 0.15 = 0.5075


def _choppy_displaced_df(spread: float, n: int = 3):
    """n candles that are choppy and below the safe zone, at a given spread.

    range_ = 0.08 > 0.05 threshold; body = 0.01 < range_/2 → choppy.
    close = 0.45 < safe_low 0.5075 → outside the safe zone.
    Everything except the book width is held constant so the gate is the only
    variable under test.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "open": [0.46] * n,
            "high": [0.52] * n,
            "low": [0.44] * n,
            "close": [0.45] * n,
            "spread": [spread] * n,
        }
    )


def _exit_fires(spread: float) -> bool:
    strategy = PoliticsEdgeStrategy()
    df = _choppy_displaced_df(spread)
    metadata = {"market_id": "MKT-1", "entry_price": _ENTRY, "p_est": _P_EST}
    df = strategy.populate_indicators(df, metadata)
    df = strategy.populate_exit_trend(df, metadata)
    return bool(df["exit_long"].iloc[-1])


def test_exit_fires_on_choppy_displacement_in_a_liquid_book() -> None:
    """Control: with a tradeable book the exit still fires as before."""
    assert _exit_fires(_TIGHT_SPREAD) is True


def test_exit_suppressed_when_book_is_too_wide() -> None:
    """The TIKT book: same displacement, same chop, but nobody is actually trading."""
    assert _exit_fires(_TIKT_SPREAD) is False


@pytest.mark.parametrize(
    ("spread", "expected"),
    [
        (0.0745, True),   # just inside effective_max_spread
        (0.075, True),    # exactly at the threshold — inclusive
        (0.0755, False),  # just outside
    ],
)
def test_liquidity_threshold_boundary(spread: float, expected: bool) -> None:
    """Gate is `spread <= effective_max_spread`, matching the entry gate."""
    assert PoliticsEdgeStrategy.config.effective_max_spread == pytest.approx(0.075)
    assert _exit_fires(spread) is expected


def test_single_wide_candle_breaks_the_three_candle_run() -> None:
    """Liquidity is per-candle, not an average — one hollow candle resets persistence.

    Otherwise a burst of real chop either side of a liquidity hole would still
    add up to an exit.
    """
    import pandas as pd

    strategy = PoliticsEdgeStrategy()
    df = _choppy_displaced_df(_TIGHT_SPREAD, n=3)
    # Make the middle candle hollow; the run of 3 can no longer be completed.
    df.loc[1, "spread"] = _TIKT_SPREAD
    metadata = {"market_id": "MKT-1", "entry_price": _ENTRY, "p_est": _P_EST}
    df = strategy.populate_indicators(df, metadata)
    df = strategy.populate_exit_trend(df, metadata)
    assert bool(df["exit_long"].iloc[-1]) is False
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# End-to-end through force_exit(), for both directions.
# _invert_ohlc flips OHLC for NO positions but must NOT flip spread — ask - bid
# is the same width in either frame. These prove the gate survives that path.
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)


def _feed_choppy_ticks(strategy: PoliticsEdgeStrategy, market_id: str, spread: float,
                       *, direction: str) -> None:
    """Feed 4 five-minute buckets (the 4th is dropped as partial → 3 candles).

    Mids are expressed in *contract value* for the given direction and converted
    to YES terms on the way in, so both directions see the identical candle shape.
    """
    contract_mids = [0.46, 0.52, 0.44, 0.45]
    for bucket in range(4):
        for i, cv_mid in enumerate(contract_mids):
            yes_mid = cv_mid if direction == "YES" else round(1.0 - cv_mid, 4)
            ts = _BASE_TS + timedelta(minutes=5 * bucket, seconds=15 * i)
            strategy.ingest_tick(
                market_id,
                round(yes_mid - spread / 2, 4),
                round(yes_mid + spread / 2, 4),
                ts,
            )


def _make_position(direction: str) -> Position:
    p_est_yes = _P_EST if direction == "YES" else round(1.0 - _P_EST, 4)
    return Position(
        id=str(uuid.uuid4()),
        market_id="KXTRUMPSAY-001",
        signal_id=str(uuid.uuid4()),
        strategy_name="PoliticsEdgeStrategy",
        strategy_version="1.0",
        signal_confidence=0.85,
        signal_edge=0.31,
        signal_estimated_prob=p_est_yes,
        direction=direction,
        contracts=8,
        entry_price=_ENTRY,
        entry_time=_BASE_TS,
        mode="live",
        status="open",
    )


@pytest.mark.parametrize("direction", ["YES", "NO"])
def test_force_exit_fires_on_liquid_book_both_directions(direction: str) -> None:
    strategy = PoliticsEdgeStrategy()
    market = _make_market(market_id="KXTRUMPSAY-001")
    _feed_choppy_ticks(strategy, market.id, _TIGHT_SPREAD, direction=direction)
    assert strategy.force_exit(_make_position(direction), market) == "algo_exit"


@pytest.mark.parametrize("direction", ["YES", "NO"])
def test_force_exit_suppressed_on_wide_book_both_directions(direction: str) -> None:
    """Spread is direction-invariant — the gate must hold for NO as well as YES."""
    strategy = PoliticsEdgeStrategy()
    market = _make_market(market_id="KXTRUMPSAY-001")
    _feed_choppy_ticks(strategy, market.id, _TIKT_SPREAD, direction=direction)
    assert strategy.force_exit(_make_position(direction), market) is None
