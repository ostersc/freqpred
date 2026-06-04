"""Unit tests for PoliticsEdgeStrategy — factbase phrase cache gate."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
    now = datetime.now(tz=timezone.utc)
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
