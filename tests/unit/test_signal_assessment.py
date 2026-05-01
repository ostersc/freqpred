"""Unit tests for signal assessment helpers."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.metrics.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401

from freqpred.markets.models import Market
from freqpred.metrics.assessment import (
    assess_signal_context,
    _parse_assessment_response,
    _trust_score_to_multiplier,
)
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig

NOW = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)


def _make_market() -> Market:
    return Market(
        id="MKT-1",
        platform="kalshi",
        question="Will Trump say Predict before Apr 20, 2026?\nFull rule text.",
        category="politics",
        status="open",
        result=None,
        close_time=NOW + timedelta(days=2),
        yes_bid=0.48,
        yes_ask=0.52,
        mid_price=0.50,
        volume_24h=1000.0,
        open_interest=5000.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
        series_ticker="KXTRUMPSAY",
    )


def _make_signal() -> Signal:
    return Signal(
        id=str(uuid.uuid4()),
        market_id="MKT-1",
        estimated_probability=0.63,
        confidence=0.80,
        edge=0.13,
        market_mid_at_signal=0.50,
        direction="YES",
        reasoning="test",
        sources=[],
        retrieval_hash="abc123",
        model_used="claude-sonnet-4-6",
        prompt_version="signal-v1",
        trigger="scheduled",
        created_at=NOW,
        raw_context="{}",
    )


class _StubStrategy(IPredictionStrategy):
    config = StrategyConfig(
        name="TestStrategy",
        min_edge=0.10,
        min_confidence=0.70,
        max_exposure_per_market=0.10,
        kelly_fraction=0.25,
        categories=["politics"],
        min_volume_24h=0.0,
        max_days_to_close=30.0,
        min_days_to_close=1.0,
    )

    def should_trade(self, signal: Signal, market: Market) -> bool:
        return True


def test_trust_score_to_multiplier_maps_neutral_and_edges() -> None:
    assert _trust_score_to_multiplier(0.5, scale_min=0.8, scale_max=1.2) == pytest.approx(1.0)
    assert _trust_score_to_multiplier(0.0, scale_min=0.8, scale_max=1.2) == pytest.approx(0.8)
    assert _trust_score_to_multiplier(1.0, scale_min=0.8, scale_max=1.2) == pytest.approx(1.2)


def _valid_llm_json(**overrides: object) -> str:
    import json

    base: dict[str, object] = {
        "trust_score": 0.5,
        "reasoning": "looks fine",
        "key_factors": ["factor1"],
        "warnings": [],
    }
    base.update(overrides)
    return json.dumps(base)


def test_parse_assessment_response_derives_size_up_from_trust_score() -> None:
    parsed = _parse_assessment_response(_valid_llm_json(trust_score=0.60))
    assert parsed["verdict"] == "size_up"
    assert parsed["trust_score"] == pytest.approx(0.60)


def test_parse_assessment_response_derives_size_down_from_trust_score() -> None:
    parsed = _parse_assessment_response(_valid_llm_json(trust_score=0.38))
    assert parsed["verdict"] == "size_down"


def test_parse_assessment_response_derives_size_up_high_score() -> None:
    parsed = _parse_assessment_response(_valid_llm_json(trust_score=0.72))
    assert parsed["verdict"] == "size_up"


def test_parse_assessment_response_derives_neutral_at_exactly_half() -> None:
    parsed = _parse_assessment_response(_valid_llm_json(trust_score=0.5))
    assert parsed["verdict"] == "neutral"


def test_parse_assessment_response_neutral_dead_band_above() -> None:
    # 0.52 is in the dead band (0.45–0.55) — must show neutral, not size_up
    parsed = _parse_assessment_response(_valid_llm_json(trust_score=0.52))
    assert parsed["verdict"] == "neutral"


def test_parse_assessment_response_neutral_dead_band_below() -> None:
    # 0.48 is in the dead band — must show neutral, not size_down
    parsed = _parse_assessment_response(_valid_llm_json(trust_score=0.48))
    assert parsed["verdict"] == "neutral"


def test_parse_assessment_response_size_up_just_outside_dead_band() -> None:
    parsed = _parse_assessment_response(_valid_llm_json(trust_score=0.56))
    assert parsed["verdict"] == "size_up"


def test_parse_assessment_response_size_down_just_outside_dead_band() -> None:
    parsed = _parse_assessment_response(_valid_llm_json(trust_score=0.44))
    assert parsed["verdict"] == "size_down"


@pytest.mark.asyncio
async def test_assess_signal_context_skips_llm_when_no_data() -> None:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    llm_client = MagicMock()
    llm_client.complete = AsyncMock()

    with patch(
        "freqpred.metrics.assessment._load_source_breakdown",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "freqpred.metrics.assessment._load_similar_market_summary",
        new_callable=AsyncMock,
        return_value={"available": False, "reason": "insufficient_matched_history"},
    ):
        assessment = await assess_signal_context(
            session,
            _make_signal(),
            _make_market(),
            _StubStrategy(),
            llm_client,
            "claude-opus-4-6",
        )

    assert assessment.trust_score == pytest.approx(0.5)
    assert assessment.size_multiplier == pytest.approx(1.0)
    assert assessment.verdict == "neutral"
    llm_client.complete.assert_not_awaited()
    session.commit.assert_awaited_once()
