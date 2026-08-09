"""Unit tests for signal assessment helpers."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
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
    _ASSESSMENT_TOOL,
    _SYSTEM_PROMPT,
    _bucketed_history_sample,
    _build_prompt_payload,
    _direction_changes,
    _edge_calibration_row_summary,
    _load_edge_band_calibration,
    _load_market_reevaluation_history,
    _merge_rows,
    _parse_assessment_response,
    _trust_score_to_multiplier,
    assess_signal_context,
)
from freqpred.metrics.models import SignalAssessmentRow
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig

NOW = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)


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


class TestV8ProfitEdgeFraming:
    """v8's core change. Measured 2026-07-24 across 34 calibration cells with
    n>=50: profit-vs-price and the model-overconfidence gap correlate only
    +0.397, and 8 cells covering 2,880 signals point opposite ways. On
    KXTRUMPSAY/NO/>40 the cell earns +0.043 over the price while showing a
    -0.558 overconfidence gap, so a judge told to treat that gap as the warning
    sign sizes DOWN a profitable band — observed live before the fix."""

    def _row(self, hit_rate: float, market_p: float, model_p: float, n: int = 150):
        return SimpleNamespace(
            n_signals=n, n_markets=n // 4, hit_rate=hit_rate,
            avg_market_implied_p=market_p, avg_model_implied_p=model_p,
        )

    def test_summary_reports_profit_over_price(self) -> None:
        out = _edge_calibration_row_summary(self._row(0.281, 0.238, 0.839))
        assert out["profit_edge_vs_price"] == pytest.approx(0.043, abs=1e-3)
        # Profitable, yet deeply "overconfident" — the two disagree in sign.
        assert out["hit_rate"] - out["avg_model_implied_p"] < -0.5

    def test_merge_weights_by_sample_size_and_recomputes_profit(self) -> None:
        merged = _merge_rows([
            self._row(0.90, 0.60, 0.85, n=100),
            self._row(0.50, 0.40, 0.85, n=100),
        ])
        assert merged["hit_rate"] == pytest.approx(0.70)
        assert merged["avg_market_implied_p"] == pytest.approx(0.50)
        # Recomputed from the merged aggregates, not averaged from the parts.
        assert merged["profit_edge_vs_price"] == pytest.approx(0.20)

    def test_merge_ignores_empty_rows(self) -> None:
        assert _merge_rows([]) == {"n_signals": 0, "n_markets": 0}
        assert _merge_rows([None]) == {"n_signals": 0, "n_markets": 0}

    def test_prompt_directs_at_profit_not_self_consistency(self) -> None:
        p = _SYSTEM_PROMPT
        assert "profit_edge_vs_price" in p
        assert "this_direction_all_bands" in p
        assert "Do NOT size down on an overconfidence gap alone" in p
        assert "where the two conflict, profit wins" in p

    def test_prompt_no_longer_caps_score_on_warnings(self) -> None:
        """v6 forced trust_score <= 0.5 whenever any warning was present. opus-5
        emitted warnings on 30/30 calls, so it could never score above neutral —
        across 60 calls the single response above 0.5 was the only one with zero
        warnings. Severing that coupling produced 19 size_ups on the frozen set,
        at a 78.9% hit rate against a 54.1% base."""
        p = _SYSTEM_PROMPT
        assert "MUST be ≤ 0.5" not in p
        assert "NO arithmetic relationship" in p

    def test_prompt_orders_trust_score_first_for_truncation_safety(self) -> None:
        assert "FIRST field" in _SYSTEM_PROMPT


def test_assessment_tool_is_strict_with_no_additional_properties() -> None:
    """Without strict mode, a forced tool call has no server-side guarantee
    on the shape of tool_use.input — an Opus 5 screen (2026-07-24) showed
    this concretely: 5/30 live calls wrapped the whole payload in a bogus
    envelope key ("parameters", "body", "paramName", or a hallucinated
    {"tool_use_id", "input"} pair) instead of the flat schema, silently
    failing _parse_assessment_response. strict + additionalProperties=false
    makes that structurally impossible regardless of which model is behind
    the call."""
    assert _ASSESSMENT_TOOL["strict"] is True
    assert _ASSESSMENT_TOOL["input_schema"]["additionalProperties"] is False


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


def _make_prior_assessment_row(
    *,
    trust_score: float = 0.2,
    size_multiplier: float = 0.88,
    verdict: str = "size_down",
) -> SignalAssessmentRow:
    row = MagicMock(spec=SignalAssessmentRow)
    row.signal_id = uuid.uuid4()
    row.trust_score = trust_score
    row.size_multiplier = size_multiplier
    row.verdict = verdict
    row.reasoning = "prior reasoning"
    row.key_factors = ["low trust source"]
    row.warnings = ["sparse data"]
    row.source_breakdown = [{"source": "twitter"}]
    row.similar_market_summary = {"available": False}
    row.llm_query_id = None
    row.created_at = NOW
    return row


@pytest.mark.asyncio
async def test_price_moved_clone_carries_forward_prior_assessment() -> None:
    """A price_moved clone with no doc links reuses the prior signal's assessment
    instead of falling back to neutral (multiplier=1.0)."""
    prior_row = _make_prior_assessment_row(size_multiplier=0.88, verdict="size_down")
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    llm_client = MagicMock()
    llm_client.complete = AsyncMock()

    clone_signal = _make_signal()
    clone_signal = Signal(
        **{**clone_signal.__dict__, "trigger": "price_moved"},
    )

    with patch(
        "freqpred.metrics.assessment._load_source_breakdown",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "freqpred.metrics.assessment._load_similar_market_summary",
        new_callable=AsyncMock,
        return_value={"available": False, "reason": "insufficient_matched_history"},
    ), patch(
        "freqpred.metrics.assessment._load_prior_assessment_by_hash",
        new_callable=AsyncMock,
        return_value=prior_row,
    ):
        assessment = await assess_signal_context(
            session,
            clone_signal,
            _make_market(),
            _StubStrategy(),
            llm_client,
            "claude-opus-4-6",
        )

    assert assessment.size_multiplier == pytest.approx(0.88)
    assert assessment.verdict == "size_down"
    assert assessment.trust_score == pytest.approx(0.2)
    llm_client.complete.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_price_moved_clone_falls_back_to_neutral_when_no_prior() -> None:
    """A price_moved clone with no doc links and no prior assessment falls back to neutral."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    llm_client = MagicMock()
    llm_client.complete = AsyncMock()

    clone_signal = _make_signal()
    clone_signal = Signal(
        **{**clone_signal.__dict__, "trigger": "price_moved"},
    )

    with patch(
        "freqpred.metrics.assessment._load_source_breakdown",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "freqpred.metrics.assessment._load_similar_market_summary",
        new_callable=AsyncMock,
        return_value={"available": False, "reason": "insufficient_matched_history"},
    ), patch(
        "freqpred.metrics.assessment._load_prior_assessment_by_hash",
        new_callable=AsyncMock,
        return_value=None,
    ):
        assessment = await assess_signal_context(
            session,
            clone_signal,
            _make_market(),
            _StubStrategy(),
            llm_client,
            "claude-opus-4-6",
        )

    assert assessment.size_multiplier == pytest.approx(1.0)
    assert assessment.verdict == "neutral"
    llm_client.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_signal_with_no_data_still_returns_neutral() -> None:
    """Non-price_moved signals with no data still return neutral (unchanged behaviour)."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    llm_client = MagicMock()
    llm_client.complete = AsyncMock()

    scheduled_signal = _make_signal()  # trigger="scheduled" by default

    with patch(
        "freqpred.metrics.assessment._load_source_breakdown",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "freqpred.metrics.assessment._load_similar_market_summary",
        new_callable=AsyncMock,
        return_value={"available": False, "reason": "insufficient_matched_history"},
    ), patch(
        "freqpred.metrics.assessment._load_prior_assessment_by_hash",
        new_callable=AsyncMock,
        return_value=_make_prior_assessment_row(),
    ) as mock_prior:
        assessment = await assess_signal_context(
            session,
            scheduled_signal,
            _make_market(),
            _StubStrategy(),
            llm_client,
            "claude-opus-4-6",
        )

    assert assessment.size_multiplier == pytest.approx(1.0)
    assert assessment.verdict == "neutral"
    mock_prior.assert_not_awaited()  # prior lookup not called for scheduled signals


# ---------------------------------------------------------------------------
# _build_prompt_payload — phrase_data injection
# ---------------------------------------------------------------------------


def _make_phrase_data() -> object:
    from datetime import UTC

    from freqpred.ingestion.fetchers.factbase import FactbasePhraseData

    return FactbasePhraseData(
        display_phrase="witch hunt",
        api_query='"witch hunt"',
        speaker_slug="trump",
        in_market_count=3,
        count_7d=8,
        count_30d=22,
        count_365d=150,
        top_quotes=[{"date": "2026-05-01", "text": "A witch hunt!", "event_type": "speech"}],
        fetched_at=datetime.now(UTC),
    )


def test_phrase_data_injected_into_payload() -> None:
    phrase_data = _make_phrase_data()
    payload = _build_prompt_payload(
        _make_signal(),
        _make_market(),
        "TestStrategy",
        source_breakdown=[],
        similar_market_summary={"available": False},
        phrase_data=phrase_data,
    )
    assert "phrase_frequency" in payload
    pf = payload["phrase_frequency"]
    assert pf["phrase"] == "witch hunt"
    assert pf["in_market_count"] == 3
    assert pf["count_7d"] == 8
    assert pf["count_30d"] == 22
    assert pf["count_365d"] == 150
    assert pf["weekly_rate_30d"] == pytest.approx(22 / 4.3, rel=0.01)


def test_no_phrase_data_no_key() -> None:
    payload = _build_prompt_payload(
        _make_signal(),
        _make_market(),
        "TestStrategy",
        source_breakdown=[],
        similar_market_summary={"available": False},
        phrase_data=None,
    )
    assert "phrase_frequency" not in payload


# ---------------------------------------------------------------------------
# T94 — edge_band_calibration / market_reevaluation_history
# ---------------------------------------------------------------------------


def _edge_calibration_row(
    n_signals: int,
    n_markets: int,
    hit_rate: float,
    avg_market_p: float,
    avg_model_p: float,
) -> MagicMock:
    row = MagicMock()
    row.n_signals = n_signals
    row.n_markets = n_markets
    row.hit_rate = hit_rate
    row.avg_market_implied_p = avg_market_p
    row.avg_model_implied_p = avg_model_p
    return row


def _compile_sql(stmt: object) -> str:
    from sqlalchemy.dialects import postgresql

    return str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


def _make_edge_calibration_session(rows_by_key: dict[tuple[str, str], object]) -> AsyncMock:
    """rows_by_key: {(direction, scope): row_or_None}; scope is 'global' or 'series'."""

    async def _execute(stmt: object, *args: object, **kwargs: object) -> MagicMock:
        sql = _compile_sql(stmt)
        direction = "YES" if "direction = 'YES'" in sql else "NO"
        scope = "global" if "series_ticker IS NULL" in sql else "series"
        result = MagicMock()
        result.scalar_one_or_none.return_value = rows_by_key.get((direction, scope))
        return result

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=_execute)
    return session


@pytest.mark.asyncio
async def test_load_edge_band_calibration_yes_and_no() -> None:
    """NO-side band assignment uses the signal's own edge (both directions covered)."""
    yes_global = _edge_calibration_row(20, 15, 0.55, 0.50, 0.65)
    no_global = _edge_calibration_row(12, 10, 0.40, 0.45, 0.60)
    session = _make_edge_calibration_session(
        {("YES", "global"): yes_global, ("NO", "global"): no_global}
    )
    market_no_series = Market(**{**_make_market().__dict__, "series_ticker": None})

    yes_signal = _make_signal()  # direction="YES", edge=0.13 -> band "0-15"
    result = await _load_edge_band_calibration(session, yes_signal, market_no_series)
    assert result is not None
    assert result["this_signal_edge_band"] == "0-15"
    assert result["same_direction_only"]["n_signals"] == 20
    assert result["same_direction_only"]["hit_rate"] == pytest.approx(0.55)
    assert result["all_directions"]["n_signals"] == 32
    assert result["all_directions"]["hit_rate"] == pytest.approx(
        round((0.55 * 20 + 0.40 * 12) / 32, 3)
    )

    no_signal = Signal(**{**_make_signal().__dict__, "direction": "NO", "edge": 0.13})
    result_no = await _load_edge_band_calibration(session, no_signal, market_no_series)
    assert result_no is not None
    assert result_no["this_signal_edge_band"] == "0-15"
    assert result_no["same_direction_only"]["n_signals"] == 12


@pytest.mark.asyncio
async def test_load_edge_band_calibration_below_min_n_omitted() -> None:
    """Global row with n_signals < 10 -> section omitted entirely."""
    thin_global = _edge_calibration_row(4, 3, 0.5, 0.5, 0.5)
    session = _make_edge_calibration_session({("YES", "global"): thin_global})

    result = await _load_edge_band_calibration(session, _make_signal(), _make_market())
    assert result is None


@pytest.mark.asyncio
async def test_load_edge_band_calibration_prefers_series_scope_when_adequate() -> None:
    """A series-scoped row with n_signals >= 10 overrides the global row."""
    global_row = _edge_calibration_row(20, 15, 0.55, 0.50, 0.65)
    series_row = _edge_calibration_row(11, 9, 0.90, 0.60, 0.55)
    session = _make_edge_calibration_session(
        {("YES", "global"): global_row, ("YES", "series"): series_row}
    )

    result = await _load_edge_band_calibration(session, _make_signal(), _make_market())
    assert result is not None
    assert result["same_direction_only"]["n_signals"] == 11
    assert result["same_direction_only"]["hit_rate"] == pytest.approx(0.90)


@pytest.mark.asyncio
async def test_load_market_reevaluation_history_first_signal() -> None:
    result_mock = MagicMock()
    result_mock.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    result = await _load_market_reevaluation_history(session, _make_signal())
    assert result == {"prior_signal_count": 0, "note": "First signal on this market."}


# ---------------------------------------------------------------------------
# T95 — verbatim bucketed market_reevaluation_history (issue #95 shape)
# ---------------------------------------------------------------------------

# The builder tests operate relative to the assessed signal's created_at (NOW);
# no wall clock is involved anywhere in the history code, by construction.
_HISTORY_T0 = NOW - timedelta(hours=400)


def _prior_row(
    hours_after_t0: float,
    direction: str,
    edge: float,
    p: float,
    mid: float,
    ask: float | None,
    conf: float = 0.7,
    trigger: str = "scheduled",
) -> SimpleNamespace:
    return SimpleNamespace(
        created_at=_HISTORY_T0 + timedelta(hours=hours_after_t0),
        direction=direction,
        edge=edge,
        estimated_probability=p,
        market_mid_at_signal=mid,
        market_ask_at_signal=ask,
        confidence=conf,
        trigger=trigger,
    )


def test_bucketed_history_returns_all_when_under_cap() -> None:
    rows = [_prior_row(i * 10, "YES", 0.1, 0.6, 0.5, 0.52) for i in range(5)]
    points = _bucketed_history_sample(rows, NOW)
    assert len(points) == 5
    assert all("bucket_summary" not in p for p in points)
    hours = [p["hours_before_current_signal"] for p in points]
    assert hours == sorted(hours, reverse=True)  # chronological, oldest first


def test_bucketed_history_always_anchors_first_and_latest() -> None:
    rows = [_prior_row(i * 10, "YES", 0.1, 0.6, 0.5, 0.52) for i in range(35)]
    points = _bucketed_history_sample(rows, NOW)
    assert len(points) <= 10
    assert points[0]["hours_before_current_signal"] == 400.0  # first prior, verbatim
    assert points[-1]["hours_before_current_signal"] == 60.0  # latest prior, verbatim
    assert "bucket_summary" not in points[0]
    assert "bucket_summary" not in points[-1]


def test_bucketed_history_buckets_are_time_even_not_index_even() -> None:
    # 30 signals inside hour one, then 5 spread over a week: index-even sampling
    # would spread points across the cluster; time-even bucketing must keep the
    # cluster inside one bucket.
    rows = [_prior_row(i / 30, "YES", 0.1, 0.6, 0.5, 0.52) for i in range(30)] + [
        _prior_row(24.0 * (d + 1), "YES", 0.1, 0.6, 0.5, 0.52) for d in range(5)
    ]
    points = _bucketed_history_sample(rows, NOW)
    covered = 2 + sum(
        p["bucket_summary"]["signals_in_bucket"] for p in points if "bucket_summary" in p
    )
    assert covered == 35
    first_bucket = next(p for p in points if "bucket_summary" in p)
    assert first_bucket["bucket_summary"]["signals_in_bucket"] >= 28


def test_bucketed_history_counts_cover_all_signals() -> None:
    # Aliasing regression: direction alternates every signal, so any ≤10-point
    # render could look calm — the per-bucket flip counts must expose it.
    rows = [
        _prior_row(i * 10, "YES" if i % 2 == 0 else "NO", 0.1 + i * 0.001, 0.6, 0.5, 0.52)
        for i in range(35)
    ]
    points = _bucketed_history_sample(rows, NOW)
    covered = 2 + sum(
        p["bucket_summary"]["signals_in_bucket"] for p in points if "bucket_summary" in p
    )
    assert covered == 35
    assert any(
        p["bucket_summary"]["direction_changes_in_bucket"] > 0
        for p in points
        if "bucket_summary" in p
    )


def test_direction_change_count_total() -> None:
    # SKIP <-> traded transitions count; the total is bucketing-independent.
    assert _direction_changes(["YES", "YES", "NO", "SKIP", "NO", "NO"]) == 3
    assert _direction_changes(["YES"] * 5) == 0
    assert _direction_changes(["YES", "NO"] * 17 + ["YES"]) == 34


def test_point_shape_yes_no_and_skip() -> None:
    rows = [
        _prior_row(370, "YES", 0.10, 0.62, 0.55, 0.57),
        _prior_row(380, "NO", 0.08, 0.62, 0.55, 0.46),
        _prior_row(390, "SKIP", 0.05, 0.62, 0.55, None),
    ]
    yes_point, no_point, skip_point = _bucketed_history_sample(rows, NOW)

    assert yes_point["model_p_traded_side"] == 0.62
    assert yes_point["market_cost_traded_side"] == 0.57
    assert yes_point["edge_pct_traded_side"] == 10.0

    # NO side: model probability converted to the traded side, cost is the NO ask
    assert no_point["model_p_traded_side"] == pytest.approx(0.38)
    assert no_point["market_cost_traded_side"] == 0.46
    assert no_point["edge_pct_traded_side"] == 8.0

    # SKIP: side-specific fields null, never converted as if NO
    assert skip_point["model_p_traded_side"] is None
    assert skip_point["market_cost_traded_side"] is None
    assert skip_point["edge_pct_traded_side"] is None

    # YES-space fields are direction-independent in all three cases
    for point in (yes_point, no_point, skip_point):
        assert point["model_p_event_yes"] == 0.62
        assert point["market_p_event_yes"] == 0.55


@pytest.mark.asyncio
async def test_load_market_reevaluation_history_omits_old_derived_fields() -> None:
    prior_rows = [
        _prior_row(0, "NO", 0.05, 0.40, 0.45, 0.62),
        _prior_row(200, "YES", 0.20, 0.70, 0.50, 0.52),
    ]
    result_mock = MagicMock()
    result_mock.all.return_value = prior_rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    result = await _load_market_reevaluation_history(session, _make_signal())
    assert result["prior_signal_count"] == 2
    assert result["direction_change_count"] == 1
    assert result["history_span_hours"] == 400.0
    assert len(result["sampled_history"]) == 2
    for gone in (
        "direction_consistent_with_this_signal",
        "edge_trend_across_prior_signals",
        "prior_directions",
        "note",
    ):
        assert gone not in result


@pytest.mark.asyncio
async def test_build_prompt_payload_includes_sampled_history_description() -> None:
    result_mock = MagicMock()
    result_mock.all.return_value = [_prior_row(0, "YES", 0.1, 0.6, 0.5, 0.52)]
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result_mock)

    history = await _load_market_reevaluation_history(session, _make_signal())
    payload = _build_prompt_payload(
        _make_signal(),
        _make_market(),
        "TestStrategy",
        source_breakdown=[],
        similar_market_summary={"available": False},
        edge_band_calibration=None,
        market_reevaluation_history=history,
    )
    description = payload["market_reevaluation_history"]["description"]
    assert "_event_yes" in description
    assert "_traded_side" in description
    assert "bucket_summary" in description


def test_build_prompt_payload_includes_new_sections() -> None:
    edge_band_calibration = {
        "description": "desc",
        "this_signal_edge_band": "0-15",
        "all_directions": {"n_signals": 32},
        "same_direction_only": {"n_signals": 20},
    }
    market_reevaluation_history = {
        "prior_signal_count": 0,
        "note": "First signal on this market.",
    }
    payload = _build_prompt_payload(
        _make_signal(),
        _make_market(),
        "TestStrategy",
        source_breakdown=[],
        similar_market_summary={"available": False},
        edge_band_calibration=edge_band_calibration,
        market_reevaluation_history=market_reevaluation_history,
    )
    assert payload["edge_band_calibration"] == edge_band_calibration
    assert payload["market_reevaluation_history"] == market_reevaluation_history


def test_build_prompt_payload_omits_edge_band_calibration_when_none() -> None:
    payload = _build_prompt_payload(
        _make_signal(),
        _make_market(),
        "TestStrategy",
        source_breakdown=[],
        similar_market_summary={"available": False},
        edge_band_calibration=None,
        market_reevaluation_history={"prior_signal_count": 0, "note": "x"},
    )
    assert "edge_band_calibration" not in payload
    assert "market_reevaluation_history" in payload


@pytest.mark.asyncio
async def test_assess_signal_context_passes_sections_to_llm() -> None:
    """Wiring test: the prompt actually sent to LLMClient.complete carries both
    T94 sections, not just that the loaders return the right shape in isolation."""
    import json

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()

    captured: dict[str, object] = {}

    async def _complete(**kwargs: object) -> MagicMock:
        captured["prompt"] = kwargs["prompt"]
        captured["max_tokens"] = kwargs["max_tokens"]
        response = MagicMock()
        response.llm_query_id = 42
        response.content = json.dumps(
            {"trust_score": 0.5, "reasoning": "ok", "key_factors": [], "warnings": []}
        )
        return response

    llm_client = MagicMock()
    llm_client.complete = AsyncMock(side_effect=_complete)

    edge_band_calibration = {
        "description": "desc",
        "this_signal_edge_band": "0-15",
        "all_directions": {"n_signals": 32},
        "same_direction_only": {"n_signals": 20},
    }
    market_reevaluation_history = {
        "prior_signal_count": 0,
        "note": "First signal on this market.",
    }

    with patch(
        "freqpred.metrics.assessment._load_source_breakdown",
        new_callable=AsyncMock,
        return_value=[
            {"source_name": "Tavily", "document_share": 1.0, "delta_vs_overall": 0.0}
        ],
    ), patch(
        "freqpred.metrics.assessment._load_similar_market_summary",
        new_callable=AsyncMock,
        return_value={"available": True, "exact_question_subset": {}},
    ), patch(
        "freqpred.metrics.assessment._load_edge_band_calibration",
        new_callable=AsyncMock,
        return_value=edge_band_calibration,
    ), patch(
        "freqpred.metrics.assessment._load_market_reevaluation_history",
        new_callable=AsyncMock,
        return_value=market_reevaluation_history,
    ):
        await assess_signal_context(
            session,
            _make_signal(),
            _make_market(),
            _StubStrategy(),
            llm_client,
            "claude-opus-4-6",
        )

    assert "prompt" in captured
    assert "edge_band_calibration" in captured["prompt"]
    assert "market_reevaluation_history" in captured["prompt"]
    # 6000 is the audited budget for the z-ai/glm-5.2 judgment model adopted
    # 2026-08-09. This is a FLOOR, not a preference: GLM spends far more of its
    # budget on reasoning tokens than the opus-5 it replaced, and at the previous
    # 1024 it burned the whole cap and returned a tool_use block with input={} —
    # no trust_score at all, which fails open to a neutral 1.0x while still
    # paying for the call. Measured at 6000 on the frozen set: max 3072 output
    # tokens, p95 1923, 0/76 at cap. Lowering this needs its own audit
    # (freqpred/metrics/CLAUDE.md); changing the judgment model needs a fresh
    # measurement rather than an assumption that the new one fits.
    assert captured["max_tokens"] == 6000
