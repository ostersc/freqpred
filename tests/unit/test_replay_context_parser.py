"""Unit tests for freqpred/replay/context_parser.py.

The property that matters is the round trip: inputs → build_prompt render →
parse back → identical inputs (and hence an identical re-render). Tests build
prompts with the real ``build_prompt`` so the parser stays coupled to the
actual block formats.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import freqpred.ingestion.models  # noqa: F401 — registers mappers
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.replay.context_parser import (
    FrozenContextParseError,
    parse_phrase_block,
    parse_series_block,
)
from freqpred.replay.engine import render_prompt_from_inputs
from freqpred.replay.fixtures import (
    FixtureDocument,
    FixtureInputs,
    FixtureMarket,
    FixturePhraseData,
    FixtureSeriesCounts,
    FixtureSeriesHistory,
)

FROZEN_NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
MARKET_ID = "KXTEST-26JUL06-FOO"


def _inputs(
    series_history: FixtureSeriesHistory | None = None,
    phrase_data: FixturePhraseData | None = None,
) -> FixtureInputs:
    return FixtureInputs(
        now=FROZEN_NOW,
        market=FixtureMarket(
            id=MARKET_ID,
            question="Will the test event happen?",
            category="Mentions",
            close_time=FROZEN_NOW + timedelta(days=3),
            open_time=FROZEN_NOW - timedelta(days=4),
            yes_bid=0.48,
            yes_ask=0.50,
            mid_price=0.49,
        ),
        documents=[
            FixtureDocument(
                id="11111111-1111-1111-1111-111111111111",
                source_url="https://example.com/a",
                title="Doc A",
                body="Body of document A.",
                source_type="news",
                source_name="Reuters",
                published_at=FROZEN_NOW - timedelta(days=1),
                fetched_at=FROZEN_NOW,
            )
        ],
        series_history=series_history,
        phrase_data=phrase_data,
        llm_response="{}",  # parser tests never parse the response
    )


def _roundtrip(
    series_history: FixtureSeriesHistory | None,
    phrase_data: FixturePhraseData | None,
) -> str:
    """Render → parse → re-render; assert byte-identical. Returns the render."""
    rendered = render_prompt_from_inputs(_inputs(series_history, phrase_data))
    parsed_series = parse_series_block(rendered, MARKET_ID)
    parsed_phrase = parse_phrase_block(rendered, fetched_at=FROZEN_NOW)
    rerendered = render_prompt_from_inputs(_inputs(parsed_series, parsed_phrase))
    assert rerendered == rendered
    return rendered


# ---------------------------------------------------------------------------
# Series block round trips
# ---------------------------------------------------------------------------


def test_series_roundtrip_with_option_row() -> None:
    _roundtrip(
        FixtureSeriesHistory(
            series_ticker="KXTEST",
            option_code="FOO",
            series_row=FixtureSeriesCounts(option_label="KXTEST", yes_count=157, no_count=148),
            option_row=FixtureSeriesCounts(option_label="Foo / Bar", yes_count=6, no_count=3),
        ),
        None,
    )


def test_series_roundtrip_small_sample_option() -> None:
    # n < 3 renders the "— small sample, treat as weak signal." suffix.
    _roundtrip(
        FixtureSeriesHistory(
            series_ticker="KXTEST",
            option_code="FOO",
            series_row=FixtureSeriesCounts(option_label="KXTEST", yes_count=10, no_count=10),
            option_row=FixtureSeriesCounts(option_label="Foo", yes_count=1, no_count=1),
        ),
        None,
    )


def test_series_roundtrip_no_option_row() -> None:
    rendered = _roundtrip(
        FixtureSeriesHistory(
            series_ticker="KXTEST",
            option_code="FOO",
            series_row=FixtureSeriesCounts(option_label="KXTEST", yes_count=5, no_count=7),
            option_row=None,
        ),
        None,
    )
    assert "No per-option history available" in rendered


def test_series_roundtrip_no_history_at_all() -> None:
    rendered = _roundtrip(
        FixtureSeriesHistory(
            series_ticker="KXTEST", option_code="FOO", series_row=None, option_row=None
        ),
        None,
    )
    assert "No series history available." in rendered


def test_no_series_block_returns_none() -> None:
    rendered = render_prompt_from_inputs(_inputs(None, None))
    assert parse_series_block(rendered, MARKET_ID) is None


# ---------------------------------------------------------------------------
# Phrase block round trips
# ---------------------------------------------------------------------------


def _phrase(
    quotes: list[dict] | None = None,
    in_market: int = 0,
) -> FixturePhraseData:
    return FixturePhraseData(
        display_phrase='Landslide / "Total Landslide"',
        in_market_count=in_market,
        count_7d=2,
        count_30d=14,
        count_365d=53,
        top_quotes=quotes or [],
        fetched_at=FROZEN_NOW,
    )


def test_phrase_roundtrip_with_quotes_and_awkward_characters() -> None:
    # Quote text with embedded quotes and parens; event_type with parens.
    quotes = [
        {"date": "2026-06-22", "text": 'He said "landslide (again)" twice', "event_type": "rally (live)"},
        {"date": "2026-06-25", "text": "Plain quote", "event_type": "truth_social"},
    ]
    _roundtrip(None, _phrase(quotes=quotes))


def test_phrase_roundtrip_no_quotes_and_drought_flags() -> None:
    # count_30d=0 triggers the RECENT DROUGHT derived lines — all recomputed.
    data = FixturePhraseData(
        display_phrase="Moscow",
        in_market_count=0,
        count_7d=0,
        count_30d=0,
        count_365d=5,
        top_quotes=[],
        fetched_at=FROZEN_NOW,
    )
    rendered = _roundtrip(None, data)
    assert "RECENT DROUGHT" in rendered


def test_phrase_roundtrip_truncated_quote_is_idempotent() -> None:
    # build_prompt truncates quote text to 120 chars; parsing the truncated
    # text and re-rendering truncates again — a no-op.
    long_quote = [{"date": "2026-06-01", "text": "x" * 300, "event_type": "speech"}]
    _roundtrip(None, _phrase(quotes=long_quote))


def test_both_blocks_roundtrip_together() -> None:
    _roundtrip(
        FixtureSeriesHistory(
            series_ticker="KXTEST",
            option_code="FOO",
            series_row=FixtureSeriesCounts(option_label="KXTEST", yes_count=157, no_count=148),
            option_row=FixtureSeriesCounts(option_label="Foo", yes_count=9, no_count=1),
        ),
        _phrase(quotes=[{"date": "2026-06-22", "text": "hi", "event_type": "rally"}], in_market=1),
    )


def test_malformed_block_raises() -> None:
    with pytest.raises(FrozenContextParseError):
        parse_series_block("=== HISTORICAL BASE RATE ===\ngarbage\n=== EVIDENCE ===", MARKET_ID)
    with pytest.raises(FrozenContextParseError):
        parse_phrase_block(
            "=== PHRASE FREQUENCY DATA (FactBase) ===\ngarbage\n=== EVIDENCE ===",
            fetched_at=FROZEN_NOW,
        )


def test_parse_market_context() -> None:
    from freqpred.replay.context_parser import parse_market_context

    inputs = _inputs(None, None)
    # Multi-line question with rules text, including an embedded "Category:"
    # line that must not false-match (the real one is anchored to the
    # Current Date line that follows it).
    inputs.market.question = "Will X?\n\nRules:\nCategory: fake\nMore rules text"
    parsed = parse_market_context(render_prompt_from_inputs(inputs))
    assert parsed["question"] == inputs.market.question
    assert parsed["category"] == "Mentions"
    assert parsed["open_time"] == inputs.market.open_time
    assert parsed["close_time"] == inputs.market.close_time

    # Unknown issuance date round-trips to None.
    inputs.market.open_time = None
    rendered = render_prompt_from_inputs(inputs)
    assert "unknown" in rendered
    assert parse_market_context(rendered)["open_time"] is None

    with pytest.raises(FrozenContextParseError):
        parse_market_context("no market context here")


def test_parse_evidence_docs_roundtrip() -> None:
    from freqpred.replay.context_parser import parse_evidence_docs

    inputs = _inputs(None, None)
    # Awkward title with parens and pipes; body with newlines that the render
    # normalizes into a single-line excerpt.
    inputs.documents[0].title = "Trump | Robert Reich (opinion)"
    inputs.documents[0].body = "line one\nline two (with parens)\nline three"
    rendered = render_prompt_from_inputs(inputs)

    parsed = parse_evidence_docs(rendered)
    assert len(parsed) == 1
    doc = parsed[0]
    assert doc["id"] == inputs.documents[0].id
    assert doc["title"] == "Trump | Robert Reich (opinion)"
    assert doc["source_name"] == "Reuters"
    assert doc["source_type"] == "news"
    assert doc["published_at"] == inputs.documents[0].published_at
    assert doc["excerpt"] == "line one line two (with parens) line three"

    # Store the excerpt back as the body → re-render is byte-identical.
    inputs.documents[0].body = doc["excerpt"]
    inputs.documents[0].summary = None
    assert render_prompt_from_inputs(inputs) == rendered

    with pytest.raises(FrozenContextParseError):
        parse_evidence_docs("no evidence header")
