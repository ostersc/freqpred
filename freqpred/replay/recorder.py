"""Record replay fixtures from real production signals (T66).

Everything a fixture needs already exists in the DB: the signal row carries the
rendered prompt (``raw_context``), the ordered retrieved-document IDs
(``sources``), and prices at signal time; the ``llm_queries`` audit row carries
the verbatim LLM response; ``document_market_links`` carries per-document
retrieval scores. The recorder reassembles those into structured inputs, then
computes the expectations with the same engine the replay tests use — so a
fixture is verified replayable at record time.

Reconstruction notes:
- ``now`` is recovered from the "Current Date (UTC): ..." line inside
  ``raw_context`` (minute precision — the prompt was rendered seconds before
  the signal row was created), falling back to ``signal.created_at``.
- ``yes_bid``/``yes_ask`` at signal time are reconstructed from the stored
  ``market_mid_at_signal`` and side-specific ``market_ask_at_signal``.
- Series-history and FactBase blocks are included only when the original
  prompt contained them, using *current* DB rows — content drift against the
  historical prompt is surfaced as a warning, and the fixture's expectations
  are computed from the fixture's own (current) inputs, so it stays
  self-consistent.
"""
from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow, FactbasePhraseRow
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import MarketRow
from freqpred.metrics.series_history import get_series_history_for_market
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.replay.engine import compute_expectations
from freqpred.replay.fixtures import (
    FixtureDecisionContext,
    FixtureDocument,
    FixtureInputs,
    FixtureMarket,
    FixturePhraseData,
    FixturePriorScheduledSignal,
    FixtureSeriesCounts,
    FixtureSeriesHistory,
    ReplayFixture,
)
from freqpred.signal.models import SignalRow

_CURRENT_DATE_RE = re.compile(r"Current Date \(UTC\): (\d{4}-\d{2}-\d{2} \d{2}:\d{2})")

# Body truncation caps. When a summary exists the body is unused by both the
# prompt (which prefers summary) and BM25 (COALESCE(summary, body)); without a
# summary the prompt uses body[:500] and BM25 scores the body text.
_BODY_CAP_WITH_SUMMARY = 500
_BODY_CAP_NO_SUMMARY = 4000

class RecordingError(Exception):
    """Raised when a signal cannot be turned into a replayable fixture."""


def _reconstruct_prices(
    direction: str,
    mid: float,
    ask_at_signal: float | None,
    probability: float,
    edge: float,
) -> tuple[float, float]:
    """Recover (yes_bid, yes_ask) at signal time from the signal's stored fields."""
    if direction == "YES":
        if ask_at_signal is None:
            raise RecordingError("YES signal missing market_ask_at_signal")
        yes_ask = ask_at_signal
        yes_bid = round(2.0 * mid - yes_ask, 4)
    elif direction == "NO":
        if ask_at_signal is None:
            raise RecordingError("NO signal missing market_ask_at_signal")
        yes_bid = round(1.0 - ask_at_signal, 4)
        yes_ask = round(2.0 * mid - yes_bid, 4)
    else:  # SKIP: edge was computed as probability - yes_ask
        yes_ask = round(probability - edge, 4)
        yes_bid = round(2.0 * mid - yes_ask, 4)
    return max(0.0, min(1.0, yes_bid)), max(0.0, min(1.0, yes_ask))


def _parse_prompt_now(raw_context: str) -> datetime | None:
    match = _CURRENT_DATE_RE.search(raw_context)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=UTC)


def _first_diff_preview(a: str, b: str, context: int = 60) -> str:
    for i, (ca, cb) in enumerate(zip(a, b, strict=False)):
        if ca != cb:
            lo = max(0, i - context)
            return f"char {i}: {a[lo:i + context]!r} vs {b[lo:i + context]!r}"
    return f"lengths {len(a)} vs {len(b)}"


async def record_fixture(
    session: AsyncSession,
    signal_id: uuid.UUID,
    *,
    strategy_name: str = "ConservativeDefault",
    bankroll: float = 1000.0,
    name: str | None = None,
    description: str = "",
) -> tuple[ReplayFixture, list[str]]:
    """Build a ReplayFixture from a real signal. Returns (fixture, warnings).

    The signal must have called the LLM (``llm_query_id`` set) — price-moved
    repricing clones carry no response to mock.
    """
    warnings: list[str] = []

    signal = (
        await session.execute(select(SignalRow).where(SignalRow.id == signal_id))
    ).scalar_one_or_none()
    if signal is None:
        raise RecordingError(f"signal {signal_id} not found")
    if signal.llm_query_id is None:
        raise RecordingError(
            f"signal {signal_id} has no llm_query_id (price-moved clone?) — "
            "record from an LLM-backed signal"
        )

    llm_row = (
        await session.execute(
            select(LLMQueryRow).where(LLMQueryRow.id == signal.llm_query_id)
        )
    ).scalar_one_or_none()
    if llm_row is None or not llm_row.response:
        raise RecordingError(f"llm_queries row {signal.llm_query_id} missing or empty")

    market_row = (
        await session.execute(select(MarketRow).where(MarketRow.id == signal.market_id))
    ).scalar_one_or_none()
    if market_row is None:
        raise RecordingError(f"market {signal.market_id} not found")

    now = _parse_prompt_now(signal.raw_context)
    if now is None:
        now = signal.created_at
        warnings.append(
            "could not parse 'Current Date' from raw_context — using "
            "signal.created_at as the frozen clock (window math may differ "
            "slightly from the original prompt)"
        )

    yes_bid, yes_ask = _reconstruct_prices(
        signal.direction,
        signal.market_mid_at_signal,
        signal.market_ask_at_signal,
        signal.estimated_probability,
        signal.edge,
    )

    # Documents in retrieval order (signal.sources preserves it), with the
    # blended scores persisted on the per-signal document links.
    source_ids = [uuid.UUID(s) for s in signal.sources]
    if not source_ids:
        raise RecordingError(f"signal {signal_id} has no source documents")
    doc_rows = {
        row.id: row
        for row in (
            await session.execute(select(DocumentRow).where(DocumentRow.id.in_(source_ids)))
        ).scalars()
    }
    missing = [str(s) for s in source_ids if s not in doc_rows]
    if missing:
        raise RecordingError(f"documents deleted since signal: {missing}")

    scores = {
        row.document_id: row.relevance_score
        for row in (
            await session.execute(
                select(DocumentMarketLinkRow).where(
                    DocumentMarketLinkRow.signal_id == signal_id
                )
            )
        ).scalars()
    }

    fixture_docs = []
    for doc_id in source_ids:
        row = doc_rows[doc_id]
        body_cap = _BODY_CAP_WITH_SUMMARY if row.summary else _BODY_CAP_NO_SUMMARY
        fixture_docs.append(
            FixtureDocument(
                id=str(row.id),
                source_url=row.source_url,
                content_hash=row.content_hash,
                title=row.title,
                body=row.body[:body_cap],
                summary=row.summary,
                source_type=row.source_type,
                source_name=row.source_name,
                category=row.category,
                tags=list(row.tags),
                published_at=row.published_at,
                fetched_at=row.fetched_at,
                similarity_score=scores.get(doc_id, 0.0),
            )
        )

    cat_result = await session.execute(
        select(CatalystQueryRow)
        .join(CatalystRunRow, CatalystQueryRow.run_id == CatalystRunRow.id)
        .where(
            CatalystRunRow.market_id == signal.market_id,
            CatalystRunRow.is_active.is_(True),
        )
    )
    catalyst_queries = [row.query_text for row in cat_result.scalars().all()]

    series_history = None
    if "=== HISTORICAL BASE RATE ===" in signal.raw_context and market_row.series_ticker:
        option_code = (
            signal.market_id.rsplit("-", 1)[-1] if "-" in signal.market_id else signal.market_id
        )
        history = await get_series_history_for_market(
            session, market_row.series_ticker, option_code
        )
        if history is not None:
            def _counts(row) -> FixtureSeriesCounts | None:  # noqa: ANN001 — ORM row
                if row is None:
                    return None
                return FixtureSeriesCounts(
                    option_label=row.option_label,
                    yes_count=row.yes_count,
                    no_count=row.no_count,
                )

            series_history = FixtureSeriesHistory(
                series_ticker=history["series_ticker"],
                option_code=history["option_code"],
                series_row=_counts(history["series_row"]),
                option_row=_counts(history["option_row"]),
            )
        else:
            warnings.append(
                "original prompt had a HISTORICAL BASE RATE block but no series "
                "history rows exist now — fixture omits the block"
            )

    phrase_data = None
    if "=== PHRASE FREQUENCY DATA" in signal.raw_context:
        fb_row = (
            await session.execute(
                select(FactbasePhraseRow).where(FactbasePhraseRow.market_id == signal.market_id)
            )
        ).scalar_one_or_none()
        if fb_row is not None:
            phrase_data = FixturePhraseData(
                display_phrase=fb_row.display_phrase,
                api_query=fb_row.api_query,
                speaker_slug=fb_row.speaker_slug,
                in_market_count=fb_row.in_market_count,
                count_7d=fb_row.count_7d,
                count_30d=fb_row.count_30d,
                count_365d=fb_row.count_365d,
                top_quotes=list(fb_row.top_quotes or []),
                fetched_at=fb_row.last_fetched_at,
            )
        else:
            warnings.append(
                "original prompt had a PHRASE FREQUENCY DATA block but no "
                "factbase row exists now — fixture omits the block"
            )

    prior_row = (
        await session.execute(
            select(SignalRow.retrieval_hash, SignalRow.created_at, SignalRow.confidence)
            .where(
                SignalRow.market_id == signal.market_id,
                SignalRow.trigger == "scheduled",
                SignalRow.created_at < signal.created_at,
            )
            .order_by(SignalRow.created_at.desc())
            .limit(1)
        )
    ).one_or_none()
    prior_scheduled = (
        FixturePriorScheduledSignal(
            retrieval_hash=prior_row[0],
            created_at=prior_row[1],
            confidence=prior_row[2],
            # The FactBase refresh timestamp at signal time is not recoverable
            # retroactively; None keeps the skip decision hash/age-driven.
            factbase_refreshed_at=None,
        )
        if prior_row is not None
        else None
    )

    inputs = FixtureInputs(
        now=now,
        trigger=signal.trigger,
        market=FixtureMarket(
            id=market_row.id,
            platform=market_row.platform,
            question=market_row.question,
            category=market_row.category,
            close_time=market_row.close_time,
            open_time=market_row.open_time,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            mid_price=signal.market_mid_at_signal,
            volume_24h=market_row.volume_24h,
            open_interest=market_row.open_interest,
            series_ticker=market_row.series_ticker,
        ),
        documents=fixture_docs,
        catalyst_queries=catalyst_queries,
        series_history=series_history,
        phrase_data=phrase_data,
        llm_response=llm_row.response,
        prior_scheduled_signal=prior_scheduled,
        decision_context=FixtureDecisionContext(strategy=strategy_name, bankroll=bankroll),
    )

    fixture_name = name or f"{signal.market_id}_{signal.direction}".lower().replace("/", "_")
    fixture = ReplayFixture(
        name=fixture_name,
        description=description,
        recorded_from_signal_id=str(signal_id),
        recorded_at=datetime.now(UTC),
        inputs=inputs,
        expectations=compute_expectations(inputs, fixture_name=fixture_name),
    )

    # Sanity checks against the historical record — drift here doesn't make the
    # fixture invalid (expectations are self-consistent with its inputs) but
    # the operator should know what changed.
    if fixture.expectations.rendered_prompt != signal.raw_context:
        warnings.append(
            "re-rendered prompt differs from the signal's stored raw_context "
            "(clock precision or content drift since the signal): "
            + _first_diff_preview(signal.raw_context, fixture.expectations.rendered_prompt)
        )
    if fixture.expectations.retrieval_hash != signal.retrieval_hash:
        warnings.append(
            "recomputed retrieval hash differs from the signal's stored hash — "
            "source document set no longer matches"
        )
    parsed = fixture.expectations.parsed
    if parsed.direction != signal.direction or abs(
        parsed.probability - signal.estimated_probability
    ) > 1e-6:
        warnings.append(
            f"parsed response (direction={parsed.direction}, "
            f"probability={parsed.probability}) does not match the signal row "
            f"(direction={signal.direction}, "
            f"probability={signal.estimated_probability})"
        )
    if abs(fixture.expectations.edge - signal.edge) > 1e-4:
        warnings.append(
            f"recomputed edge {fixture.expectations.edge:.6f} differs from stored "
            f"edge {signal.edge:.6f} beyond price-reconstruction rounding"
        )

    return fixture, warnings
