"""The full-body backfill that makes a frozen-context bank usable for T101.

A frozen-context fixture stores the 500-char excerpt the signal LLM was shown
as its ``body`` — which is exactly what retrieval-time extraction replaces, so
without a separate full text a prompt-mode benchmark of T101 measures nothing.
``_verified_full_body`` carries the live row's text across, but only when that
row still re-renders the frozen excerpt byte-for-byte; documents are upserted
on re-fetch and a live blog rewrites itself hourly.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import freqpred.ingestion.models  # noqa: F401 — registers mappers
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.rag.models import DocumentRow
from freqpred.replay.fixtures import FixtureDocument
from freqpred.replay.recorder import _render_excerpt, _verified_full_body

FROZEN_NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)

# Long enough that the excerpt is a true prefix, with a newline inside the cut
# so the flatten-and-strip step is actually exercised.
_LONG_BODY = (
    "Trump is scheduled to attend the Davos forum on Tuesday.\n"
    "Aides said he would take questions afterward. " + ("Filler sentence. " * 60)
)


def _row(body: str, summary: str | None = None) -> DocumentRow:
    return DocumentRow(
        id="11111111-1111-1111-1111-111111111111",
        source_url="https://example.com/a",
        content_hash="abc",
        title="Doc A",
        body=body,
        summary=summary,
        source_type="news",
        source_name="Reuters",
        category="Politics",
        tags=[],
        published_at=FROZEN_NOW - timedelta(days=1),
        fetched_at=FROZEN_NOW,
    )


def test_full_body_carried_when_live_row_reproduces_the_excerpt():
    frozen = _render_excerpt(_LONG_BODY)
    assert _verified_full_body(_row(_LONG_BODY), frozen) == _LONG_BODY


def test_full_body_rejected_when_the_document_drifted():
    """The re-fetch failure mode: same id, different article."""
    frozen = _render_excerpt(_LONG_BODY)
    drifted = "[Skip to navigation] Print subscriptions " + ("Other text. " * 80)
    assert _verified_full_body(_row(drifted), frozen) is None


def test_full_body_carried_when_the_excerpt_came_from_the_summary():
    """build_prompt preferred summary, so the summary vouches for identity."""
    summary = "Trump will attend Davos and take questions."
    frozen = _render_excerpt(summary)
    assert _verified_full_body(_row(_LONG_BODY, summary=summary), frozen) == _LONG_BODY


def test_full_body_skipped_for_short_bodies():
    """Nothing beyond the excerpt to offer — extraction would skip it anyway."""
    short = "A short body under the cap."
    assert _verified_full_body(_row(short), _render_excerpt(short)) is None


def test_full_body_capped():
    huge = "x" * 40_000
    carried = _verified_full_body(_row(huge), _render_excerpt(huge))
    assert carried is not None
    assert len(carried) == 16_000


def _fixture_doc(**kwargs) -> FixtureDocument:
    base = {
        "id": "11111111-1111-1111-1111-111111111111",
        "source_url": "https://example.com/a",
        "title": "Doc A",
        "body": "the frozen 500-char excerpt",
        "source_type": "news",
        "source_name": "Reuters",
        "published_at": FROZEN_NOW - timedelta(days=1),
        "fetched_at": FROZEN_NOW,
    }
    return FixtureDocument(**{**base, **kwargs})


def test_to_document_defaults_to_the_frozen_excerpt():
    """Prompt rendering and the expectations round-trip must see the excerpt."""
    doc = _fixture_doc(full_body=_LONG_BODY)
    assert doc.to_document().body == "the frozen 500-char excerpt"


def test_to_document_full_substitutes_the_full_body():
    doc = _fixture_doc(full_body=_LONG_BODY)
    assert doc.to_document(full=True).body == _LONG_BODY


def test_to_document_full_falls_back_when_no_full_body():
    doc = _fixture_doc()
    assert doc.to_document(full=True).body == "the frozen 500-char excerpt"
