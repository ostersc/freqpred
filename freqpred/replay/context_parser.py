"""Recover as-of-signal-time prompt inputs from a stored ``raw_context``.

Recording a fixture from an already-resolved market must NOT read the live
series-history/FactBase tables: those now contain the outcome (the settled
market's own result is in the series counts; ``in_market_count`` includes the
occurrence that resolved the market), so live-table inputs would leak the
answer into the re-rendered prompt. The as-of state is instead parsed back out
of the frozen HISTORICAL BASE RATE and PHRASE FREQUENCY blocks inside
``raw_context`` — machine-generated text with fixed formats (see
``freqpred/signal/llm.py`` ``_build_base_rate_block`` / ``_build_factbase_block``).

Only the *primary* values are parsed (counts, labels, quotes); derived lines
(percentages, Poisson baselines, drought/elevated flags, window math) are
recomputed by ``build_prompt`` on re-render. The recorder then verifies the
reconstruction by re-rendering and requiring **byte-equality** with the stored
``raw_context`` — any parser gap, format drift, or document-content upsert
fails loudly instead of producing a corrupt fixture.

Format-coupled to prompt version ``signal-v9``. If the block formats change,
the round-trip gate fails and this module must be updated alongside them.
"""
from __future__ import annotations

import re
from datetime import datetime

from freqpred.replay.fixtures import (
    FixturePhraseData,
    FixtureSeriesCounts,
    FixtureSeriesHistory,
)

_SERIES_BLOCK_HEADER = "=== HISTORICAL BASE RATE ==="
_PHRASE_BLOCK_HEADER = "=== PHRASE FREQUENCY DATA (FactBase) ==="

_HEADER_WITH_OPTION_RE = re.compile(r'recurring weekly series \((\S+) / "(.*)"\)\.')
_HEADER_SERIES_ONLY_RE = re.compile(r"recurring series \((\S+)\)\.")
_SERIES_OVERALL_RE = re.compile(r"Series overall: (\d+) YES / (\d+) NO")
_OPTION_RE = re.compile(r"This option specifically: (\d+) YES / (\d+) NO")

_PHRASE_RE = re.compile(r'^Phrase: "(.*)"$', re.MULTILINE)
_COUNT_RES = {
    "in_market_count": re.compile(r"^  Since market opened : (\d+)$", re.MULTILINE),
    "count_7d": re.compile(r"^  Last 7 days         : (\d+)$", re.MULTILINE),
    "count_30d": re.compile(r"^  Last 30 days        : (\d+)$", re.MULTILINE),
    "count_365d": re.compile(r"^  Last 365 days       : (\d+)$", re.MULTILINE),
}
# Quote text is greedy so embedded quotes survive; the trailing `"  (...)$`
# anchor makes the LAST `"  (` the separator.
_QUOTE_RE = re.compile(r'^  \[(.*?)\] "(.*)"  \((.*)\)$', re.MULTILINE)

# The question is multi-line (rules text); non-greedy up to the
# Category-then-Current-Date anchor so embedded "Category:" text can't
# false-match. close_time/open_time also live here — market rows drift after
# resolution (early determinations rewrite close_time; rules amendments
# rewrite the question), so ALL of these must come from the frozen prompt.
_MARKET_CONTEXT_RE = re.compile(
    r"=== MARKET CONTEXT ===\n"
    r"Question: (?P<question>.*?)\n"
    r"\nCategory: (?P<category>.*)\n"
    r"Current Date \(UTC\): .*\n"
    r"Market Opened \(Issuance Date\): (?P<open_time>.*)\n"
    r"Market Closes: (?P<close_time>\S+) \(",
    re.DOTALL,
)

# One evidence entry as rendered by build_prompt. The excerpt is a single line
# (newlines are replaced before rendering). Greedy source-name group makes the
# LAST " (" the name/type separator.
_DOC_ENTRY_RE = re.compile(
    r"^\[(\d+)\] (.*)\n"
    r"    Source: (.*) \((.*)\)\n"
    r"    Published: (.*)\n"
    r"    ID: (.*)\n"
    r"    (.*)$",
    re.MULTILINE,
)


class FrozenContextParseError(Exception):
    """The stored raw_context doesn't match the expected signal-v9 block format."""


def parse_evidence_docs(raw_context: str) -> list[dict]:
    """Recover the as-of document content from the EVIDENCE block.

    Live document rows are upserted on re-fetch (``ON CONFLICT (source_url) DO
    UPDATE``), so titles, bodies, and published_at drift after the signal. The
    prompt carries everything the re-render needs: title, source name/type,
    published_at, doc ID, and the exact excerpt. The excerpt is stored back as
    the document body (with no summary) — ``build_prompt`` re-truncates and
    re-normalizes it, which is idempotent on already-rendered excerpt text.

    Returns dicts with keys: id, title, source_name, source_type,
    published_at (datetime | None), excerpt — in rendered order.
    """
    start = raw_context.find("=== EVIDENCE ===")
    if start == -1:
        raise FrozenContextParseError("no EVIDENCE block")
    section = raw_context[start:]

    docs: list[dict] = []
    for match in _DOC_ENTRY_RE.finditer(section):
        _index, title, source_name, source_type, published, doc_id, excerpt = match.groups()
        docs.append(
            {
                "id": doc_id,
                "title": title,
                "source_name": source_name,
                "source_type": source_type,
                "published_at": (
                    None if published == "unknown" else datetime.fromisoformat(published)
                ),
                "excerpt": excerpt,
            }
        )
    if not docs:
        raise FrozenContextParseError("EVIDENCE block: no parseable document entries")
    return docs


def parse_market_context(raw_context: str) -> dict:
    """Recover as-of market fields from the MARKET CONTEXT block.

    The live market row drifts after resolution: Kalshi re-buckets finalized
    markets' category to "other", early determinations rewrite ``close_time``,
    and rules amendments rewrite the question text. Frozen-context recording
    takes all of them from the prompt, not the row.

    Returns dict with keys: question, category, open_time (datetime | None),
    close_time (datetime).
    """
    match = _MARKET_CONTEXT_RE.search(raw_context)
    if match is None:
        raise FrozenContextParseError("MARKET CONTEXT block: unrecognized format")
    open_time = match.group("open_time")
    return {
        "question": match.group("question"),
        "category": match.group("category"),
        "open_time": None if open_time == "unknown" else datetime.fromisoformat(open_time),
        "close_time": datetime.fromisoformat(match.group("close_time")),
    }


def _slice_block(raw_context: str, header: str) -> str | None:
    """Return the block from *header* up to the next ``===`` section header."""
    start = raw_context.find(header)
    if start == -1:
        return None
    rest = raw_context[start + len(header):]
    next_header = rest.find("\n===")
    return rest if next_header == -1 else rest[:next_header]


def parse_series_block(raw_context: str, market_id: str) -> FixtureSeriesHistory | None:
    """Parse the HISTORICAL BASE RATE block back into fixture inputs.

    Returns None when the prompt had no block. Raises FrozenContextParseError
    when the block exists but doesn't match the known format.
    """
    block = _slice_block(raw_context, _SERIES_BLOCK_HEADER)
    if block is None:
        return None

    option_code = market_id.rsplit("-", 1)[-1] if "-" in market_id else market_id

    with_option = _HEADER_WITH_OPTION_RE.search(block)
    series_only = _HEADER_SERIES_ONLY_RE.search(block)
    if with_option is None and series_only is None:
        raise FrozenContextParseError("HISTORICAL BASE RATE block: no series header line")
    series_ticker = with_option.group(1) if with_option else series_only.group(1)

    overall = _SERIES_OVERALL_RE.search(block)
    if overall is None:
        # "No series history available." variant — series_row was None.
        if "No series history available." not in block:
            raise FrozenContextParseError(
                "HISTORICAL BASE RATE block: no 'Series overall' line and no "
                "'No series history available.' marker"
            )
        return FixtureSeriesHistory(
            series_ticker=series_ticker, option_code=option_code,
            series_row=None, option_row=None,
        )

    series_row = FixtureSeriesCounts(
        option_label=series_ticker,  # aggregate rows carry the ticker as label
        yes_count=int(overall.group(1)),
        no_count=int(overall.group(2)),
    )

    option_row = None
    option = _OPTION_RE.search(block)
    if option is not None:
        if with_option is None:
            raise FrozenContextParseError(
                "HISTORICAL BASE RATE block: option counts without an option label header"
            )
        option_row = FixtureSeriesCounts(
            option_label=with_option.group(2),
            yes_count=int(option.group(1)),
            no_count=int(option.group(2)),
        )

    return FixtureSeriesHistory(
        series_ticker=series_ticker, option_code=option_code,
        series_row=series_row, option_row=option_row,
    )


def parse_phrase_block(raw_context: str, fetched_at: datetime) -> FixturePhraseData | None:
    """Parse the PHRASE FREQUENCY block back into fixture inputs.

    ``api_query``/``speaker_slug`` are not rendered into the prompt and default
    to empty; ``fetched_at`` is not rendered either, so the caller supplies a
    placeholder (none of the three affect the re-render).
    """
    block = _slice_block(raw_context, _PHRASE_BLOCK_HEADER)
    if block is None:
        return None

    phrase = _PHRASE_RE.search(block)
    if phrase is None:
        raise FrozenContextParseError("PHRASE FREQUENCY block: no Phrase line")

    counts: dict[str, int] = {}
    for name, pattern in _COUNT_RES.items():
        match = pattern.search(block)
        if match is None:
            raise FrozenContextParseError(f"PHRASE FREQUENCY block: no {name} line")
        counts[name] = int(match.group(1))

    quotes = [
        {"date": date, "text": text, "event_type": event_type}
        for date, text, event_type in _QUOTE_RE.findall(block)
    ]

    return FixturePhraseData(
        display_phrase=phrase.group(1),
        api_query="",
        speaker_slug="",
        top_quotes=quotes,
        fetched_at=fetched_at,
        **counts,
    )
