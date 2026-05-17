"""FactBase phrase frequency fetcher for KXTRUMPSAY markets.

Flow per market (all steps idempotent):
1. Scheduler checks DB for existing row.
2. If fresh (< 24h): mark_ready and skip.
3. If stale or missing: re-fetch counts from API.
4. If missing: call Haiku once to extract search terms, then insert row.
5. Upsert counts, mark_ready in in-process cache.

Haiku is called at most once per market lifetime — subsequent cycles reuse
the stored api_query.
"""
from __future__ import annotations

import asyncio
import json
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from freqpred.ingestion.models import FactbasePhraseRow

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from freqpred.llm.client import LLMClient
    from freqpred.runtime.telemetry import RuntimeTelemetry

log = structlog.get_logger(__name__)

_FACTBASE_API = "https://api.factsquared.com/json/factba.se-{speaker}-20240623.php"
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_REQUEST_TIMEOUT = 15.0
_PAGE_RATE_SLEEP = 1.0  # seconds between pages — be polite
_SCHEDULER_REFRESH_HOURS = 24
_TOP_QUOTES_MAX = 5
_HAIKU_MODEL = "claude-haiku-4-5-20251001"

_PHRASE_EXTRACT_TOOL = {
    "name": "extract_search_terms",
    "description": (
        "Extract the target phrase(s) from a KXTRUMPSAY market question and "
        "produce all search variants needed to match Trump's exact statements."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "display_phrase": {
                "type": "string",
                "description": "Human-readable phrase label, e.g. 'Communist / Communism'",
            },
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "All exact variants to search for, including plurals and possessives "
                    "of each slash-separated alternative. Each term will be wrapped in "
                    "double-quotes and OR'd together in the API query."
                ),
            },
        },
        "required": ["display_phrase", "terms"],
    },
}

_PHRASE_EXTRACT_SYSTEM = """\
You extract search terms from Kalshi prediction market questions about what Trump will say.

Rules:
- Find the quoted phrase in the question (single quotes, double quotes, or curly quotes).
- If the phrase contains slash-separated variants (e.g. "Communist / Communism" or "Doge/Dogecoin"),
  treat each side of the slash as a separate base term — either satisfies the market.
- For each base term, generate:
  - The term itself (exact)
  - Simple plural (add "s" or "es" as appropriate)
  - Possessive (add "'s")
  - Plural possessive (add "s'" or "es'")
  Do NOT add tense inflections, synonyms, or hyphenated compounds — only plurals and possessives.
- Keep multi-word phrases intact (e.g. "witch hunt" stays whole; pluralize only the last word if applicable).
- Return display_phrase as the readable label and terms as the flat list of all variants.
- If no clear phrase can be extracted, return display_phrase="" and terms=[].
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FactbaseSearchTerms:
    display_phrase: str
    api_query: str  # '"term1" OR "term2" OR ...'
    match_terms: list[str]  # flat list for in-Python matching


@dataclass
class FactbasePhraseData:
    display_phrase: str
    api_query: str
    speaker_slug: str
    in_market_count: int
    count_7d: int
    count_30d: int
    count_365d: int
    top_quotes: list[dict]
    fetched_at: datetime


# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------


class FactbasePhraseCache:
    """Tracks market_ids whose phrase data is ready in DB.

    Populated async by run_factbase_scheduler; read sync by
    PoliticsEdgeStrategy.is_market_interesting().
    """

    def __init__(self) -> None:
        self._ready: set[str] = set()

    def mark_ready(self, market_id: str) -> None:
        self._ready.add(market_id)

    def is_ready(self, market_id: str) -> bool:
        return market_id in self._ready


# ---------------------------------------------------------------------------
# Phrase extraction (Haiku — once per market)
# ---------------------------------------------------------------------------


async def extract_search_terms(
    question: str,
    llm_client: "LLMClient",
    market_id: str | None = None,
) -> FactbaseSearchTerms | None:
    """Use Haiku to extract search terms from a KXTRUMPSAY market question.

    Returns None if no phrase can be extracted (market blocked indefinitely).
    Logs one llm_queries row per call — satisfies hard constraint #2.
    """
    from freqpred.llm.client import LLMError

    try:
        response = await llm_client.complete(
            question,
            _HAIKU_MODEL,
            query_type="factbase_phrase_extract",
            system=_PHRASE_EXTRACT_SYSTEM,
            market_id=market_id,
            max_tokens=256,
            json_tool=_PHRASE_EXTRACT_TOOL,
        )
    except LLMError:
        log.warning("factbase.extract_search_terms.llm_error", market_id=market_id, exc_info=True)
        return None

    try:
        parsed = json.loads(response.content)
        display_phrase: str = parsed.get("display_phrase", "")
        terms: list[str] = parsed.get("terms") or []
    except (json.JSONDecodeError, AttributeError):
        log.warning("factbase.extract_search_terms.parse_error", content=response.content[:200])
        return None

    if not display_phrase or not terms:
        log.info("factbase.extract_search_terms.no_phrase", market_id=market_id, question=question[:120])
        return None

    api_query = " OR ".join(f'"{t}"' for t in terms)
    log.info(
        "factbase.extract_search_terms.ok",
        market_id=market_id,
        display_phrase=display_phrase,
        term_count=len(terms),
    )
    return FactbaseSearchTerms(
        display_phrase=display_phrase,
        api_query=api_query,
        match_terms=terms,
    )


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------


async def fetch_phrase_frequency(
    search_terms: FactbaseSearchTerms,
    speaker_slug: str,
    market_open_time: datetime | None,
    *,
    http_timeout: float = _REQUEST_TIMEOUT,
) -> FactbasePhraseData:
    """Fetch occurrence counts for all time windows from the FactBase API.

    Paginates until segments older than 365 days are reached, then counts
    in Python. Returns zeros + empty quotes on any error (never raises).
    """
    now = datetime.now(UTC)
    cutoff_365d = now - timedelta(days=365)
    cutoff_30d = now - timedelta(days=30)
    cutoff_7d = now - timedelta(days=7)
    market_open = market_open_time or now

    url = _FACTBASE_API.format(speaker=speaker_slug)
    encoded_q = urllib.parse.quote(search_terms.api_query)
    params_str = f"q={encoded_q}&sort=date-desc"

    all_segments: list[dict] = []
    page = 1

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=http_timeout,
    ) as client:
        while True:
            try:
                resp = await client.post(f"{url}?{params_str}&page={page}")
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                log.warning(
                    "factbase.fetch.http_error",
                    page=page,
                    phrase=search_terms.display_phrase,
                    exc_info=True,
                )
                break

            segments = data.get("data") or []
            if not segments:
                break

            total_pages = data.get("meta", {}).get("total_pages", 1)

            for seg in segments:
                seg_date_str = seg.get("date", "")
                try:
                    seg_date = datetime.fromisoformat(seg_date_str).replace(tzinfo=UTC)
                except (ValueError, AttributeError):
                    continue
                if seg_date < cutoff_365d:
                    # Sorted desc — everything from here is older than 365d
                    all_segments = all_segments  # already collected
                    page = total_pages + 1  # force exit
                    break
                all_segments.append({**seg, "_dt": seg_date})

            if page >= total_pages:
                break

            page += 1
            await asyncio.sleep(_PAGE_RATE_SLEEP)

    # Count windows and collect Trump-only quotes
    in_market_count = 0
    count_7d = 0
    count_30d = 0
    count_365d = 0
    trump_quotes: list[dict] = []

    for seg in all_segments:
        seg_dt: datetime = seg["_dt"]
        speaker = seg.get("speaker", "")
        text = str(seg.get("text") or "")

        if seg_dt >= cutoff_365d:
            count_365d += 1
        if seg_dt >= cutoff_30d:
            count_30d += 1
        if seg_dt >= cutoff_7d:
            count_7d += 1
        if seg_dt >= market_open:
            in_market_count += 1

        if speaker == "Donald Trump" and text:
            trump_quotes.append({
                "date": seg.get("date", ""),
                "text": text[:200],
                "event_type": seg.get("record_type") or seg.get("type", ""),
            })

    # Most recent first, capped
    top_quotes = trump_quotes[:_TOP_QUOTES_MAX]

    log.info(
        "factbase.fetch.ok",
        phrase=search_terms.display_phrase,
        in_market=in_market_count,
        count_7d=count_7d,
        count_30d=count_30d,
        count_365d=count_365d,
    )

    return FactbasePhraseData(
        display_phrase=search_terms.display_phrase,
        api_query=search_terms.api_query,
        speaker_slug=speaker_slug,
        in_market_count=in_market_count,
        count_7d=count_7d,
        count_30d=count_30d,
        count_365d=count_365d,
        top_quotes=top_quotes,
        fetched_at=now,
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def phrase_row_to_data(row: FactbasePhraseRow) -> FactbasePhraseData:
    """Convert a DB row to the domain dataclass used by signal/assessor."""
    return FactbasePhraseData(
        display_phrase=row.display_phrase,
        api_query=row.api_query,
        speaker_slug=row.speaker_slug,
        in_market_count=row.in_market_count,
        count_7d=row.count_7d,
        count_30d=row.count_30d,
        count_365d=row.count_365d,
        top_quotes=list(row.top_quotes or []),
        fetched_at=row.last_fetched_at,
    )


async def _upsert_phrase_row(
    session: "AsyncSession",
    market_id: str,
    data: FactbasePhraseData,
) -> None:
    stmt = (
        pg_insert(FactbasePhraseRow)
        .values(
            market_id=market_id,
            display_phrase=data.display_phrase,
            api_query=data.api_query,
            speaker_slug=data.speaker_slug,
            in_market_count=data.in_market_count,
            count_7d=data.count_7d,
            count_30d=data.count_30d,
            count_365d=data.count_365d,
            top_quotes=data.top_quotes,
            last_fetched_at=data.fetched_at,
        )
        .on_conflict_do_update(
            index_elements=["market_id"],
            set_={
                "display_phrase": data.display_phrase,
                "api_query": data.api_query,
                "in_market_count": data.in_market_count,
                "count_7d": data.count_7d,
                "count_30d": data.count_30d,
                "count_365d": data.count_365d,
                "top_quotes": data.top_quotes,
                "last_fetched_at": data.fetched_at,
            },
        )
    )
    await session.execute(stmt)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


async def run_factbase_scheduler(
    session_factory: "async_sessionmaker[AsyncSession]",
    series_allowlist: frozenset[str],
    phrase_cache: FactbasePhraseCache,
    llm_client: "LLMClient",
    interval_seconds: int = 300,
    telemetry: "RuntimeTelemetry | None" = None,
) -> None:
    """Background scheduler: keep factbase_phrase_frequency rows fresh.

    On startup, immediately marks any existing rows as ready so markets are
    unblocked after a restart without waiting for the first full cycle.
    """
    from freqpred.markets.models import MarketRow
    from freqpred.runtime.telemetry import SERVICE_FACTBASE_SCHEDULER

    log.info("factbase.scheduler.starting", allowlist=list(series_allowlist), interval_seconds=interval_seconds)

    # Warm cache from existing DB rows on startup
    try:
        async with session_factory() as session:
            existing = await session.execute(select(FactbasePhraseRow))
            for row in existing.scalars().all():
                phrase_cache.mark_ready(row.market_id)
        log.info("factbase.scheduler.cache_warmed", count=len(phrase_cache._ready))
    except Exception:
        log.error("factbase.scheduler.cache_warm_error", exc_info=True)

    while True:
        try:
            await _run_cycle(
                session_factory=session_factory,
                series_allowlist=series_allowlist,
                phrase_cache=phrase_cache,
                llm_client=llm_client,
            )
            if telemetry:
                await telemetry.mark_success(SERVICE_FACTBASE_SCHEDULER)
        except Exception:
            log.error("factbase.scheduler.cycle_error", exc_info=True)
            if telemetry:
                await telemetry.mark_error(
                    SERVICE_FACTBASE_SCHEDULER,
                    "Unhandled error in factbase scheduler cycle",
                )
        await asyncio.sleep(interval_seconds)


async def _run_cycle(
    session_factory: "async_sessionmaker[AsyncSession]",
    series_allowlist: frozenset[str],
    phrase_cache: FactbasePhraseCache,
    llm_client: "LLMClient",
) -> None:
    """One scheduler cycle: process all active allowlist markets."""
    from freqpred.markets.models import MarketRow

    now = datetime.now(UTC)
    refresh_cutoff = now - timedelta(hours=_SCHEDULER_REFRESH_HOURS)

    async with session_factory() as session:
        # Load active markets in the allowlist with their existing phrase rows
        result = await session.execute(
            select(MarketRow, FactbasePhraseRow)
            .outerjoin(
                FactbasePhraseRow,
                MarketRow.id == FactbasePhraseRow.market_id,
            )
            .where(
                MarketRow.series_ticker.in_(list(series_allowlist)),
                MarketRow.status == "active",
            )
        )
        rows = result.all()

    log.info("factbase.scheduler.cycle_start", markets=len(rows), allowlist=list(series_allowlist))

    for market_row, phrase_row in rows:
        market_id: str = market_row.id

        # Case A: fresh row — just warm the cache and move on
        if phrase_row is not None and phrase_row.last_fetched_at > refresh_cutoff:
            phrase_cache.mark_ready(market_id)
            continue

        # Case B/C: stale or missing — need to refresh counts
        if phrase_row is not None:
            # Reuse stored api_query — skip Haiku
            search_terms = FactbaseSearchTerms(
                display_phrase=phrase_row.display_phrase,
                api_query=phrase_row.api_query,
                match_terms=[],  # not needed for API call
            )
        else:
            # Case C: no row — call Haiku once to extract terms
            search_terms = await extract_search_terms(
                market_row.question,
                llm_client,
                market_id=market_id,
            )
            if search_terms is None:
                log.warning(
                    "factbase.scheduler.no_terms",
                    market_id=market_id,
                    question=market_row.question[:120],
                )
                continue

        # Fetch fresh counts from FactBase API
        data = await fetch_phrase_frequency(
            search_terms,
            speaker_slug="trump",
            market_open_time=market_row.open_time,
        )

        # Persist
        async with session_factory() as session:
            await _upsert_phrase_row(session, market_id, data)
            await session.commit()

        phrase_cache.mark_ready(market_id)
        log.info(
            "factbase.scheduler.market_updated",
            market_id=market_id,
            display_phrase=data.display_phrase,
            in_market=data.in_market_count,
            count_7d=data.count_7d,
        )
