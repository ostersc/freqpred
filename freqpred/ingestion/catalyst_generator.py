"""Catalyst Generator: derives targeted search queries per market via LLM.

For each selected market the generator produces 3–5 search query strings
(catalysts) that represent specific events or developments likely to shift
the market's probability. These are stored as CatalystRun + CatalystQuery
rows in the DB and used by the ingestion scheduler to drive fetchers.

Generation schedule:
  - First run: when a market has no prior CatalystRun today.
  - Daily re-run: uses RAG-retrieved recent documents as additional context
    so the LLM can refine or add catalysts based on what's appeared in the news.

Every LLM call is logged to llm_queries (constraint: audit log is non-negotiable).
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime
from typing import Protocol

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.models import CatalystQuery, CatalystQueryRow, CatalystRun, CatalystRunRow
from freqpred.llm.client import LLMClient, LLMError
from freqpred.markets.models import Market, MarketRow, PositionRow
from freqpred.rag.models import Document

log = structlog.get_logger(__name__)

# Model used for catalyst generation — cheap reasoning task, not primary signal.
_CATALYST_MODEL = "claude-haiku-4-5-20251001"
_PROMPT_VERSION = "catalyst-v2"

# Number of RAG documents to include as context on re-runs.
_RAG_CONTEXT_DOCS = 5
_RAG_MAX_BODY_CHARS = 500  # truncate each doc body to keep prompts short


class _Embedder(Protocol):
    async def embed_text(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_catalysts(
    market: Market,
    session: AsyncSession,
    llm_client: LLMClient,
    embedder: _Embedder | None = None,
    model: str = _CATALYST_MODEL,
) -> CatalystRun:
    """Generate (or re-generate) catalyst queries for *market*.

    On first run: prompt uses only the market question + metadata.
    On re-runs:   prompt also includes recently retrieved documents from the
                  RAG store to let the LLM refine/add catalysts.

    Writes one ``CatalystRun`` and 3–5 ``CatalystQuery`` rows. The LLM call
    is logged to ``llm_queries``.  Caller is responsible for committing.

    Args:
        market:           The market to generate catalysts for.
        session:          Open async session (caller manages commit).
        anthropic_client: Authenticated Anthropic async client.
        embedder:         Voyage AI embedder for RAG context on re-runs.
                          If None, re-runs skip the RAG context step.

    Returns:
        The new CatalystRun domain object.
    """
    prior_run = await _get_latest_run(session, market.id)
    generation = (prior_run.generation + 1) if prior_run else 1

    # Build RAG context for re-runs (skip if first run or no embedder).
    rag_docs: list[Document] = []
    if prior_run and embedder is not None:
        rag_docs = await _retrieve_recent_docs(session, embedder, market)

    prompt = _build_prompt(market, rag_docs)

    try:
        llm_resp = await llm_client.complete(
            prompt,
            model,
            "catalyst_generation",
            market_id=market.id,
            strategy="system",
            prompt_version=_PROMPT_VERSION,
        )
    except LLMError as exc:
        raise CatalystGenerationError(
            f"Catalyst generation failed for market {market.id}: {exc}"
        ) from exc

    llm_query_id = llm_resp.llm_query_id
    queries, parse_error = _parse_queries(llm_resp.content)

    if not queries:
        raise CatalystGenerationError(
            f"Catalyst generation failed for market {market.id}: {parse_error}"
        )

    # Persist the run.
    run_row = CatalystRunRow(
        id=uuid.uuid4(),
        market_id=market.id,
        generation=generation,
        llm_query_id=llm_query_id,
        is_active=True,
    )
    session.add(run_row)
    await session.flush()  # get run_row.id for FK

    for query_text, tv_query in queries:
        session.add(
            CatalystQueryRow(
                id=uuid.uuid4(),
                run_id=run_row.id,
                query_text=query_text,
                tv_query=tv_query,
            )
        )
    await session.flush()

    log.info(
        "catalyst_generator.generated",
        market_id=market.id,
        generation=generation,
        query_count=len(queries),
        rag_docs_used=len(rag_docs),
        cost_usd=round(llm_resp.cost_usd, 6),
    )

    return CatalystRun(
        id=str(run_row.id),
        market_id=market.id,
        generation=generation,
        is_active=True,
        created_at=run_row.created_at or datetime.now(UTC),
        llm_query_id=llm_query_id,
    )


async def run_catalyst_refresh(
    session: AsyncSession,
    strategies: list,
    llm_client: LLMClient,
    embedder: _Embedder | None = None,
) -> dict[str, int]:
    """Refresh catalysts for all strategy-selected active markets.

    Fetches all non-closed markets from DB, filters via strategy selection,
    generates catalysts for any market that hasn't been processed today,
    then deactivates runs for markets that are closed or no longer selected.

    Args:
        session:          Open async session (caller manages commit).
        strategies:       Registered strategy instances.
        anthropic_client: Authenticated Anthropic async client.
        embedder:         Voyage AI embedder for RAG context on re-runs.

    Returns:
        Stats dict: {"generated": int, "skipped": int, "deactivated": int}.
    """
    from freqpred.ingestion.selector import deactivate_stale_catalysts, select_markets

    now = datetime.now(UTC)

    # Fetch all markets that haven't closed yet.
    result = await session.execute(
        select(MarketRow).where(MarketRow.close_time > now)
    )
    market_rows = result.scalars().all()
    markets = [_market_row_to_domain(r) for r in market_rows]

    selected = select_markets(markets, strategies)

    # Protect markets with open positions from catalyst deactivation even if
    # they no longer pass is_market_interesting (e.g. price drifted to extreme).
    open_pos_result = await session.execute(
        select(PositionRow.market_id).where(PositionRow.status == "open").distinct()
    )
    open_market_ids: set[str] = {r.market_id for r in open_pos_result.all()}

    generated = 0
    skipped = 0

    for market in selected:
        already_run_today = await _has_run_today(session, market.id)
        if already_run_today:
            skipped += 1
            log.debug(
                "catalyst_generator.skip",
                market_id=market.id,
                reason="already_run_today",
            )
            continue

        try:
            await generate_catalysts(market, session, llm_client, embedder)
            generated += 1
        except CatalystGenerationError:
            log.warning(
                "catalyst_generator.run_error",
                market_id=market.id,
                exc_info=True,
            )

    deactivated = await deactivate_stale_catalysts(session, strategies, open_market_ids)

    log.info(
        "catalyst_generator.refresh_complete",
        total_markets=len(markets),
        selected=len(selected),
        generated=generated,
        skipped=skipped,
        deactivated=deactivated,
    )
    return {"generated": generated, "skipped": skipped, "deactivated": deactivated}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(market: Market, rag_docs: list[Document]) -> str:
    close_date = market.close_time.strftime("%Y-%m-%d")

    lines = [
        "You are a prediction market research assistant.",
        "",
        "This market has NOT resolved yet. Your job is to generate 3 to 5 web search",
        "queries that find PREDICTIVE evidence — information that helps estimate the",
        "probability of the market resolving YES before it closes.",
        "",
        "Read the resolution criteria carefully. Think about what background signals,",
        "behavioral patterns, contextual factors, or recent developments would make the",
        "YES outcome more or less likely. Do NOT search for transcripts, results, or",
        "outcomes of the event itself — the event hasn't happened yet.",
        "",
        f"Market: {market.question}",
        f"Category: {market.category}",
        f"Closes: {close_date}",
    ]

    if rag_docs:
        lines += [
            "",
            "Recent relevant news found so far (use this to refine your queries):",
        ]
        for doc in rag_docs:
            body_preview = doc.body[:_RAG_MAX_BODY_CHARS].replace("\n", " ")
            lines.append(f"- [{doc.source_name}] {doc.title}: {body_preview}...")

    lines += [
        "",
        "Return a JSON array of objects, each with two keys:",
        '  "query_text": a natural-language web search string (used by Tavily, NewsAPI, Reddit, GDELT)',
        '  "tv_query":   a Solr/Lucene boolean query for the Internet Archive TV News Archive',
        "               (finds spoken-word evidence in TV transcripts), or null if TV is not relevant.",
        "",
        "TV query rules: use AND/OR/AND NOT/( )/\" \" boolean syntax. Quote multi-word phrases.",
        "Good TV queries find people *saying* things on air — keep them tight.",
        'TV null is fine for data/price markets where TV transcripts add little signal.',
        "",
        'Example for a word-mention market (Trump says "communist"):',
        '[{"query_text": "how often does Trump use word communist in speeches", "tv_query": "trump AND (\\"communist\\" OR \\"communism\\")"}]',
        'Example for a price market:',
        '[{"query_text": "Bitcoin price forecast March 2026", "tv_query": null}, {"query_text": "BTC ETF inflows trend", "tv_query": null}]',
        'Example for a Fed policy market:',
        '[{"query_text": "Fed interest rate cut probability 2026", "tv_query": "\\"federal reserve\\" AND (\\"rate cut\\" OR \\"interest rates\\")"}]',
        "",
        "Return the JSON array and nothing else.",
    ]

    return "\n".join(lines)



def _parse_queries(text: str) -> tuple[list[tuple[str, str | None]], str | None]:
    """Extract a JSON array of {query_text, tv_query} objects from the LLM response.

    Returns a list of (query_text, tv_query) pairs and an optional error string.

    Handles:
    - Clean JSON array of objects (preferred)
    - Plain JSON array of strings (legacy — tv_query defaults to None)
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Leading explanation text before the array
    """
    import re as _re

    text = text.strip()

    # Strip markdown code fences if present.
    if "```" in text:
        text = _re.sub(r"```(?:json)?\s*", "", text).strip()

    for candidate in (text, text[text.find("["):] if "[" in text else ""):
        if not candidate:
            continue
        # Trim anything after the closing ] to handle trailing text.
        if "]" in candidate:
            candidate = candidate[: candidate.rfind("]") + 1]
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, list) or not parsed:
                continue

            # New format: array of {query_text, tv_query} objects.
            if all(isinstance(q, dict) for q in parsed):
                pairs: list[tuple[str, str | None]] = []
                for item in parsed:
                    qt = (item.get("query_text") or "").strip()
                    tv = item.get("tv_query")
                    if isinstance(tv, str):
                        tv = tv.strip() or None
                    else:
                        tv = None
                    if qt:
                        pairs.append((qt, tv))
                if pairs:
                    return pairs, None

            # Legacy format: plain array of strings (tv_query absent).
            if all(isinstance(q, str) for q in parsed):
                pairs = [(q.strip(), None) for q in parsed if q.strip()]
                if pairs:
                    return pairs, None

        except json.JSONDecodeError:
            continue
    return [], f"Could not parse JSON array from response: {text[:200]!r}"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _get_latest_run(
    session: AsyncSession, market_id: str
) -> CatalystRunRow | None:
    """Return the most recent CatalystRunRow for this market, or None."""
    result = await session.execute(
        select(CatalystRunRow)
        .where(CatalystRunRow.market_id == market_id)
        .order_by(CatalystRunRow.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _has_run_today(session: AsyncSession, market_id: str) -> bool:
    """Return True if a CatalystRun was already created today (UTC) for this market."""
    today = date.today()
    result = await session.execute(
        select(CatalystRunRow.id)
        .where(
            CatalystRunRow.market_id == market_id,
            func.date(CatalystRunRow.created_at) == today,
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def _retrieve_recent_docs(
    session: AsyncSession,
    embedder: _Embedder,
    market: Market,
    top_k: int = _RAG_CONTEXT_DOCS,
) -> list[Document]:
    """Retrieve the most relevant recent documents for this market via RAG."""
    from freqpred.rag.retriever import retrieve

    try:
        pairs = await retrieve(
            session=session,
            embedder=embedder,
            question=market.question,
            market_id=market.id,
            top_k=top_k,
            max_age_days=7,
        )
        return [doc for doc, _ in pairs]
    except Exception:
        log.warning(
            "catalyst_generator.rag_retrieve_error",
            market_id=market.id,
            exc_info=True,
        )
        return []


def _market_row_to_domain(row: MarketRow) -> Market:
    return Market(
        id=row.id,
        platform=row.platform,
        question=row.question,
        category=row.category,
        status=row.status,
        result=row.result,
        close_time=row.close_time,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
        mid_price=row.mid_price,
        last_price=row.last_price,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        liquidity=row.liquidity,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
        metadata=dict(row.metadata_),
        open_time=row.open_time,
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CatalystGenerationError(Exception):
    """Raised when catalyst generation fails and cannot produce query strings."""
