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
import time
import uuid
from datetime import UTC, date, datetime
from typing import Protocol

import anthropic
import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.models import CatalystQuery, CatalystQueryRow, CatalystRun, CatalystRunRow
from freqpred.llm.audit import calculate_cost, log_llm_query
from freqpred.markets.models import Market, MarketRow
from freqpred.rag.models import Document

log = structlog.get_logger(__name__)

# Model used for catalyst generation — cheap reasoning task, not primary signal.
_CATALYST_MODEL = "claude-haiku-4-5-20251001"
_PROMPT_VERSION = "catalyst-v1"

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
    anthropic_client: anthropic.AsyncAnthropic,
    embedder: _Embedder | None = None,
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
    queries, raw_response, tokens_in, tokens_out, latency_ms, success, error = (
        await _call_llm(anthropic_client, prompt)
    )

    cost = calculate_cost(_CATALYST_MODEL, tokens_in, tokens_out)

    llm_query_id = await log_llm_query(
        session,
        strategy="system",
        query_type="catalyst_generation",
        model_used=_CATALYST_MODEL,
        prompt_version=_PROMPT_VERSION,
        prompt=prompt,
        response=raw_response,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=cost,
        latency_ms=latency_ms,
        success=success,
        market_id=market.id,
        error_message=error,
    )

    if not success or not queries:
        raise CatalystGenerationError(
            f"Catalyst generation failed for market {market.id}: {error}"
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

    for query_text in queries:
        session.add(
            CatalystQueryRow(
                id=uuid.uuid4(),
                run_id=run_row.id,
                query_text=query_text,
            )
        )
    await session.flush()

    log.info(
        "catalyst_generator.generated",
        market_id=market.id,
        generation=generation,
        query_count=len(queries),
        rag_docs_used=len(rag_docs),
        cost_usd=round(cost, 6),
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
    anthropic_client: anthropic.AsyncAnthropic,
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
            await generate_catalysts(market, session, anthropic_client, embedder)
            generated += 1
        except CatalystGenerationError:
            log.warning(
                "catalyst_generator.run_error",
                market_id=market.id,
                exc_info=True,
            )

    deactivated = await deactivate_stale_catalysts(session, strategies)

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
        "Identify 3 to 5 specific events, data releases, or developments that would",
        "most significantly shift the probability of this market resolving YES or NO.",
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
        "Return a JSON array of search query strings only. Each query should be",
        "specific enough to find targeted news (not broad category keywords).",
        'Example: ["February CPI release 2026", "Fed Chair Powell Senate testimony"]',
        "",
        "Return the JSON array and nothing else.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


async def _call_llm(
    client: anthropic.AsyncAnthropic,
    prompt: str,
) -> tuple[list[str], str, int, int, int, bool, str | None]:
    """Call Claude Haiku and parse the JSON array response.

    Returns:
        (queries, raw_response, tokens_in, tokens_out, latency_ms, success, error)
    """
    t0 = time.monotonic()
    raw_response = ""
    try:
        message = await client.messages.create(
            model=_CATALYST_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        raw_response = message.content[0].text if message.content else ""
        tokens_in = message.usage.input_tokens
        tokens_out = message.usage.output_tokens
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return [], str(exc), 0, 0, latency_ms, False, str(exc)

    # Parse JSON array from response.
    queries, error = _parse_queries(raw_response)
    success = bool(queries) and error is None
    return queries, raw_response, tokens_in, tokens_out, latency_ms, success, error


def _parse_queries(text: str) -> tuple[list[str], str | None]:
    """Extract a JSON array of strings from the LLM response text.

    Tries the full text first, then looks for the first '[' to handle any
    leading explanation the model may have added.
    """
    text = text.strip()
    for candidate in (text, text[text.find("["):] if "[" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                queries = [q.strip() for q in parsed if q.strip()]
                if queries:
                    return queries, None
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
        return await retrieve(
            session=session,
            embedder=embedder,
            question=market.question,
            category=market.category,
            top_k=top_k,
            max_age_days=7,
        )
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
        close_time=row.close_time,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
        mid_price=row.mid_price,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
        metadata=dict(row.metadata_),
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CatalystGenerationError(Exception):
    """Raised when catalyst generation fails and cannot produce query strings."""
