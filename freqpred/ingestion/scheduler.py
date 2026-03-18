"""Ingestion scheduler: runs fetchers per-market on catalyst queries.

Each cycle:
  1. Loads all non-closed markets from the DB.
  2. Filters them through the registered strategy (select_markets).
  3. Generates catalyst queries for markets that have none yet
     (or whose last run is stale) via generate_catalysts.
  4. Deactivates catalysts for markets no longer selected.
  5. Runs Tavily + NewsAPI + Reddit fetchers against every active
     CatalystQuery and upserts results into the document store.
  6. Updates ingestion:last_run:{market_id} in Redis.

Public API:
    run_cycle(...)   — one full pass (steps 1-6).
    run_scheduler(…) — async loop that calls run_cycle every N seconds.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.ingestion.fetchers import newsapi as newsapi_fetcher
from freqpred.ingestion.fetchers import reddit as reddit_fetcher
from freqpred.ingestion.fetchers import tavily as tavily_fetcher
from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow
from freqpred.ingestion.quota import get_daily_count, increment_daily_count
from freqpred.ingestion.store import DocumentSkipped, upsert_document
from freqpred.markets.models import Market, MarketRow
from freqpred.rag.embedder import LocalEmbedder

if TYPE_CHECKING:
    from freqpred.llm.client import LLMClient
    from freqpred.ingestion.selector import StrategyProtocol

log = structlog.get_logger(__name__)

_REDIS_KEY = "ingestion:last_run:{market_id}"
_NEWSAPI_LOOKBACK_DAYS = 7

# Category → subreddit mapping used for Reddit queries.
_SUBREDDIT_MAP: dict[str, list[str]] = {
    "politics":   ["politics", "PoliticalDiscussion", "neutralpolitics"],
    "technology": ["technology", "MachineLearning", "singularity"],
    "economics":  ["economics", "investing", "stocks"],
    "fintech":    ["investing", "wallstreetbets", "stocks", "fintech"],
    "sports":     ["sports"],
    "crypto":     ["CryptoCurrency", "Bitcoin"],
    "climate":    ["climate", "environment"],
}


class _RedisClient(Protocol):
    """Minimal async Redis interface required by the scheduler."""

    async def set(self, name: str, value: str) -> None: ...


def _subreddits_for_category(category: str) -> list[str]:
    return _SUBREDDIT_MAP.get(category.lower(), ["news"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_cycle(
    session: AsyncSession,
    embedder: LocalEmbedder,
    redis_client: _RedisClient,
    strategy: "StrategyProtocol | None" = None,
    llm_client: "LLMClient | None" = None,
    tavily_api_key: str = "",
    newsapi_api_key: str = "",
    newsapi_enabled: bool = True,
    newsapi_max_daily_requests: int = 90,
    reddit_user_agent: str = "freqpred/0.1",
    domain_blacklist: frozenset[str] = frozenset({"kalshi.com"}),
) -> dict[str, int]:
    """Run one full ingestion cycle.

    Steps:
      1. Load all non-closed markets from the DB.
      2. Filter through strategy.is_market_interesting() (if strategy provided).
      3. Generate catalyst queries for selected markets with no active run.
      4. Deactivate catalysts for markets no longer selected.
      5. Fetch documents for every market with active catalyst queries.
      6. Update Redis last-run timestamps.

    Fetcher errors are caught per-fetcher; one failure does not abort others.

    Args:
        session:                    Open async SQLAlchemy session (caller manages commit).
        embedder:                   Local embedder for document storage.
        redis_client:               Async Redis client.
        strategy:                   Strategy used to filter markets. If None, all markets
                                    with active catalyst runs are processed.
        llm_client:                 LLM client for catalyst generation. If None, catalyst
                                    generation is skipped.
        tavily_api_key:             Tavily API key. Tavily is skipped if empty.
        newsapi_api_key:            NewsAPI key. NewsAPI is skipped if empty.
        newsapi_enabled:            When False, the NewsAPI fetcher is skipped entirely
                                    even if newsapi_api_key is set (default: True).
        newsapi_max_daily_requests: Daily request cap tracked in Postgres
                                    ``api_daily_counters`` (default: 90).
        reddit_user_agent:          User-Agent for Reddit requests.
        domain_blacklist:           Domains to exclude from Tavily and NewsAPI results.
                                    Matched as a substring of each URL (default: kalshi.com).

    Returns:
        Stats dict with keys: markets_processed, catalysts_generated,
        docs_fetched, docs_stored, docs_error.
    """
    catalysts_generated = 0

    # --- Phase 1: catalyst generation for newly-selected markets ---------------
    if strategy is not None and llm_client is not None:
        catalysts_generated = await _ensure_catalysts(
            session, strategy, llm_client, embedder
        )

    # --- Phase 2: load markets that have active catalyst queries ---------------
    market_queries = await _load_active_market_queries(session)

    if not market_queries:
        log.debug("scheduler.run_cycle.no_active_markets")
        return {
            "markets_processed": 0,
            "catalysts_generated": catalysts_generated,
            "docs_fetched": 0,
            "docs_stored": 0,
            "docs_error": 0,
        }

    total_fetched = 0
    total_stored = 0
    total_error = 0
    now = datetime.now(UTC)
    newsapi_from = now - timedelta(days=_NEWSAPI_LOOKBACK_DAYS)

    for market_id, category, query_texts in market_queries:
        market_fetched = 0
        market_stored = 0
        market_error = 0

        for query_text in query_texts:
            raw_docs = []

            # Tavily
            if tavily_api_key:
                try:
                    docs = await tavily_fetcher.fetch(
                        api_key=tavily_api_key,
                        query=query_text,
                        excluded_domains=domain_blacklist,
                    )
                    raw_docs.extend(docs)
                except Exception:
                    log.warning(
                        "scheduler.fetcher_error",
                        market_id=market_id,
                        fetcher="tavily",
                        query=query_text,
                        exc_info=True,
                    )

            # NewsAPI — guarded by enabled flag and daily Postgres quota
            if newsapi_api_key and newsapi_enabled:
                daily_count = await get_daily_count(session, "newsapi", now.date())
                if daily_count >= newsapi_max_daily_requests:
                    log.warning(
                        "newsapi_daily_limit_reached",
                        date=now.date().isoformat(),
                        count=daily_count,
                        max_daily_requests=newsapi_max_daily_requests,
                    )
                else:
                    try:
                        docs = await newsapi_fetcher.fetch(
                            api_key=newsapi_api_key,
                            query=query_text,
                            from_date=newsapi_from,
                            excluded_domains=domain_blacklist,
                        )
                        raw_docs.extend(docs)
                        await increment_daily_count(session, "newsapi", now.date())
                    except Exception:
                        log.warning(
                            "scheduler.fetcher_error",
                            market_id=market_id,
                            fetcher="newsapi",
                            query=query_text,
                            exc_info=True,
                        )

            # Reddit
            try:
                subreddits = _subreddits_for_category(category)
                docs = await reddit_fetcher.fetch(
                    subreddits=subreddits,
                    query=query_text,
                    user_agent=reddit_user_agent,
                )
                raw_docs.extend(docs)
            except Exception:
                log.warning(
                    "scheduler.fetcher_error",
                    market_id=market_id,
                    fetcher="reddit",
                    query=query_text,
                    exc_info=True,
                )

            market_fetched += len(raw_docs)

            # Upsert each document.
            # Each upsert is wrapped in a savepoint so that a single document
            # failure (e.g. constraint violation) does not abort the entire
            # PostgreSQL transaction and roll back all other documents.
            for raw_doc in raw_docs:
                raw_doc.category = category
                try:
                    async with session.begin_nested():
                        await upsert_document(session, embedder, raw_doc)
                    market_stored += 1
                except DocumentSkipped:
                    pass  # empty body after cleaning — already logged at debug
                except Exception:
                    log.warning(
                        "scheduler.upsert_error",
                        market_id=market_id,
                        source_url=raw_doc.source_url,
                        exc_info=True,
                    )
                    market_error += 1

        # Update Redis last-run timestamp.
        try:
            redis_key = _REDIS_KEY.format(market_id=market_id)
            await redis_client.set(redis_key, now.isoformat())
        except Exception:
            log.warning(
                "scheduler.redis_error",
                market_id=market_id,
                exc_info=True,
            )

        log.info(
            "scheduler.market_cycle_complete",
            market_id=market_id,
            queries=len(query_texts),
            docs_fetched=market_fetched,
            docs_stored=market_stored,
            docs_error=market_error,
        )

        total_fetched += market_fetched
        total_stored += market_stored
        total_error += market_error

    stats = {
        "markets_processed": len(market_queries),
        "catalysts_generated": catalysts_generated,
        "docs_fetched": total_fetched,
        "docs_stored": total_stored,
        "docs_error": total_error,
    }

    log.info("scheduler.cycle_complete", **stats)
    return stats


async def run_scheduler(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: LocalEmbedder,
    redis_client: _RedisClient,
    interval_seconds: int = 1800,
    strategy: "StrategyProtocol | None" = None,
    llm_client: "LLMClient | None" = None,
    tavily_api_key: str = "",
    newsapi_api_key: str = "",
    newsapi_enabled: bool = True,
    newsapi_max_daily_requests: int = 90,
    reddit_user_agent: str = "freqpred/0.1",
    domain_blacklist: frozenset[str] = frozenset({"kalshi.com"}),
) -> None:
    """Async loop: runs run_cycle every *interval_seconds*.

    Designed to be launched as an asyncio background task alongside the
    market watcher. Logs and continues on cycle-level errors — never exits.

    Args:
        session_factory:            Async SQLAlchemy session factory.
        embedder:                   Voyage AI embedder.
        redis_client:               Async Redis client.
        interval_seconds:           Sleep duration between cycles (default 1800 = 30 min).
        strategy:                   Strategy used to filter markets for catalyst generation.
        llm_client:                 LLM client for catalyst generation.
        tavily_api_key:             Tavily API key.
        newsapi_api_key:            NewsAPI key.
        newsapi_enabled:            When False, NewsAPI fetcher is skipped entirely.
        newsapi_max_daily_requests: Daily request cap for the NewsAPI fetcher.
        reddit_user_agent:          User-Agent for Reddit requests.
        domain_blacklist:           Domains to exclude from Tavily and NewsAPI results.
    """
    log.info("scheduler.started", interval_seconds=interval_seconds)

    while True:
        try:
            async with session_factory() as session:
                await run_cycle(
                    session=session,
                    embedder=embedder,
                    redis_client=redis_client,
                    strategy=strategy,
                    llm_client=llm_client,
                    tavily_api_key=tavily_api_key,
                    newsapi_api_key=newsapi_api_key,
                    newsapi_enabled=newsapi_enabled,
                    newsapi_max_daily_requests=newsapi_max_daily_requests,
                    reddit_user_agent=reddit_user_agent,
                    domain_blacklist=domain_blacklist,
                )
                await session.commit()
        except Exception:
            log.error("scheduler.cycle_error", exc_info=True)

        await asyncio.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


# How old a catalyst run must be before it is regenerated with fresh RAG context.
_CATALYST_REFRESH_INTERVAL = timedelta(hours=24)


async def _ensure_catalysts(
    session: AsyncSession,
    strategy: "StrategyProtocol",
    llm_client: "LLMClient",
    embedder: LocalEmbedder,
) -> int:
    """Generate or refresh catalyst queries for selected markets.

    Two cases trigger generation for a market:
    - No active CatalystRun exists (generation 1 — first time seen).
    - The latest active run is older than _CATALYST_REFRESH_INTERVAL
      (generation N+1 — daily refresh with RAG-informed context).

    Returns the number of markets that had catalysts generated/refreshed.
    """
    from freqpred.ingestion.catalyst_generator import CatalystGenerationError, generate_catalysts
    from freqpred.ingestion.selector import deactivate_stale_catalysts, select_markets

    now = datetime.now(UTC)
    stale_cutoff = now - _CATALYST_REFRESH_INTERVAL

    # Load all non-closed markets.
    result = await session.execute(
        select(MarketRow).where(MarketRow.close_time > now)
    )
    all_rows = result.scalars().all()
    all_markets: list[Market] = [_market_row_to_domain(r) for r in all_rows]

    selected = select_markets(all_markets, [strategy])

    # Deactivate catalysts for markets that are no longer interesting.
    await deactivate_stale_catalysts(session, [strategy])

    if not selected:
        return 0

    # For each selected market, find its latest active CatalystRun (if any)
    # so we can decide whether it needs a fresh run.
    selected_ids = {m.id for m in selected}
    existing_result = await session.execute(
        select(CatalystRunRow.market_id, func.max(CatalystRunRow.created_at).label("latest"))
        .where(
            CatalystRunRow.market_id.in_(selected_ids),
            CatalystRunRow.is_active.is_(True),
        )
        .group_by(CatalystRunRow.market_id)
    )
    latest_run_by_market: dict[str, datetime] = {
        row.market_id: row.latest for row in existing_result.all()
    }

    # A market needs (re)generation if it has no active run OR its run is stale.
    needs_generation = [
        m for m in selected
        if m.id not in latest_run_by_market
        or latest_run_by_market[m.id] < stale_cutoff
    ]

    generated = 0
    for market in needs_generation:
        is_refresh = market.id in latest_run_by_market
        try:
            await generate_catalysts(market, session, llm_client, embedder)
            generated += 1
            log.info(
                "scheduler.catalysts_generated",
                market_id=market.id,
                refresh=is_refresh,
            )
        except CatalystGenerationError:
            log.warning(
                "scheduler.catalyst_generation_failed",
                market_id=market.id,
                exc_info=True,
            )

    return generated


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


async def _load_active_market_queries(
    session: AsyncSession,
) -> list[tuple[str, str, list[str]]]:
    """Return (market_id, category, [query_text, ...]) for all active catalyst runs.

    Only the latest active CatalystRun per market is considered.
    Markets whose latest run has is_active=False are excluded.
    """
    # Subquery: latest created_at per market where is_active=True.
    latest_subq = (
        select(
            CatalystRunRow.market_id,
            func.max(CatalystRunRow.created_at).label("max_created_at"),
        )
        .where(CatalystRunRow.is_active.is_(True))
        .group_by(CatalystRunRow.market_id)
        .subquery()
    )

    # Join runs → markets → queries.
    result = await session.execute(
        select(CatalystRunRow.id, MarketRow.id, MarketRow.category, CatalystQueryRow.query_text)
        .join(
            latest_subq,
            (CatalystRunRow.market_id == latest_subq.c.market_id)
            & (CatalystRunRow.created_at == latest_subq.c.max_created_at),
        )
        .join(MarketRow, MarketRow.id == CatalystRunRow.market_id)
        .join(CatalystQueryRow, CatalystQueryRow.run_id == CatalystRunRow.id)
        .where(CatalystRunRow.is_active.is_(True))
    )
    rows = result.all()

    # Group by market_id preserving insertion order.
    grouped: dict[str, tuple[str, list[str]]] = {}
    for _run_id, market_id, category, query_text in rows:
        if market_id not in grouped:
            grouped[market_id] = (category, [])
        grouped[market_id][1].append(query_text)

    return [(mid, cat, queries) for mid, (cat, queries) in grouped.items()]
