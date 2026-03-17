"""Ingestion scheduler: runs fetchers per-market on catalyst queries.

Reads the latest active CatalystRun per market, runs Tavily + NewsAPI +
Reddit fetchers against each CatalystQuery, and upserts results into the
document store. Tracks last-run per market in Redis.

Public API:
    run_cycle(...)   — one fetch pass over all active-catalyst markets.
    run_scheduler(…) — async loop that calls run_cycle every N seconds.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.ingestion.fetchers import newsapi as newsapi_fetcher
from freqpred.ingestion.fetchers import reddit as reddit_fetcher
from freqpred.ingestion.fetchers import tavily as tavily_fetcher
from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow
from freqpred.ingestion.store import upsert_document
from freqpred.markets.models import MarketRow
from freqpred.rag.embedder import VoyageEmbedder

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
    embedder: VoyageEmbedder,
    redis_client: _RedisClient,
    tavily_api_key: str = "",
    newsapi_api_key: str = "",
    reddit_user_agent: str = "freqpred/0.1",
) -> dict[str, int]:
    """Run one ingestion cycle for all markets with an active CatalystRun.

    For each market:
      - Reads CatalystQuery rows from its latest active CatalystRun.
      - Runs Tavily, NewsAPI, and Reddit fetchers per query.
      - Upserts results via the document store.
      - Updates ``ingestion:last_run:{market_id}`` in Redis.

    Fetcher errors are caught per-fetcher; one failure does not abort others.

    Args:
        session:          Open async SQLAlchemy session (caller manages commit).
        embedder:         Voyage AI embedder for document storage.
        redis_client:     Async Redis client.
        tavily_api_key:   Tavily API key. Tavily is skipped if empty.
        newsapi_api_key:  NewsAPI key. NewsAPI is skipped if empty.
        reddit_user_agent: User-Agent for Reddit requests.

    Returns:
        Stats dict with keys: markets_processed, docs_fetched,
        docs_stored, docs_error.
    """
    market_queries = await _load_active_market_queries(session)

    if not market_queries:
        log.debug("scheduler.run_cycle.no_active_markets")
        return {"markets_processed": 0, "docs_fetched": 0, "docs_stored": 0, "docs_error": 0}

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

            # NewsAPI
            if newsapi_api_key:
                try:
                    docs = await newsapi_fetcher.fetch(
                        api_key=newsapi_api_key,
                        query=query_text,
                        from_date=newsapi_from,
                    )
                    raw_docs.extend(docs)
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
            for raw_doc in raw_docs:
                raw_doc.category = category
                try:
                    await upsert_document(session, embedder, raw_doc)
                    market_stored += 1
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
        "docs_fetched": total_fetched,
        "docs_stored": total_stored,
        "docs_error": total_error,
    }

    log.info("scheduler.cycle_complete", **stats)
    return stats


async def run_scheduler(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: VoyageEmbedder,
    redis_client: _RedisClient,
    interval_seconds: int = 1800,
    tavily_api_key: str = "",
    newsapi_api_key: str = "",
    reddit_user_agent: str = "freqpred/0.1",
) -> None:
    """Async loop: runs run_cycle every *interval_seconds*.

    Designed to be launched as an asyncio background task alongside the
    market watcher. Logs and continues on cycle-level errors — never exits.

    Args:
        session_factory:  Async SQLAlchemy session factory.
        embedder:         Voyage AI embedder.
        redis_client:     Async Redis client.
        interval_seconds: Sleep duration between cycles (default 1800 = 30 min).
        tavily_api_key:   Tavily API key.
        newsapi_api_key:  NewsAPI key.
        reddit_user_agent: User-Agent for Reddit requests.
    """
    log.info("scheduler.started", interval_seconds=interval_seconds)

    while True:
        try:
            async with session_factory() as session:
                await run_cycle(
                    session=session,
                    embedder=embedder,
                    redis_client=redis_client,
                    tavily_api_key=tavily_api_key,
                    newsapi_api_key=newsapi_api_key,
                    reddit_user_agent=reddit_user_agent,
                )
                await session.commit()
        except Exception:
            log.error("scheduler.cycle_error", exc_info=True)

        await asyncio.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


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
