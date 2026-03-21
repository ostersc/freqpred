"""Ingestion scheduler: runs fetchers per-market on catalyst queries.

Each cycle:
  1. Loads all non-closed markets from the DB.
  2. Filters them through the registered strategy (select_markets).
  3. Generates catalyst queries for markets that have none yet
     (or whose last run is stale) via generate_catalysts.
  4. Deactivates catalysts for markets no longer selected.
  5. Runs Tavily + NewsAPI + Reddit fetchers against every active
     CatalystQuery and upserts results into the document store.
  6. Updates fetcher_cursors in Postgres for Truth Social account feeds.

Public API:
    run_cycle(...)   — one full pass (steps 1-6).
    run_scheduler(…) — async loop that calls run_cycle every N seconds.
"""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.ingestion.backoff import record_rate_limit, record_success, tick_and_load
from freqpred.ingestion.cursors import get_cursor, set_cursor
from freqpred.ingestion.fetchers import gdelt as gdelt_fetcher
from freqpred.ingestion.fetchers import newsapi as newsapi_fetcher
from freqpred.ingestion.fetchers import reddit as reddit_fetcher
from freqpred.ingestion.fetchers import tavily as tavily_fetcher
from freqpred.ingestion.fetchers import truthsocial as truthsocial_fetcher
from freqpred.ingestion.fetchers import tv_archive as tv_archive_fetcher
from freqpred.ingestion.fetchers.gdelt import GDELTRateLimitError
from freqpred.ingestion.fetchers.newsapi import NewsAPIRateLimitError
from freqpred.ingestion.fetchers.truthsocial import (
    LoginErrorException as TruthSocialLoginError,
    patch_api_for_block_detection,
)
from tavily.errors import ForbiddenError, UsageLimitExceededError
from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow
from freqpred.ingestion.quota import get_daily_count, increment_daily_count
from freqpred.ingestion.store import DocumentSkipped, link_document_to_market, upsert_document
from freqpred.markets.models import Market, MarketRow
from freqpred.rag.embedder import LocalEmbedder

if TYPE_CHECKING:
    from freqpred.llm.client import LLMClient
    from freqpred.ingestion.selector import StrategyProtocol
    from freqpred.config import TruthSocialAccountConfig

log = structlog.get_logger(__name__)

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


_TS_ACCOUNT_FETCHER = "truthsocial_account"


def _subreddits_for_category(category: str) -> list[str]:
    return _SUBREDDIT_MAP.get(category.lower(), ["news"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_cycle(
    session: AsyncSession,
    embedder: LocalEmbedder,
    strategy: "StrategyProtocol | None" = None,
    llm_client: "LLMClient | None" = None,
    tavily_api_key: str = "",
    newsapi_api_key: str = "",
    newsapi_enabled: bool = True,
    newsapi_max_daily_requests: int = 90,
    reddit_user_agent: str = "freqpred/0.1",
    domain_blacklist: frozenset[str] = frozenset({"kalshi.com"}),
    truthsocial_enabled: bool = False,
    truthsocial_username: str = "",
    truthsocial_password: str = "",
    truthsocial_accounts: "list[TruthSocialAccountConfig] | None" = None,
) -> dict[str, int]:
    """Run one full ingestion cycle.

    Steps:
      1. Load all non-closed markets from the DB.
      2. Filter through strategy.is_market_interesting() (if strategy provided).
      3. Generate catalyst queries for selected markets with no active run.
      4. Deactivate catalysts for markets no longer selected.
      5. Fetch documents for every market with active catalyst queries.
      6. Update postgres last-run timestamps.

    Fetcher errors are caught per-fetcher; one failure does not abort others.

    Args:
        session:                    Open async SQLAlchemy session (caller manages commit).
        embedder:                   Local embedder for document storage.
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
        truthsocial_enabled:        When True, Truth Social fetchers are active.
        truthsocial_username:       Truth Social account username (env: TRUTHSOCIAL_USERNAME).
        truthsocial_password:       Truth Social account password (env: TRUTHSOCIAL_PASSWORD).
        truthsocial_accounts:       Account feeds to poll each cycle, with category mappings
                                    used to link fetched docs to active markets.
    Returns:
        Stats dict with keys: markets_processed, catalysts_generated,
        docs_fetched, docs_stored, docs_error.
    """
    catalysts_generated = 0

    # --- Phase 1: catalyst generation for newly-selected markets ---------------
    # Each market's catalysts are committed inside _ensure_catalysts so rows
    # are durable even if the process dies during the generation loop.
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

    # Load persistent backoff state from DB (ticks down all counters by 1).
    # Returns {service: is_backed_off} for services with existing rows.
    backoff_state = await tick_and_load(session)

    # In-memory flags for within-cycle short-circuit. Initialized from DB
    # state so a restart respects backoff that was set in a previous cycle.
    tavily_limit_hit: bool = backoff_state.get("tavily", False)
    newsapi_limit_hit: bool = backoff_state.get("newsapi", False)
    newsapi_limit_logged: bool = newsapi_limit_hit
    gdelt_limit_hit: bool = backoff_state.get("gdelt", False)
    truthsocial_login_failed: bool = backoff_state.get("truthsocial", False)
    tv_archive_limit_hit: bool = backoff_state.get("tv_archive", False)

    # Track which services had a successful call this cycle so we only write
    # record_success once per service (it's idempotent but avoids extra DB hits).
    success_recorded: set[str] = set()

    # Build Truth Social Api object once for the cycle (avoids re-auth per call).
    ts_api: object | None = None
    if truthsocial_enabled and truthsocial_username:
        from truthbrush.api import Api as TruthSocialApi  # noqa: PLC0415

        ts_api = TruthSocialApi(
            username=truthsocial_username, password=truthsocial_password
        )
        patch_api_for_block_detection(ts_api)

    # --- Phase 2a: Truth Social account feeds (once per cycle, per account) ----
    # Build a category → [market_id] index from the already-loaded active markets.
    _category_markets: dict[str, list[str]] = {}
    for _mid, _cat, _ct, _qpairs in market_queries:
        _category_markets.setdefault(_cat.lower(), []).append(_mid)

    if ts_api is not None and truthsocial_accounts:
        for ts_account in truthsocial_accounts:
            ts_username = ts_account.username
            if truthsocial_login_failed:
                break
            try:
                last_run = await get_cursor(session, _TS_ACCOUNT_FETCHER, ts_username)
                ts_created_after = last_run if last_run is not None else now - timedelta(hours=48)

                docs = await truthsocial_fetcher.fetch_account(
                    api=ts_api,
                    username=ts_username,
                    created_after=ts_created_after,
                    excluded_domains=domain_blacklist,
                )

                # Collect the market IDs this account's categories map to.
                account_market_ids: list[str] = []
                for cat in ts_account.categories:
                    account_market_ids.extend(_category_markets.get(cat.lower(), []))

                for raw_doc in docs:
                    try:
                        async with session.begin_nested():
                            doc = await upsert_document(session, embedder, raw_doc)
                            for mid in account_market_ids:
                                await link_document_to_market(session, doc.id, mid)
                        total_stored += 1
                    except DocumentSkipped:
                        pass
                    except Exception:
                        log.warning(
                            "scheduler.upsert_error",
                            source_url=raw_doc.source_url,
                            exc_info=True,
                        )
                        total_error += 1
                total_fetched += len(docs)
                await set_cursor(session, _TS_ACCOUNT_FETCHER, ts_username, now)
                if "truthsocial" not in success_recorded:
                    await record_success(session, "truthsocial")
                    success_recorded.add("truthsocial")
                log.info(
                    "scheduler.truthsocial_account_fetched",
                    username=ts_username,
                    docs_fetched=len(docs),
                    markets_linked=len(account_market_ids),
                )
            except TruthSocialLoginError as exc:
                truthsocial_login_failed = True
                skip_cycles = await record_rate_limit(session, "truthsocial")
                from freqpred.ingestion.fetchers.truthsocial import TruthSocialBlockedError  # noqa: PLC0415
                if isinstance(exc, TruthSocialBlockedError):
                    log.warning(
                        "scheduler.truthsocial_cloudflare_blocked",
                        username=ts_username,
                        skip_cycles=skip_cycles,
                        hint="IP temporarily banned by Cloudflare (error 1015); backing off",
                    )
                else:
                    log.error(
                        "scheduler.truthsocial_login_failed",
                        username=ts_username,
                        skip_cycles=skip_cycles,
                        exc_info=True,
                    )
            except Exception:
                log.warning(
                    "scheduler.fetcher_error",
                    fetcher="truthsocial_account",
                    username=ts_username,
                    exc_info=True,
                )

    for market_id, category, close_time, query_pairs in market_queries:
        market_start = time.monotonic()
        market_fetched = 0
        market_stored = 0
        market_error = 0

        for query_text, tv_query in query_pairs:
            # --- Build non-GDELT fetch coroutines to run in parallel ---
            # GDELT (doc + TV) are run sequentially afterwards because they share
            # a 1 req/5 s rate limit across all their API endpoints. Running them
            # concurrently in the same gather would fire simultaneous requests and
            # trigger rate limiting despite each function's own sleep guard.
            fetch_names: list[str] = []
            fetch_coros = []

            if tavily_api_key and not tavily_limit_hit:
                fetch_names.append("tavily")
                fetch_coros.append(tavily_fetcher.fetch(
                    api_key=tavily_api_key,
                    query=query_text,
                    excluded_domains=domain_blacklist,
                ))

            # NewsAPI quota check requires a DB round-trip; do it before the gather.
            newsapi_fetched = False
            if newsapi_api_key and newsapi_enabled and not newsapi_limit_hit:
                daily_count = await get_daily_count(session, "newsapi", now.date())
                if daily_count >= newsapi_max_daily_requests:
                    if not newsapi_limit_logged:
                        log.warning(
                            "newsapi_daily_limit_reached",
                            date=now.date().isoformat(),
                            count=daily_count,
                            max_daily_requests=newsapi_max_daily_requests,
                        )
                        newsapi_limit_logged = True
                else:
                    newsapi_fetched = True
                    fetch_names.append("newsapi")
                    fetch_coros.append(newsapi_fetcher.fetch(
                        api_key=newsapi_api_key,
                        query=query_text,
                        from_date=newsapi_from,
                        excluded_domains=domain_blacklist,
                    ))

            fetch_names.append("reddit")
            fetch_coros.append(reddit_fetcher.fetch(
                subreddits=_subreddits_for_category(category),
                query=query_text,
                user_agent=reddit_user_agent,
            ))

            if tv_query and not tv_archive_limit_hit:
                fetch_names.append("tv_archive")
                fetch_coros.append(tv_archive_fetcher.fetch(
                    query=tv_query,
                    close_time=close_time,
                ))

            # --- Run non-GDELT fetchers in parallel ---
            results = await asyncio.gather(*fetch_coros, return_exceptions=True)

            raw_docs = []
            for name, result in zip(fetch_names, results):
                if isinstance(result, BaseException):
                    if name == "tavily" and isinstance(result, (ForbiddenError, UsageLimitExceededError)):
                        tavily_limit_hit = True
                        skip_cycles = await record_rate_limit(session, "tavily")
                        log.warning("scheduler.tavily_limit_reached", reason=str(result), skip_cycles=skip_cycles)
                    elif name == "newsapi" and isinstance(result, NewsAPIRateLimitError):
                        newsapi_limit_hit = True
                        newsapi_limit_logged = True
                        skip_cycles = await record_rate_limit(session, "newsapi")
                        log.warning("scheduler.newsapi_rate_limited", reason=str(result), skip_cycles=skip_cycles)
                    else:
                        log.warning(
                            "scheduler.fetcher_error",
                            market_id=market_id,
                            fetcher=name,
                            query=query_text,
                            error=str(result),
                        )
                else:
                    raw_docs.extend(result)
                    if name == "newsapi" and newsapi_fetched:
                        await increment_daily_count(session, "newsapi", now.date())
                    # Clear backoff on first success this cycle for this service.
                    if name not in success_recorded:
                        await record_success(session, name)
                        success_recorded.add(name)

            # --- Run GDELT fetcher ---
            # Runs sequentially after the parallel gather (has its own rate-limit sleep).
            if not gdelt_limit_hit:
                try:
                    gdelt_docs = await gdelt_fetcher.fetch(
                        query=query_text,
                        excluded_domains=domain_blacklist,
                    )
                    raw_docs.extend(gdelt_docs)
                    if "gdelt" not in success_recorded:
                        await record_success(session, "gdelt")
                        success_recorded.add("gdelt")
                except GDELTRateLimitError as exc:
                    gdelt_limit_hit = True
                    skip_cycles = await record_rate_limit(session, "gdelt")
                    log.warning("scheduler.gdelt_rate_limited", reason=str(exc), skip_cycles=skip_cycles)
                except Exception:
                    log.warning(
                        "scheduler.fetcher_error",
                        market_id=market_id,
                        fetcher="gdelt",
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
                        doc = await upsert_document(session, embedder, raw_doc)
                        await link_document_to_market(session, doc.id, market_id)
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

        log.info(
            "scheduler.market_cycle_complete",
            market_id=market_id,
            elapsed_s=round(time.monotonic() - market_start, 2),
            queries=len(query_pairs),
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
    interval_seconds: int = 1800,
    strategy: "StrategyProtocol | None" = None,
    llm_client: "LLMClient | None" = None,
    tavily_api_key: str = "",
    newsapi_api_key: str = "",
    newsapi_enabled: bool = True,
    newsapi_max_daily_requests: int = 90,
    reddit_user_agent: str = "freqpred/0.1",
    domain_blacklist: frozenset[str] = frozenset({"kalshi.com"}),
    truthsocial_enabled: bool = False,
    truthsocial_username: str = "",
    truthsocial_password: str = "",
    truthsocial_accounts: "list[TruthSocialAccountConfig] | None" = None,
) -> None:
    """Async loop: runs run_cycle every *interval_seconds*.

    Designed to be launched as an asyncio background task alongside the
    market watcher. Logs and continues on cycle-level errors — never exits.

    Args:
        session_factory:            Async SQLAlchemy session factory.
        embedder:                   Local embedder for document storage.
        interval_seconds:           Sleep duration between cycles (default 1800 = 30 min).
        strategy:                   Strategy used to filter markets for catalyst generation.
        llm_client:                 LLM client for catalyst generation.
        tavily_api_key:             Tavily API key.
        newsapi_api_key:            NewsAPI key.
        newsapi_enabled:            When False, NewsAPI fetcher is skipped entirely.
        newsapi_max_daily_requests: Daily request cap for the NewsAPI fetcher.
        reddit_user_agent:          User-Agent for Reddit requests.
        domain_blacklist:           Domains to exclude from Tavily and NewsAPI results.
        truthsocial_enabled:        When True, Truth Social fetchers are active.
        truthsocial_username:       Truth Social account username.
        truthsocial_password:       Truth Social account password.
        truthsocial_accounts:       Standing account feeds to poll each cycle.
    """
    active_fetchers = ["reddit", "gdelt"]
    if tavily_api_key:
        active_fetchers.insert(0, "tavily")
    if newsapi_api_key and newsapi_enabled:
        active_fetchers.insert(0 if not tavily_api_key else 1, "newsapi")
    if truthsocial_enabled and truthsocial_username:
        active_fetchers.append("truthsocial")

    log.info(
        "scheduler.started",
        interval_seconds=interval_seconds,
        fetchers=active_fetchers,
        newsapi_daily_cap=newsapi_max_daily_requests if newsapi_api_key and newsapi_enabled else None,
    )

    while True:
        try:
            async with session_factory() as session:
                await run_cycle(
                    session=session,
                    embedder=embedder,
                    strategy=strategy,
                    llm_client=llm_client,
                    tavily_api_key=tavily_api_key,
                    newsapi_api_key=newsapi_api_key,
                    newsapi_enabled=newsapi_enabled,
                    newsapi_max_daily_requests=newsapi_max_daily_requests,
                    reddit_user_agent=reddit_user_agent,
                    domain_blacklist=domain_blacklist,
                    truthsocial_enabled=truthsocial_enabled,
                    truthsocial_username=truthsocial_username,
                    truthsocial_password=truthsocial_password,
                    truthsocial_accounts=truthsocial_accounts,
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
            # Commit per-market so catalysts are durable immediately; if the
            # process dies mid-loop the already-generated rows are preserved.
            await session.commit()
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
        except Exception:
            log.warning(
                "scheduler.catalyst_generation_failed",
                market_id=market.id,
                exc_info=True,
            )
            await session.rollback()

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
) -> list[tuple[str, str, datetime, list[tuple[str, str | None]]]]:
    """Return (market_id, category, close_time, [(query_text, tv_query), ...]) for all active catalyst runs.

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
        select(
            CatalystRunRow.id,
            MarketRow.id,
            MarketRow.category,
            MarketRow.close_time,
            CatalystQueryRow.query_text,
            CatalystQueryRow.tv_query,
        )
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
    grouped: dict[str, tuple[str, datetime, list[tuple[str, str | None]]]] = {}
    for _run_id, market_id, category, close_time, query_text, tv_query in rows:
        if market_id not in grouped:
            grouped[market_id] = (category, close_time, [])
        grouped[market_id][2].append((query_text, tv_query))

    return [(mid, cat, ct, queries) for mid, (cat, ct, queries) in grouped.items()]
