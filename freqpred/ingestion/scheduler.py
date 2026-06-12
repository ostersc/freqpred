"""Ingestion scheduler: runs fetchers per-market on catalyst queries.

Each cycle:
  1. Loads all non-closed markets from the DB.
  2. Filters them through the registered strategy (select_markets).
  3. Generates catalyst queries for markets that have none yet
     (or whose last run is stale) via generate_catalysts.
  4. Deactivates catalysts for markets no longer selected.
  5. Runs Tavily + NewsAPI + Guardian + Reddit + GDELT + TV Archive fetchers
     against every active CatalystQuery and upserts results into the document store.

Near-real-time sources (TV chyrons, Truth Social account feeds) run on their
own faster cadence in ``realtime_scheduler.py``.

Public API:
    run_cycle(...)   — one full pass (steps 1-5).
    run_scheduler(…) — async loop that calls run_cycle every N seconds.
"""
from __future__ import annotations

import asyncio
import random
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.ingestion.backoff import record_rate_limit, record_success, tick_and_load
from freqpred.ingestion.cursors import delete_cursors, get_cursor, set_cursor
from freqpred.ingestion.fetchers import gdelt as gdelt_fetcher
from freqpred.ingestion.fetchers import guardian as guardian_fetcher
from freqpred.ingestion.fetchers import newsapi as newsapi_fetcher
from freqpred.ingestion.fetchers import reddit as reddit_fetcher
from freqpred.ingestion.fetchers import tavily as tavily_fetcher
from freqpred.ingestion.fetchers import tv_archive as tv_archive_fetcher
from freqpred.ingestion.fetchers.gdelt import GDELTRateLimitError
from freqpred.ingestion.fetchers.guardian import GuardianRateLimitError
from freqpred.ingestion.fetchers.newsapi import NewsAPIRateLimitError
from freqpred.ingestion.fetchers.reddit import RedditBlockedError
from tavily.errors import ForbiddenError, UsageLimitExceededError
from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow
from freqpred.ingestion.quota import current_window, get_daily_count, get_window_count, increment_window_count
from freqpred.ingestion.store import DocumentSkipped, UpsertStatus, link_document_to_market, upsert_document
from freqpred.markets.models import Market, MarketRow, PositionRow
from freqpred.rag.embedder import LocalEmbedder

if TYPE_CHECKING:
    from freqpred.llm.client import LLMClient
    from freqpred.ingestion.selector import StrategyProtocol
    from freqpred.runtime.telemetry import RuntimeTelemetry

# Backoff services owned by this scheduler — passed to tick_and_load so the
# realtime scheduler's counters (truthsocial) are not affected.
_MAIN_SCHEDULER_SERVICES: frozenset[str] = frozenset(
    {"tavily", "newsapi", "guardian", "gdelt", "tv_archive", "reddit"}
)

# Heartbeat service name for an individual ingestion fetcher (matches the
# SERVICE_FETCHER_* constants in freqpred.runtime.telemetry).
def _fetcher_service(name: str) -> str:
    return f"fetcher_{name}"

log = structlog.get_logger(__name__)

_NEWSAPI_LOOKBACK_DAYS = 7
_GUARDIAN_LOOKBACK_DAYS = 7

# Category → subreddit mapping used for Reddit queries.
# Keys are matched lowercase. Kalshi politics markets arrive under three
# categories — "Politics", "Elections", and "Mentions" (word-mention series
# like KXTRUMPSAY) — all of which need the politics subreddits. Without an
# entry a category silently falls back to ["news"], which is both a weak
# match for niche catalyst queries and a poor-Brier source.
#
# Selection principle (per source_quality_scores Brier on Mentions markets):
# moderated discussion subs far outperform link firehoses — NeutralPolitics
# 0.064 and PoliticalDiscussion 0.091 vs the r/politics firehose at 0.275.
# moderatepolitics and NeutralNews match the winning profile; Conservative is
# a base-sentiment hypothesis for "will Trump say X" markets (he amplifies
# what resonates there) — prune any of these if their Brier comes back bad.
_POLITICS_SUBREDDITS: list[str] = [
    "politics",
    "PoliticalDiscussion",
    "neutralpolitics",
    "moderatepolitics",
    "NeutralNews",
    "Conservative",
]
_SUBREDDIT_MAP: dict[str, list[str]] = {
    "politics":   _POLITICS_SUBREDDITS,
    "mentions":   _POLITICS_SUBREDDITS,
    "elections":  _POLITICS_SUBREDDITS,
    "technology": ["technology", "MachineLearning", "singularity"],
    "economics":  ["economics", "investing", "stocks"],
    "fintech":    ["investing", "wallstreetbets", "stocks", "fintech"],
    "sports":     ["sports"],
    "crypto":     ["CryptoCurrency", "Bitcoin"],
    "climate":    ["climate", "environment"],
}



def _subreddits_for_category(category: str) -> list[str]:
    return _SUBREDDIT_MAP.get(category.lower(), ["news"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_cycle(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: LocalEmbedder,
    strategy: "StrategyProtocol | None" = None,
    llm_client: "LLMClient | None" = None,
    cheap_model: str = "claude-haiku-4-5-20251001",
    tavily_api_key: str = "",
    tavily_daily_cap: int = 33,
    tavily_min_fetch_interval_hours: float = 1.0,
    newsapi_api_key: str = "",
    newsapi_enabled: bool = True,
    newsapi_max_window_requests: int = 45,
    newsapi_min_fetch_interval_hours: float = 1.0,
    guardian_api_key: str = "",
    guardian_enabled: bool = True,
    guardian_daily_cap: int = 490,
    guardian_min_fetch_interval_hours: float = 1.0,
    reddit_user_agent: str = "freqpred/0.1",
    reddit_min_fetch_interval_hours: float = 2.0,
    domain_blacklist: frozenset[str] = frozenset({"kalshi.com"}),
    telemetry: "RuntimeTelemetry | None" = None,
) -> dict[str, int]:
    """Run one full ingestion cycle.

    Steps:
      1. Load all non-closed markets from the DB.
      2. Filter through strategy.is_market_interesting() (if strategy provided).
      3. Generate catalyst queries for selected markets with no active run.
      4. Deactivate catalysts for markets no longer selected.
      5. Fetch documents for every market with active catalyst queries.

    Fetcher errors are caught per-fetcher; one failure does not abort others.
    Near-real-time sources (chyrons, Truth Social) run in realtime_scheduler.py.

    Session ownership: this function manages its own sessions via session_factory.
    A single setup session handles catalyst generation and per-cycle state loading;
    then one session per market is opened and committed after each market completes
    so documents are immediately visible to the signal pipeline.

    Args:
        session_factory:            Async SQLAlchemy session factory.
        embedder:                   Local embedder for document storage.
        strategy:                   Strategy used to filter markets. If None, all markets
                                    with active catalyst runs are processed.
        llm_client:                 LLM client for catalyst generation. If None, catalyst
                                    generation is skipped.
        tavily_api_key:             Tavily API key. Tavily is skipped if empty.
        tavily_daily_cap:           Hard daily request cap for Tavily (default: 33; 1,000/month ÷ 30).
        tavily_min_fetch_interval_hours: Floor on per-market adaptive interval for Tavily (default: 1.0h).
        newsapi_api_key:            NewsAPI key. NewsAPI is skipped if empty.
        newsapi_enabled:             When False, the NewsAPI fetcher is skipped entirely
                                     even if newsapi_api_key is set (default: True).
        newsapi_max_window_requests: Per-12-hour-window request cap tracked in Postgres
                                     ``api_daily_counters`` (default: 45; NewsAPI allows 50).
        newsapi_min_fetch_interval_hours: Floor on per-market adaptive interval for NewsAPI (default: 1.0h).
        guardian_api_key:           Guardian Content API key. Guardian is skipped if empty.
        guardian_enabled:           When False, the Guardian fetcher is skipped entirely
                                    even if guardian_api_key is set (default: True).
        guardian_daily_cap:         Hard daily request cap for Guardian (default: 490).
        guardian_min_fetch_interval_hours: Floor on the per-market adaptive fetch interval
                                    (default: 1.0h). The actual interval scales up automatically
                                    with market count to stay within the daily cap.
        reddit_user_agent:          User-Agent for Reddit requests.
        reddit_min_fetch_interval_hours: Floor on the per-market Reddit fetch interval
                                    (default: 2.0h). Each due market costs
                                    subreddits x queries unauthenticated RSS requests,
                                    so Reddit is cursor-gated like the API fetchers
                                    to stay under its ~1 req/2s tolerance.
        domain_blacklist:           Domains to exclude from fetcher results.
                                    Matched as a substring of each URL (default: kalshi.com).
    Returns:
        Stats dict with keys: markets_processed, catalysts_generated,
        docs_fetched, docs_stored, docs_error.
    """
    now = datetime.now(UTC)
    catalysts_generated = 0

    # --- Phase 1: setup session — catalyst generation + per-cycle state --------
    # Committed once at the end of this block. _ensure_catalysts also commits
    # per-market internally so catalysts are durable before fetching begins.
    async with session_factory() as session:
        if strategy is not None and llm_client is not None:
            catalysts_generated = await _ensure_catalysts(
                session, strategy, llm_client, embedder, model=cheap_model
            )

        market_queries = await _load_active_market_queries(session)

        if not market_queries:
            log.debug("scheduler.run_cycle.no_active_markets")
            await session.commit()
            stats = {
                "markets_processed": 0,
                "catalysts_generated": catalysts_generated,
                "docs_fetched": 0,
                "docs_stored": 0,
                "docs_error": 0,
            }
            if telemetry is not None:
                from freqpred.runtime.telemetry import SERVICE_INGESTION_SCHEDULER  # noqa: PLC0415

                await telemetry.mark_success(SERVICE_INGESTION_SCHEDULER, details=stats)
            return stats

        newsapi_from = now - timedelta(days=_NEWSAPI_LOOKBACK_DAYS)
        guardian_from = now - timedelta(days=_GUARDIAN_LOOKBACK_DAYS)
        newsapi_window_date, newsapi_hour_slot = current_window(now)

        # Load persistent backoff state from DB (ticks down this scheduler's counters).
        backoff_state = await tick_and_load(session, services=_MAIN_SCHEDULER_SERVICES)

        # In-memory flags for within-cycle short-circuit. Initialized from DB
        # state so a restart respects backoff that was set in a previous cycle.
        tavily_limit_hit: bool = backoff_state.get("tavily", False)
        newsapi_limit_hit: bool = backoff_state.get("newsapi", False)
        newsapi_limit_logged: bool = newsapi_limit_hit
        guardian_limit_hit: bool = backoff_state.get("guardian", False)
        gdelt_limit_hit: bool = backoff_state.get("gdelt", False)
        tv_archive_limit_hit: bool = backoff_state.get("tv_archive", False)
        reddit_limit_hit: bool = backoff_state.get("reddit", False)

        # Per-fetcher daily quota checks — if already at cap, skip for the whole cycle.
        tavily_daily_count: int = 0
        if tavily_api_key and not tavily_limit_hit:
            tavily_daily_count = await get_daily_count(session, "tavily", now.date())
            if tavily_daily_count >= tavily_daily_cap:
                tavily_limit_hit = True
                log.warning("scheduler.tavily_daily_cap_reached_at_cycle_start",
                            count=tavily_daily_count, cap=tavily_daily_cap)

        guardian_daily_count: int = 0
        if guardian_api_key and guardian_enabled and not guardian_limit_hit:
            guardian_daily_count = await get_daily_count(session, "guardian", now.date())
            if guardian_daily_count >= guardian_daily_cap:
                guardian_limit_hit = True
                log.warning("scheduler.guardian_daily_cap_reached_at_cycle_start",
                            count=guardian_daily_count, cap=guardian_daily_cap)

        # Adaptive per-market fetch intervals.
        # Formula: max(min_interval, min(24h, total_queries × 24h / daily_cap))
        total_active_queries = sum(len(qp) for _, _, _, _, qp in market_queries)

        def _compute_interval(daily_cap: float, min_hours: float) -> timedelta:
            if daily_cap > 0 and total_active_queries > 0:
                hours = max(min_hours, min(24.0, total_active_queries * 24.0 / daily_cap))
            else:
                hours = 24.0
            return timedelta(hours=hours)

        tavily_fetch_interval = _compute_interval(tavily_daily_cap, tavily_min_fetch_interval_hours)
        newsapi_fetch_interval = _compute_interval(
            newsapi_max_window_requests * 2, newsapi_min_fetch_interval_hours
        )
        guardian_fetch_interval = _compute_interval(guardian_daily_cap, guardian_min_fetch_interval_hours)

        log.debug(
            "scheduler.fetch_intervals",
            total_queries=total_active_queries,
            tavily_hours=round(tavily_fetch_interval.total_seconds() / 3600, 2),
            newsapi_hours=round(newsapi_fetch_interval.total_seconds() / 3600, 2),
            guardian_hours=round(guardian_fetch_interval.total_seconds() / 3600, 2),
        )

        await session.commit()

    # --- Phase 2: market loop — one session per market, committed after each ---
    total_fetched = 0
    total_stored = 0
    total_error = 0
    total_fetcher_errors = 0
    last_fetcher_error = ""
    # Last error message per fetcher this cycle (includes rate-limit trips) —
    # flushed to per-fetcher telemetry heartbeats at cycle end.
    fetcher_error_messages: dict[str, str] = {}

    # Track which services had a successful call this cycle so we only write
    # record_success once per service (it's idempotent but avoids extra DB hits).
    success_recorded: set[str] = set()

    for market_id, category, market_question, close_time, query_pairs in market_queries:
        market_start = time.monotonic()
        market_fetched = 0
        market_stored = 0
        market_deduped = 0
        market_error = 0

        async with session_factory() as market_session:
            # --- Per-market cursor checks (once per market, not per query) ---
            tavily_due_this_market = False
            if tavily_api_key and not tavily_limit_hit:
                last_tavily = await get_cursor(market_session, "tavily", market_id)
                tavily_due_this_market = (
                    last_tavily is None or (now - last_tavily) >= tavily_fetch_interval
                )
            tavily_fetched_this_market = False

            newsapi_due_this_market = False
            if newsapi_api_key and newsapi_enabled and not newsapi_limit_hit:
                last_newsapi = await get_cursor(market_session, "newsapi", market_id)
                newsapi_due_this_market = (
                    last_newsapi is None or (now - last_newsapi) >= newsapi_fetch_interval
                )
            newsapi_fetched_this_market = False

            guardian_due_this_market = False
            if guardian_api_key and guardian_enabled and not guardian_limit_hit:
                last_guardian = await get_cursor(market_session, "guardian", market_id)
                guardian_due_this_market = (
                    last_guardian is None or (now - last_guardian) >= guardian_fetch_interval
                )
            guardian_fetched_this_market = False

            reddit_due_this_market = False
            if not reddit_limit_hit:
                last_reddit = await get_cursor(market_session, "reddit", market_id)
                # Jitter the interval +/-25% (i.e. +/-30 min at the 2h default)
                # so markets desynchronize. Without it, every market fetched in
                # the same cycle becomes due together again one interval later —
                # 3 idle cycles, then one cycle bursting subreddits x queries x
                # all-markets requests. Symmetric jitter preserves the average
                # cadence at the configured base.
                jittered = timedelta(
                    hours=reddit_min_fetch_interval_hours * random.uniform(0.75, 1.25)
                )
                reddit_due_this_market = (
                    last_reddit is None or (now - last_reddit) >= jittered
                )
            reddit_fetched_this_market = False
            # Reddit's unauthenticated budget is 1 request/min per IP, so a due
            # market gets exactly one search — a randomly chosen catalyst query.
            # Rotation across fetches (every ~2h per market) covers all queries
            # over the course of a day.
            reddit_query_index = random.randrange(len(query_pairs)) if query_pairs else -1

            for query_index, (query_text, tv_query) in enumerate(query_pairs):
                # --- Build non-GDELT fetch coroutines to run in parallel ---
                # GDELT (doc + TV) are run sequentially afterwards because they share
                # a 1 req/5 s rate limit across all their API endpoints.
                fetch_names: list[str] = []
                fetch_coros = []

                if tavily_due_this_market and not tavily_limit_hit:
                    fetch_names.append("tavily")
                    fetch_coros.append(tavily_fetcher.fetch(
                        api_key=tavily_api_key,
                        query=query_text,
                        excluded_domains=domain_blacklist,
                    ))

                # NewsAPI: adaptive cursor gate first, then per-query window cap check.
                newsapi_queued = False
                if newsapi_due_this_market and not newsapi_limit_hit:
                    window_count = await get_window_count(market_session, "newsapi", newsapi_window_date, newsapi_hour_slot)
                    if window_count >= newsapi_max_window_requests:
                        if not newsapi_limit_logged:
                            log.warning(
                                "newsapi_window_limit_reached",
                                window_date=newsapi_window_date.isoformat(),
                                hour_slot=newsapi_hour_slot,
                                count=window_count,
                                max_window_requests=newsapi_max_window_requests,
                            )
                            newsapi_limit_logged = True
                    else:
                        newsapi_queued = True
                        fetch_names.append("newsapi")
                        fetch_coros.append(newsapi_fetcher.fetch(
                            api_key=newsapi_api_key,
                            query=query_text,
                            from_date=newsapi_from,
                            excluded_domains=domain_blacklist,
                        ))

                # Guardian: use tv_query (Solr syntax) when available, else query_text.
                if guardian_due_this_market and not guardian_limit_hit:
                    fetch_names.append("guardian")
                    fetch_coros.append(guardian_fetcher.fetch(
                        api_key=guardian_api_key,
                        query=tv_query or query_text,
                        from_date=guardian_from,
                        excluded_domains=domain_blacklist,
                    ))

                if (
                    reddit_due_this_market
                    and not reddit_limit_hit
                    and query_index == reddit_query_index
                ):
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
                        fetcher_error_messages[name] = str(result)
                        if name == "tavily" and isinstance(result, (ForbiddenError, UsageLimitExceededError)):
                            tavily_limit_hit = True
                            tavily_due_this_market = False
                            skip_cycles = await record_rate_limit(market_session, "tavily")
                            log.warning("scheduler.tavily_limit_reached", reason=str(result), skip_cycles=skip_cycles)
                        elif name == "newsapi" and isinstance(result, NewsAPIRateLimitError):
                            newsapi_limit_hit = True
                            newsapi_due_this_market = False
                            newsapi_limit_logged = True
                            skip_cycles = await record_rate_limit(market_session, "newsapi")
                            log.warning("scheduler.newsapi_rate_limited", reason=str(result), skip_cycles=skip_cycles)
                        elif name == "guardian" and isinstance(result, GuardianRateLimitError):
                            guardian_limit_hit = True
                            guardian_due_this_market = False
                            skip_cycles = await record_rate_limit(market_session, "guardian")
                            log.warning("scheduler.guardian_rate_limited", reason=str(result), skip_cycles=skip_cycles)
                        elif name == "reddit" and isinstance(result, RedditBlockedError):
                            reddit_limit_hit = True
                            skip_cycles = await record_rate_limit(market_session, "reddit")
                            log.error(
                                "scheduler.reddit_blocked",
                                reason=str(result),
                                skip_cycles=skip_cycles,
                            )
                        else:
                            total_fetcher_errors += 1
                            last_fetcher_error = f"{name}: {result}"
                            log.warning(
                                "scheduler.fetcher_error",
                                market_id=market_id,
                                fetcher=name,
                                query=query_text,
                                error=str(result),
                            )
                    else:
                        raw_docs.extend(result)
                        if name == "tavily":
                            tavily_fetched_this_market = True
                            tavily_daily_count += 1
                            await increment_window_count(market_session, "tavily", now.date(), now.hour // 12)
                            if tavily_daily_count >= tavily_daily_cap:
                                tavily_limit_hit = True
                                tavily_due_this_market = False
                                log.warning("scheduler.tavily_daily_cap_reached",
                                            count=tavily_daily_count, cap=tavily_daily_cap)
                        if name == "newsapi" and newsapi_queued:
                            newsapi_fetched_this_market = True
                            await increment_window_count(market_session, "newsapi", newsapi_window_date, newsapi_hour_slot)
                        if name == "reddit":
                            reddit_fetched_this_market = True
                        if name == "guardian":
                            guardian_fetched_this_market = True
                            guardian_daily_count += 1
                            await increment_window_count(market_session, "guardian", now.date(), now.hour // 12)
                            if guardian_daily_count >= guardian_daily_cap:
                                guardian_limit_hit = True
                                guardian_due_this_market = False
                                log.warning("scheduler.guardian_daily_cap_reached",
                                            count=guardian_daily_count, cap=guardian_daily_cap)
                        # Clear backoff on first success this cycle for this service.
                        if name not in success_recorded:
                            await record_success(market_session, name)
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
                            await record_success(market_session, "gdelt")
                            success_recorded.add("gdelt")
                    except GDELTRateLimitError as exc:
                        gdelt_limit_hit = True
                        fetcher_error_messages["gdelt"] = str(exc)
                        skip_cycles = await record_rate_limit(market_session, "gdelt")
                        log.warning("scheduler.gdelt_rate_limited", reason=str(exc), skip_cycles=skip_cycles)
                    except Exception as exc:
                        fetcher_error_messages["gdelt"] = str(exc)
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
                        async with market_session.begin_nested():
                            doc, status = await upsert_document(
                                market_session,
                                embedder,
                                raw_doc,
                                llm_client=llm_client,
                                query_text=query_text,
                                market_question=market_question,
                                summary_model=cheap_model,
                            )
                            await link_document_to_market(market_session, doc.id, market_id)
                        if status == UpsertStatus.DEDUPED:
                            market_deduped += 1
                        else:
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

            # Update per-market cursors after all queries complete.
            if tavily_fetched_this_market:
                await set_cursor(market_session, "tavily", market_id, now)
            if newsapi_fetched_this_market:
                await set_cursor(market_session, "newsapi", market_id, now)
            if guardian_fetched_this_market:
                await set_cursor(market_session, "guardian", market_id, now)
            if reddit_fetched_this_market:
                await set_cursor(market_session, "reddit", market_id, now)

            await market_session.commit()

        log.info(
            "scheduler.market_cycle_complete",
            market_id=market_id,
            elapsed_s=round(time.monotonic() - market_start, 2),
            queries=len(query_pairs),
            docs_fetched=market_fetched,
            docs_stored=market_stored,
            docs_deduped=market_deduped,
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
        "fetcher_errors": total_fetcher_errors,
    }

    log.info("scheduler.cycle_complete", **stats)
    if telemetry is not None:
        from freqpred.runtime.telemetry import SERVICE_INGESTION_SCHEDULER  # noqa: PLC0415

        if total_fetcher_errors > 0:
            await telemetry.mark_error(
                SERVICE_INGESTION_SCHEDULER,
                f"{total_fetcher_errors} fetcher error(s): {last_fetcher_error}",
                details=stats,
            )
        else:
            await telemetry.mark_success(SERVICE_INGESTION_SCHEDULER, details=stats)

        # Per-fetcher heartbeats: each fetcher reports independently so a dead
        # source surfaces as stale in system health even while the scheduler
        # loop itself stays green. Success = at least one error-free call this
        # cycle (zero docs is still success).
        for name in success_recorded:
            await telemetry.mark_success(_fetcher_service(name))
        for name, message in fetcher_error_messages.items():
            await telemetry.mark_error(_fetcher_service(name), message)
    return stats


async def run_scheduler(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: LocalEmbedder,
    interval_seconds: int = 1800,
    strategy: "StrategyProtocol | None" = None,
    llm_client: "LLMClient | None" = None,
    cheap_model: str = "claude-haiku-4-5-20251001",
    tavily_api_key: str = "",
    tavily_daily_cap: int = 33,
    tavily_min_fetch_interval_hours: float = 1.0,
    newsapi_api_key: str = "",
    newsapi_enabled: bool = True,
    newsapi_max_window_requests: int = 45,
    newsapi_min_fetch_interval_hours: float = 1.0,
    guardian_api_key: str = "",
    guardian_enabled: bool = True,
    guardian_daily_cap: int = 490,
    guardian_min_fetch_interval_hours: float = 1.0,
    reddit_user_agent: str = "freqpred/0.1",
    reddit_min_fetch_interval_hours: float = 2.0,
    domain_blacklist: frozenset[str] = frozenset({"kalshi.com"}),
    telemetry: "RuntimeTelemetry | None" = None,
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
        newsapi_max_window_requests: Per-12-hour-window request cap for the NewsAPI fetcher.
        guardian_api_key:           Guardian Content API key.
        guardian_enabled:           When False, Guardian fetcher is skipped entirely.
        guardian_daily_cap:         Hard daily request cap for Guardian (default: 490).
        guardian_min_fetch_interval_hours: Floor on per-market adaptive fetch interval.
        reddit_user_agent:          User-Agent for Reddit requests.
        domain_blacklist:           Domains to exclude from fetcher results.
    """
    active_fetchers = ["reddit", "gdelt"]
    if tavily_api_key:
        active_fetchers.insert(0, "tavily")
    if newsapi_api_key and newsapi_enabled:
        active_fetchers.insert(0 if not tavily_api_key else 1, "newsapi")
    if guardian_api_key and guardian_enabled:
        active_fetchers.append("guardian")

    log.info(
        "scheduler.started",
        interval_seconds=interval_seconds,
        fetchers=active_fetchers,
        newsapi_window_cap=newsapi_max_window_requests if newsapi_api_key and newsapi_enabled else None,
    )

    while True:
        try:
            await run_cycle(
                session_factory=session_factory,
                embedder=embedder,
                strategy=strategy,
                llm_client=llm_client,
                cheap_model=cheap_model,
                tavily_api_key=tavily_api_key,
                tavily_daily_cap=tavily_daily_cap,
                tavily_min_fetch_interval_hours=tavily_min_fetch_interval_hours,
                newsapi_api_key=newsapi_api_key,
                newsapi_enabled=newsapi_enabled,
                newsapi_max_window_requests=newsapi_max_window_requests,
                newsapi_min_fetch_interval_hours=newsapi_min_fetch_interval_hours,
                guardian_api_key=guardian_api_key,
                guardian_enabled=guardian_enabled,
                guardian_daily_cap=guardian_daily_cap,
                guardian_min_fetch_interval_hours=guardian_min_fetch_interval_hours,
                reddit_user_agent=reddit_user_agent,
                reddit_min_fetch_interval_hours=reddit_min_fetch_interval_hours,
                domain_blacklist=domain_blacklist,
                telemetry=telemetry,
            )
        except Exception as exc:
            log.error("scheduler.cycle_error", exc_info=True)
            if telemetry is not None:
                from freqpred.runtime.telemetry import SERVICE_INGESTION_SCHEDULER  # noqa: PLC0415

                await telemetry.mark_error(
                    SERVICE_INGESTION_SCHEDULER,
                    str(exc),
                )

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
    *,
    model: str = "claude-haiku-4-5-20251001",
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

    # Protect markets with open positions from catalyst deactivation even if
    # they no longer pass is_market_interesting (e.g. price drifted to extreme).
    open_pos_result = await session.execute(
        select(PositionRow.market_id).where(PositionRow.status == "open").distinct()
    )
    open_market_ids: set[str] = {r.market_id for r in open_pos_result.all()}

    # Deactivate catalysts for markets that are no longer interesting, then
    # clean up their Guardian cursors so stale rows don't accumulate.
    deactivated_ids = await deactivate_stale_catalysts(session, [strategy], open_market_ids)
    if deactivated_ids:
        for fetcher in ("tavily", "newsapi", "guardian"):
            await delete_cursors(session, fetcher, deactivated_ids)

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
            await generate_catalysts(market, session, llm_client, embedder, model=model)
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
        status=row.status,
        result=row.result,
        close_time=row.close_time,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
        mid_price=row.mid_price,
        last_price=row.last_price,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        yes_bid_size=row.yes_bid_size,
        yes_ask_size=row.yes_ask_size,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
        metadata=dict(row.metadata_),
        open_time=row.open_time,
        series_ticker=row.series_ticker,
    )


async def _load_active_market_queries(
    session: AsyncSession,
) -> list[tuple[str, str, str, datetime, list[tuple[str, str | None]]]]:
    """Return (market_id, category, market_question, close_time, [(query_text, tv_query), ...]) for all active catalyst runs.

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
            MarketRow.question,
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
    grouped: dict[str, tuple[str, str, datetime, list[tuple[str, str | None]]]] = {}
    for _run_id, market_id, category, market_question, close_time, query_text, tv_query in rows:
        if market_id not in grouped:
            grouped[market_id] = (category, market_question, close_time, [])
        grouped[market_id][3].append((query_text, tv_query))

    return [(mid, cat, mq, ct, queries) for mid, (cat, mq, ct, queries) in grouped.items()]
