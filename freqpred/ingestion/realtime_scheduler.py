"""Real-time news ingestion scheduler.

Runs a fast cycle (default every 5 minutes) for cursor-based, near-real-time
sources that benefit from frequent polling:
  - TV chyrons via the Internet Archive Third Eye API
  - Truth Social account feeds

These sources use fetcher_cursors for deduplication so running them more
frequently than the main ingestion cycle does not cause double-processing.

The realtime scheduler manages its own backoff counters independently of the
main scheduler by passing ``services=_REALTIME_SERVICES`` to ``tick_and_load``.
This prevents the faster realtime cycle from draining backoff counters that
belong to the main scheduler (tavily, newsapi, gdelt, tv_archive).

Public API:
    run_realtime_cycle(...)    — one full pass.
    run_realtime_scheduler(…)  — async loop calling run_realtime_cycle every N seconds.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.ingestion.backoff import record_rate_limit, record_success, tick_and_load
from freqpred.ingestion.cursors import get_cursor, set_cursor
from freqpred.ingestion.fetchers import tv_chyron as tv_chyron_fetcher
from freqpred.ingestion.fetchers import truthsocial as truthsocial_fetcher
from freqpred.ingestion.fetchers.truthsocial import (
    LoginErrorException as TruthSocialLoginError,
    patch_api_for_block_detection,
)
from freqpred.ingestion.scheduler import _load_active_market_queries
from freqpred.ingestion.store import DocumentSkipped, UpsertStatus, link_document_to_market, upsert_document
from freqpred.rag.embedder import LocalEmbedder

if TYPE_CHECKING:
    from freqpred.config import TruthSocialAccountConfig

log = structlog.get_logger(__name__)

_TS_ACCOUNT_FETCHER = "truthsocial_account"

# Services whose backoff counters are managed by this scheduler.
# Passed to tick_and_load so the main scheduler's counters are not affected.
_REALTIME_SERVICES: frozenset[str] = frozenset({"truthsocial"})


async def run_realtime_cycle(
    session: AsyncSession,
    embedder: LocalEmbedder,
    tv_chyron_enabled: bool = True,
    truthsocial_enabled: bool = False,
    truthsocial_username: str = "",
    truthsocial_password: str = "",
    truthsocial_accounts: "list[TruthSocialAccountConfig] | None" = None,
    domain_blacklist: frozenset[str] = frozenset({"kalshi.com"}),
) -> dict[str, int]:
    """Run one real-time ingestion cycle (chyrons + Truth Social account feeds).

    Chyrons are fetched in bulk once per cycle and distributed to every active
    market whose tv_query matches at least one chyron.  The fetcher_cursors row
    for ('tv_chyron', 'global') is updated at the end of the cycle so only new
    chyrons are processed on the next run.

    Truth Social account feeds are polled per-account with per-account cursors,
    then distributed to markets via the account's configured categories.

    Args:
        session:                Open async SQLAlchemy session (caller manages commit).
        embedder:               Local embedder for document storage.
        tv_chyron_enabled:      When True, pull Third Eye chyron data (default: True).
        truthsocial_enabled:    When True, Truth Social account feeds are active.
        truthsocial_username:   Truth Social account username.
        truthsocial_password:   Truth Social account password.
        truthsocial_accounts:   Account feeds to poll, with category mappings.
        domain_blacklist:       Domains to exclude from Truth Social results.

    Returns:
        Stats dict with keys: docs_fetched, docs_stored, docs_error.
    """
    now = datetime.now(UTC)
    total_fetched = 0
    total_stored = 0
    total_error = 0

    # Tick and load backoff state only for services managed by this scheduler.
    backoff_state = await tick_and_load(session, services=_REALTIME_SERVICES)
    truthsocial_login_failed: bool = backoff_state.get("truthsocial", False)
    success_recorded: set[str] = set()

    # Load active market queries — needed for chyron distribution and TS category mapping.
    market_queries = await _load_active_market_queries(session)

    # Build category → [market_id] index for Truth Social category routing.
    _category_markets: dict[str, list[str]] = {}
    for _mid, _cat, _mq, _ct, _qpairs in market_queries:
        _category_markets.setdefault(_cat.lower(), []).append(_mid)

    # --- TV chyron phase ---
    if tv_chyron_enabled:
        last_tv_cursor = await get_cursor(session, "tv_chyron", "global")
        all_chyrons = await tv_chyron_fetcher.fetch_all(lookback_hours=1)
        chyrons_matched = 0

        for market_id, category, _market_question, _close_time, query_pairs in market_queries:
            for _query_text, tv_query in query_pairs:
                if not tv_query:
                    continue
                and_groups = tv_chyron_fetcher.parse_and_groups(tv_query)
                matched = tv_chyron_fetcher.filter_chyrons(
                    all_chyrons, and_groups, since=last_tv_cursor
                )
                chyrons_matched += len(matched)
                total_fetched += len(matched)
                for raw_doc in matched:
                    raw_doc.category = category
                    try:
                        async with session.begin_nested():
                            doc, status = await upsert_document(session, embedder, raw_doc)
                            await link_document_to_market(session, doc.id, market_id)
                        if status != UpsertStatus.DEDUPED:
                            total_stored += 1
                    except DocumentSkipped:
                        pass
                    except Exception:
                        log.warning(
                            "realtime_scheduler.upsert_error",
                            market_id=market_id,
                            source_url=raw_doc.source_url,
                            exc_info=True,
                        )
                        total_error += 1

        log.info(
            "realtime_scheduler.chyron_phase_complete",
            chyrons_fetched=len(all_chyrons),
            chyrons_matched=chyrons_matched,
            markets=len(market_queries),
        )
        await set_cursor(session, "tv_chyron", "global", now)

    # --- Truth Social account feeds ---
    ts_api: object | None = None
    if truthsocial_enabled and truthsocial_username:
        from truthbrush.api import Api as TruthSocialApi  # noqa: PLC0415

        ts_api = TruthSocialApi(
            username=truthsocial_username, password=truthsocial_password
        )
        patch_api_for_block_detection(ts_api)

    if ts_api is not None and truthsocial_accounts:
        for ts_account in truthsocial_accounts:
            ts_username = ts_account.username
            if truthsocial_login_failed:
                break
            try:
                last_run = await get_cursor(session, _TS_ACCOUNT_FETCHER, ts_username)
                ts_created_after = (
                    last_run if last_run is not None else now - timedelta(hours=48)
                )

                docs = await truthsocial_fetcher.fetch_account(
                    api=ts_api,
                    username=ts_username,
                    created_after=ts_created_after,
                    excluded_domains=domain_blacklist,
                )

                account_market_ids: list[str] = []
                for cat in ts_account.categories:
                    account_market_ids.extend(_category_markets.get(cat.lower(), []))

                for raw_doc in docs:
                    try:
                        async with session.begin_nested():
                            doc, status = await upsert_document(session, embedder, raw_doc)
                            for mid in account_market_ids:
                                await link_document_to_market(session, doc.id, mid)
                        if status != UpsertStatus.DEDUPED:
                            total_stored += 1
                    except DocumentSkipped:
                        pass
                    except Exception:
                        log.warning(
                            "realtime_scheduler.upsert_error",
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
                    "realtime_scheduler.truthsocial_account_fetched",
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
                        "realtime_scheduler.truthsocial_cloudflare_blocked",
                        username=ts_username,
                        skip_cycles=skip_cycles,
                        hint="IP temporarily banned by Cloudflare (error 1015); backing off",
                    )
                else:
                    log.error(
                        "realtime_scheduler.truthsocial_login_failed",
                        username=ts_username,
                        skip_cycles=skip_cycles,
                        exc_info=True,
                    )
            except Exception:
                log.warning(
                    "realtime_scheduler.fetcher_error",
                    fetcher="truthsocial_account",
                    username=ts_username,
                    exc_info=True,
                )

    stats = {
        "docs_fetched": total_fetched,
        "docs_stored": total_stored,
        "docs_error": total_error,
    }
    log.info("realtime_scheduler.cycle_complete", **stats)
    return stats


async def run_realtime_scheduler(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: LocalEmbedder,
    interval_seconds: int = 300,
    tv_chyron_enabled: bool = True,
    truthsocial_enabled: bool = False,
    truthsocial_username: str = "",
    truthsocial_password: str = "",
    truthsocial_accounts: "list[TruthSocialAccountConfig] | None" = None,
    domain_blacklist: frozenset[str] = frozenset({"kalshi.com"}),
) -> None:
    """Async loop: runs run_realtime_cycle every *interval_seconds*.

    Designed to be launched as an asyncio background task alongside the main
    ingestion scheduler. Logs and continues on cycle-level errors — never exits.

    Args:
        session_factory:        Async SQLAlchemy session factory.
        embedder:               Local embedder for document storage.
        interval_seconds:       Sleep duration between cycles (default 300 = 5 min).
        tv_chyron_enabled:      When True, pull Third Eye chyron data each cycle.
        truthsocial_enabled:    When True, Truth Social account feeds are active.
        truthsocial_username:   Truth Social account username.
        truthsocial_password:   Truth Social account password.
        truthsocial_accounts:   Standing account feeds to poll each cycle.
        domain_blacklist:       Domains to exclude from Truth Social results.
    """
    active_sources: list[str] = []
    if tv_chyron_enabled:
        active_sources.append("tv_chyron")
    if truthsocial_enabled and truthsocial_username:
        active_sources.append("truthsocial")

    log.info(
        "realtime_scheduler.started",
        interval_seconds=interval_seconds,
        sources=active_sources,
    )

    while True:
        try:
            async with session_factory() as session:
                await run_realtime_cycle(
                    session=session,
                    embedder=embedder,
                    tv_chyron_enabled=tv_chyron_enabled,
                    truthsocial_enabled=truthsocial_enabled,
                    truthsocial_username=truthsocial_username,
                    truthsocial_password=truthsocial_password,
                    truthsocial_accounts=truthsocial_accounts,
                    domain_blacklist=domain_blacklist,
                )
                await session.commit()
        except Exception:
            log.error("realtime_scheduler.cycle_error", exc_info=True)

        await asyncio.sleep(interval_seconds)
