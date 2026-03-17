"""Manual smoke test: pick a random Kalshi market, generate catalysts via LLM,
then fetch targeted news from Tavily + NewsAPI + Reddit against those catalysts.

Usage:
    uv run python scripts/test_fetchers.py
    uv run python scripts/test_fetchers.py --category politics
    uv run python scripts/test_fetchers.py --max-results 5
    uv run python scripts/test_fetchers.py --dry-run   # catalysts only, no fetching
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import anthropic
import click

# Register all ORM models before any SQLAlchemy mapper is instantiated.
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models        # noqa: F401
import freqpred.rag.models        # noqa: F401
import freqpred.signal.models     # noqa: F401

from freqpred.config import load_config
from freqpred.ingestion.catalyst_generator import generate_catalysts
from freqpred.ingestion.fetchers import newsapi, reddit, tavily
from freqpred.ingestion.store import RawDocument
from freqpred.markets.kalshi import KalshiClient

# From SPEC.md §9 — subreddit targets by category
_CATEGORY_SUBREDDITS: dict[str, list[str]] = {
    "politics":   ["politics", "PoliticalDiscussion", "neutralpolitics"],
    "technology": ["technology", "MachineLearning", "singularity"],
    "fintech":    ["investing", "wallstreetbets", "stocks", "fintech"],
    "economics":  ["economics", "investing", "stocks"],
    "sports":     ["sports"],
    "crypto":     ["CryptoCurrency", "Bitcoin"],
    "climate":    ["climate", "environment"],
}


def _print_docs(docs: list[RawDocument], max_body: int = 200) -> None:
    if not docs:
        click.echo("  (no results)")
        return
    for i, doc in enumerate(docs, 1):
        age = datetime.now(timezone.utc) - doc.published_at
        age_str = f"{int(age.total_seconds() // 3600)}h ago"
        click.echo(f"  [{i}] {doc.title or '(no title)'}")
        click.echo(f"      {doc.source_url}")
        click.echo(f"      {doc.source_name} · {age_str}")
        snippet = doc.body[:max_body].replace("\n", " ")
        if len(doc.body) > max_body:
            snippet += "…"
        click.echo(f"      {snippet}")
        click.echo()


def _make_stub_session() -> MagicMock:
    """Build a minimal stub AsyncSession for the catalyst generator.

    In this script we don't need to persist to DB — we just want the LLM
    call and the returned CatalystRun domain object. The session stubs allow
    generate_catalysts() to run without a real DB connection.
    """
    session = AsyncMock()

    # _get_latest_run: return None (treat every run as a first run)
    first_result = MagicMock()
    first_result.scalar_one_or_none.return_value = None

    # log_llm_query flush: simulate auto-increment id
    flush_result = MagicMock()
    flush_result.scalar_one_or_none.return_value = None

    call_count = 0

    async def execute_side_effect(*_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return first_result
        return flush_result

    session.execute.side_effect = execute_side_effect
    session.flush = AsyncMock()
    session.add = MagicMock()

    # Give each added row a fake id so log_llm_query doesn't return None.
    session.add.side_effect = lambda row: (
        setattr(row, "id", 999),
        setattr(row, "created_at", datetime.now(timezone.utc)),
    )

    return session


@click.command()
@click.option("--category", default=None, help="Kalshi category to filter markets.")
@click.option("--max-results", default=5, show_default=True, help="Results per source per catalyst.")
@click.option("--dry-run", is_flag=True, default=False, help="Generate catalysts only, skip news fetching.")
def main(category: str | None, max_results: int, dry_run: bool) -> None:
    asyncio.run(_run(category=category, max_results=max_results, dry_run=dry_run))


async def _run(category: str | None, max_results: int, dry_run: bool) -> None:
    cfg = load_config()

    # ── 1. Pull a random market from Kalshi ──────────────────────────────────
    click.echo("Fetching markets from Kalshi…")
    async with KalshiClient(
        api_key=cfg.kalshi.api_key,
        base_url=cfg.kalshi.base_url,
        private_key_path=cfg.kalshi.private_key_path,
    ) as client:
        markets = await client.list_markets(category=category)

    if not markets:
        click.echo("No markets found — check your Kalshi API key or category filter.")
        return

    market = random.choice(markets)
    click.echo(f"\n{'═' * 70}")
    click.echo(f"  MARKET   {market.id}")
    click.echo(f"  {'─' * 66}")
    click.echo(f"  Question : {market.question}")
    click.echo(f"  Category : {market.category}")
    click.echo(f"  Mid price: {market.mid_price:.1%}  (bid {market.yes_bid:.1%} / ask {market.yes_ask:.1%})")
    click.echo(f"  Closes   : {market.close_time.strftime('%Y-%m-%d')}")
    click.echo(f"{'═' * 70}\n")

    # ── 2. Generate catalysts via LLM ────────────────────────────────────────
    click.echo(f"Generating catalyst queries (Claude Haiku)…")
    anthropic_client = anthropic.AsyncAnthropic(api_key=cfg.anthropic.api_key)
    stub_session = _make_stub_session()

    try:
        run = await generate_catalysts(market, stub_session, anthropic_client, embedder=None)
    except Exception as exc:
        click.echo(f"Catalyst generation failed: {exc}", err=True)
        return

    # Retrieve query texts from the stub session's add() calls.
    # generate_catalysts calls session.add() for: 1 CatalystRunRow + N CatalystQueryRow.
    from freqpred.ingestion.models import CatalystQueryRow
    added_objects = [call.args[0] for call in stub_session.add.call_args_list]
    catalyst_queries = [obj.query_text for obj in added_objects if isinstance(obj, CatalystQueryRow)]

    click.echo(f"\nCatalysts generated (generation {run.generation}):")
    for i, q in enumerate(catalyst_queries, 1):
        click.echo(f"  {i}. {q}")

    if dry_run or not catalyst_queries:
        click.echo("\n(dry-run: skipping news fetch)")
        return

    # ── 3. Fetch news against each catalyst query ────────────────────────────
    subreddits = _CATEGORY_SUBREDDITS.get(market.category, ["news", "worldnews"])
    subs_display = ", ".join(f"r/{s}" for s in subreddits)
    from_date = datetime.now(timezone.utc) - timedelta(days=7)

    for query in catalyst_queries:
        click.echo(f"\n{'─' * 70}")
        click.echo(f'Catalyst: "{query}"')
        click.echo(f"{'─' * 70}")

        # Tavily
        click.echo(f"\n[Tavily]")
        tavily_docs = await tavily.fetch(cfg.tavily.api_key, query, max_results=max_results)
        click.echo(f"{len(tavily_docs)} result(s)\n")
        _print_docs(tavily_docs)

        # NewsAPI
        click.echo(f"[NewsAPI]")
        newsapi_docs = await newsapi.fetch(
            cfg.newsapi.api_key, query, from_date=from_date, max_results=max_results
        )
        click.echo(f"{len(newsapi_docs)} result(s)\n")
        _print_docs(newsapi_docs)

        # Reddit
        click.echo(f"[Reddit] ({subs_display})")
        reddit_docs = await reddit.fetch(
            subreddits=subreddits,
            query=query,
            user_agent=cfg.reddit.user_agent,
            limit=max_results * 4,
        )
        click.echo(f"{len(reddit_docs)} result(s) (filtered: score≥10, age≤7d)\n")
        _print_docs(reddit_docs)


if __name__ == "__main__":
    main()
