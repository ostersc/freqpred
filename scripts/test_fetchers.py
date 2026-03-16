"""Manual smoke test: pick a random Kalshi market, fetch news from Tavily + NewsAPI.

Usage:
    uv run python scripts/test_fetchers.py
    uv run python scripts/test_fetchers.py --category politics
    uv run python scripts/test_fetchers.py --max-results 5
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

import click

from freqpred.config import load_config
from freqpred.ingestion.fetchers import newsapi, tavily
from freqpred.ingestion.store import RawDocument
from freqpred.markets.kalshi import KalshiClient


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


@click.command()
@click.option("--category", default=None, help="Kalshi category to filter markets.")
@click.option("--max-results", default=5, show_default=True, help="Results per source.")
def main(category: str | None, max_results: int) -> None:
    asyncio.run(_run(category=category, max_results=max_results))


async def _run(category: str | None, max_results: int) -> None:
    cfg = load_config()

    # ── 1. Pull markets from Kalshi ──────────────────────────────────────────
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
    click.echo(f"  MARKET  {market.id}")
    click.echo(f"  {'═' * 66}")
    click.echo(f"  Question : {market.question}")
    click.echo(f"  Category : {market.category}")
    click.echo(f"  Mid price: {market.mid_price:.1%}  (bid {market.yes_bid:.1%} / ask {market.yes_ask:.1%})")
    click.echo(f"  Closes   : {market.close_time.strftime('%Y-%m-%d')}")
    click.echo(f"{'═' * 70}\n")

    query = market.question

    # ── 2. Tavily ─────────────────────────────────────────────────────────────
    click.echo(f'[Tavily] Searching: "{query}"...')
    tavily_docs = await tavily.fetch(cfg.tavily.api_key, query, max_results=max_results)
    click.echo(f"[Tavily] {len(tavily_docs)} result(s)\n")
    _print_docs(tavily_docs)

    # ── 3. NewsAPI ────────────────────────────────────────────────────────────
    from_date = datetime.now(timezone.utc) - timedelta(days=7)
    click.echo(f'[NewsAPI] Searching: "{query}" (last 7 days)...')
    newsapi_docs = await newsapi.fetch(
        cfg.newsapi.api_key, query, from_date=from_date, max_results=max_results
    )
    click.echo(f"[NewsAPI] {len(newsapi_docs)} result(s)\n")
    _print_docs(newsapi_docs)


if __name__ == "__main__":
    main()
