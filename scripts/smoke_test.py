"""Manual smoke test: pick a random Kalshi market, generate catalysts, fetch
targeted news, then run the full signal pipeline to produce a probability estimate.

Usage:
    uv run python scripts/smoke_test.py
    uv run python scripts/smoke_test.py --category politics
    uv run python scripts/smoke_test.py --max-results 5
    uv run python scripts/smoke_test.py --dry-run   # catalysts only, no fetching/signal
"""
from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timezone
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
from freqpred.llm.client import LLMClient
from freqpred.markets.kalshi import KalshiClient
from freqpred.markets.models import Market
from freqpred.rag.models import Document
from freqpred.signal.llm import SYSTEM_PROMPT, build_prompt, parse_signal_response

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

_SIGNAL_MODEL = "claude-sonnet-4-6"


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
    """Build a minimal stub AsyncSession for components that need a DB session.

    In this smoke test we don't persist to DB — we just want the LLM calls
    and their return values. The session stubs satisfy the audit-logging
    interface without a real DB connection.
    """
    session = AsyncMock()

    first_result = MagicMock()
    first_result.scalar_one_or_none.return_value = None

    call_count = 0

    async def execute_side_effect(*_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return first_result
        return MagicMock()

    session.execute.side_effect = execute_side_effect
    session.flush = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    session.add.side_effect = lambda row: (
        setattr(row, "id", 999),
        setattr(row, "created_at", datetime.now(timezone.utc)),
    )

    return session


def _make_stub_session_factory() -> MagicMock:
    """Return an async session factory that always yields a fresh stub session."""
    factory = MagicMock()
    stub = _make_stub_session()
    factory.return_value.__aenter__ = AsyncMock(return_value=stub)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory


def _raw_to_document(raw: RawDocument, category: str) -> Document:
    """Convert a RawDocument to a Document domain object for signal analysis.

    Uses a synthetic UUID and zero embedding since we're bypassing the
    vector store in this smoke test.
    """
    return Document(
        id=str(uuid.uuid4()),
        source_url=raw.source_url,
        content_hash="",
        title=raw.title or "",
        body=raw.body,
        source_type=raw.source_type,
        source_name=raw.source_name,
        category=category,
        tags=[],
        published_at=raw.published_at,
        fetched_at=datetime.now(timezone.utc),
        embedding=[],
        embedding_model="none",
        summary=None,
    )


async def _run_signal_analysis(
    market: Market,
    docs: list[Document],
    anthropic_client: anthropic.AsyncAnthropic,
) -> None:
    """Call Claude Sonnet with fetched docs as evidence and print the signal."""
    click.echo(f"\n{'═' * 70}")
    click.echo(f"  SIGNAL ANALYSIS  ({_SIGNAL_MODEL})")
    click.echo(f"{'═' * 70}")
    click.echo(f"  Evidence docs: {len(docs)}")

    if not docs:
        click.echo("  No evidence to analyze — skipping signal.")
        return

    prompt = build_prompt(market, docs)
    session_factory = _make_stub_session_factory()
    llm_client = LLMClient(
        anthropic_client,
        session_factory,
        default_strategy="smoke_test",
        prompt_version="signal-v1",
    )

    click.echo("  Calling LLM…")
    try:
        response = await llm_client.complete(
            prompt,
            _SIGNAL_MODEL,
            query_type="market_analysis",
            system=SYSTEM_PROMPT,
            market_id=market.id,
            max_tokens=1024,
        )
    except Exception as exc:
        click.echo(f"  LLM call failed: {exc}", err=True)
        return

    parsed = parse_signal_response(response.content)
    if parsed is None:
        click.echo("  Failed to parse LLM response:")
        click.echo(f"  {response.content[:400]}")
        return

    edge = parsed["probability"] - market.mid_price
    edge_pct = f"+{edge:.1%}" if edge >= 0 else f"{edge:.1%}"

    click.echo()
    click.echo(f"  Probability : {parsed['probability']:.1%}")
    click.echo(f"  Market mid  : {market.mid_price:.1%}")
    click.echo(f"  Edge        : {edge_pct}")
    click.echo(f"  Confidence  : {parsed['confidence']:.1%}")
    click.echo(f"  Direction   : {parsed['direction']}")
    click.echo(f"  Reasoning   : {parsed['reasoning']}")
    click.echo(f"  Cost        : ${response.cost_usd:.4f}  ({response.latency_ms}ms)")
    click.echo()


@click.command()
@click.option("--market-id", default=None, help="Specific Kalshi market ID to analyze (e.g. kxpresmention-djt26mar17c).")
@click.option("--category", default=None, help="Kalshi category to filter when picking a random market.")
@click.option("--max-results", default=5, show_default=True, help="Results per source per catalyst.")
@click.option("--dry-run", is_flag=True, default=False, help="Generate catalysts only, skip fetching and signal.")
def main(market_id: str | None, category: str | None, max_results: int, dry_run: bool) -> None:
    asyncio.run(_run(market_id=market_id, category=category, max_results=max_results, dry_run=dry_run))


async def _run(market_id: str | None, category: str | None, max_results: int, dry_run: bool) -> None:
    cfg = load_config()

    # ── 1. Resolve market ────────────────────────────────────────────────────
    async with KalshiClient(
        api_key=cfg.kalshi.api_key,
        base_url=cfg.kalshi.base_url,
        private_key_path=cfg.kalshi.private_key_path,
    ) as client:
        if market_id:
            market_id = market_id.upper()
            click.echo(f"Fetching market {market_id}…")
            try:
                market = await client.get_market(market_id)
            except Exception as exc:
                click.echo(f"Market '{market_id}' not found: {exc}", err=True)
                return
        else:
            click.echo("Fetching markets from Kalshi…")
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
    click.echo("Generating catalyst queries (Claude Haiku)…")
    anthropic_client = anthropic.AsyncAnthropic(api_key=cfg.anthropic.api_key)
    stub_session = _make_stub_session()
    llm_client = LLMClient(
        anthropic_client,
        _make_stub_session_factory(),
        default_strategy="smoke_test",
        prompt_version="catalyst-v1",
    )

    try:
        run = await generate_catalysts(market, stub_session, llm_client, embedder=None)
    except Exception as exc:
        click.echo(f"Catalyst generation failed: {exc}", err=True)
        return

    from freqpred.ingestion.models import CatalystQueryRow
    added_objects = [call.args[0] for call in stub_session.add.call_args_list]
    catalyst_queries = [obj.query_text for obj in added_objects if isinstance(obj, CatalystQueryRow)]

    click.echo(f"\nCatalysts generated (generation {run.generation}):")
    for i, q in enumerate(catalyst_queries, 1):
        click.echo(f"  {i}. {q}")

    if dry_run or not catalyst_queries:
        click.echo("\n(dry-run: skipping news fetch and signal analysis)")
        return

    # ── 3. Fetch news against each catalyst query ────────────────────────────
    subreddits = _CATEGORY_SUBREDDITS.get(market.category, ["news", "worldnews"])
    subs_display = ", ".join(f"r/{s}" for s in subreddits)

    all_raw_docs: list[RawDocument] = []

    for query in catalyst_queries:
        click.echo(f"\n{'─' * 70}")
        click.echo(f'Catalyst: "{query}"')
        click.echo(f"{'─' * 70}")

        # Tavily
        click.echo("\n[Tavily]")
        tavily_docs = await tavily.fetch(cfg.tavily.api_key, query, max_results=max_results)
        click.echo(f"{len(tavily_docs)} result(s)\n")
        _print_docs(tavily_docs)
        all_raw_docs.extend(tavily_docs)

        # NewsAPI — omit from_date on free tier (restricts to past 24h regardless)
        click.echo("[NewsAPI]")
        newsapi_docs = await newsapi.fetch(
            cfg.newsapi.api_key, query, max_results=max_results
        )
        click.echo(f"{len(newsapi_docs)} result(s)\n")
        _print_docs(newsapi_docs)
        all_raw_docs.extend(newsapi_docs)

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
        all_raw_docs.extend(reddit_docs)

    # ── 4. Run signal analysis on all fetched evidence ───────────────────────
    # Deduplicate by URL, then convert to Document objects for the signal prompt
    seen_urls: set[str] = set()
    unique_raw: list[RawDocument] = []
    for raw in all_raw_docs:
        if raw.source_url not in seen_urls:
            seen_urls.add(raw.source_url)
            unique_raw.append(raw)

    docs = [_raw_to_document(raw, market.category) for raw in unique_raw]
    await _run_signal_analysis(market, docs, anthropic_client)


if __name__ == "__main__":
    main()
