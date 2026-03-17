"""freqpred CLI entry point."""
from __future__ import annotations

import asyncio

import click

from freqpred.config import load_config


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """freqpred — LLM-driven prediction market trading framework."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config()


@main.command()
@click.option("--strategy", required=True, help="Strategy class name to run.")
@click.option(
    "--mode",
    type=click.Choice(["paper", "live", "signal-only"]),
    default="paper",
    show_default=True,
    help="Trading mode.",
)
@click.pass_context
def run(ctx: click.Context, strategy: str, mode: str) -> None:
    """Run the freqpred trading loop."""
    config = ctx.obj["config"]
    click.echo(f"Starting freqpred | strategy={strategy} | mode={mode}")
    click.echo(f"Trading mode from config: {config.trading.mode}")
    click.echo("(Not yet implemented — scaffold only)")


@main.group()
def markets() -> None:
    """Manage and inspect Kalshi markets."""


@markets.command(name="list")
@click.option(
    "--category",
    default=None,
    help="Filter by category (e.g. politics, technology).",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="Skip writing results to the database.",
)
@click.pass_context
def markets_list(ctx: click.Context, category: str | None, no_db: bool) -> None:
    """Fetch active Kalshi markets and write them to the database."""
    config = ctx.obj["config"]
    asyncio.run(_markets_list(config, category=category, skip_db=no_db))


async def _markets_list(config: object, category: str | None, skip_db: bool) -> None:
    from freqpred.db import make_engine, make_session_factory
    from freqpred.markets.kalshi import KalshiClient
    from freqpred.markets.repository import upsert_markets
    import freqpred.signal.models  # noqa: F401 — register SignalRow with SQLAlchemy mapper
    import freqpred.rag.models  # noqa: F401 — register DocumentMarketLinkRow with SQLAlchemy mapper

    async with KalshiClient(
        api_key=config.kalshi.api_key,
        base_url=config.kalshi.base_url,
        private_key_path=config.kalshi.private_key_path,
    ) as client:
        click.echo(
            f"Fetching markets from Kalshi"
            + (f" [category={category}]" if category else "")
            + " ..."
        )
        market_list = await client.list_markets(category=category)

    if not market_list:
        click.echo("No markets found.")
        return

    # Write to DB
    if not skip_db and config.database.url:
        engine = make_engine(config.database.url)
        session_factory = make_session_factory(engine)
        async with session_factory() as session:
            written = await upsert_markets(session, market_list)
        await engine.dispose()
        click.echo(f"Wrote {written} market(s) to database.")

    # Print table to stdout
    header = f"{'TICKER':<30} {'CATEGORY':<14} {'BID':>6} {'ASK':>6} {'MID':>6}  QUESTION"
    click.echo(header)
    click.echo("-" * min(120, len(header) + 40))
    for m in market_list:
        question_preview = m.question[:60] + "…" if len(m.question) > 60 else m.question
        click.echo(
            f"{m.id:<30} {m.category:<14} "
            f"{m.yes_bid:>6.3f} {m.yes_ask:>6.3f} {m.mid_price:>6.3f}  {question_preview}"
        )
    click.echo(f"\nTotal: {len(market_list)} market(s)")


@main.group()
def ingestion() -> None:
    """Ingestion pipeline commands."""


@ingestion.command(name="run")
@click.option(
    "--category",
    default=None,
    help="Only process markets in this category (e.g. politics, economics).",
)
@click.option(
    "--limit",
    default=3,
    show_default=True,
    help="Maximum number of markets to process.",
)
@click.option(
    "--min-volume",
    default=0.0,
    show_default=True,
    help="Minimum 24h volume filter.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Generate catalysts but skip the news fetching step.",
)
@click.pass_context
def ingestion_run(
    ctx: click.Context,
    category: str | None,
    limit: int,
    min_volume: float,
    dry_run: bool,
) -> None:
    """Generate catalysts for selected markets then fetch targeted news.

    Pulls active markets from the DB, generates 3-5 targeted search queries
    per market (via Claude Haiku), then runs Tavily + NewsAPI + Reddit
    fetchers against those queries and stores results in the document store.

    Use --dry-run to generate and print catalysts without fetching news.
    """
    config = ctx.obj["config"]
    asyncio.run(_ingestion_run(config, category, limit, min_volume, dry_run))


async def _ingestion_run(
    config: object,
    category: str | None,
    limit: int,
    min_volume: float,
    dry_run: bool,
) -> None:
    import freqpred.ingestion.models  # noqa: F401
    import freqpred.signal.models     # noqa: F401
    import freqpred.rag.models        # noqa: F401

    import anthropic
    from datetime import UTC, datetime

    from sqlalchemy import select

    from freqpred.db import make_engine, make_session_factory
    from freqpred.ingestion.catalyst_generator import CatalystGenerationError, generate_catalysts
    from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow
    from freqpred.ingestion.store import RawDocument, upsert_document
    from freqpred.markets.models import MarketRow
    from freqpred.rag.embedder import VoyageEmbedder

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return

    anthropic_api_key = config.anthropic.api_key
    if not anthropic_api_key:
        click.echo("ERROR: ANTHROPIC_API_KEY not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)
    anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

    embedder = None
    if config.voyage.api_key:
        embedder = VoyageEmbedder(api_key=config.voyage.api_key, model=config.voyage.model)

    now = datetime.now(UTC)

    async with session_factory() as session:
        # Fetch non-closed markets from DB.
        stmt = select(MarketRow).where(MarketRow.close_time > now)
        if category:
            stmt = stmt.where(MarketRow.category == category)
        if min_volume > 0:
            stmt = stmt.where(MarketRow.volume_24h >= min_volume)
        stmt = stmt.order_by(MarketRow.volume_24h.desc()).limit(limit)

        result = await session.execute(stmt)
        market_rows = result.scalars().all()

    if not market_rows:
        click.echo("No markets found in DB. Run `freqpred markets list` first.")
        await engine.dispose()
        return

    click.echo(f"Processing {len(market_rows)} market(s)...")

    total_docs = 0

    for mrow in market_rows:
        from freqpred.markets.models import Market
        market = Market(
            id=mrow.id,
            platform=mrow.platform,
            question=mrow.question,
            category=mrow.category,
            close_time=mrow.close_time,
            yes_bid=mrow.yes_bid,
            yes_ask=mrow.yes_ask,
            mid_price=mrow.mid_price,
            volume_24h=mrow.volume_24h,
            open_interest=mrow.open_interest,
            last_fetched_at=mrow.last_fetched_at,
            price_updated_at=mrow.price_updated_at,
            metadata_fetched_at=mrow.metadata_fetched_at,
            current_signal_id=str(mrow.current_signal_id) if mrow.current_signal_id else None,
            metadata=dict(mrow.metadata_),
        )

        click.echo(f"\n{'─'*70}")
        click.echo(f"Market : {market.id}")
        click.echo(f"Question: {market.question}")
        click.echo(f"Category: {market.category}  |  Volume: {market.volume_24h:.0f}  |  Mid: {market.mid_price:.3f}")
        click.echo(f"Closes : {market.close_time.strftime('%Y-%m-%d')}")

        # Generate catalysts.
        async with session_factory() as session:
            try:
                run = await generate_catalysts(market, session, anthropic_client, embedder)
                await session.commit()

                # Fetch the query texts we just wrote.
                q_result = await session.execute(
                    select(CatalystQueryRow).where(CatalystQueryRow.run_id == run.id)
                )
                query_rows = q_result.scalars().all()
                queries = [q.query_text for q in query_rows]
            except CatalystGenerationError as exc:
                click.echo(f"  ✗ Catalyst generation failed: {exc}", err=True)
                continue

        click.echo(f"\nCatalysts (generation {run.generation}):")
        for i, q in enumerate(queries, 1):
            click.echo(f"  {i}. {q}")

        if dry_run:
            click.echo("  (dry-run: skipping news fetch)")
            continue

        if not queries:
            continue

        # Run fetchers against each catalyst query.
        click.echo("\nFetching news...")
        from freqpred.ingestion.fetchers import tavily as tavily_fetcher
        from freqpred.ingestion.fetchers import newsapi as newsapi_fetcher
        from freqpred.ingestion.fetchers import reddit as reddit_fetcher
        from datetime import timedelta

        raw_docs: list[RawDocument] = []

        for query in queries:
            # Tavily
            if config.tavily.api_key:
                tavily_docs = await tavily_fetcher.fetch(
                    api_key=config.tavily.api_key,
                    query=query,
                    max_results=5,
                )
                raw_docs.extend(tavily_docs)

            # NewsAPI
            if config.newsapi.api_key:
                newsapi_docs = await newsapi_fetcher.fetch(
                    api_key=config.newsapi.api_key,
                    query=query,
                    from_date=datetime.now(UTC) - timedelta(days=7),
                    max_results=5,
                )
                raw_docs.extend(newsapi_docs)

            # Reddit — use category to pick subreddits
            subreddits = _subreddits_for_category(market.category)
            reddit_docs = await reddit_fetcher.fetch(
                subreddits=subreddits,
                query=query,
                limit=20,
            )
            raw_docs.extend(reddit_docs)

        click.echo(f"  Fetched {len(raw_docs)} raw document(s) across all sources.")

        if not raw_docs or not embedder:
            if not embedder:
                click.echo("  (skipping store: VOYAGE_API_KEY not configured)")
            continue

        # Upsert into document store.
        stored = 0
        skipped = 0
        async with session_factory() as session:
            for raw_doc in raw_docs:
                raw_doc.category = market.category
                try:
                    doc = await upsert_document(session, embedder, raw_doc)
                    stored += 1
                except Exception as exc:
                    click.echo(f"  ✗ Store error: {exc}", err=True)
                    skipped += 1
            await session.commit()

        click.echo(f"  Stored {stored} doc(s) ({skipped} errors).")
        total_docs += stored

    click.echo(f"\n{'═'*70}")
    click.echo(f"Done. Total documents stored: {total_docs}")
    await engine.dispose()


def _subreddits_for_category(category: str) -> list[str]:
    _MAP = {
        "politics":    ["politics", "PoliticalDiscussion", "neutralpolitics"],
        "technology":  ["technology", "MachineLearning", "singularity"],
        "economics":   ["economics", "investing", "stocks"],
        "fintech":     ["investing", "wallstreetbets", "stocks", "fintech"],
        "sports":      ["sports"],
        "crypto":      ["CryptoCurrency", "Bitcoin"],
        "climate":     ["climate", "environment"],
    }
    return _MAP.get(category.lower(), ["news"])


@main.group()
def signal() -> None:
    """Signal pipeline commands."""


@signal.command(name="analyze")
@click.option("--market-id", required=True, help="Kalshi market ID to analyze.")
@click.pass_context
def signal_analyze(ctx: click.Context, market_id: str) -> None:
    """Run signal analysis for a specific market."""
    click.echo(f"Analyzing market: {market_id}")
    click.echo("(Not yet implemented — scaffold only)")
