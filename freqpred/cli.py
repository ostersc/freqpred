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
def signal() -> None:
    """Signal pipeline commands."""


@signal.command(name="analyze")
@click.option("--market-id", required=True, help="Kalshi market ID to analyze.")
@click.pass_context
def signal_analyze(ctx: click.Context, market_id: str) -> None:
    """Run signal analysis for a specific market."""
    click.echo(f"Analyzing market: {market_id}")
    click.echo("(Not yet implemented — scaffold only)")
