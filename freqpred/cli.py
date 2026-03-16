"""freqpred CLI entry point."""
from __future__ import annotations

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
@click.pass_context
def markets_list(ctx: click.Context) -> None:
    """List active Kalshi markets."""
    click.echo("(Not yet implemented — scaffold only)")


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
