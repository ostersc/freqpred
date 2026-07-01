"""Backfill settlement_sources into Market.metadata for past (non-active) markets.

T88 added settlement_sources capture going forward via list_markets(), which only
re-syncs active markets on its regular poll. Markets that are already settled/
closed/etc. will never be touched by that poll again, so their stored metadata
predates the field unless backfilled here.

Scoped to the small "recently past" status buckets (settled, closed, determined,
inactive, initialized) rather than the much larger "finalized" bucket (millions
of old markets) — see GH issue #88 discussion.

Usage (dry run — shows what would change):
    uv run python scripts/backfill_settlement_sources.py

Apply the fix:
    uv run python scripts/backfill_settlement_sources.py --apply
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

# Register all ORM models before any SQLAlchemy mapper is instantiated.
sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.cli import _kalshi_credentials
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.markets.kalshi import KalshiClient
from freqpred.markets.models import MarketRow

_BACKFILL_STATUSES = ("settled", "closed", "determined", "inactive", "initialized")


async def _find_targets(session: AsyncSession) -> list[dict]:
    """Return markets in scope that are missing settlement_sources in metadata."""
    result = await session.execute(
        select(MarketRow.id, MarketRow.metadata_)
        .where(MarketRow.status.in_(_BACKFILL_STATUSES))
    )
    rows = result.all()

    targets = []
    for market_id, metadata in rows:
        metadata = metadata or {}
        if "settlement_sources" in metadata:
            continue
        event_ticker = metadata.get("event_ticker")
        if not event_ticker:
            continue
        targets.append(
            {"market_id": market_id, "event_ticker": event_ticker, "metadata": metadata}
        )
    return targets


async def _apply(
    session: AsyncSession,
    targets: list[dict],
    sources_by_event: dict[str, list[dict[str, str]]],
) -> int:
    """Write settlement_sources into metadata for each target market. Returns count updated.

    The ``metadata`` column is plain ``json`` (not ``jsonb``), so it doesn't
    support the ``||`` concatenation operator — the merged dict is built in
    Python and written back whole.
    """
    updated = 0
    for target in targets:
        sources = sources_by_event.get(target["event_ticker"])
        if sources is None:
            continue
        merged = {**target["metadata"], "settlement_sources": sources}
        await session.execute(
            update(MarketRow)
            .where(MarketRow.id == target["market_id"])
            .values(metadata_=merged)
        )
        updated += 1
    await session.commit()
    return updated


async def _run(apply: bool) -> None:
    config = load_config()
    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        sys.exit(1)

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    async with session_factory() as session:
        targets = await _find_targets(session)

    if not targets:
        click.echo(
            f"No markets in {_BACKFILL_STATUSES} are missing settlement_sources."
        )
        return

    event_tickers = sorted({t["event_ticker"] for t in targets})
    click.echo(
        f"\n{len(targets)} market(s) across {len(event_tickers)} event(s) "
        f"missing settlement_sources in statuses {_BACKFILL_STATUSES}.\n"
    )

    api_key, private_key_path = _kalshi_credentials(config)
    async with KalshiClient(
        api_key=api_key,
        base_url=config.kalshi.base_url,
        private_key_path=private_key_path,
    ) as kalshi_client:
        events = await kalshi_client.get_events_by_tickers(event_tickers)

    sources_by_event = {
        ev.event_ticker: [s.model_dump() for s in ev.settlement_sources]
        for ev in events
    }
    missing_events = set(event_tickers) - set(sources_by_event)
    if missing_events:
        click.echo(
            f"Warning: {len(missing_events)} event ticker(s) not returned by "
            f"Kalshi (purged/404) — their markets will be skipped: "
            f"{sorted(missing_events)[:10]}{'...' if len(missing_events) > 10 else ''}\n"
        )

    with_sources = sum(1 for s in sources_by_event.values() if s)
    click.echo(
        f"Fetched {len(sources_by_event)} event(s); "
        f"{with_sources} have a non-empty settlement_sources list.\n"
    )

    if not apply:
        click.echo("Re-run with --apply to write these changes to the database.")
        return

    async with session_factory() as session:
        updated = await _apply(session, targets, sources_by_event)

    click.echo(f"\n{updated} market row(s) updated.")

    await engine.dispose()


@click.command()
@click.option("--apply", is_flag=True, default=False, help="Write changes to DB (default: dry run).")
def main(apply: bool) -> None:
    """Backfill settlement_sources for past markets that list_markets() will never re-sync."""
    asyncio.run(_run(apply))


if __name__ == "__main__":
    main()
