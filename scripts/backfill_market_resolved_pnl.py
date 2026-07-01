"""Backfill exit_price / pnl / pnl_pct for positions that were incorrectly closed
at mid_price instead of the correct contract payout (1.0 or 0.0).

Affected rows: exit_reason='market_resolved' where the market has a known result
and exit_price ≠ correct payout.

Usage (dry run — shows what would change):
    uv run python scripts/backfill_market_resolved_pnl.py

Apply the fix:
    uv run python scripts/backfill_market_resolved_pnl.py --apply
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
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.markets.models import MarketRow, PositionRow


async def _find_affected(session: AsyncSession) -> list[dict]:
    """Return rows that need correction.

    Each dict contains the current and corrected values so the caller can
    print a preview and/or apply the update.
    """
    result = await session.execute(
        select(
            PositionRow.id,
            PositionRow.market_id,
            PositionRow.direction,
            PositionRow.contracts,
            PositionRow.entry_price,
            PositionRow.entry_fee_usd,
            PositionRow.exit_price,
            PositionRow.pnl,
            PositionRow.pnl_pct,
            PositionRow.mode,
            MarketRow.result,
            MarketRow.question,
        )
        .join(MarketRow, PositionRow.market_id == MarketRow.id)
        .where(
            PositionRow.status == "closed",
            PositionRow.exit_reason == "market_resolved",
            MarketRow.result.is_not(None),
        )
    )
    rows = result.all()

    affected = []
    for row in rows:
        wins = row.direction.upper() == row.result.upper()
        correct_exit_price = 1.0 if wins else 0.0

        # Skip rows already at the correct price.
        if abs((row.exit_price or 0.0) - correct_exit_price) < 1e-6:
            continue

        fee = row.entry_fee_usd or 0.0
        gross_pnl = (correct_exit_price - row.entry_price) * row.contracts
        correct_pnl = round(gross_pnl - fee, 4)
        cost_basis = row.entry_price * row.contracts + fee
        correct_pnl_pct = round(correct_pnl / cost_basis, 6) if cost_basis else 0.0

        affected.append({
            "id": str(row.id),
            "market_id": row.market_id,
            "question": row.question,
            "direction": row.direction,
            "result": row.result,
            "contracts": row.contracts,
            "entry_price": row.entry_price,
            "mode": row.mode,
            # Current (wrong)
            "old_exit_price": row.exit_price,
            "old_pnl": row.pnl,
            # Corrected
            "new_exit_price": correct_exit_price,
            "new_pnl": correct_pnl,
            "new_pnl_pct": correct_pnl_pct,
        })

    return affected


async def _apply(session: AsyncSession, affected: list[dict]) -> None:
    """Write corrected values to DB for all affected rows."""
    for row in affected:
        await session.execute(
            update(PositionRow)
            .where(PositionRow.id == row["id"])
            .values(
                exit_price=row["new_exit_price"],
                pnl=row["new_pnl"],
                pnl_pct=row["new_pnl_pct"],
            )
        )
    await session.commit()


async def _run(apply: bool) -> None:
    config = load_config()
    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        sys.exit(1)

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    async with session_factory() as session:
        affected = await _find_affected(session)

    if not affected:
        click.echo("No affected rows found — all market_resolved exits already have correct payout prices.")
        return

    # Print preview table.
    click.echo(f"\n{'DRY RUN' if not apply else "APPLYING FIX"} — {len(affected)} position(s) to correct:\n")
    fmt = "{:36s}  {:4s}  {:6s}  {:3s}  {:>8s}  {:>8s}  {:>9s}  {:>9s}  {:8s}"
    click.echo(fmt.format(
        "position_id", "dir", "result", "qty",
        "entry", "exit(old)", "pnl(old)", "pnl(new)", "mode",
    ))
    click.echo("-" * 100)
    for r in affected:
        click.echo(fmt.format(
            r["id"],
            r["direction"],
            r["result"],
            str(r["contracts"]),
            f"{r['entry_price']:.4f}",
            f"{r['old_exit_price']:.4f}" if r["old_exit_price"] is not None else "None",
            f"{r['old_pnl']:+.4f}" if r["old_pnl"] is not None else "None",
            f"{r['new_pnl']:+.4f}",
            r["mode"],
        ))

    pnl_delta = sum(r["new_pnl"] - (r["old_pnl"] or 0.0) for r in affected)
    click.echo(f"\nTotal P&L correction: {pnl_delta:+.4f}")

    if not apply:
        click.echo("\nRe-run with --apply to write these changes to the database.")
        return

    async with session_factory() as session:
        await _apply(session, affected)

    click.echo(f"\n{len(affected)} row(s) updated.")

    await engine.dispose()


@click.command()
@click.option("--apply", is_flag=True, default=False, help="Write corrections to DB (default: dry run).")
def main(apply: bool) -> None:
    """Backfill market_resolved positions closed at wrong exit_price (mid instead of 1.0/0.0)."""
    asyncio.run(_run(apply))


if __name__ == "__main__":
    main()
