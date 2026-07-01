"""Dump Kalshi market info + local position history for a given ticker.

Usage:
    uv run python scripts/market_info.py KXTRUMPSAY-26MAR23-COMM
"""
from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.markets.kalshi import KalshiClient


async def run(ticker: str) -> None:
    config = load_config()
    c = config.kalshi

    async with KalshiClient(
        api_key=c.api_key,
        base_url=c.base_url,
        private_key_path=c.private_key_path,
    ) as client:
        market = await client.get_market(ticker)

    print(f"{'─' * 60}")
    print(f"  {ticker}")
    print(f"{'─' * 60}")
    print(f"  status:     {market.status}")
    print(f"  result:     {market.result or '—'}")
    print(f"  yes_bid:    {market.yes_bid:.4f}   yes_ask: {market.yes_ask:.4f}")
    print(f"  mid_price:  {market.mid_price:.4f}")
    print(f"  close_time: {market.close_time}")
    print(f"  question:   {market.question[:120]}")
    print()

    engine = make_engine(config.database.url)
    sf = make_session_factory(engine)
    async with sf() as session:
        result = await session.execute(
            text("""
                SELECT direction, contracts, entry_price, exit_price,
                       exit_reason, pnl, pnl_pct, status,
                       entry_time::timestamptz, exit_time::timestamptz
                FROM positions
                WHERE market_id = :ticker
                ORDER BY entry_time
            """),
            {"ticker": ticker},
        )
        rows = result.fetchall()

    if not rows:
        print("  No positions on record.")
        return

    print(
        f"  {'dir':<4} {'qty':>5} {'entry':>7} {'exit':>7} {'reason':<26} "
        f"{'pnl':>8} {'pnl%':>7}  {'status':<10} entry_time"
    )
    print(f"  {'─'*4} {'─'*5} {'─'*7} {'─'*7} {'─'*26} {'─'*8} {'─'*7}  {'─'*10} {'─'*20}")
    for r in rows:
        exit_px  = f"{r.exit_price:.4f}" if r.exit_price is not None else "      —"
        pnl_str  = f"{r.pnl:+8.4f}" if r.pnl is not None else "       —"
        pnl_pct  = f"{r.pnl_pct*100:+6.1f}%" if r.pnl_pct is not None else "      —"
        print(
            f"  {r.direction:<4} {r.contracts:>5} {r.entry_price:>7.4f} {exit_px:>7} "
            f"{(r.exit_reason or ''):<26} {pnl_str} {pnl_pct}  "
            f"{r.status:<10} {str(r.entry_time)[:19]}"
        )
    await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/market_info.py <TICKER>")
        sys.exit(1)
    asyncio.run(run(sys.argv[1]))
