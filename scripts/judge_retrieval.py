"""Measure retrieval quality using T101's extractor as a relevance oracle.

For each market, the rank-1 retrieved document is extracted against that
market's own question and labelled ``direct`` / ``contextual`` / ``none``. The
``none`` rate at rank 1 is the headline: a top-ranked document judged unrelated
to the question is a wasted retrieval slot, and no amount of prompt work
recovers it.

Two modes:

    live   — rank-1 relevance for recent signals, straight from the DB. Costs
             nothing: it reads labels the signal pipeline already wrote.
    pair   — judge two candidate documents per market from a TSV of
             ``market_id<TAB>doc_a<TAB>doc_b``, for before/after comparisons of
             a ranking change. Extracts only what is not already cached.

**Fallbacks are excluded from every count.** ``extract_for_documents`` returns a
``DocumentExtract`` for every document, and for bodies at or under
``MIN_EXTRACT_BODY_CHARS`` it returns one labelled ``contextual`` *without
calling the model* — the placeholder that keeps a short document in the prompt.
Counting those as judgments inflated a 2026-08-11 comparison from "9 improved,
0 regressed" to "12 improved, 3 regressed". They are not opinions; they are the
absence of one.

Usage:
    DATABASE_URL=... uv run python scripts/judge_retrieval.py live --days 7
    DATABASE_URL=... uv run python scripts/judge_retrieval.py pair --tsv /tmp/top1.tsv
"""
from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

import anthropic
import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, text

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.llm.client import LLMClient
from freqpred.llm.provider import maybe_openrouter_client
from freqpred.markets.models import Market, MarketRow
from freqpred.rag.models import Document, DocumentRow
from freqpred.signal.extractor import extract_for_documents

_RANK = {"none": 0, "contextual": 1, "direct": 2}


def _market_from_row(row: MarketRow) -> Market:
    return Market(
        id=row.id, platform=row.platform, question=row.question, category=row.category,
        status=row.status, result=row.result, close_time=row.close_time,
        yes_bid=row.yes_bid, yes_ask=row.yes_ask, mid_price=row.mid_price,
        last_price=row.last_price, volume_24h=row.volume_24h,
        open_interest=row.open_interest, yes_bid_size=row.yes_bid_size,
        yes_ask_size=row.yes_ask_size, last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at, metadata_fetched_at=row.metadata_fetched_at,
        metadata=dict(row.metadata_), open_time=row.open_time,
        series_ticker=row.series_ticker,
    )


def _doc_from_row(row: DocumentRow) -> Document:
    return Document(
        id=str(row.id), source_url=row.source_url, content_hash=row.content_hash,
        title=row.title, body=row.body, source_type=row.source_type,
        source_name=row.source_name, category=row.category, tags=list(row.tags),
        published_at=row.published_at, fetched_at=row.fetched_at,
        embedding=[], embedding_model="", summary=row.summary,
    )


_LIVE_SQL = """
WITH ranked AS (
  SELECT l.document_id, s.market_id,
         row_number() OVER (PARTITION BY l.signal_id ORDER BY l.relevance_score DESC) AS rnk
  FROM document_market_links l JOIN signals s ON s.id = l.signal_id
  WHERE l.signal_id IS NOT NULL
    AND s.created_at > now() - make_interval(days => :days)
)
SELECT coalesce(e.relevance, 'unjudged (short body — extraction skipped)') AS label,
       count(*) AS n
FROM ranked r
LEFT JOIN document_extracts e
       ON e.document_id = r.document_id AND e.market_id = r.market_id
WHERE r.rnk = 1
GROUP BY 1 ORDER BY 2 DESC
"""


async def _live(config, days: int) -> None:
    engine = make_engine(config.database.url)
    sf = make_session_factory(engine)
    try:
        async with sf() as s:
            rows = (await s.execute(text(_LIVE_SQL), {"days": days})).all()
    finally:
        await engine.dispose()

    judged = sum(n for label, n in rows if label in _RANK)
    click.echo(f"rank-1 relevance over the last {days} day(s):\n")
    for label, n in rows:
        pct = f"{100 * n / judged:5.1f}%" if label in _RANK else "     —"
        click.echo(f"  {label:46} {n:4}  {pct}")
    if judged:
        none_n = next((n for label, n in rows if label == "none"), 0)
        click.echo(f"\n  rank-1 judged unrelated: {none_n}/{judged} ({100 * none_n / judged:.0f}%)")
    click.echo("\n  (unjudged rows are short bodies extraction skips by design — "
               "excluded from the rate, not counted as relevant)")


async def _pair(config, tsv: Path) -> None:
    pairs = [r for r in csv.reader(tsv.open(), delimiter="\t") if len(r) == 3]
    engine = make_engine(config.database.url)
    sf = make_session_factory(engine)
    client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key), sf,
        default_strategy="system",
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
        openrouter_client=maybe_openrouter_client(config.openrouter.api_key),
    )

    improved = regressed = same = skipped = 0
    try:
        async with sf() as s:
            for market_id, a_id, b_id in pairs:
                mrow = (await s.execute(
                    select(MarketRow).where(MarketRow.id == market_id)
                )).scalar_one_or_none()
                if mrow is None:
                    skipped += 1
                    continue
                rows = {
                    str(r.id): r for r in (await s.execute(
                        select(DocumentRow).where(DocumentRow.id.in_([a_id, b_id]))
                    )).scalars()
                }
                if a_id not in rows or b_id not in rows:
                    skipped += 1
                    continue
                ex = await extract_for_documents(
                    s, client, _market_from_row(mrow),
                    [_doc_from_row(rows[a_id]), _doc_from_row(rows[b_id])],
                    model=config.anthropic.cheap_model,
                )
                a, b = ex.get(a_id), ex.get(b_id)
                # A fallback is the absence of a judgment, not a judgment of
                # "contextual" — counting it as one fabricates evidence.
                if a is None or b is None or a.fallback or b.fallback:
                    skipped += 1
                    click.echo(f"{market_id[:34]:34} SKIPPED (no judgment on one side)")
                    continue
                if _RANK[b.relevance] > _RANK[a.relevance]:
                    improved += 1
                elif _RANK[b.relevance] < _RANK[a.relevance]:
                    regressed += 1
                else:
                    same += 1
                click.echo(f"{market_id[:34]:34} A={a.relevance:11} B={b.relevance}")
    finally:
        await engine.dispose()

    judged = improved + regressed + same
    click.echo(f"\n{'=' * 70}\njudged pairs: {judged}   skipped (unjudgeable): {skipped}")
    click.echo(f"  B better : {improved}\n  A better : {regressed}\n  same     : {same}")
    if improved + regressed:
        from math import comb
        n, k = improved + regressed, max(improved, regressed)
        p = 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
        click.echo(f"  sign test over {n} discriminating pairs: two-sided p = {min(1.0, p):.4f}")


@click.group()
def cli() -> None:
    """Judge retrieval quality with T101 relevance labels."""


@cli.command()
@click.option("--days", type=int, default=7, show_default=True)
def live(days: int) -> None:
    """Rank-1 relevance for recent signals. Free — reads existing labels."""
    config = load_config()
    if not config.database.url:
        raise click.ClickException("DATABASE_URL not configured.")
    asyncio.run(_live(config, days))


@cli.command()
@click.option("--tsv", type=click.Path(exists=True, path_type=Path), required=True,
              help="market_id<TAB>doc_a<TAB>doc_b per line.")
def pair(tsv: Path) -> None:
    """Judge two candidate documents per market. Costs API calls when uncached."""
    config = load_config()
    if not config.database.url:
        raise click.ClickException("DATABASE_URL not configured.")
    asyncio.run(_pair(config, tsv))


if __name__ == "__main__":
    cli()
