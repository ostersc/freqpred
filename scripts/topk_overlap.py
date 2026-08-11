"""Measure how much the retrieval top-K moves when the index changes.

Descriptive only. There is no outcome-based retrieval metric until T101 labels
relevance, so a low overlap is not by itself a failure — it just quantifies how
much the change actually moved retrieval.

Workflow around a reindex (T100):

    # 1. BEFORE the reindex — snapshot the current top-K for a sample of markets
    DATABASE_URL=... uv run python scripts/topk_overlap.py snapshot \
        --out /tmp/topk_before.json --sample 60

    # 2. Run the reindex (see scripts/reindex_embeddings.py --all --apply)

    # 3. AFTER — re-snapshot the SAME markets and questions
    DATABASE_URL=... uv run python scripts/topk_overlap.py snapshot \
        --out /tmp/topk_after.json --markets-from /tmp/topk_before.json

    # 4. Report
    uv run python scripts/topk_overlap.py compare \
        /tmp/topk_before.json /tmp/topk_after.json

``--markets-from`` replays the exact market/question set from an earlier snapshot,
so the two sides differ only by the index, never by which markets were sampled.
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

# Register all ORM models before any SQLAlchemy mapper is instantiated.
from sqlalchemy import func, select

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.markets.models import MarketRow
from freqpred.rag.embedder import make_embedder
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.rag.retriever import retrieve


async def _candidate_markets(
    session: Any,
    min_linked_docs: int,
    max_age_days: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Return (market_id, question) for markets retrieval can actually score.

    Counts only documents published inside the same ``max_age_days`` window
    ``retrieve()`` applies — a market whose linked corpus is all older than the
    cutoff retrieves nothing, and a snapshot of empty result sets would compare
    two empty sets as perfect overlap and quietly overstate stability.
    """
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    linked_counts = (
        select(
            DocumentMarketLinkRow.market_id.label("market_id"),
            func.count(func.distinct(DocumentMarketLinkRow.document_id)).label("n"),
        )
        .join(DocumentRow, DocumentRow.id == DocumentMarketLinkRow.document_id)
        .where(DocumentRow.published_at >= cutoff)
        .group_by(DocumentMarketLinkRow.market_id)
        .having(func.count(func.distinct(DocumentMarketLinkRow.document_id)) >= min_linked_docs)
        .subquery()
    )
    result = await session.execute(
        select(MarketRow.id, MarketRow.question)
        .join(linked_counts, MarketRow.id == linked_counts.c.market_id)
        .order_by(MarketRow.id)
    )
    candidates = [(str(mid), q) for mid, q in result.all()]

    # Deterministic order so a re-run without --markets-from still lines up.
    random.Random(seed).shuffle(candidates)
    return candidates


async def _snapshot(
    session_factory: Any,
    embedder: Any,
    out: Path,
    sample: int,
    top_k: int,
    min_linked_docs: int,
    max_age_days: int,
    seed: int,
    markets_from: Path | None,
) -> None:
    async with session_factory() as session:
        replay = markets_from is not None
        if replay:
            prior = json.loads(markets_from.read_text())  # type: ignore[union-attr]
            candidates = [(e["market_id"], e["question"]) for e in prior["entries"]]
            click.echo(f"Replaying {len(candidates)} markets from {markets_from}")
        else:
            candidates = await _candidate_markets(session, min_linked_docs, max_age_days, seed)
            click.echo(
                f"{len(candidates)} candidate markets with >= {min_linked_docs} linked docs "
                f"published within {max_age_days}d (seed={seed}); target sample {sample}"
            )

        if not candidates:
            click.echo("No markets matched — nothing to snapshot.", err=True)
            sys.exit(1)

        entries: list[dict[str, Any]] = []
        empty = 0
        for market_id, question in candidates:
            results = await retrieve(
                session, embedder, question, market_id, top_k=top_k, max_age_days=max_age_days
            )
            if not results and not replay:
                # Retrieves nothing → contributes no information, and two empty sets
                # would score as perfect overlap. Skip and draw the next candidate.
                empty += 1
                continue
            entries.append(
                {
                    "market_id": market_id,
                    "question": question,
                    "doc_ids": [str(doc.id) for doc, _ in results],
                    "scores": [round(score, 6) for _, score in results],
                }
            )
            if len(entries) % 10 == 0:
                click.echo(f"  {len(entries)}/{sample if not replay else len(candidates)}")
            if not replay and len(entries) >= sample:
                break

    if empty:
        click.echo(f"Skipped {empty} markets that retrieved zero documents.")
    if not replay and len(entries) < sample:
        click.echo(
            f"WARNING: only {len(entries)} markets retrieved documents "
            f"(asked for {sample}). Lower --min-linked-docs or raise --max-age-days.",
            err=True,
        )

    payload = {
        "captured_at": datetime.now(UTC).isoformat(),
        "embedding_model": embedder.model_name,
        "embedding_column": embedder.embedding_column,
        "max_embed_chars": embedder.max_embed_chars,
        "top_k": top_k,
        "max_age_days": max_age_days,
        "entries": entries,
    }
    out.write_text(json.dumps(payload, indent=2))
    click.echo(f"Wrote {len(entries)} entries to {out}")


def _report(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_by_id = {e["market_id"]: e for e in before["entries"]}
    after_by_id = {e["market_id"]: e for e in after["entries"]}
    shared = sorted(set(before_by_id) & set(after_by_id))

    if not shared:
        click.echo("No markets in common between the two snapshots.", err=True)
        sys.exit(1)

    click.echo(
        f"Before: {before['captured_at']}  model={before['embedding_model']} "
        f"max_embed_chars={before.get('max_embed_chars')}\n"
        f"After:  {after['captured_at']}  model={after['embedding_model']} "
        f"max_embed_chars={after.get('max_embed_chars')}\n"
        f"Markets compared: {len(shared)}  top_k={before['top_k']}\n"
    )

    overlaps: list[float] = []
    top1_same = 0
    unchanged = 0
    both_empty = 0
    for market_id in shared:
        b = before_by_id[market_id]["doc_ids"]
        a = after_by_id[market_id]["doc_ids"]
        denom = max(len(b), len(a))
        if denom == 0:
            # Two empty result sets carry no information about the index change;
            # scoring them 1.0 would inflate the overlap.
            both_empty += 1
            continue
        overlaps.append(len(set(b) & set(a)) / denom)
        if b and a and b[0] == a[0]:
            top1_same += 1
        if b == a:
            unchanged += 1

    if both_empty:
        click.echo(f"Excluded {both_empty} markets that retrieved nothing on both sides.\n")
    if not overlaps:
        click.echo("No market retrieved documents on either side — nothing to report.", err=True)
        sys.exit(1)

    overlaps.sort()
    n = len(overlaps)

    def pct(p: float) -> float:
        return overlaps[min(n - 1, int(p * n))]

    click.echo(
        f"Top-{before['top_k']} overlap (set intersection / k):\n"
        f"  mean   {sum(overlaps) / n:.3f}\n"
        f"  p10    {pct(0.10):.3f}\n"
        f"  p50    {pct(0.50):.3f}\n"
        f"  p90    {pct(0.90):.3f}\n"
        f"  min    {overlaps[0]:.3f}   max {overlaps[-1]:.3f}\n"
        f"\n"
        f"Identical top-1:    {top1_same}/{n} ({top1_same / n:.1%})\n"
        f"Identical full set: {unchanged}/{n} ({unchanged / n:.1%})\n"
        f"Fully disjoint:     {sum(1 for o in overlaps if o == 0.0)}/{n}"
    )


@click.group()
def cli() -> None:
    """Snapshot and compare retrieval top-K across an index change."""


@cli.command()
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output JSON path.")
@click.option("--sample", type=int, default=60, help="Markets to sample (issue T100 wants >= 50).")
@click.option("--top-k", type=int, default=10, help="Documents retrieved per market.")
@click.option(
    "--min-linked-docs",
    type=int,
    default=10,
    help="Skip markets with fewer linked docs inside the age window.",
)
@click.option(
    "--max-age-days",
    type=int,
    default=30,
    help="Retrieval age window; also filters which markets are eligible.",
)
@click.option("--seed", type=int, default=100, help="Sampling seed (deterministic).")
@click.option(
    "--markets-from",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Replay the exact market/question set from an earlier snapshot.",
)
def snapshot(
    out: Path,
    sample: int,
    top_k: int,
    min_linked_docs: int,
    max_age_days: int,
    seed: int,
    markets_from: Path | None,
) -> None:
    """Capture the current top-K document IDs for a sample of real markets."""
    config = load_config()
    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        sys.exit(1)

    embedder = make_embedder(config.embedding)
    engine = make_engine(config.database.url)
    factory = make_session_factory(engine)

    async def _run() -> None:
        try:
            await _snapshot(
                factory, embedder, out, sample, top_k,
                min_linked_docs, max_age_days, seed, markets_from,
            )
        finally:
            await engine.dispose()

    asyncio.run(_run())


@cli.command()
@click.argument("before", type=click.Path(exists=True, path_type=Path))
@click.argument("after", type=click.Path(exists=True, path_type=Path))
def compare(before: Path, after: Path) -> None:
    """Report top-K overlap between two snapshots."""
    _report(json.loads(before.read_text()), json.loads(after.read_text()))


if __name__ == "__main__":
    cli()
