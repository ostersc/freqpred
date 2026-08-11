"""Eyeball T101 extraction on a handful of real markets before the full run.

This is the cheap sanity check that sits *before* the prompt-mode benchmark:
it runs live retrieval for a few markets, extracts each retrieved document
against that market's question, and prints the raw ``[:500]`` cut next to what
the extractor produced. What it is for:

* catching false ``relevance="none"`` labels — the failure mode that silently
  deletes evidence from the prompt;
* confirming boilerplate (app-download blocks, cookie notices, live-blog
  sign-offs) actually disappears;
* pricing the feature — calls per signal, cost per signal, added latency.

It does NOT measure whether signals get better. Only
``scripts/benchmark_signals.py --prompt-mode`` does that, and only against a
bank recorded with full document bodies.

Usage:
    # free — call volume and token projection, no API calls
    DATABASE_URL=... uv run python scripts/probe_extracts.py --markets 5 --dry-run

    # the real probe
    DATABASE_URL=... uv run python scripts/probe_extracts.py --markets 5

    # one specific market
    DATABASE_URL=... uv run python scripts/probe_extracts.py --market-id KXTRUMPSAY-26AUG12-MELA
"""
from __future__ import annotations

import asyncio
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anthropic
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
from freqpred.llm.client import LLMClient
from freqpred.llm.provider import maybe_openrouter_client
from freqpred.markets.models import Market, MarketRow
from freqpred.rag.embedder import make_embedder
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.rag.retriever import retrieve
from freqpred.signal.extractor import (
    MAX_EXTRACT_INPUT_CHARS,
    MIN_EXTRACT_BODY_CHARS,
    build_extract_prompt,
    extract_document,
    legacy_excerpt,
    needs_extraction,
)

_MAX_AGE_DAYS = 30

#: PoliticsEdgeStrategy's ``factbase_series_allowlist`` — the whole universe it
#: will ever trade, so probing anything else describes evidence no signal in
#: production is built from.
_POLITICS_SERIES = (
    "KXTRUMPSAY",
    "KXTRUMPSAYMONTH",
    "KXTRUMPSAYNICKNAME",
    "KXTRUMPSAYTRUMP",
)


def _market_from_row(row: MarketRow) -> Market:
    return Market(
        id=row.id,
        platform=row.platform,
        question=row.question,
        category=row.category,
        status=row.status,
        result=row.result,
        close_time=row.close_time,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
        mid_price=row.mid_price,
        last_price=row.last_price,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        yes_bid_size=row.yes_bid_size,
        yes_ask_size=row.yes_ask_size,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
        metadata=dict(row.metadata_),
        open_time=row.open_time,
        series_ticker=row.series_ticker,
    )


async def _pick_markets(
    session: Any,
    market_ids: list[str],
    series: list[str],
    count: int,
    seed: int,
) -> list[MarketRow]:
    """Markets retrieval can actually score, sampled deterministically.

    Mirrors ``topk_overlap.py``'s candidate filter: a market whose linked
    corpus is entirely outside the retriever's age window retrieves nothing
    and would probe nothing.
    """
    if market_ids:
        rows = list(
            (
                await session.execute(
                    select(MarketRow).where(MarketRow.id.in_(market_ids))
                )
            )
            .scalars()
            .all()
        )
        found = {r.id for r in rows}
        if missing := [m for m in market_ids if m not in found]:
            raise click.ClickException(f"Markets not found in DB: {', '.join(missing)}")
        # Preserve the order the caller asked for.
        by_id = {r.id: r for r in rows}
        return [by_id[m] for m in market_ids]

    cutoff = datetime.now(UTC) - timedelta(days=_MAX_AGE_DAYS)
    linked = (
        select(DocumentMarketLinkRow.market_id.label("market_id"))
        .join(DocumentRow, DocumentRow.id == DocumentMarketLinkRow.document_id)
        .where(DocumentRow.published_at >= cutoff)
        .group_by(DocumentMarketLinkRow.market_id)
        .having(func.count(func.distinct(DocumentMarketLinkRow.document_id)) >= 10)
        .subquery()
    )
    stmt = (
        select(MarketRow)
        .join(linked, MarketRow.id == linked.c.market_id)
        .where(MarketRow.status == "active")
        .order_by(MarketRow.id)
    )
    if series:
        # The probe should look at the universe a strategy actually trades;
        # PoliticsEdgeStrategy never leaves its factbase_series_allowlist.
        stmt = stmt.where(MarketRow.series_ticker.in_(series))
    rows = list((await session.execute(stmt)).scalars().all())
    if not rows:
        raise click.ClickException(
            "No active markets with a fresh linked corpus — nothing to probe."
        )
    random.Random(seed).shuffle(rows)
    return rows[:count]


def _preview(text: str, width: int) -> str:
    flat = text.replace("\n", " ").strip()
    return flat[:width] + ("…" if len(flat) > width else "")


async def _probe(
    config: Any,
    market_ids: list[str],
    series: list[str],
    count: int,
    top_k: int,
    seed: int,
    model: str,
    dry_run: bool,
    width: int,
) -> None:
    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)
    embedder = make_embedder(config.embedding)
    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        session_factory,
        default_strategy="system",
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
        # The extraction model may name an OpenRouter slug like any other
        # model setting; without this the transport lookup raises here.
        openrouter_client=maybe_openrouter_client(config.openrouter.api_key),
    )

    relevance_counts: Counter[str] = Counter()
    skipped_short = 0
    fallbacks = 0
    total_calls = 0
    prompt_chars = 0
    before_chars = 0
    after_chars = 0
    per_market_latency: list[float] = []

    try:
        async with session_factory() as session:
            market_rows = await _pick_markets(session, market_ids, series, count, seed)
            click.echo(
                f"Probing {len(market_rows)} market(s) — top_k={top_k}, model={model}, "
                f"embedder={embedder.model_name}\n"
            )

            for row in market_rows:
                market = _market_from_row(row)
                doc_pairs = await retrieve(
                    session,
                    embedder,
                    market.question,
                    market.id,
                    top_k=top_k,
                    max_age_days=_MAX_AGE_DAYS,
                )
                docs = [doc for doc, _ in doc_pairs]
                click.echo("═" * 100)
                click.echo(f"{market.id} — {market.question}")
                click.echo(f"  retrieved {len(docs)} documents")
                click.echo("═" * 100)

                if dry_run:
                    for doc in docs:
                        if not needs_extraction(doc):
                            skipped_short += 1
                            continue
                        total_calls += 1
                        prompt_chars += len(build_extract_prompt(market, doc))
                        click.echo(
                            f"  would extract {str(doc.id)[:8]} "
                            f"body={len(doc.body or '')} chars "
                            f"(sent {min(len(doc.body or ''), MAX_EXTRACT_INPUT_CHARS)}) "
                            f"— {_preview(doc.title, 60)}"
                        )
                    continue

                started = time.perf_counter()
                results = await asyncio.gather(
                    *(
                        extract_document(llm_client, market, doc, model=model)
                        for doc in docs
                    )
                )
                per_market_latency.append(time.perf_counter() - started)

                for i, (doc, result) in enumerate(zip(docs, results, strict=True), start=1):
                    raw = legacy_excerpt(doc)
                    before_chars += len(raw)
                    if not needs_extraction(doc):
                        skipped_short += 1
                        click.echo(
                            f"\n[{i}] {_preview(doc.title, 80)}\n"
                            f"    {doc.source_name} · body={len(doc.body or '')} chars "
                            f"· SHORT — rendered whole, no call"
                        )
                        after_chars += len(raw)
                        continue

                    relevance_counts[result.relevance] += 1
                    if result.fallback:
                        fallbacks += 1
                    after_chars += 0 if result.relevance == "none" else len(result.extract)

                    tag = "FALLBACK" if result.fallback else result.relevance.upper()
                    click.echo(
                        f"\n[{i}] {_preview(doc.title, 80)}\n"
                        f"    {doc.source_name} · body={len(doc.body or '')} chars · {tag}"
                    )
                    click.echo(f"    BEFORE: {_preview(raw, width)}")
                    if result.relevance == "none":
                        click.echo("    AFTER : (dropped from prompt)")
                    else:
                        click.echo(f"    AFTER : {_preview(result.extract, width)}")
                click.echo("")

    finally:
        await engine.dispose()

    click.echo("═" * 100)
    if dry_run:
        # 3.2 chars/token, calibrated against the 2026-08-11 probe (85,431
        # actual input tokens on 273,376 prompt chars). The usual 4.0
        # approximation under-projected that run by 25% — TV transcripts and
        # URL-dense article bodies tokenize worse than prose.
        est_input_tokens = int(prompt_chars / 3.2)
        click.echo(
            f"DRY RUN — {total_calls} extraction calls, {skipped_short} skipped "
            f"(body <= {MIN_EXTRACT_BODY_CHARS} chars)\n"
            f"  projected input ~{est_input_tokens:,} tokens "
            f"(+ ~{total_calls * 130:,} output tokens)"
        )
        return

    total_docs = sum(relevance_counts.values()) + skipped_short
    click.echo(
        f"SUMMARY over {total_docs} documents in {len(per_market_latency)} market(s)\n"
        f"  relevance: " + ", ".join(f"{k}={relevance_counts[k]}" for k in ("direct", "contextual", "none"))
        + f"\n  short-body skips: {skipped_short}   extraction fallbacks: {fallbacks}"
        + f"\n  evidence chars: {before_chars:,} before → {after_chars:,} after"
        + (
            f"\n  latency per market (all docs concurrent): "
            f"{min(per_market_latency):.1f}s–{max(per_market_latency):.1f}s"
            if per_market_latency
            else ""
        )
    )
    click.echo(
        "  cost: check `freqpred llm spend` — every call is audited as "
        "query_type=evidence_extraction"
    )


@click.command()
@click.option("--market-id", "market_ids", multiple=True,
              help="Probe these specific markets (repeatable), in the order given.")
@click.option("--series", "series", multiple=True, default=_POLITICS_SERIES,
              show_default=True,
              help="Restrict sampling to these series — defaults to the "
                   "PoliticsEdgeStrategy allowlist, the only markets it trades. "
                   "Ignored when --market-id is given.")
@click.option("--markets", "count", type=int, default=5, show_default=True,
              help="How many markets to sample when --market-id is not given.")
@click.option("--top-k", type=int, default=10, show_default=True,
              help="Documents retrieved per market (matches the signal pipeline).")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Seed for the market sample — same seed, same markets.")
@click.option("--model", default=None,
              help="Extraction model (default: config anthropic.cheap_model).")
@click.option("--dry-run", is_flag=True,
              help="Print call volume and a token projection without calling the API.")
@click.option("--width", type=int, default=300, show_default=True,
              help="Characters of before/after text to print per document.")
def main(
    market_ids: tuple[str, ...],
    series: tuple[str, ...],
    count: int,
    top_k: int,
    seed: int,
    model: str | None,
    dry_run: bool,
    width: int,
) -> None:
    config = load_config()
    if not config.database.url:
        raise click.ClickException("DATABASE_URL not configured.")
    asyncio.run(
        _probe(
            config,
            list(market_ids),
            list(series),
            count,
            top_k,
            seed,
            model or config.anthropic.cheap_model,
            dry_run,
            width,
        )
    )


if __name__ == "__main__":
    main()
