"""Re-embed documents using the configured embedder.

Embed text is derived by ``freqpred.ingestion.store.derive_embed_text`` — the same
function ``upsert_document`` uses — so a reindex can never disagree with live
ingestion about what a document's vector represents.

By default only documents that need it are re-processed (missing embedding for the
ollama backend, mismatched embedding_model otherwise); already-correct documents are
skipped, so the default mode is idempotent and safe to re-run.

``--all`` re-embeds every document regardless. Use it after a change to how embed text
is derived (e.g. T100, which switched from summary-derived to body-derived vectors) —
the existing vectors are present and model-matched, so the default filter would find
nothing to do.

**A full reindex leaves the index heterogeneous while it runs**: unprocessed rows still
hold vectors from the old regime while processed rows hold the new one, so every
retrieval is comparing two representations inside one cosine space. Pause the signal
loop, or run in a low-activity window and exclude signals created inside the printed
start/end window from any before/after comparison.

Usage:
    # Dry run — shows how many docs would be re-embedded
    DATABASE_URL=... uv run python scripts/reindex_embeddings.py

    # Apply
    DATABASE_URL=... uv run python scripts/reindex_embeddings.py --apply

    # Full reindex of every document (after an embed-text derivation change)
    DATABASE_URL=... uv run python scripts/reindex_embeddings.py --all --apply
"""
from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

# Register all ORM models before any SQLAlchemy mapper is instantiated.
from sqlalchemy import select, update

import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.ingestion.store import derive_embed_text
from freqpred.rag.embedder import make_embedder
from freqpred.rag.models import DocumentRow

_BATCH_SIZE = 50


async def _reindex(
    session_factory: object,
    embedder: object,
    apply: bool,
    reindex_all: bool = False,
) -> None:

    engine_session = session_factory
    embed_col: str = embedder.embedding_column  # type: ignore[union-attr]
    target_model: str = embedder.model_name  # type: ignore[union-attr]
    max_chars: int = embedder.max_embed_chars  # type: ignore[union-attr]
    embed_attr = getattr(DocumentRow, embed_col)

    async with engine_session() as session:  # type: ignore[operator]
        stmt = select(DocumentRow.id).order_by(DocumentRow.fetched_at.desc())
        if not reindex_all:
            # For Ollama: find docs where embedding_768 is NULL (never reindexed for this backend).
            # For sentence_transformers: find docs where embedding_model differs (switched back from Ollama).
            if embed_col == "embedding_768":
                filter_cond = embed_attr.is_(None)
            else:
                filter_cond = DocumentRow.embedding_model != target_model
            stmt = stmt.where(filter_cond)

        result = await session.execute(stmt)
        ids = [row[0] for row in result.all()]

    total = len(ids)
    scope = "ALL documents (--all)" if reindex_all else "documents needing re-embed"
    click.echo(
        f"Backend: {embed_col!r}  Model: {target_model!r}  max_embed_chars: {max_chars}\n"
        f"Scope: {scope}\n"
        f"Documents to re-embed: {total}"
    )

    if not apply:
        click.echo("Dry run — pass --apply to commit changes.")
        return

    started_at = datetime.now(UTC)
    click.echo(f"Reindex started: {started_at.isoformat()}")
    if reindex_all:
        click.echo(
            "  NOTE: the index is heterogeneous until this finishes — signals created "
            "in this window mix old and new representations. Exclude them from any "
            "before/after retrieval comparison."
        )

    done = 0
    skipped = 0
    for batch_start in range(0, total, _BATCH_SIZE):
        batch_ids = ids[batch_start : batch_start + _BATCH_SIZE]

        async with engine_session() as session:  # type: ignore[operator]
            rows_result = await session.execute(
                select(DocumentRow).where(DocumentRow.id.in_(batch_ids))
            )
            rows: list[DocumentRow] = list(rows_result.scalars().all())

            for row in rows:
                # Shared with ingestion's upsert_document — never duplicate this choice.
                embed_src = derive_embed_text(row.body, max_chars)
                try:
                    embedding = await embedder.embed_text(embed_src)  # type: ignore[union-attr]
                except Exception as exc:
                    click.echo(
                        f"  SKIP {row.id} ({row.source_url[:60]}): {exc}",
                        err=True,
                    )
                    skipped += 1
                    continue
                update_vals = {embed_col: embedding, "embedding_model": target_model}
                await session.execute(
                    update(DocumentRow)
                    .where(DocumentRow.id == row.id)
                    .values(**update_vals)
                )
                done += 1

            await session.commit()

        click.echo(f"  {done}/{total} re-embedded, {skipped} skipped")

    finished_at = datetime.now(UTC)
    click.echo(
        f"Done. {done} re-embedded, {skipped} skipped.\n"
        f"Reindex window: {started_at.isoformat()} → {finished_at.isoformat()} "
        f"({(finished_at - started_at).total_seconds() / 3600:.2f}h)"
    )


@click.command()
@click.option("--apply", is_flag=True, default=False, help="Commit changes (default: dry run).")
@click.option(
    "--all",
    "reindex_all",
    is_flag=True,
    default=False,
    help="Re-embed every document, not just those missing or model-mismatched. "
         "Use after a change to how embed text is derived.",
)
def main(apply: bool, reindex_all: bool) -> None:
    config = load_config()

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        sys.exit(1)

    embedder = make_embedder(config.embedding)

    engine = make_engine(config.database.url)
    factory = make_session_factory(engine)

    async def _run() -> None:
        try:
            await _reindex(factory, embedder, apply, reindex_all)
        finally:
            await engine.dispose()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
