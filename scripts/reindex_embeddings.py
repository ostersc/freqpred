"""Re-embed all documents whose embedding_model differs from the configured embedder.

Idempotent — safe to re-run. Only documents with a mismatched embedding_model
are re-processed; already-correct documents are skipped.

Run this after applying migration 0046 (vector(768)) and switching the config
to embedding.backend=ollama / embedding.model=nomic-embed-text.

Usage:
    # Dry run — shows how many docs would be re-embedded
    DATABASE_URL=... uv run python scripts/reindex_embeddings.py

    # Apply
    DATABASE_URL=... uv run python scripts/reindex_embeddings.py --apply
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

# Register all ORM models before any SQLAlchemy mapper is instantiated.
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models        # noqa: F401
import freqpred.markets.models    # noqa: F401
import freqpred.rag.models        # noqa: F401
import freqpred.signal.models     # noqa: F401

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.rag.embedder import make_embedder
from freqpred.rag.models import DocumentRow

_BATCH_SIZE = 50


async def _reindex(
    session_factory: object,
    embedder: object,
    apply: bool,
) -> None:
    from freqpred.rag.embedder import LocalEmbedder, OllamaEmbedder  # noqa: PLC0415

    engine_session = session_factory
    embed_col: str = embedder.embedding_column  # type: ignore[union-attr]
    target_model: str = embedder.model_name  # type: ignore[union-attr]
    embed_attr = getattr(DocumentRow, embed_col)

    async with engine_session() as session:  # type: ignore[operator]
        # For Ollama: find docs where embedding_768 is NULL (never reindexed for this backend).
        # For sentence_transformers: find docs where embedding_model differs (switched back from Ollama).
        if embed_col == "embedding_768":
            filter_cond = embed_attr.is_(None)
        else:
            filter_cond = DocumentRow.embedding_model != target_model

        result = await session.execute(
            select(DocumentRow.id)
            .where(filter_cond)
            .order_by(DocumentRow.fetched_at.desc())
        )
        ids = [row[0] for row in result.all()]

    total = len(ids)
    click.echo(
        f"Backend: {embed_col!r}  Model: {target_model!r}\n"
        f"Documents needing re-embed: {total}"
    )

    if not apply:
        click.echo("Dry run — pass --apply to commit changes.")
        return

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
                embed_src = (row.summary or row.body)[: embedder.max_embed_chars]  # type: ignore[union-attr]
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

    click.echo(f"Done. {done} re-embedded, {skipped} skipped.")


@click.command()
@click.option("--apply", is_flag=True, default=False, help="Commit changes (default: dry run).")
def main(apply: bool) -> None:
    config = load_config()

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        sys.exit(1)

    embedder = make_embedder(config.embedding)

    from freqpred.db import make_engine, make_session_factory  # noqa: PLC0415
    engine = make_engine(config.database.url)
    factory = make_session_factory(engine)

    async def _run() -> None:
        try:
            await _reindex(factory, embedder, apply)
        finally:
            await engine.dispose()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
