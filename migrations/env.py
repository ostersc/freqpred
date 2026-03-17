"""Alembic environment — async SQLAlchemy engine."""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Import Base and all ORM models so Alembic can detect schema changes.
# The order matters: db.Base must be imported before any model that uses it.
from freqpred.db import Base  # noqa: F401
import freqpred.markets.models  # noqa: F401 — registers MarketRow, PositionRow
import freqpred.signal.models  # noqa: F401 — registers SignalRow
import freqpred.rag.models  # noqa: F401 — registers DocumentRow, DocumentMarketLinkRow
import freqpred.llm.models  # noqa: F401 — registers LLMQueryRow
import freqpred.ingestion.models  # noqa: F401 — registers CatalystRunRow, CatalystQueryRow

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it before running alembic commands."
        )
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout, no DB connection)."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode against an async engine."""
    connectable = create_async_engine(get_database_url())
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
