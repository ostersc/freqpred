"""Async database engine, session factory, and declarative base.

Usage:
    engine = make_engine(settings.database_url)
    SessionFactory = make_session_factory(engine)

    async with SessionFactory() as session:
        ...
"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# alembic.ini lives at the project root (one level above this package directory).
_ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def run_migrations(database_url: str) -> None:
    """Apply all pending Alembic migrations (upgrade head) at startup.

    Uses the project's alembic.ini and env.py so the same migration logic
    runs here as via ``uv run alembic upgrade head``. DATABASE_URL is set in
    the environment so env.py can read it.

    Safe to call on every startup — Alembic is a no-op when already up to date.
    """
    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig

    os.environ.setdefault("DATABASE_URL", database_url)
    cfg = AlembicConfig(str(_ALEMBIC_INI))
    alembic_command.upgrade(cfg, "head")


def make_engine(database_url: str):
    """Create an async SQLAlchemy engine."""
    return create_async_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the given engine."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager / dependency that yields a session."""
    async with session_factory() as session:
        yield session
