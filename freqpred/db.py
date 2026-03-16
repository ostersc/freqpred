"""Async database engine, session factory, and declarative base.

Usage:
    engine = make_engine(settings.database_url)
    SessionFactory = make_session_factory(engine)

    async with SessionFactory() as session:
        ...
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


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
