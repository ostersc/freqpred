"""FastAPI application factory for the freqpred dashboard."""
from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .routes import router


def create_app(
    session_factory: async_sessionmaker[AsyncSession],
    redis_url: str | None = None,
    daily_cap_usd: float = 10.0,
) -> FastAPI:
    """Create and configure the dashboard FastAPI application.

    Args:
        session_factory: Async SQLAlchemy session factory.
        redis_url:        Redis URL for health check (optional).
        daily_cap_usd:    LLM daily spend cap from config, used in /api/llm/cost.
    """
    app = FastAPI(
        title="freqpred dashboard",
        description="Read-only JSON API for signals, positions, calibration, and cost.",
        version="1.0.0",
    )
    app.state.session_factory = session_factory
    app.state.redis_url = redis_url
    app.state.daily_cap_usd = daily_cap_usd

    app.include_router(router, prefix="/api")
    return app
