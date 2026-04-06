"""FastAPI application factory for the freqpred dashboard."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .routes import router

if TYPE_CHECKING:
    from freqpred.config import RiskConfig


def create_app(
    session_factory: async_sessionmaker[AsyncSession],
    daily_cap_usd: float = 10.0,
    risk_config: "RiskConfig | None" = None,
    bankroll_usd: float = 0.0,
) -> FastAPI:
    """Create and configure the dashboard FastAPI application.

    Args:
        session_factory:  Async SQLAlchemy session factory.
        daily_cap_usd:    LLM daily spend cap from config, used in /api/llm/cost.
        risk_config:      Risk engine config; used by /api/system/health circuit-breaker
                          fields.  Pass ``None`` to use default thresholds.
        bankroll_usd:     Trading bankroll; used by /api/system/health to compute loss %.

    Trading mode and active strategy are both discovered at request time from the
    ``run_state`` DB table written by ``freqpred run`` — no mode or strategy
    arguments needed here.
    """
    app = FastAPI(
        title="freqpred dashboard",
        description="JSON API for signals, positions, calibration, cost, and system health.",
        version="1.0.0",
    )
    app.state.session_factory = session_factory
    app.state.daily_cap_usd = daily_cap_usd
    app.state.risk_config = risk_config
    app.state.bankroll_usd = bankroll_usd
    app.state.started_at = datetime.now(UTC)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    app.include_router(router, prefix="/api")
    return app
