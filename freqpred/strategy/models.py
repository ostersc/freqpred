"""ORM model for persisting runtime strategy config overrides."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Integer, VARCHAR
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from freqpred.db import Base


class RuntimeConfigOverrideRow(Base):
    """Persists mutable StrategyConfig overrides applied at runtime via the
    dashboard PUT /api/strategy/config endpoint.

    One row per strategy name.  The ``overrides`` JSONB column stores only the
    fields that were explicitly changed — it is merged on top of the strategy's
    defaults at dashboard startup and whenever the run-loop reloads overrides.
    """

    __tablename__ = "runtime_config_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False, unique=True)
    overrides: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="now()"
    )
