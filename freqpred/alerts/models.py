"""ORM model for persisting run-loop state."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import VARCHAR, Integer
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from freqpred.db import Base

VALID_RUN_STATES = ("running", "paused", "stopped")


class RunStateRow(Base):
    """Singleton table that persists the run-loop state across restarts.

    At most one row exists (id=1). Use ``run_state.get_run_state`` /
    ``run_state.set_run_state`` to read/write it.

    The risk-window fields (drawdown baseline, daily-loss acknowledgement) are
    partitioned per trading mode: paper and live bankrolls evolve on entirely
    different scales, so a baseline captured in one mode is always wrong for
    the other. Loop-control fields (state, cb_*, strategy_name, mode) stay
    global — only one mode runs per process.
    """

    __tablename__ = "run_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    state: Mapped[str] = mapped_column(VARCHAR(20), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    drawdown_reset_at_paper: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True, default=None
    )
    drawdown_reset_bankroll_paper: Mapped[float | None] = mapped_column(
        nullable=True, default=None
    )
    drawdown_reset_at_live: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True, default=None
    )
    drawdown_reset_bankroll_live: Mapped[float | None] = mapped_column(
        nullable=True, default=None
    )
    strategy_name: Mapped[str | None] = mapped_column(
        VARCHAR(255), nullable=True, default=None
    )
    mode: Mapped[str | None] = mapped_column(VARCHAR(20), nullable=True, default=None)
    cb_active: Mapped[bool] = mapped_column(nullable=False, default=False)
    cb_reason: Mapped[str | None] = mapped_column(nullable=True, default=None)
    daily_loss_ack_at_paper: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True, default=None
    )
    daily_loss_ack_at_live: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True, default=None
    )
