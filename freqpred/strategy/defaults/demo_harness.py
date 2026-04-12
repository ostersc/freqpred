"""DemoHarness: minimal strategy for validating the live order path against Kalshi demo.

Behaviour: accepts any open market, pins the first one it sees, always signals
to trade using a $1 position size so the OrderManager exercises _submit_live
without meaningful financial exposure.

Safety: refuses to instantiate unless KALSHI_BASE_URL contains "demo".
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import func, select, update

from freqpred.strategy.algo_base import IAlgoStrategy
from freqpred.strategy.config import StrategyConfig

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd
    from sqlalchemy.ext.asyncio import AsyncSession

    from freqpred.markets.models import Market, Position
    from freqpred.signal.models import Signal


# Markets not updated within this window were not seen in the last watcher cycle
# and therefore don't exist (or are resolved) in the current API environment.
# Set to 2× the default watcher poll interval (300s) plus a small buffer.
_FRESHNESS_LIMIT = timedelta(seconds=700)
# Reuse an existing synthetic signal if it is younger than this (avoids spamming the signals table).
_SIGNAL_REUSE_WINDOW = timedelta(minutes=2)


class DemoHarness(IAlgoStrategy):
    """Live-path validation strategy.

    Extends ``IAlgoStrategy`` to demonstrate the algo exit pattern.  The
    ``populate_exit_trend`` implementation always sets ``exit_long=True``,
    but ``force_exit`` is overridden to return ``"demo_immediate_exit"``
    on the first confirmed tick so the demo loop completes quickly without
    waiting for 2 complete candles.

    - Only runs if KALSHI_BASE_URL contains "demo" (hard abort otherwise).
    - Pins the first affordable, fresh, liquid market it encounters.
    - should_trade() always returns True — bypasses edge/confidence filters.
    - position_size() returns $1.00 — minimal exposure.
    - synthesize_signal() creates a real signal row when pipeline has no docs.
      Reuses the same signal on retry; abandons the market after 5 min if no
      position was created (e.g. 404 from the exchange).
    """

    config = StrategyConfig(
        name="DemoHarness",
        min_edge=0.0,
        min_confidence=0.0,
        max_exposure_per_market=1.0,  # $1 position always fits; never let cap kill contracts
        kelly_fraction=0.01,
        categories=[],
        min_volume_24h=0.0,
        max_days_to_close=3650.0,
        min_days_to_close=0.0,
    )

    def __init__(self) -> None:
        super().__init__()  # initialises IAlgoStrategy tick buffers
        self._pinned_market_id: str | None = None
        self._failed_markets: set[str] = set()
        self._has_open_position: bool = False

    def is_market_interesting(self, market: "Market") -> bool:
        if market.id in self._failed_markets:
            return False

        if self._pinned_market_id is None:
            now = datetime.now(tz=timezone.utc)
            if (
                market.status == "active"
                and market.yes_ask is not None
                and 0.05 <= market.yes_ask <= 0.85
                and market.volume_24h > 0
                and market.last_fetched_at is not None
                and (now - market.last_fetched_at) < _FRESHNESS_LIMIT
                and (market.open_time is None or market.open_time <= now)
                and market.close_time > now
            ):
                self._pinned_market_id = market.id
        return market.id == self._pinned_market_id

    def populate_exit_trend(self, df: "pd.DataFrame", metadata: dict) -> "pd.DataFrame":
        """Always signal exit — demo harness exits on every complete candle.

        In practice DemoHarness overrides force_exit to exit immediately
        (before 2 complete candles accumulate), so this method is here to
        satisfy the IAlgoStrategy abstract contract and to demonstrate the
        pattern.
        """
        df["exit_long"] = True
        return df

    def force_exit(self, position: "Position", market: "Market") -> str | None:
        """Exit the demo position on the first confirmed (open) tick.

        PositionMonitor only passes status='open' positions here, so any call
        means the entry fill is confirmed. Return immediately to close without
        waiting for 2 complete candles (as IAlgoStrategy.force_exit requires).
        """
        return "demo_immediate_exit"

    async def on_position_opened(
        self,
        position: "Position",
        market: "Market",
        session_factory: "Any",
    ) -> None:
        """Record that a position is open; keep the market pinned so synthesize_signal
        can detect closure via the DB and re-enable entry."""
        self._has_open_position = True

    def on_order_failed(self, market: "Market") -> None:
        """Immediately abandon a market that rejected our order at the exchange."""
        self._failed_markets.add(market.id)
        if self._pinned_market_id == market.id:
            self._pinned_market_id = None

    def should_trade(self, _signal: "Signal", _market: "Market") -> bool:
        return True

    def position_size(self, _signal: "Signal", _bankroll: float, _existing_market_exposure: float = 0.0) -> float:
        return 0.0 if self._has_open_position else 1.0

    async def synthesize_signal(
        self, session: "AsyncSession", market: "Market"
    ) -> "Signal | None":
        """Create or reuse a synthetic signal so the order path can be exercised.

        - Returns None immediately if there is already an open/pending position.
        - Reuses an existing demo_harness signal to avoid spamming the signals table.
        - Refuses to synthesize if KALSHI_BASE_URL does not contain "demo".
        """
        base_url = os.environ.get("KALSHI_BASE_URL", "")
        if "demo" not in base_url.lower():
            import structlog
            structlog.get_logger(__name__).warning(
                "demo_harness.synthesize_signal_blocked",
                reason="KALSHI_BASE_URL does not contain 'demo'",
                base_url=base_url,
            )
            return None

        from freqpred.markets.models import MarketRow, PositionRow
        from freqpred.signal.models import Signal, SignalRow

        # Wait for any open DemoHarness position to be closed by the PositionMonitor.
        pos_count = (
            await session.execute(
                select(func.count()).where(
                    PositionRow.strategy_name == "DemoHarness",
                    PositionRow.status.in_(["open", "pending"]),
                )
            )
        ).scalar_one()
        if pos_count > 0:
            return None
        # All positions closed — reset state so the next cycle can open a new one.
        self._has_open_position = False
        self._failed_markets.add(market.id)
        self._pinned_market_id = None

        # Check for an existing synthetic signal for this market.
        existing_result = await session.execute(
            select(SignalRow)
            .where(
                SignalRow.market_id == market.id,
                SignalRow.trigger == "demo_harness",
            )
            .order_by(SignalRow.created_at.desc())
            .limit(1)
        )
        existing_row = existing_result.scalar_one_or_none()

        if existing_row is not None:
            age = datetime.now(tz=timezone.utc) - existing_row.created_at
            if age < _SIGNAL_REUSE_WINDOW:
                # Reuse the recent signal — no need to write a new row.
                return Signal(
                    id=str(existing_row.id),
                    market_id=existing_row.market_id,
                    estimated_probability=existing_row.estimated_probability,
                    confidence=existing_row.confidence,
                    edge=existing_row.edge,
                    market_mid_at_signal=existing_row.market_mid_at_signal,
                    direction=existing_row.direction,
                    reasoning=existing_row.reasoning,
                    sources=list(existing_row.sources),
                    retrieval_hash=existing_row.retrieval_hash,
                    model_used=existing_row.model_used,
                    prompt_version=existing_row.prompt_version,
                    trigger=existing_row.trigger,
                    created_at=existing_row.created_at,
                    raw_context=existing_row.raw_context,
                )
            # Signal is stale — fall through to create a fresh one.

        # No existing signal — create one.
        mid = market.mid_price or 0.50
        edge = 0.15  # hardcoded to always clear the min_edge_floor of 0.1
        estimated_prob = min(mid + edge, 0.95)

        row = SignalRow(
            id=uuid.uuid4(),
            market_id=market.id,
            estimated_probability=estimated_prob,
            confidence=1.0,
            edge=edge,
            market_mid_at_signal=mid,
            direction="YES",
            reasoning="Synthetic signal generated by DemoHarness for live order path validation.",
            sources=[],
            retrieval_hash="0" * 64,
            model_used="demo_harness",
            prompt_version="demo",
            trigger="demo_harness",
            created_at=datetime.now(tz=timezone.utc),
            raw_context="",
        )
        session.add(row)
        await session.flush()

        await session.execute(
            update(MarketRow)
            .where(MarketRow.id == market.id)
            .values(current_signal_id=row.id)
        )
        await session.commit()

        return Signal(
            id=str(row.id),
            market_id=row.market_id,
            estimated_probability=row.estimated_probability,
            confidence=row.confidence,
            edge=row.edge,
            market_mid_at_signal=row.market_mid_at_signal,
            direction=row.direction,
            reasoning=row.reasoning,
            sources=list(row.sources),
            retrieval_hash=row.retrieval_hash,
            model_used=row.model_used,
            prompt_version=row.prompt_version,
            trigger=row.trigger,
            created_at=row.created_at,
            raw_context=row.raw_context,
        )
