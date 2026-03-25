"""IPredictionStrategy abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from freqpred.markets.models import Market, Position
    from freqpred.signal.models import Signal
    from freqpred.strategy.config import StrategyConfig


class IPredictionStrategy(ABC):
    """Base class for all freqpred trading strategies.

    Users subclass this, implement the required methods, and point
    freqpred at their strategy file via config.

    Example::

        class MyPoliticsStrategy(IPredictionStrategy):
            config = StrategyConfig(
                name="my_politics_v1",
                min_edge=0.18,
                min_confidence=0.72,
                kelly_fraction=0.25,
                categories=["politics"],
                ...
            )

            def should_trade(self, signal, market):
                return signal.edge >= self.config.min_edge

            def position_size(self, signal, bankroll):
                kelly = signal.edge / (1 - signal.estimated_probability)
                return bankroll * kelly * self.config.kelly_fraction
    """

    config: StrategyConfig

    @abstractmethod
    def should_trade(self, signal: Signal, market: Market) -> bool:
        """Return True if this signal warrants opening a position."""
        ...

    @abstractmethod
    def position_size(self, signal: Signal, bankroll: float) -> float:
        """Return dollar amount to risk on this position."""
        ...

    def is_market_interesting(self, market: Market) -> bool:
        """Return True if this strategy wants the ingestion pipeline to monitor this market.

        The Market Selector calls this on all registered strategies. A market is
        selected for catalyst generation if *any* strategy returns True.

        Default implementation applies StrategyConfig filters (category, volume,
        days-to-close). Override for custom market selection logic.
        """
        now = datetime.now(tz=timezone.utc)
        days_to_close = (market.close_time - now).total_seconds() / 86400
        if self.config.min_mid_price is not None and market.mid_price < self.config.min_mid_price:
            return False
        if self.config.max_mid_price is not None and market.mid_price > self.config.max_mid_price:
            return False
        return (
            (not self.config.categories or market.category in self.config.categories)
            and market.volume_24h >= self.config.min_volume_24h
            and self.config.min_days_to_close
                <= days_to_close
                <= self.config.max_days_to_close
        )

    def filter_markets(self, markets: list[Market]) -> list[Market]:
        """Pre-filter markets before signal analysis.

        Delegates to is_market_interesting. Override is_market_interesting for
        custom logic rather than overriding this method.
        """
        return [m for m in markets if self.is_market_interesting(m)]

    def should_exit(self, position: Position, signal: Signal, market: Market) -> bool:
        """Signal-driven exit.

        Called after LLM re-analysis of a market with an open position.
        Default: exit if new signal direction != position direction AND
        confidence >= min_confidence.
        """
        return (
            signal.direction not in ("SKIP", position.direction)
            and signal.confidence >= self.config.min_confidence
        )

    def force_exit(self, position: "Position", market: "Market") -> str | None:
        """Signal-independent exit hook. Called on every PositionMonitor tick.

        Return a non-None exit reason string to force exit regardless of signal
        state. Return None to proceed with normal exit logic.

        Use this when a strategy needs to exit on its own initiative — e.g.
        immediately after entry in demo/test strategies, or on a time-based rule
        that doesn't depend on LLM re-analysis.

        Default: None (no forced exit).
        """
        return None

    def custom_exit(self, position: "Position", signal: "Signal", market: "Market") -> str | None:
        """Signal-informed custom exit hook.

        Return a non-None exit reason tag to force exit.
        Return None to proceed with normal logic.
        Default: None (no custom exit).
        """
        return None

    async def synthesize_signal(
        self, session: "AsyncSession", market: "Market"
    ) -> "Signal | None":
        """Optional hook: generate and persist a synthetic signal when the pipeline has no docs.

        Called by the run loop as a fallback when ``pipeline.analyze()`` returns ``None``.
        The returned Signal must already be committed to the DB (FK constraint on positions).

        Default: returns ``None`` (no-op for all production strategies).
        Only override this in testing/demo strategies.
        """
        return None

    async def on_position_opened(
        self,
        position: "Position",
        market: "Market",
        session_factory: Any,
    ) -> None:
        """Optional async hook called immediately after a position is opened.

        Called by the signal loop after ``order_manager.submit()`` returns a
        non-None Position. Use this to trigger immediate exits (e.g. in demo/test
        strategies) or to record supplemental data.

        Default: no-op.
        """

    def on_order_failed(self, market: "Market") -> None:
        """Optional hook called when order placement fails for a market.

        Called by the signal loop when ``order_manager.submit()`` returns None
        after an exchange-side error (e.g. market_closed, market_not_found).
        Allows strategies to immediately abandon a bad market rather than
        waiting for a timeout.

        Default: no-op.
        """

    def on_resolution(self, position: Position) -> None:
        """Optional hook called when a market resolves."""
