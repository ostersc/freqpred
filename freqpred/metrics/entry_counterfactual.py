"""Replay unentered markets through the strategy's real entry and exit rules.

The problem this fixes
----------------------
The weekly review's entry-gate section scores a blocked signal as
``hit_rate - market_ask_at_signal`` — buy at the signal-time ask, hold to
settlement. Both halves are wrong for `PoliticsEdgeStrategy`, and wrong in the
same direction, so the section overstates what the gates are refusing:

* **Entry.** ``order_types.entry == "limit"`` posts a resting bid at
  ``estimated_probability - min_edge`` (0.15 below the model's own estimate),
  good for ``limit_order_timeout_hours``. It is not a market order at the ask.
  Measured on signal-v11 markets with candle coverage: **31 of 62 (50%) would
  have filled within the 2h window.** Assuming a certain fill at the ask invents
  half the trades and prices the rest ~15c too high.
* **Exit.** Admitted markets were *actively managed* — signal exits, algo exits,
  and (historically) stoplosses. Blocked markets were scored as buy-and-hold.
  Comparing a managed book against an unmanaged one is not a like-for-like test
  of the gate.

This module replays a signal through the same rules the live path uses, over the
real candle price path, and reports what the trade would actually have returned.

What is modelled, and what is not
---------------------------------
Modelled, in the framework's exit priority order:

1. **Limit entry with timeout** — filled only if the traded side's ask reaches
   the limit before the timeout. Unfilled is a real, common outcome worth $0.
2. **Stoploss** — first candle whose exit price reaches ``entry + stoploss``.
   Currently inert: `PoliticsEdgeStrategy` set ``stoploss=-1.0`` on 2026-06-04,
   unreachable on a 0-1 price scale. The last stoploss exit in the whole dataset
   is 2026-06-03, the day before. Kept because the sweep's whole purpose is
   asking whether re-enabling it would pay.
3. **should_exit** — a later real signal on the same market flips direction with
   ``confidence >= min_confidence`` and the held side's estimated probability has
   dropped by more than ``min_edge``. Replayed from the actual stored signals, so
   this is not a model of the LLM, it is the LLM's real output.
4. **Resolution** — settles at 1.0 or 0.0 in the traded side's own space.

**Not modelled: `force_exit`** (the algo displacement/choppiness rule). It runs on
5-minute candles and the stored path here is hourly; a choppiness threshold tuned
for 5-minute ranges fires constantly on 60-minute ones, so applying it would
manufacture exits rather than approximate them. It accounted for 15 of 91 live
closed positions (~16%), so results here are best read as "everything except the
algo exit". Backfilling 1-minute candles and resampling to 5 minutes is the
upgrade path; `freqpred candles backfill --interval 1` already supports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class Candle:
    """One period of a market's book, in YES space."""

    end_ts: datetime
    yes_bid_high: float | None
    yes_bid_low: float | None
    yes_ask_high: float | None
    yes_ask_low: float | None

    def entry_cost_low(self, direction: str) -> float | None:
        """Cheapest price the traded side could have been BOUGHT at this period.

        YES buys at the yes_ask; NO buys at the no_ask = 1 - yes_bid. A limit
        entry fills when this reaches the limit price.
        """
        if direction == "YES":
            return self.yes_ask_low
        return None if self.yes_bid_high is None else 1.0 - self.yes_bid_high

    def exit_price_low(self, direction: str) -> float | None:
        """Worst price the traded side could have been SOLD at this period.

        YES sells into the yes_bid; NO sells into the no_bid = 1 - yes_ask.
        Returns None for an empty book — a 0.0 bid is "no bid", not a price, and
        treating it as tradeable stops out every position on illiquidity.
        """
        level = (
            self.yes_bid_low
            if direction == "YES"
            else (None if self.yes_ask_high is None else 1.0 - self.yes_ask_high)
        )
        return None if level is None or level <= 0.0 else level


@dataclass(frozen=True)
class LaterSignal:
    """A subsequent real signal on the same market, for should_exit replay."""

    created_at: datetime
    direction: str
    confidence: float
    estimated_probability: float


@dataclass(frozen=True)
class CounterfactualTrade:
    market_id: str
    signal_id: str
    direction: str
    limit_price: float
    filled: bool
    entry_price: float | None
    exit_reason: str
    exit_price: float | None
    pnl_per_contract: float
    hit: bool

    @property
    def traded(self) -> bool:
        return self.filled


def limit_entry_price(
    direction: str, estimated_probability: float, min_edge: float
) -> float:
    """The resting bid `order_manager` would post for this signal.

    Mirrors `OrderManager.submit`: YES posts at ``p_est - min_edge``, NO at
    ``(1 - p_est) - min_edge`` — both in the traded side's own price space.
    """
    base = (
        estimated_probability if direction == "YES" else 1.0 - estimated_probability
    )
    return round(base - min_edge, 4)


def simulate_trade(
    *,
    market_id: str,
    signal_id: str,
    direction: str,
    signal_time: datetime,
    estimated_probability: float,
    result: str,
    candles: list[Candle],
    later_signals: list[LaterSignal],
    min_edge: float,
    min_confidence: float,
    stoploss: float,
    entry_mode: str = "limit",
    market_ask_at_signal: float | None = None,
    limit_timeout_hours: float = 2.0,
) -> CounterfactualTrade:
    """Replay one signal through entry, exits, and settlement over a price path.

    Candles must be ordered by end_ts. Returns a trade whose `pnl_per_contract`
    is 0.0 when the limit never filled — an unfilled order is not a loss, but it
    is also not the profit the buy-and-hold approximation credits it with.
    """
    won = (direction == "YES" and result == "yes") or (
        direction == "NO" and result == "no"
    )
    limit_px = limit_entry_price(direction, estimated_probability, min_edge)

    def _unfilled(reason: str) -> CounterfactualTrade:
        return CounterfactualTrade(
            market_id=market_id,
            signal_id=signal_id,
            direction=direction,
            limit_price=limit_px,
            filled=False,
            entry_price=None,
            exit_reason=reason,
            exit_price=None,
            pnl_per_contract=0.0,
            hit=won,
        )

    # --- entry -----------------------------------------------------------
    if entry_mode == "limit":
        if limit_px <= 0.0:
            # The model is so bearish on its own side that the limit is at or
            # below zero; no order could ever be placed.
            return _unfilled("limit_unplaceable")
        deadline = signal_time + timedelta(hours=limit_timeout_hours)
        entry_price: float | None = None
        entry_idx = 0
        for i, candle in enumerate(candles):
            if candle.end_ts <= signal_time:
                continue
            if candle.end_ts > deadline:
                break
            cost = candle.entry_cost_low(direction)
            if cost is not None and cost <= limit_px:
                entry_price, entry_idx = limit_px, i
                break
        if entry_price is None:
            return _unfilled("unfilled")
    else:
        if market_ask_at_signal is None:
            return _unfilled("no_price")
        entry_price = market_ask_at_signal
        entry_idx = next(
            (i for i, c in enumerate(candles) if c.end_ts > signal_time), len(candles)
        )

    # --- exits, in the framework's priority order ------------------------
    for candle in candles[entry_idx + 1 :]:
        exit_px = candle.exit_price_low(direction)

        # 1. Hard stoploss. Inert while stoploss is -1.0, by design.
        if exit_px is not None and exit_px <= entry_price + stoploss:
            stop_px = entry_price + stoploss
            return CounterfactualTrade(
                market_id=market_id,
                signal_id=signal_id,
                direction=direction,
                limit_price=limit_px,
                filled=True,
                entry_price=entry_price,
                exit_reason="stoploss",
                exit_price=stop_px,
                pnl_per_contract=stop_px - entry_price,
                hit=won,
            )

        # 2. should_exit — a real later signal flipped direction with conviction.
        for later in later_signals:
            if not (candle.end_ts >= later.created_at > signal_time):
                continue
            if later.direction in ("SKIP", direction):
                continue
            if later.confidence < min_confidence:
                continue
            sign = 1.0 if direction == "YES" else -1.0
            drop = sign * (estimated_probability - later.estimated_probability)
            if drop > min_edge and exit_px is not None:
                return CounterfactualTrade(
                    market_id=market_id,
                    signal_id=signal_id,
                    direction=direction,
                    limit_price=limit_px,
                    filled=True,
                    entry_price=entry_price,
                    exit_reason="signal",
                    exit_price=exit_px,
                    pnl_per_contract=exit_px - entry_price,
                    hit=won,
                )

    # 3. Never exited early — settles at 1.0 or 0.0 in the traded side's space.
    settle = 1.0 if won else 0.0
    return CounterfactualTrade(
        market_id=market_id,
        signal_id=signal_id,
        direction=direction,
        limit_price=limit_px,
        filled=True,
        entry_price=entry_price,
        exit_reason="resolution",
        exit_price=settle,
        pnl_per_contract=settle - entry_price,
        hit=won,
    )


_CANDLES_SQL = text(
    """
    SELECT market_id, end_period_ts, yes_bid_high, yes_bid_low,
           yes_ask_high, yes_ask_low
    FROM market_candles
    WHERE period_interval = :interval
      AND market_id = ANY(:market_ids)
    ORDER BY market_id, end_period_ts
    """
)

_LATER_SIGNALS_SQL = text(
    """
    SELECT market_id, created_at, direction, confidence, estimated_probability
    FROM signals
    WHERE market_id = ANY(:market_ids)
      AND direction IN ('YES', 'NO', 'SKIP')
    ORDER BY market_id, created_at
    """
)


async def load_paths_and_signals(
    session: AsyncSession,
    market_ids: list[str],
    *,
    period_interval: int = 60,
) -> tuple[dict[str, list[Candle]], dict[str, list[LaterSignal]]]:
    """Bulk-load candle paths and later signals for a set of markets."""
    if not market_ids:
        return {}, {}

    candles: dict[str, list[Candle]] = {}
    for row in (
        await session.execute(
            _CANDLES_SQL, {"interval": period_interval, "market_ids": market_ids}
        )
    ).mappings():
        candles.setdefault(row["market_id"], []).append(
            Candle(
                end_ts=row["end_period_ts"],
                yes_bid_high=row["yes_bid_high"],
                yes_bid_low=row["yes_bid_low"],
                yes_ask_high=row["yes_ask_high"],
                yes_ask_low=row["yes_ask_low"],
            )
        )

    later: dict[str, list[LaterSignal]] = {}
    for row in (
        await session.execute(_LATER_SIGNALS_SQL, {"market_ids": market_ids})
    ).mappings():
        later.setdefault(row["market_id"], []).append(
            LaterSignal(
                created_at=row["created_at"],
                direction=row["direction"],
                confidence=row["confidence"],
                estimated_probability=row["estimated_probability"],
            )
        )
    return candles, later
