"""Kalshi price-grid alignment for computed limit prices.

Kalshi rejects limit orders whose price is not on the market's tick grid.
The grid is published per market as the ``price_ranges`` array
(``{start, end, step}`` bands in fixed-point dollars) and varies by the
market's ``price_level_structure``: as of the July 2026 rollout there are
ten structures, with center-band ticks of 1c, 0.5c or 0.2c and edge-band
ticks down to 0.1c (edge bands are $0.00-$0.10 and $0.90-$1.00).

We deliberately snap to a **whole cent** rather than consuming
``price_ranges``.  A whole cent is a valid price under every structure
Kalshi publishes, because every center and edge tick (1c, 0.5c, 0.2c,
0.1c) divides evenly into 1c.  That makes whole-cent alignment correct
without per-market grid data, at the cost of not exploiting sub-cent
ticks on markets that offer them.

Only *computed* prices need this.  Prices taken from the order book
(bids, asks) are already on that market's grid by construction, and
re-rounding them would push a sell below the bid or a buy above the ask
on any market using sub-cent ticks — so callers must not use this helper
on book-sourced prices.
"""
from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_UP, Decimal

# One whole cent, on the 0-1 dollar scale used throughout freqpred.
TICK = Decimal("0.01")


def round_to_tick(price: float, *, action: str) -> float:
    """Snap a computed limit price onto Kalshi's price grid.

    Rounds away from crossing the intended price: a ``"buy"`` rounds down
    so the order never pays more than the caller asked for (preserving the
    edge the price was derived from), a ``"sell"`` rounds up so it never
    accepts less.

    Returns a value on the 0-1 dollar scale.  A price that rounds to zero
    or below is returned as ``0.0`` for the caller to reject — this helper
    does not clamp into a tradeable range, because a computed price at or
    below one tick means the edge that produced it has evaporated.
    """
    if action not in ("buy", "sell"):
        raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")

    rounding = ROUND_DOWN if action == "buy" else ROUND_UP
    # Decimal(str(...)) avoids the binary-float artefacts that make e.g.
    # 0.685 quantize as 0.68 in one direction and 0.69 in the other.
    snapped = (Decimal(str(price)) / TICK).quantize(Decimal("1"), rounding=rounding) * TICK

    if snapped <= 0:
        return 0.0
    if snapped >= 1:
        return 1.0
    return float(snapped)
