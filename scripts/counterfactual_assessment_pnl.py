"""Counterfactual P&L: what if assessment sizing had not been applied?

For every closed position where the entry-signal assessment had a non-neutral
size_multiplier, this script reconstructs the base-Kelly contract count and
computes the P&L delta vs what actually happened.

With --scale-min / --scale-max you can reproject every trust_score onto a
different multiplier range and see what P&L would have been.

Usage:
    uv run python scripts/counterfactual_assessment_pnl.py
    uv run python scripts/counterfactual_assessment_pnl.py --mode live
    uv run python scripts/counterfactual_assessment_pnl.py --verdict size_down
    uv run python scripts/counterfactual_assessment_pnl.py --scale-min 0.5 --scale-max 1.5
"""
from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass

from sqlalchemy import select

from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.markets.models import MarketRow, PositionRow
from freqpred.metrics.assessment import _trust_score_to_multiplier
from freqpred.metrics.models import SignalAssessmentRow
from freqpred.rag.models import DocumentMarketLinkRow  # noqa: F401 — registers mapper
from freqpred.signal.models import SignalRow  # noqa: F401 — registers mapper


@dataclass
class Row:
    position_id: str
    market_id: str
    market_question: str
    direction: str
    mode: str
    entry_price: float
    exit_price: float
    actual_contracts: int
    actual_pnl: float
    actual_multiplier: float
    reprojected_multiplier: float
    verdict: str
    trust_score: float
    counterfactual_contracts: int
    counterfactual_pnl: float
    pnl_delta: float   # counterfactual - actual (positive = CF earned more)


def _reproject(
    trust_score: float,
    actual_multiplier: float,
    scale_min: float,
    scale_max: float,
) -> float:
    """Return multiplier under a different scale, or 1.0 (no-assessment baseline)."""
    if scale_min is None:
        return 1.0
    return _trust_score_to_multiplier(trust_score, scale_min=scale_min, scale_max=scale_max)


async def main(
    mode: str,
    verdict_filter: str | None,
    scale_min: float | None,
    scale_max: float | None,
) -> None:
    config = load_config()
    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    # Determine what we're comparing against
    using_custom_scale = scale_min is not None and scale_max is not None
    cf_label = (
        f"scale {scale_min:.1f}–{scale_max:.1f}x" if using_custom_scale
        else "no assessment (1.0x)"
    )

    async with session_factory() as session:
        result = await session.execute(
            select(
                PositionRow,
                SignalAssessmentRow,
                MarketRow.question,
            )
            .join(
                SignalAssessmentRow,
                SignalAssessmentRow.signal_id == PositionRow.signal_id,
            )
            .join(MarketRow, MarketRow.id == PositionRow.market_id)
            .where(
                PositionRow.status == "closed",
                PositionRow.mode == mode,
                PositionRow.pnl.isnot(None),
                PositionRow.exit_price.isnot(None),
            )
            .order_by(PositionRow.entry_time)
        )
        rows_raw = result.all()

    rows: list[Row] = []
    for pos, assessment, question in rows_raw:
        actual_mult = assessment.size_multiplier
        trust = assessment.trust_score

        if using_custom_scale:
            cf_mult = _trust_score_to_multiplier(
                trust, scale_min=scale_min, scale_max=scale_max
            )
            # Only skip if both are effectively equal — show all when rescaling
            if abs(cf_mult - actual_mult) < 1e-4:
                continue
        else:
            # Baseline: no assessment — only rows that were actually adjusted
            if abs(actual_mult - 1.0) < 1e-4:
                continue
            cf_mult = 1.0

        if verdict_filter and assessment.verdict != verdict_filter:
            continue

        # Reconstruct base-Kelly contracts from the actual multiplier, then
        # reapply the counterfactual multiplier.
        base_contracts = pos.contracts / actual_mult
        cf_contracts = max(1, math.floor(base_contracts * cf_mult))

        cf_pnl = (pos.exit_price - pos.entry_price) * cf_contracts
        actual_pnl = pos.pnl

        rows.append(Row(
            position_id=str(pos.id),
            market_id=pos.market_id,
            market_question=question,
            direction=pos.direction,
            mode=pos.mode,
            entry_price=pos.entry_price,
            exit_price=pos.exit_price,
            actual_contracts=pos.contracts,
            actual_pnl=actual_pnl,
            actual_multiplier=actual_mult,
            reprojected_multiplier=cf_mult,
            verdict=assessment.verdict,
            trust_score=trust,
            counterfactual_contracts=cf_contracts,
            counterfactual_pnl=cf_pnl,
            pnl_delta=cf_pnl - actual_pnl,
        ))

    if not rows:
        print(f"No qualifying closed {mode} positions found.")
        return

    # ── Summary ─────────────────────────────────────────────────────────────
    total_actual = sum(r.actual_pnl for r in rows)
    total_cf     = sum(r.counterfactual_pnl for r in rows)
    total_delta  = total_cf - total_actual
    sized_down   = [r for r in rows if r.verdict == "size_down"]
    sized_up     = [r for r in rows if r.verdict == "size_up"]

    scale_note = (
        f"actual scale 0.80–1.20x  →  counterfactual {scale_min:.2f}–{scale_max:.2f}x"
        if using_custom_scale
        else "actual assessment  →  no assessment (1.0x baseline)"
    )

    print(f"\n{'='*72}")
    print(f"  Counterfactual assessment P&L — mode={mode}")
    print(f"  {scale_note}")
    print(f"{'='*72}")
    print(f"  Positions in comparison                : {len(rows)}")
    print(f"    size_down (original verdict)         : {len(sized_down)}")
    print(f"    size_up   (original verdict)         : {len(sized_up)}")
    print(f"\n  Actual P&L                             : ${total_actual:+.2f}")
    print(f"  Counterfactual P&L ({cf_label:<14})  : ${total_cf:+.2f}")
    print(f"  Delta (CF minus actual)                : ${total_delta:+.2f}")
    if total_delta > 0:
        print(f"  Interpretation: counterfactual would have earned ${total_delta:.2f} MORE")
    else:
        print(f"  Interpretation: actual beat counterfactual by ${-total_delta:.2f}")

    # ── Per-verdict breakdown ────────────────────────────────────────────────
    for label, subset in [("size_down", sized_down), ("size_up", sized_up)]:
        if not subset:
            continue
        sub_actual = sum(r.actual_pnl for r in subset)
        sub_cf     = sum(r.counterfactual_pnl for r in subset)
        sub_delta  = sub_cf - sub_actual
        print(f"\n  [{label}] {len(subset)} positions")
        print(f"    Actual P&L        : ${sub_actual:+.2f}")
        print(f"    Counterfactual    : ${sub_cf:+.2f}")
        print(f"    Delta             : ${sub_delta:+.2f}")

    # ── Per-position detail ──────────────────────────────────────────────────
    mult_header = "Act.M  CF.M" if using_custom_scale else "Mult "
    print(f"\n{'─'*76}")
    print(
        f"  {'Market':<32} {'Dir':<4} {mult_header}  {'Act.C':>5} {'CF.C':>5}  "
        f"{'Act.P&L':>8} {'CF.P&L':>8} {'Delta':>8}"
    )
    print(f"{'─'*76}")
    for r in rows:
        question_short = r.market_question[:30].rstrip()
        if using_custom_scale:
            mult_str = f"{r.actual_multiplier:>4.2f}x {r.reprojected_multiplier:>4.2f}x"
        else:
            mult_str = f"{r.actual_multiplier:>5.2f}x"
        print(
            f"  {question_short:<32} {r.direction:<4} {mult_str}"
            f"  {r.actual_contracts:>5} {r.counterfactual_contracts:>5}"
            f"  ${r.actual_pnl:>7.2f} ${r.counterfactual_pnl:>7.2f} ${r.pnl_delta:>+7.2f}"
        )
    print(f"{'─'*76}")
    print(f"  {'TOTAL':<32}              "
          f"  {'':>5} {'':>5}"
          f"  ${total_actual:>7.2f} ${total_cf:>7.2f} ${total_delta:>+7.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--verdict", default=None, choices=["size_down", "size_up"],
                        help="Filter to only one verdict type")
    parser.add_argument("--scale-min", type=float, default=None,
                        help="Counterfactual scale_min (e.g. 0.5). Omit for no-assessment baseline.")
    parser.add_argument("--scale-max", type=float, default=None,
                        help="Counterfactual scale_max (e.g. 1.5). Omit for no-assessment baseline.")
    args = parser.parse_args()

    if (args.scale_min is None) != (args.scale_max is None):
        parser.error("--scale-min and --scale-max must be provided together")

    asyncio.run(main(args.mode, args.verdict, args.scale_min, args.scale_max))
