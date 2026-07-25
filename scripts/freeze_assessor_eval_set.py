"""Freeze a reusable evaluation set for the assessor audit — no LLM calls.

Why this exists
---------------
Every assessor run so far drew its own sample: `_pick_sample` shuffles all
assessed signals under SEED=42, and that pool grows as new assessments land. The
consequence was measured on 2026-07-24 — the IDENTICAL production package
(assessment-v6 on opus-4-7) scored corr +0.529 on one draw and +0.246 on the
next. Run-to-run differences were sample composition, not package quality, which
makes the adoption metric nearly unusable at n=30.

A frozen set fixes both halves of that:
  * the SAMPLE stops moving, so two runs are comparable;
  * the rendered PAYLOAD is stored verbatim, so harness changes (the liquidity
    mask, the crossed-book mask) cannot silently alter the inputs underneath a
    cached score. Each payload carries a hash; a fixture whose hash no longer
    matches is stale and must be re-scored rather than reused.

Why it is free
--------------
The same premise as freqpred.replay.recorder: everything needed is already in
the DB. Selection, PIT payload construction, and hashing are pure computation,
and the `current` arm's responses are harvested from `llm_queries` where they
were already paid for. Only genuinely new packages cost money.

Caching the current arm is sound because the judge is effectively deterministic:
7 signals scored 2-3x by the identical package returned identical trust_scores
(one 0.05 deviation across ~9 repeat pairs). On frozen inputs, re-running it
would buy nothing.

Stratification
--------------
KXTRUMPSAY only (109 of the last 111 production positions), signal-v11 only (the
current signal prompt cohort — profit edge is strongly version-dependent), and
balanced on direction. Direction balance is the point: NO earns +7.3pp over the
price paid while YES loses 7.6pp across 8,410 resolved signals, making it the
strongest per-signal discriminator in the data, and the natural mix (22 YES / 8
NO in the last draw) was far too thin on NO to resolve it. NO is the binding
constraint at 38 available, so the set takes all of them and band-matches YES.

Accepted trade: a direction-balanced set no longer mirrors production's natural
mix, so absolute AUC is less representative of live performance. The audit exists
to compare packages against each other, and for that the contrast matters more
than the level.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import random
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select, text

import freqpred.ingestion.models  # noqa: F401 — registers mapper
import freqpred.rag.models  # noqa: F401
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import MarketRow
from freqpred.metrics.assessment import _build_prompt_payload
from freqpred.signal.models import SignalRow
from freqpred.strategy.loader import load_strategy

_AUDIT_PATH = Path(__file__).resolve().parent / "audit_assessor_enhancement.py"
_spec = importlib.util.spec_from_file_location("audit_assessor_enhancement", _AUDIT_PATH)
audit = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(audit)  # type: ignore[union-attr]

SEED = 20260724
SERIES = "KXTRUMPSAY"
STRATEGY_NAME = "PoliticsEdgeStrategy"  # matches the audit harness
SIGNAL_PROMPT_VERSION = "signal-v11"
CURRENT_ARM_VERSION = "assessment-v6-audit-pit-current"
DEFAULT_OUT = Path("scripts/.audit_output/frozen_eval_set.json")


def _payload_hash(payload: dict) -> str:
    """Stable hash of the rendered prompt bytes — the cache-validity key."""
    return hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode()
    ).hexdigest()[:16]


async def _select_balanced(session) -> list[dict]:
    """Deterministic, direction-balanced, band-matched selection."""
    rows = (
        await session.execute(
            text("""
        SELECT s.id::text AS signal_id, s.direction,
          CASE WHEN s.edge*100 < 0 THEN '<0' WHEN s.edge*100 < 15 THEN '0-15'
               WHEN s.edge*100 < 40 THEN '15-40' ELSE '>40' END AS band,
          CASE WHEN (s.direction='YES' AND m.result='yes')
                 OR (s.direction='NO'  AND m.result='no') THEN 1 ELSE 0 END AS hit
        FROM signals s
        JOIN markets m ON m.id = s.market_id
        JOIN signal_assessments sa ON sa.signal_id = s.id
        WHERE m.series_ticker = :series AND m.status='finalized' AND m.result IS NOT NULL
          AND s.prompt_version = :pv AND s.direction IN ('YES','NO')
          AND s.market_ask_at_signal > 0 AND s.market_ask_at_signal < 1
          AND sa.llm_query_id IS NOT NULL
        ORDER BY s.id
    """).bindparams(series=SERIES, pv=SIGNAL_PROMPT_VERSION)
        )
    ).mappings().all()
    df = pd.DataFrame(rows)
    rng = random.Random(SEED)
    picked: list[dict] = []
    # NO is the scarce side: take all of it, then match YES band-for-band.
    for band, grp in df[df.direction == "NO"].groupby("band"):
        take = grp.to_dict("records")
        picked += take
        pool = df[(df.direction == "YES") & (df.band == band)].to_dict("records")
        rng.shuffle(pool)
        picked += pool[: len(take)]
    return picked


async def main(out_path: Path) -> None:
    config = load_config()
    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)
    strategy = load_strategy(STRATEGY_NAME)

    async with session_factory() as session:
        picked = await _select_balanced(session)
        print(f"selected {len(picked)} signals ({SERIES}, {SIGNAL_PROMPT_VERSION})")

        calib_df = await audit.build_calibration_pool(session)
        print(f"  calibration pool: {len(calib_df)} resolved signals")

        cached = {
            str(r.signal_id): r.response
            for r in (
                await session.execute(
                    select(LLMQueryRow.signal_id, LLMQueryRow.response).where(
                        LLMQueryRow.query_type == "model_eval",
                        LLMQueryRow.prompt_version == CURRENT_ARM_VERSION,
                        LLMQueryRow.success.is_(True),
                    )
                )
            ).all()
        }

        entries: list[dict[str, Any]] = []
        for i, row in enumerate(picked, 1):
            sid = uuid.UUID(row["signal_id"])
            sig_row = (
                await session.execute(select(SignalRow).where(SignalRow.id == sid))
            ).scalar_one()
            mkt_row = (
                await session.execute(
                    select(MarketRow).where(MarketRow.id == sig_row.market_id)
                )
            ).scalar_one()
            signal = audit._row_to_signal(sig_row)
            market = audit._row_to_market_at_signal_time(mkt_row, signal)

            base = _build_prompt_payload(
                signal,
                market,
                strategy.config.name,
                await audit._load_source_breakdown_pit(
                    session, signal, market, signal.created_at
                ),
                await audit._load_similar_market_summary_pit(
                    session, market, strategy.config.name, signal.created_at,
                    min_signals=strategy.config.similar_market_min_signals,
                    min_trades=strategy.config.similar_market_min_trades,
                ),
                scale_min=strategy.config.assessment_scale_min,
                scale_max=strategy.config.assessment_scale_max,
                phrase_data=None,
            )
            base = audit._fix_days_to_close(base, market, signal.created_at)
            base = audit._mask_unreconstructible_liquidity(
                base, book_reliable=audit._book_reconstruction_is_consistent(signal)
            )
            history = await audit._load_market_reevaluation_history(session, signal)

            cur = json.loads(json.dumps(base))
            cur["edge_band_calibration"] = audit._edge_band_calibration(
                calib_df, signal, signal.created_at
            )
            cur["market_reevaluation_history"] = history

            chal = json.loads(json.dumps(base))
            chal["edge_band_calibration"] = audit._add_profit_edge(
                audit._edge_band_calibration_v8(calib_df, signal, signal.created_at)
            )
            chal["market_reevaluation_history"] = history

            entries.append(
                {
                    "signal_id": row["signal_id"],
                    "market_id": signal.market_id,
                    "direction": row["direction"],
                    "edge_band": row["band"],
                    "edge_pct": signal.edge * 100.0,
                    "confidence": signal.confidence,
                    "hit": bool(row["hit"]),
                    "baseline_prior": audit._pit_baseline_prior(
                        calib_df, signal, signal.created_at
                    ),
                    "baseline_profit_edge": audit._pit_baseline_profit_edge(
                        calib_df, signal, signal.created_at
                    ),
                    "payloads": {
                        "current": {"payload": cur, "hash": _payload_hash(cur)},
                        "challenger": {"payload": chal, "hash": _payload_hash(chal)},
                    },
                    # Already-paid response, reusable because the judge is deterministic.
                    "cached_current_response": cached.get(row["signal_id"]),
                }
            )
            if i % 10 == 0:
                print(f"  rendered {i}/{len(picked)}")

    await engine.dispose()

    n_cached = sum(1 for e in entries if e["cached_current_response"])
    doc = {
        "seed": SEED,
        "series": SERIES,
        "signal_prompt_version": SIGNAL_PROMPT_VERSION,
        "n": len(entries),
        "stratification": "direction-balanced, band-matched; all available NO",
        "entries": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True))

    hits = sum(e["hit"] for e in entries)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  n={len(entries)}  hits={hits}  ({hits / len(entries):.0%})")
    for d in ("NO", "YES"):
        sub = [e for e in entries if e["direction"] == d]
        print(f"  {d:3s}: n={len(sub):2d} hits={sum(e['hit'] for e in sub):2d}")
    print(f"  current-arm responses already paid for: {n_cached}/{len(entries)}")
    print(f"  still to score: current={len(entries) - n_cached}, challenger={len(entries)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    asyncio.run(main(p.parse_args().out))
