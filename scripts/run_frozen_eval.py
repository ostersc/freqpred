"""Score a frozen assessor eval set. Payloads are replayed verbatim, never rebuilt.

The frozen set already contains the fully-rendered point-in-time payload for each
arm, so this runner does no DB reconstruction: it replays stored prompt bytes,
which is what makes runs byte-for-byte reproducible and cached scores safe to
reuse. A cached response is reused only when its payload hash still matches, so a
harness change that alters the inputs invalidates the cache instead of silently
mixing old scores with new prompts.

Resumable: an interrupted run can be re-run and will skip anything already
scored in the output CSV, so a mid-flight stop costs only the calls not yet made.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import anthropic
import pandas as pd

import freqpred.ingestion.models  # noqa: F401 — registers mapper
import freqpred.rag.models  # noqa: F401
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.llm.client import LLMClient
from freqpred.metrics.assessment import (
    _ASSESSMENT_TOOL,
    _PROMPT_VERSION,
    _SYSTEM_PROMPT,
    _clamp_multiplier,
    _parse_assessment_response,
    _trust_score_to_multiplier,
)
from freqpred.strategy.loader import load_strategy

_AUDIT = Path(__file__).resolve().parent / "audit_assessor_enhancement.py"
_spec = importlib.util.spec_from_file_location("audit_assessor_enhancement", _AUDIT)
audit = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(audit)  # type: ignore[union-attr]

STRATEGY_NAME = "PoliticsEdgeStrategy"
MAX_TOKENS = 1024
CONCURRENCY = 4


def _score(parsed: dict, scale_min: float, scale_max: float) -> dict:
    mult = _clamp_multiplier(
        _trust_score_to_multiplier(
            parsed["trust_score"], scale_min=scale_min, scale_max=scale_max
        ),
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return {
        "trust_score": parsed["trust_score"],
        "multiplier": mult,
        "verdict": parsed["verdict"],
    }


async def main(eval_set: Path, out_path: Path, arms: set[str]) -> None:
    config = load_config()
    strategy = load_strategy(STRATEGY_NAME)
    smin = strategy.config.assessment_scale_min
    smax = strategy.config.assessment_scale_max
    doc = json.loads(eval_set.read_text())
    entries = doc["entries"]

    done: dict[str, dict] = {}
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = {r["signal_id"]: r for r in prev.to_dict("records")}
        print(f"resuming: {len(done)} rows already in {out_path}")

    engine = make_engine(config.database.url)
    sf = make_session_factory(engine)
    llm = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        sf,
        default_strategy="model_eval",
        prompt_version="frozen-eval",
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
        max_consecutive_errors=config.risk.max_consecutive_llm_errors,
    )

    # Fail loudly rather than sending system=None, which would silently score the
    # challenger arm with no system prompt at all and look like a real result.
    if "challenger" in arms and not (
        audit.CHALLENGER_SYSTEM_PROMPT and audit.CHALLENGER_VERSION
    ):
        raise SystemExit(
            "ERROR: the challenger arm is not defined. Set CHALLENGER_VERSION, "
            "CHALLENGER_SYSTEM_PROMPT, CHALLENGER_MODEL and _challenger_payload in "
            "scripts/audit_assessor_enhancement.py before screening a package. "
            "(It is deliberately disarmed after an adoption so a re-run cannot "
            "measure current-vs-current and bill for a guaranteed null result.)"
        )

    arm_cfg = {
        "current": (_SYSTEM_PROMPT, f"{_PROMPT_VERSION}-frozen-current", None),
        "challenger": (
            audit.CHALLENGER_SYSTEM_PROMPT,
            f"{audit.CHALLENGER_VERSION}-frozen",
            audit.CHALLENGER_MODEL,
        ),
    }
    sem = asyncio.Semaphore(CONCURRENCY)
    rows: list[dict] = []
    counter = {"n": 0, "cached": 0, "failed": 0}

    async def run_entry(e: dict) -> dict:
        row = {
            k: e[k]
            for k in (
                "signal_id", "market_id", "direction", "edge_pct",
                "confidence", "hit", "baseline_prior", "baseline_profit_edge",
            )
        }
        prior = done.get(e["signal_id"], {})
        for arm in ("current", "challenger"):
            if arm not in arms:
                continue
            if pd.notna(prior.get(f"{arm}_trust_score", None)):
                for f in ("trust_score", "multiplier", "verdict"):
                    row[f"{arm}_{f}"] = prior[f"{arm}_{f}"]
                continue
            # Reuse an already-paid response only if the prompt bytes are unchanged.
            cached = e.get("cached_current_response")
            if arm == "current" and cached and e.get("cached_hash") == e["payloads"]["current"]["hash"]:
                try:
                    row.update(
                        {f"current_{k}": v for k, v in _score(
                            _parse_assessment_response(cached), smin, smax).items()}
                    )
                    counter["cached"] += 1
                    continue
                except Exception:  # noqa: BLE001 — stale cache, just re-score
                    pass
            system, version, model = arm_cfg[arm]
            payload = e["payloads"][arm]["payload"]
            async with sem:
                try:
                    resp = await llm.complete(
                        prompt=json.dumps(payload, indent=2, sort_keys=True),
                        model=model or config.anthropic.judgment_model,
                        query_type="model_eval",
                        system=system,
                        market_id=e["market_id"],
                        signal_id=e["signal_id"],
                        strategy=f"frozen_{arm}",
                        prompt_version=version,
                        max_tokens=MAX_TOKENS,
                        json_tool=_ASSESSMENT_TOOL,
                    )
                    row.update(
                        {f"{arm}_{k}": v for k, v in _score(
                            _parse_assessment_response(resp.content), smin, smax).items()}
                    )
                except Exception as exc:  # noqa: BLE001
                    counter["failed"] += 1
                    print(f"  {e['market_id'][:28]:28s} {arm} FAILED: {str(exc)[:90]}")
                    for f in ("trust_score", "multiplier", "verdict"):
                        row[f"{arm}_{f}"] = None
        counter["n"] += 1
        if counter["n"] % 10 == 0:
            print(f"  scored {counter['n']}/{len(entries)}")
        return row

    rows = await asyncio.gather(*(run_entry(e) for e in entries))
    await engine.dispose()

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {len(df)} rows to {out_path}")
    print(f"  reused cached: {counter['cached']}   failed calls: {counter['failed']}")
    for arm in sorted(arms):
        col = f"{arm}_trust_score"
        if col in df:
            print(f"  {arm:11s} scored={int(df[col].notna().sum())}/{len(df)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", type=Path, default=Path("scripts/.audit_output/frozen_eval_set.json"))
    p.add_argument("--out", type=Path, default=Path("scripts/.audit_output/frozen_eval_results.csv"))
    p.add_argument("--arms", default="current,challenger")
    a = p.parse_args()
    asyncio.run(main(a.eval_set, a.out, set(a.arms.split(","))))
