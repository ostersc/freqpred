"""Non-inferiority analysis for an assessor challenger package.

analyze_audit.py answers "is the challenger BETTER?" (two-sided CI vs zero).
That is the wrong question when the goal is migrating off an ageing judgment
model: there the bar is "is it materially WORSE?", and a wash is an adopt.

It also leads with corr(trust_score, hit), which is an abstraction over the
thing actually being bought. What the sizing multiplier does economically is
tilt capital toward winners and away from losers, so the headline here is that
tilt measured in multiplier space:

    capital_tilt = mean(multiplier | hit) - mean(multiplier | miss)

Positive means more size on winners than losers. It is reported alongside the
correlation because the two can disagree: correlation is invariant to location
and scale, so an arm can rank signals perfectly while barely moving any money,
and the dead band plus the scale_min/scale_max clamp mean rank order does not
translate linearly into dollars.

Usage:
    uv run python scripts/.audit_output/analyze_noninferiority.py <csv> [margin]

`margin` is the non-inferiority tolerance in correlation points (default 0.10):
how much worse the challenger's ranking may be before we call it material.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_BOOT = 5000
DEAD_BAND = (0.45, 0.55)


def _corr(a: np.ndarray, h: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(h) == 0:
        return 0.0
    return float(np.corrcoef(a, h)[0, 1])


def _tilt(mult: np.ndarray, h: np.ndarray) -> float:
    """Mean multiplier on winners minus mean multiplier on losers."""
    if h.sum() == 0 or (~h.astype(bool)).sum() == 0:
        return float("nan")
    return float(mult[h.astype(bool)].mean() - mult[~h.astype(bool)].mean())


def bootstrap_paired(
    df: pd.DataFrame, a: str, b: str, stat: str
) -> dict:
    """Paired bootstrap of (challenger - current) for a given statistic."""
    col = "trust_score" if stat == "corr" else "multiplier"
    sub = df.dropna(subset=[f"{a}_{col}", f"{b}_{col}", "hit"])
    av = sub[f"{a}_{col}"].to_numpy(float)
    bv = sub[f"{b}_{col}"].to_numpy(float)
    h = sub["hit"].astype(float).to_numpy()
    fn = _corr if stat == "corr" else _tilt
    obs_a, obs_b = fn(av, h), fn(bv, h)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, len(sub), len(sub))
        diffs[i] = fn(bv[idx], h[idx]) - fn(av[idx], h[idx])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n": len(sub),
        "a": obs_a,
        "b": obs_b,
        "diff": obs_b - obs_a,
        "lo": lo,
        "hi": hi,
        "p_worse": float((diffs < 0).mean()),
    }


def describe(df: pd.DataFrame, arm: str) -> None:
    t = df[f"{arm}_trust_score"].dropna()
    m = df.dropna(subset=[f"{arm}_multiplier", "hit"])
    mult = m[f"{arm}_multiplier"].to_numpy(float)
    h = m["hit"].astype(float).to_numpy()
    v = t.map(
        lambda x: "size_down"
        if x < DEAD_BAND[0]
        else ("size_up" if x > DEAD_BAND[1] else "neutral")
    )
    print(
        f"  {arm:11s} n={len(t):2d}  trust[{t.min():.2f}-{t.max():.2f}] "
        f"mean={t.mean():.3f} sd={t.std():.4f} distinct={t.nunique():2d}  "
        f"corr={_corr(t.to_numpy(float), df.loc[t.index, 'hit'].astype(float).to_numpy()):+.3f}  "
        f"tilt={_tilt(mult, h):+.4f}x"
    )
    print(
        f"              verdicts: down={int((v == 'size_down').sum()):2d} "
        f"neutral={int((v == 'neutral').sum()):2d} up={int((v == 'size_up').sum()):2d}"
        f"   mean_mult(hit)={mult[h.astype(bool)].mean():.3f} "
        f"mean_mult(miss)={mult[~h.astype(bool)].mean():.3f}"
    )


def _auc(s: np.ndarray, h: np.ndarray) -> float:
    wins, losses = s[h], s[~h]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    gt = (wins[:, None] > losses[None, :]).sum()
    tie = (wins[:, None] == losses[None, :]).sum()
    return float((gt + 0.5 * tie) / (len(wins) * len(losses)))


def incremental_auc(df: pd.DataFrame, arms: list[str]) -> None:
    """Does the LLM beat a free, LLM-free base-rate lookup?

    Absolute AUC flatters every arm, because most of it is reproducible from a
    (direction, edge band) prior that costs nothing. Adoption should turn on the
    INCREMENT over that prior, not the raw number.
    """
    if "baseline_prior" not in df.columns:
        print("\n(no baseline_prior column — re-run the audit to record it)")
        return
    sub = df.dropna(subset=["baseline_prior", "hit"])
    h = sub["hit"].astype(bool).to_numpy()
    prior = sub["baseline_prior"].to_numpy(float)
    base = _auc(prior, h)
    print("\n=== Incremental value over the free base-rate prior ===")
    print(f"  {'PRIOR (direction x edge band, no LLM)':42s} AUC={base:.3f}")
    for a in arms:
        # Drop only the rows this arm failed to score, rather than skipping the
        # arm entirely — a couple of failed calls must not silently hide a result.
        ok = sub[f"{a}_trust_score"].notna().to_numpy()
        s = sub.loc[ok, f"{a}_trust_score"].to_numpy(float)
        h_a = h[ok]
        p_a = prior[ok]
        auc_a = _auc(s, h_a)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = RNG.integers(0, len(s), len(s))
            diffs[i] = _auc(s[idx], h_a[idx]) - _auc(p_a[idx], h_a[idx])
        diffs = diffs[~np.isnan(diffs)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        verdict = "BEATS prior" if lo > 0 else ("BELOW prior" if hi < 0 else "indistinguishable from free")
        print(
            f"  {a:42s} AUC={auc_a:.3f}  incr={auc_a - base:+.3f} "
            f"CI=({lo:+.3f},{hi:+.3f})  -> {verdict}"
        )


def main() -> None:
    path = sys.argv[1]
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    df = pd.read_csv(path)
    arms = [a for a in ("current", "challenger") if f"{a}_trust_score" in df.columns]
    print(f"Source: {path}   rows={len(df)}   hits={int(df.hit.sum())}")
    print(f"Non-inferiority margin: {margin:.2f} correlation points\n")

    print("=== Per-arm behaviour ===")
    for a in arms:
        describe(df, a)
    incremental_auc(df, arms)

    if len(arms) < 2:
        print("\n(only one arm present — no contrast)")
        return

    print("\n=== Does it push winners harder than losers? (capital tilt) ===")
    r = bootstrap_paired(df, "current", "challenger", "tilt")
    print(
        f"  current={r['a']:+.4f}x  challenger={r['b']:+.4f}x  "
        f"diff={r['diff']:+.4f}x  95% CI=({r['lo']:+.4f}, {r['hi']:+.4f})"
    )
    print(f"  P(challenger tilts less) = {r['p_worse']:.1%}   n={r['n']}")

    print("\n=== Ranking quality: corr(trust_score, hit) ===")
    c = bootstrap_paired(df, "current", "challenger", "corr")
    print(
        f"  current={c['a']:+.3f}  challenger={c['b']:+.3f}  "
        f"diff={c['diff']:+.3f}  95% CI=({c['lo']:+.3f}, {c['hi']:+.3f})   n={c['n']}"
    )

    print("\n=== VERDICT ===")
    superior = c["lo"] > 0
    non_inferior = c["lo"] > -margin
    worse = c["hi"] < 0
    if superior:
        print("  ADOPT — challenger is significantly BETTER (CI excludes zero, above).")
    elif worse:
        print("  REJECT — challenger is significantly WORSE (CI excludes zero, below).")
    elif non_inferior:
        print(
            f"  ADOPT — non-inferior. CI lower bound {c['lo']:+.3f} is inside the "
            f"{margin:.2f} tolerance, so a materially worse challenger is ruled out."
        )
    else:
        print(
            f"  INCONCLUSIVE — CI lower bound {c['lo']:+.3f} breaches the "
            f"-{margin:.2f} tolerance. Cannot rule out a material regression; "
            "more samples needed."
        )
    if not np.isnan(r["diff"]):
        direction = "MORE" if r["diff"] > 0 else "LESS"
        print(
            f"  Capital tilt: challenger puts {direction} relative size on winners "
            f"({r['b']:+.4f}x vs {r['a']:+.4f}x)."
        )


if __name__ == "__main__":
    main()
