import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_BOOT = 5000

# Arm ladder, oldest package first. Only arms whose columns exist in the CSV
# are analyzed, so this script works on historical (v2/v3) result files as
# well as current-vs-challenger runs.
ARM_LABELS = {
    "existing": "existing   (production history, v4 era)",
    "control": "control    (live, pre-T94: v4 prompt, no new sections)",
    "enhanced": "t94-proto  (morning v2 run: v4 prompt + prototype sections)",
    "t94": "t94        (as shipped: v5 prompt + shipped sections)",
    "t95": "t95        (adopted as v6: prompt + bucketed verbatim history)",
    "current": "current    (production package at run time)",
    "challenger": "challenger (proposed package under screen)",
}
BOOT_PAIRS = [
    ("control", "t94"),
    ("t94", "t95"),
    ("control", "t95"),
    ("enhanced", "t95"),
    ("current", "challenger"),
    # model-swap screen: t95 = v6 package on opus-4-7 (adoption run, 768 tok)
    ("t95", "challenger"),
]


def point_biserial(scores: pd.Series, hits: pd.Series) -> float:
    return float(pd.Series(scores).corr(pd.Series(hits.astype(float))))


def bootstrap_corr_diff(df: pd.DataFrame, col_a: str, col_b: str, hit_col: str = "hit") -> dict:
    sub = df.dropna(subset=[col_a, col_b, hit_col])
    n = len(sub)
    a = sub[col_a].to_numpy()
    b = sub[col_b].to_numpy()
    h = sub[hit_col].astype(float).to_numpy()
    obs_a = np.corrcoef(a, h)[0, 1]
    obs_b = np.corrcoef(b, h)[0, 1]
    obs_diff = obs_b - obs_a
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        ca = np.corrcoef(a[idx], h[idx])[0, 1] if np.std(a[idx]) > 0 and np.std(h[idx]) > 0 else 0.0
        cb = np.corrcoef(b[idx], h[idx])[0, 1] if np.std(b[idx]) > 0 and np.std(h[idx]) > 0 else 0.0
        diffs[i] = cb - ca
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n": n,
        "corr_a": obs_a,
        "corr_b": obs_b,
        "diff": obs_diff,
        "ci_95": (lo, hi),
        "significant": (lo > 0) or (hi < 0),
    }


def verdict_counts(df: pd.DataFrame, arm: str) -> dict[str, int] | None:
    """Verdict distribution; uses the recorded verdict column when present,
    otherwise derives it from trust_score with the production dead band."""
    vcol, tcol = f"{arm}_verdict", f"{arm}_trust_score"
    if vcol in df.columns and df[vcol].notna().any():
        series = df[vcol].dropna()
    elif tcol in df.columns:
        ts = df[tcol].dropna()
        series = ts.map(lambda t: "size_down" if t < 0.45 else ("size_up" if t > 0.55 else "neutral"))
    else:
        return None
    return {v: int((series == v).sum()) for v in ("size_down", "neutral", "size_up")}


def stake_sim(df: pd.DataFrame, mult_col: str, bankroll=1000.0, kelly_fraction=0.25, max_exposure_per_market=0.20):
    market_budget = bankroll * max_exposure_per_market

    def ideal_total(r):
        if r["direction"] == "NO":
            p_market = 1.0 - (r["estimated_probability"] + r["edge"])
            p_est = 1.0 - r["estimated_probability"]
        else:
            p_market = r["estimated_probability"] - r["edge"]
            p_est = r["estimated_probability"]
        if p_market <= 0 or p_market >= 1:
            return 0.0, p_market
        b = (1.0 - p_market) / p_market
        conf = r["confidence"]
        p_adj = conf * p_est + (1.0 - conf) * p_market
        f_star = (b * p_adj - (1.0 - p_adj)) / b
        if f_star <= 0.0:
            return 0.0, p_market
        base = f_star * kelly_fraction * market_budget
        return base * r[mult_col], p_market

    staked, pnl = 0.0, 0.0
    for _, r in df.iterrows():
        if pd.isna(r[mult_col]):
            continue
        stake, p_market = ideal_total(r)
        if stake <= 0:
            continue
        cost = r["market_p_side_cost"]
        pnl_row = stake * ((1.0 - cost) / cost) if r["hit"] else -stake
        staked += stake
        pnl += pnl_row
    roi = 100 * pnl / staked if staked > 0 else float("nan")
    return staked, pnl, roi


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "scripts/.audit_output/assessor_enhancement_audit_pit_v3.csv"
    df = pd.read_csv(path)
    arms = [a for a in ARM_LABELS if f"{a}_trust_score" in df.columns]
    print(f"Source: {path}")
    print(f"Loaded {len(df)} sampled signals; arms present: {arms}")
    print()

    print("=== corr(trust_score, hit) per arm ===")
    for arm in arms:
        sub = df.dropna(subset=[f"{arm}_trust_score", "hit"])
        c = point_biserial(sub[f"{arm}_trust_score"], sub["hit"])
        print(f"{ARM_LABELS[arm]}: r={c:+.3f}  (n={len(sub)})")
    print()
    print("=== corr(size_multiplier, hit) per arm ===")
    for arm in arms:
        sub = df.dropna(subset=[f"{arm}_multiplier", "hit"])
        c = point_biserial(sub[f"{arm}_multiplier"], sub["hit"])
        print(f"{ARM_LABELS[arm]}: r={c:+.3f}  (n={len(sub)})")
    print()

    print("=== Verdict distribution per arm (dead band: <0.45 down, >0.55 up) ===")
    for arm in arms:
        vc = verdict_counts(df, arm)
        if vc is None:
            continue
        print(
            f"{ARM_LABELS[arm]}: size_down={vc['size_down']:2d}  "
            f"neutral={vc['neutral']:2d}  size_up={vc['size_up']:2d}"
        )
    print()

    print("=== Bootstrap paired corr diffs (trust_score), 95% CI ===")
    for a, b in BOOT_PAIRS:
        if a not in arms or b not in arms:
            continue
        res = bootstrap_corr_diff(df, f"{a}_trust_score", f"{b}_trust_score")
        print(
            f"{b} - {a}: corr_{a}={res['corr_a']:+.3f} corr_{b}={res['corr_b']:+.3f} "
            f"diff={res['diff']:+.3f} 95% CI={tuple(round(x, 3) for x in res['ci_95'])} "
            f"significant={res['significant']} (n={res['n']})"
        )
    print()

    # Stake-weighted P&L: join per-signal economics from the DB export.
    econ = pd.read_csv("scripts/.audit_output/signal_econ.csv")
    df = df.merge(econ, on="signal_id", how="left", suffixes=("", "_econ"))
    # market_ask_at_signal is already the traded side's cost per contract
    df["market_p_side_cost"] = df["market_ask_at_signal"]
    df["edge"] = df["edge_econ"] if "edge_econ" in df else df["edge_pct"] / 100.0
    print("=== Stake-weighted P&L within the audit sample ===")
    df["_baseline_mult"] = 1.0
    sim_arms = [("no multiplier (1.0x baseline)", "_baseline_mult")] + [
        (ARM_LABELS[a], f"{a}_multiplier") for a in arms
    ]
    for label, col in sim_arms:
        staked, pnl, roi = stake_sim(df, col)
        print(f"{label:60s}: staked=${staked:8.2f}  pnl=${pnl:8.2f}  roi={roi:6.1f}%")
