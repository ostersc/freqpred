import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_BOOT = 5000


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

    path = sys.argv[1] if len(sys.argv) > 1 else "scripts/.audit_output/assessor_enhancement_audit.csv"
    df = pd.read_csv(path)
    print(f"Source: {path}")
    print(f"Loaded {len(df)} sampled signals")
    print(f"  usable existing+control+enhanced triples: "
          f"{df.dropna(subset=['existing_trust_score','control_trust_score','enhanced_trust_score']).shape[0]}")
    print()

    for label, col in [
        ("existing (production, historical)", "existing_trust_score"),
        ("control (live now, unmodified)", "control_trust_score"),
        ("enhanced (live now, + calibration data)", "enhanced_trust_score"),
    ]:
        sub = df.dropna(subset=[col, "hit"])
        c = point_biserial(sub[col], sub["hit"])
        print(f"corr(trust_score, hit) — {label}: r={c:.3f}  (n={len(sub)})")
    print()
    for label, col in [
        ("existing", "existing_multiplier"),
        ("control", "control_multiplier"),
        ("enhanced", "enhanced_multiplier"),
    ]:
        sub = df.dropna(subset=[col, "hit"])
        c = point_biserial(sub[col], sub["hit"])
        print(f"corr(size_multiplier, hit) — {label}: r={c:.3f}  (n={len(sub)})")
    print()

    print("=== Bootstrap: enhanced vs control (same live model, only the calibration data differs) ===")
    res_ts = bootstrap_corr_diff(df, "control_trust_score", "enhanced_trust_score")
    print(f"trust_score:   corr_control={res_ts['corr_a']:.3f} corr_enhanced={res_ts['corr_b']:.3f} "
          f"diff={res_ts['diff']:+.3f} 95% CI={tuple(round(x,3) for x in res_ts['ci_95'])} "
          f"significant={res_ts['significant']} (n={res_ts['n']})")
    res_mult = bootstrap_corr_diff(df, "control_multiplier", "enhanced_multiplier")
    print(f"multiplier:    corr_control={res_mult['corr_a']:.3f} corr_enhanced={res_mult['corr_b']:.3f} "
          f"diff={res_mult['diff']:+.3f} 95% CI={tuple(round(x,3) for x in res_mult['ci_95'])} "
          f"significant={res_mult['significant']} (n={res_mult['n']})")
    print()

    print("=== Bootstrap: enhanced vs existing (production historical baseline) ===")
    res_ts2 = bootstrap_corr_diff(df, "existing_trust_score", "enhanced_trust_score")
    print(f"trust_score:   corr_existing={res_ts2['corr_a']:.3f} corr_enhanced={res_ts2['corr_b']:.3f} "
          f"diff={res_ts2['diff']:+.3f} 95% CI={tuple(round(x,3) for x in res_ts2['ci_95'])} "
          f"significant={res_ts2['significant']} (n={res_ts2['n']})")
    print()

    # Stake-weighted P&L: join per-signal economics from the DB export.
    econ = pd.read_csv("scripts/.audit_output/signal_econ.csv")
    df = df.merge(econ, on="signal_id", how="left", suffixes=("", "_econ"))
    # market_ask_at_signal is already the traded side's cost per contract
    df["market_p_side_cost"] = df["market_ask_at_signal"]
    df["edge"] = df["edge_econ"] if "edge_econ" in df else df["edge_pct"] / 100.0
    print("=== Stake-weighted P&L within the audit sample ===")
    for label, col in [
        ("no multiplier (1.0x baseline)", None),
        ("existing (production)", "existing_multiplier"),
        ("control (live, unmodified)", "control_multiplier"),
        ("enhanced (live, + calibration)", "enhanced_multiplier"),
    ]:
        d = df.copy()
        if col is None:
            d["_mult"] = 1.0
            mcol = "_mult"
        else:
            mcol = col
        staked, pnl, roi = stake_sim(d, mcol)
        print(f"{label:35s}: staked=${staked:8.2f}  pnl=${pnl:8.2f}  roi={roi:6.1f}%")
