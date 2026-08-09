# Does our probability estimate add anything beyond the market price?

**Date:** 2026-07-28
**Status:** research — no code changed. Findings require the signal-LLM
validation gates in `freqpred/signal/CLAUDE.md` before anything is adopted.
**Origin:** fell out of the backtest in
[trump-speaking-opportunities.md](trump-speaking-opportunities.md), which found
the LLM estimate scoring worse than the Poisson baseline it is handed.

---

## TL;DR

1. **Our estimate carries no information beyond the price.** Regressing outcome
   on `logit(mid) + logit(ours)`, the coefficient on ours is **+0.033, 95% CI
   [−0.152, +0.214]** — contains zero. The optimal blend weight on our estimate
   is **0.00** (CI [0.00, 0.14]).
2. **It is worse than a constant.** Brier 0.2684 vs 0.2270 for always predicting
   the base rate, and 0.1569 for the market mid.
3. **`edge` therefore measures our error, not alpha** — and the largest edges are
   *anti*-predictive. Where we claim edge > +0.20 (n=2,076 signals, 143 markets):
   price says 0.379, we say 0.823, **the truth is 0.292**. Below the price.
4. **Confidence is inverted.** Filtering to `confidence >= 0.8` makes our Brier
   *worse* (0.3211 vs 0.2684 unfiltered). No subset tested beats the price.
5. **The `edge` band is already configured** (`min_edge=0.15, max_edge=0.40`,
   enforced live) and sweeping the cap does not help — see §6. Most of the
   anti-predictive bucket in point 3 is already outside the traded population.
6. **A fade-YES bias is visible but not established**: +0.044/contract across 298
   markets, CI [−0.003, +0.091], against ~1.5–1.8c of fees. Our NO-direction
   calls score +0.123 [+0.017, +0.226], but their advantage over blanket-NO is
   not significant (§7). Worth tracking, not trading.

---

## Data

All signals on resolved markets in the last 90 days, joined to outcomes.
Primary sample **KXTRUMPSAY: 6,624 signals over 298 markets**, base rate 34.8%.
Repeated across all series (8,266 signals, 443 markets) with identical results.

Every confidence interval is **block-bootstrapped by market** — signals re-fire
on the same market every 30 minutes and are nowhere near independent. Treating
6,624 signals as 6,624 observations would overstate precision by roughly 5×.

## 1. Incremental information

The trading question is not "is our estimate well calibrated" but "conditional
on the price, does it move the posterior in the right direction".

```
outcome ~ logit(mid) + logit(ours)
```

| term | coefficient | 95% CI (block bootstrap) |
|---|---|---|
| `logit(mid)` | **+1.120** | [+0.922, +1.400] |
| `logit(ours)` | **+0.033** | [−0.152, +0.214] |

A coefficient of 1.0 on the price means the price is already a sufficient
statistic — the CI contains 1.0. The coefficient on our estimate contains zero.
Conditional on the price being known, our estimate is redundant.

Optimal blend, `logit(q) = (1−w)·logit(px) + w·logit(ours)`: **w\* = 0.00**,
CI [0.00, 0.14]. Brier 0.1569 (price only) vs 0.2684 (ours only).

## 2. Edge calibration — the important table

| edge bucket | n | mkts | mean price | mean ours | **actual** | our bias | price bias |
|---|---|---|---|---|---|---|---|
| [−1.00, −0.20) | 1007 | 94 | 0.555 | 0.186 | 0.430 | −0.244 | +0.125 |
| [−0.20, −0.10) | 688 | 113 | 0.332 | 0.183 | 0.205 | −0.022 | +0.127 |
| [−0.10, −0.03) | 781 | 159 | 0.328 | 0.262 | 0.256 | +0.006 | +0.071 |
| [−0.03, +0.03) | 831 | 171 | 0.458 | 0.454 | 0.425 | +0.030 | +0.033 |
| [+0.03, +0.10) | 581 | 151 | 0.481 | 0.542 | 0.465 | +0.078 | +0.017 |
| [+0.10, +0.20) | 660 | 136 | 0.520 | 0.670 | 0.461 | +0.210 | +0.060 |
| **[+0.20, +1.00)** | **2076** | **143** | **0.379** | **0.823** | **0.292** | **+0.530** | +0.087 |

Read the bottom row carefully. In the bucket where we are most convinced YES is
underpriced — and where sizing is largest — the market says 0.379, we say 0.823,
and the realised rate is **0.292**. Not merely wrong: the outcome is *below* the
price we thought was too low. Buying that signal is worse than buying at random.

The negative-edge side is different. At [−1.00, −0.20) the price is 0.555 and the
truth is 0.430 — the price *was* too high and our direction was right, though we
overshot to 0.186. Whatever signal exists is on the fade-YES side.

Note the `price bias` column is positive in every row: the price sits above the
realised rate throughout. That is the thread picked up in §5.

## 3. No subset beats the price

| subset | n | mkts | Brier ours | Brier price |
|---|---|---|---|---|
| all | 6624 | 298 | 0.2684 | 0.1569 |
| confidence ≥ 0.7 | 3147 | 229 | 0.3019 | 0.1428 |
| confidence ≥ 0.8 | 1565 | 153 | 0.3211 | 0.1471 |
| confidence ≥ 0.9 | 749 | 87 | 0.2858 | 0.1262 |
| days_left > 3 | 3833 | 249 | 0.2844 | 0.2038 |
| price < 0.20 | 1603 | 177 | 0.2078 | 0.0562 |
| trigger=scheduled | 3571 | 297 | 0.2283 | 0.1373 |
| trigger=price_moved | 3016 | 231 | 0.3138 | 0.1818 |

**Confidence is inverted** — the more confident we are, the worse we score. Any
gate that raises the confidence threshold makes selection worse, not better.

### The price's fairest test

The price looks unbeatable partly by construction: near resolution it converges
to 0/1, and signals fire close to resolution. So I tested the price's *weakest*
case — long horizon and a thin book, where the mid carries least information:

| subset | n | mkts | Brier ours | Brier price |
|---|---|---|---|---|
| days_left > 3 AND spread > 5c | 2093 | 221 | 0.2084 | 0.1719 |
| days_left > 5 AND spread > 5c | 940 | 147 | 0.2115 | 0.1737 |

The gap narrows (from +0.111 to +0.037) but never closes, and the incremental
coefficient there is +0.112, CI [−0.082, +0.282] — still contains zero. This is
the most favourable framing I could construct for our model and it still loses.

## 4. What it cost

Live book, 90 days: **96 closed positions, −$6.67**. By direction and entry edge:

| entry edge | dir | n | P&L | wins |
|---|---|---|---|---|
| 0.10–0.20 | NO | 14 | +$1.41 | 9 |
| 0.10–0.20 | YES | 14 | −$0.29 | 7 |
| 0.20–0.35 | NO | 14 | −$0.57 | 7 |
| **0.20–0.35** | **YES** | **33** | **−$9.49** | 10 |
| ≥0.35 | NO | 3 | +$0.02 | 1 |
| ≥0.35 | YES | 18 | +$2.24 | 9 |

Directionally consistent with the calibration story — the moderate-edge YES
bucket is where the loss is concentrated — but **n=96 is far too small to
conclude from on its own**, and the ≥0.35 YES bucket is positive, which cuts the
other way. The statistical evidence in §1–§3 (6,624 signals, 298 markets) is what
carries weight here; the P&L is corroboration, not proof. Paper book over the
same style of period: 298 closed, −$381.16.

The position mix is **65 YES vs 31 NO** — better than 2:1 long YES.

## 5. The bias we are on the wrong side of

Across every edge bucket the price sits above the realised rate. At market level
(one observation per market, so long-lived markets do not get extra weight):

| series | markets | mean price | actual YES | price bias |
|---|---|---|---|---|
| KXTRUMPSAY | 298 | 0.494 | 0.450 | **+0.044** |
| KXTRUMPSAYMONTH | 18 | 0.165 | 0.111 | +0.054 |
| KXTRUMPACT | 18 | 0.634 | 0.500 | +0.134 |
| KXTRUMPENDORSEMENTS | 17 | 0.503 | 0.353 | +0.150 |
| KXTRUTHSOCIAL | 33 | 0.173 | 0.182 | −0.009 |

Fitting a correction `q = expit(k·logit(px) + c)` on the first half of the window
and scoring it on the second, for `days_left > 3`:

- signal level: k=1.03, c=−0.60 → test Brier 0.2140 → 0.1984 (**+7.27%**)
- market level: k=0.88, c=−0.45 → test Brier 0.1888 → 0.1833 (**+2.94%**, 125 test markets)

The fitted correction is almost entirely a **negative offset**, not a shrink —
the market is not "too extreme", it is **too high on YES**. This is the classic
lottery-ticket bias one would expect on "will he say X" contracts.

**This is a hypothesis, not a finding to trade.** A Brier improvement is not
profit: it has to clear the spread and Kalshi's fees, and +0.044 at market level
on KXTRUMPSAY is thin. It needs its own backtest with transaction costs before
anyone acts on it. I am flagging it because it is the only thing in this dataset
that survived an out-of-sample test — and because our model leans the opposite way.

## 6. The edge band is already set, and tightening it does not help

`PoliticsEdgeStrategy` runs `min_edge=0.15, max_edge=0.40`, enforced live at
`freqpred/strategy/base.py:66`. The `StrategyConfig` docstring already states the
rationale this analysis rediscovered ("very high edge means the market is right
and the model is wrong"). So §2's headline bucket is **mostly outside the traded
population** and the obvious remedy is already in place.

Scoring only the ~admitted population (edge 0.15–0.40, conf ≥ 0.60, mid
0.10–0.90, 0.25–7d): **1,866 signals over 161 markets**.

Sweeping the cap, other gates held fixed:

| max_edge | signals | markets | Brier ours | Brier price |
|---|---|---|---|---|
| 0.20 | 534 | 123 | 0.2556 | 0.2214 |
| 0.30 | 1284 | 147 | 0.2936 | 0.2334 |
| **0.40 (live)** | 1866 | 161 | 0.3181 | 0.2306 |
| 0.50 | 2231 | 168 | 0.3351 | 0.2248 |
| none | 2796 | 184 | 0.4096 | 0.2017 |

Tightening improves our Brier but never reaches the price, and it does not fix
the underlying problem. **The edge cap is not the lever.**

## 7. Direction — and a correction to my own first read

At *signal* level the admitted population looked starkly asymmetric: YES-side
gross −0.132/contract, NO-side +0.158. **That was an artifact of signal-count
weighting** — markets that stay unresolved accumulate 30-minute signals for days
and disproportionately settle NO. Collapsing to one observation per
(market, direction), bootstrapped by market:

| direction | markets | gross/contract | 95% CI |
|---|---|---|---|
| YES | 101 | +0.0030 | [−0.0897, +0.0952] |
| NO | 62 | **+0.1250** | [+0.0169, +0.2254] |

YES is flat, not negative. NO is positive with the CI excluding zero. But the
**difference** — NO minus YES, +0.1220 — has CI [−0.0179, +0.2571] and
**contains zero**, so the asymmetry itself is not established.

Does our NO call beat simply buying NO everywhere?

| | markets | gross/contract | 95% CI |
|---|---|---|---|
| blanket NO, all admitted markets | 161 | +0.0484 | [−0.0219, +0.1174] |
| blanket NO, markets we called NO | 62 | +0.1230 | [+0.0161, +0.2260] |
| blanket NO, markets we did not | 99 | +0.0017 | [−0.0922, +0.0941] |
| **selection value of our NO call** | | **+0.1213** | **[−0.0222, +0.2602]** ← contains zero |
| blanket NO, all 298 KXTRUMPSAY markets, no gates | 298 | +0.0443 | [−0.0027, +0.0910] |

The point estimates line up with the fade-YES story and hint that our NO calls
concentrate it. None of it clears significance at market level, and after ~1.5–1.8
cents of Kalshi fees the blanket version is thin.

## Recommendations

**Do not change the entry gates on the strength of this.** The edge band exists,
sweeping it does not help, and the direction asymmetry is not established.

What actually holds up, in order of evidential strength:

1. **Our estimate adds no information beyond the price** (§1) — 298 markets,
   block-bootstrapped, reproduced across all series. This is the robust result
   and it is a statement about the signal, not the gates.
2. **Do not raise the confidence threshold** (§3). It is inverted — a higher bar
   selects worse trades. This is the one gate change worth *avoiding* explicitly.
3. **Track the fade-YES bias rather than trade it.** +0.044/contract across 298
   markets with a CI grazing zero is not an edge yet; it is the only
   positive-expectancy thing in the data and deserves a proper cost-aware
   backtest and more markets. A natural fit for the weekly review as a tracked
   metric.
4. **Re-examine why the LLM runs hot on YES.** Worth checking whether retrieved
   evidence is structurally one-sided — documents explain why a phrase is
   topical, never why it will not be said. Pairs with the finding that the LLM
   scores worse than the Poisson baseline it is handed.

Caveat on all of §5–§7: this analysis has now sliced the same 298 markets many
ways. Treat the surviving point estimates as hypotheses to re-test on fresh
data, not as measurements. Signal changes go through
`freqpred/signal/CLAUDE.md`; sizing through `freqpred/metrics/CLAUDE.md`.

## Reproducing

`edge_value.py` in the session scratchpad; input export:

```sql
SELECT s.created_at, s.market_id,
       regexp_replace(s.market_id, '-[0-9]{2}[A-Z]{3}[0-9]{2}-.*$','') AS series,
       s.estimated_probability, s.market_mid_at_signal, s.market_ask_at_signal,
       s.edge, s.confidence, s.direction, s.trigger,
       extract(epoch FROM (m.close_time - s.created_at))/86400.0 AS days_left,
       m.result
FROM signals s JOIN markets m ON m.id = s.market_id
WHERE m.result IS NOT NULL AND s.created_at > now() - interval '90 days';
```
