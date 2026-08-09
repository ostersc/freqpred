# Estimating Trump speaking opportunities for the KXTRUMPSAY Poisson baseline

**Date:** 2026-07-28
**Status:** research — **recommendation is DO NOT BUILD.** The backtest in
§"Backtest" falsifies the idea, including against an oracle that cheats. Kept
because the sources survey and the two incidental findings are worth having.
**Question:** the Poisson baseline in `freqpred/signal/llm.py:_build_factbase_block()`
spreads the remaining window evenly (`1 - exp(-rate * days_to_close)`). Real
utterance opportunities are not evenly spread — a week with three rallies is not
a week at Bedminster. Is there a source that lets us weight the remaining days?

---

## TL;DR

1. **No source gives a multi-week forward view, because the White House does not
   publish one.** Every downstream tracker (Factbase, @WHPressPool, Kadoa,
   C-SPAN) is derived from the same daily press guidance, which drops the night
   before. The horizon is **T+1**, extending to **T+3 on Fridays** (the Friday
   guidance covers the weekend).
2. **The Factbase calendar has a clean, free, unauthenticated JSON feed** we are
   not currently using: `calendar-full.json`. 5,438 events over 557 days plus
   tomorrow, structured (time, location, type, press-coverage level).
3. **Calendar density does predict utterances**, at ~1.8× from quiet to busy days
   — but the evidence is marginal (p ≈ 0.055–0.07) and it only applies to the
   1–2 days we can actually see.
4. **Beyond T+1 the schedule is close to unpredictable.** Day-of-week and holiday
   explain **4.3%** of the variance in daily event count (leave-one-out R²);
   day-over-day autocorrelation is r = +0.07. There is no useful day-type prior
   to build.
5. **Backtested on 2,288 logged signals, the weighting makes the forecast
   worse** — out-of-sample −0.62% Brier, and an oracle with perfect knowledge of
   the future schedule is worse still (−2.7% at β=1.0). The idea is falsified,
   not merely unsupported by data availability. **Do not build it.**
6. Two bigger findings fell out of the same table: the LLM's estimate is 7.9%
   worse than the flat Poisson it is given, and the market mid beats both by a
   wide margin. See §"Two incidental findings".

---

## Sources evaluated

| Source | Forward horizon | Format | Auth | Verdict |
|---|---|---|---|---|
| [`media-cdn.factba.se/rss/json/trump/calendar-full.json`](https://media-cdn.factba.se/rss/json/trump/calendar-full.json) | **T+1** | JSON, 2.3 MB | none | **Recommended.** Full history + tomorrow, structured, ETag/`max-age=60` |
| `media-cdn.factba.se/rss/csv/trump/calendar.csv` | T+1 | CSV, 100 KB | none | Same data, lossy (drops `day_summary` flags) |
| Factbase Google-Calendar iCal feed | T+1 | ICS, 10.6 MB | none | Same data, 4× the bytes, needs an ICS parser |
| [`rollcall.com/factbase/trump/topic/calendar/`](https://rollcall.com/factbase/trump/topic/calendar/) | T+1 | HTML | none | The page we already read — the JSON above is its backing feed |
| @WHPressPool (X) | T+1, **T+3 on Fri** | prose in tweets | X API | The upstream guidance. Factbase transcribes it; not worth scraping X |
| [`c-span.org/schedule`](https://www.c-span.org/schedule/) | T+7 | HTML | none | Horizon looks better than it is — days T+2 onward are generic placeholders ("Public affairs events…"), not named presidential events |
| [`kadoa.com/potus/schedule`](https://www.kadoa.com/potus/schedule) | T+1 | HTML | none | Third-party rescrape of the same guidance, no API. No reason to add a hop |
| `whitehouse.gov/live/` | none | HTML | none | Retrospective stream archive; `wp-json` is 403 |
| Factbase future-month pages (`/calendar/august-2026/`) | none | HTML | none | Render empty — confirms no forward data exists upstream |

Two things that *are* knowable far in advance, and are not daily-density data:

- **Congressional calendar**, published a year ahead — both chambers are out for
  nearly all of October 2026 and the first week of November (Election Day is
  Nov 3). Session status shifts where Trump is, not how much he talks.
- **Fixed political anchors** for late 2026: the GOP midterm convention in Dallas
  in September, UNGA in late September, the midterm run-up. NPR reports Trump is
  campaigning more at this point in 2026 than he did in 2018.

These are worth encoding as coarse *regime* flags eventually. They are not a
substitute for daily density and they are not what this note recommends.

### The feed

```json
{
  "date": "2026-07-29", "time": "08:00:00", "day_of_week": "Wednesday",
  "type": "President Schedule",
  "details": "The President participates in Executive Time",
  "location": "The White House", "coverage": "Closed Press",
  "day_summary": {"trump_property": null, "political_rally": null,
                  "golf": null, "fundraiser": null, "international": null}
}
```

`coverage` is the useful field — it separates `Closed Press` / `Executive Time`
from `Open Press`, `On Camera`, and the various pool designations. `day_summary`
carries golf / rally / fundraiser / international flags. Archive spans
2025-01-18 → tomorrow; mean 9.7 events/day (median 8, range 1–29).

---

## Does calendar density actually predict utterances?

### Measuring *when* the phrase was said

Kalshi's `close_time` is administrative and badly lagged — **median 43.9 hours**
after the fact, and batched (several markets stamped within the same second).
Only 9% of YES markets closed on the day the phrase was actually said. Any
analysis keyed on `close_time` is measuring Kalshi's ops schedule, not Trump's.

Instead, hits are dated from `market_candles`: the first hour the YES side priced
≥ 0.97, **restricted to markets that had traded below 0.85 earlier in the window**
so we catch a genuine repricing rather than a market that simply opened high.
That restriction matters — at a 0.60 threshold, 87% of "spikes" land on Monday,
which is just the Mon→Mon markets opening. 154 of 160 YES markets survive it.

This limits the sample to the candle era (from 2026-05-18, a rolling ~67-day
Kalshi window): **1,481 market-days, 154 hits**.

### Result

Every KXTRUMPSAY window runs Mon 14:00 → Mon 14:00, so weekday is perfectly
collinear with position-in-window. Raw weekday hazards therefore cannot be read
as an exposure effect — they mix in survival selection, since the phrases still
unresolved on Friday are the harder ones. The clean design is to compare **busy
vs quiet days within the same weekday**, which holds both fixed.

Mean within-stratum percentile rank of the day's exposure on hit days (0.5 = no
effect), permutation test, 5,000 shuffles within strata. All six metrics tried
are reported:

| Exposure metric | mean pct-rank | p |
|---|---|---|
| press-accessible events | 0.532 | **0.055** |
| all events | 0.531 | 0.071 |
| non-Executive-Time events | 0.530 | 0.075 |
| pool-coverage events | 0.523 | 0.124 |
| speaking-keyword events | 0.512 | 0.269 |
| on-camera / open-press events | 0.491 | 0.611 |

Monotone dose-response on event count:

| day's scheduled events | market-days | hits | hazard | weight |
|---|---|---|---|---|
| quiet (≤ 8) | 437 | 33 | 7.55% | 0.73× |
| mid (9–15) | 663 | 70 | 10.56% | 1.02× |
| busy (16+) | 381 | 51 | 13.39% | 1.29× |

Five of six metrics point the same way, the dose-response is monotone, and the
best two are p ≈ 0.06. That is real but not established — worth building on,
not worth betting size on yet.

### What I am explicitly *not* claiming

Weekend market-days show a 0.34× hazard vs 1.18× on weekdays, which looks like a
huge effect and is the intuitive version of the user's hypothesis. **I do not
believe it as an exposure effect.** Two reasons:

- It is fully confounded with survival position (Sat/Sun are days 5–6 of 7).
- The calendar contradicts it: Sunday averages **9.8** events (1.00× the overall
  mean) and 4.28 press-accessible events (0.94×). Saturday is 7.7 (0.79×). The
  event data says weekends are only slightly quieter, nothing like 3×.

I tried to break the collinearity using the long-window series
(`KXTRUMPSAYMONTH`/`NICKNAME`/`COMPANY`, where weekday floats freely within the
window) and it failed for lack of data — those markets mostly resolved before
candle coverage began, leaving 37 markets and **2 hits**. Not usable. Revisit
once ~3 more months of candles accumulate.

### Why there is no day-type prior beyond T+1

| predictor → daily event count | leave-one-out R² |
|---|---|
| weekday + federal holiday | **0.043** |
| weekday → press-accessible count | 0.047 |

Autocorrelation of daily event count: r = +0.07 (lag 1d), +0.11 (2d), −0.02
(3d), +0.13 (7d). Holidays average 10.5 events — *above* the 9.7 mean, because
Trump's holidays are Mar-a-Lago trips with press gaggles, not silence.

The schedule is near-white-noise around a stable mean. Nothing to extrapolate,
which is the honest answer to "what does a scheduled vacation look like": it
mostly does not look quiet, and we cannot see it coming anyway.

---

## Backtest — the decisive test

The hazard analysis above says "there is a signal". The only question that
matters is whether feeding it into the baseline produces a *better forecast*.
That is directly testable: the signal prompts are in `llm_queries`, so the flat
baseline we actually used is recoverable, and λ can be inverted exactly from the
printed percentage (`p = 1 − exp(−λ·days)`).

**2,288 logged signals over 178 resolved KXTRUMPSAY markets, 2026-05-17 →
2026-07-26.** Weighting is a one-parameter family so it can be fit honestly:

```
w_d = (events_d / mean_events) ** beta        # beta = 0 reproduces flat exactly
```

Point-in-time discipline: only days within the guidance horizon (today +
tomorrow, +2 on Fridays) get a weight; later days stay at 1.0.

### Undecided markets (`in_market_count == 0`), n = 2,117

| model | Brier | vs flat | LogLoss |
|---|---|---|---|
| **flat Poisson (current)** | **0.1997** | — | **0.6259** |
| exposure-weighted β=0.25 | 0.1995 | +0.11% | 0.6291 |
| exposure-weighted β=0.5 | 0.1997 | −0.02% | 0.6350 |
| exposure-weighted β=1.0 | 0.2015 | −0.92% | 0.6564 |

Restricting to ≤ 2.5 days left — the only window guidance covers, where the
effect must concentrate — gives the same shape: +0.34% at β=0.25, then negative.

**Out-of-sample:** β fit on 2026-05-17→06-30 picks 0.55; applied to
06-30→07-26 it makes Brier **0.62% worse** (0.2294 → 0.2308).

### It is not underpowered, and an oracle doesn't save it

The weighting genuinely moves the number — mean |Δp| is 1.5 pts at β=0.5 and
3.2 pts at β=1.0, with a max of 29 pts. So this is a real negative, not a null
from a too-small perturbation.

Running it as an **oracle** — using the full archive for every remaining day,
i.e. a perfect crystal ball for Trump's schedule, the upper bound on what any
forward-looking source could ever deliver:

| oracle β | Brier | vs flat |
|---|---|---|
| 0.25 | 0.2000 | −0.16% |
| 0.5 | 0.2010 | −0.66% |
| 1.0 | 0.2051 | −2.71% |
| 1.5 | 0.2128 | −6.57% |

Monotonically worse. **The ceiling on this idea is negative**, so the T+1
horizon was never the binding constraint — the hypothesis itself is wrong at the
magnitude that matters.

### Reconciling with the hazard result

Both can be true. The stratified hazard finding (1.29× busy / 0.73× quiet,
p≈0.06) is a *conditional* effect among markets surviving to that day. The
baseline's error, though, is dominated by λ — the 30-day rate — not by the shape
of the remaining window. A multiplicative weight that small adds more variance
than the bias it removes. The measured spread corresponds to roughly β ≈ 0.3,
and β = 0.25 is exactly where the effect vanishes into noise.

## Two incidental findings that matter more

Both fall out of the same table and are larger than anything above.

**1. The LLM's estimate is worse than the flat Poisson it is handed.**
On undecided markets: Brier 0.2155 vs 0.1997 (7.9% worse), LogLoss 0.6576 vs
0.6259. This independently reconfirms **F2 from the 2026-07-21 weekly review** on
a different window and a larger sample. The prompt tells the model to anchor on
the baseline and adjust from evidence; the adjustment is net destructive.

**2. The market mid beats every model we have, by a lot.**

| | Brier | LogLoss |
|---|---|---|
| market mid at signal | **0.1651** | **0.4923** |
| flat Poisson | 0.1997 | 0.6259 |
| LLM estimate | 0.2155 | 0.6576 |

On the ≤2.5-day subset the gap widens to 0.0575 vs 0.1284 vs 0.1446 — the price
is ~2.5× better than our estimate. If our probability is systematically worse
calibrated than the price we are transacting against, computed "edge" on this
series is largely noise, and acting on it is adverse selection. That is the
question worth spending the next block of effort on, not calendar weighting.

Caveat: mid-at-signal is a strong benchmark partly by construction, and signals
fire when something has already moved. The gap is large enough to be worth a
proper look regardless.

**Followed up in [signal-estimate-vs-market-price.md](signal-estimate-vs-market-price.md):**
across 6,624 signals the estimate's incremental coefficient over the price is
+0.033 (CI [−0.152, +0.214]) — redundant. The caveat above was tested by
restricting to long horizons and thin books; the price still wins.

## Proposed design (NOT recommended — falsified above)

*This is the design the hazard analysis pointed to, and it is exactly what the
backtest scored and rejected. Retained so that anyone revisiting the idea can
see precisely what was tested rather than assuming a weaker version was tried.*

Replace the flat term with an exposure-weighted one:

```
P(≥1 in remaining window) = 1 - exp(-λ · Σ_{d ∈ remaining} w_d)
```

Two tiers, because that is all the information that exists:

- **Tier 1 — days covered by published guidance (T+0, T+1; T+0…T+2 on Fridays):**
  `w_d` from the observed event count, calibrated to the dose-response above.
  Clamp to roughly [0.7, 1.3] — the measured range, not something wider.
- **Tier 2 — every later day:** `w_d = 1.0`. Flat, exactly as today. The R² of
  0.04 says we have earned no right to vary it.

Since Σw over a long window ≈ the day count, this is a no-op for markets with a
week left and only bites in the last day or two — where the whole edge is.

### Worked example

Weekly market, 30-day rate of 6 mentions (λ = 0.2/day), Monday close, priced
Saturday with Sat+Sun remaining. Flat baseline: **33.0%**. If the published
guidance shows two quiet days (w = 0.73 each) the baseline falls to **25.3%**;
if it shows two busy ones (w = 1.29) it rises to **40.3%**. On a contract
trading at 30¢ that spans pass → buy, from information visible the night before.

### Implementation sketch

- Extend `freqpred/ingestion/fetchers/factbase.py` with a `calendar-full.json`
  fetch, or add a sibling `factbase_calendar.py`. Conditional GET on the ETag —
  the feed sets `cache-control: max-age=60` and a strong ETag, so a daily poll
  is cheap and polite. Fetch shortly after midnight ET, when it refreshes.
- New table `factbase_calendar_days` (date PK, event counts by coverage class,
  the `day_summary` flags, `fetched_at`), plus the raw events if we want to
  revisit metric choice later without re-scraping.
- Per `CLAUDE.md`: new scheduled task ⇒ new `SERVICE_*` constant and
  `FreshnessSpec` in `freqpred/runtime/telemetry.py:build_freshness_specs()`,
  with its own heartbeat — not shared with `SERVICE_FACTBASE_SCHEDULER`.
- The weight enters `_build_factbase_block()` alongside the existing Poisson
  lines. Changing that prompt is a **signal-LLM change** and goes through the
  benchmark gates in `freqpred/signal/CLAUDE.md` — it cannot be adopted on
  inspection.

### Why it looked plausible right up until it was scored

Worth naming, because the failure mode generalises: the hazard analysis was a
*conditional* result (busy days differ from quiet days among surviving markets)
and I read it as a *forecasting* result (weighting by density improves the
estimate). Those are different claims, and only the second one pays. Nothing in
§"Does calendar density predict utterances" was wrong — it just did not imply
what the design assumed. The scoring step is what separated them, and it cost
far less than the fetcher would have.

### Caveats that would have applied before shipping

- **Statistical strength.** p ≈ 0.06 on the best metric, one series, 154 hits,
  10 weeks. The dose-response and the consistency across metrics are what make
  it credible; a single p-value here would not.
- **Sample is one regime.** May–July 2026. Campaign season may well behave
  differently — which is an argument for re-fitting, not for assuming.
- **Polling etiquette.** `rollcall.com/robots.txt` blocks `anthropic-ai`,
  `GPTBot`, etc. from *model training*; it allows general crawling, and
  `media-cdn.factba.se` is a separate CDN host with no robots.txt. We already
  call the Factbase phrase API. Daily conditional GETs are proportionate, but
  this is a courtesy dependency on a free feed with no SLA — treat a fetch
  failure as "fall back to flat weights", never as a blocking error.

## Reproducing

Scripts used are in the session scratchpad (`analyze*.py`, `forecast.py`);
they read `calendar-full.json` plus two DB exports. Core queries:

```sql
-- spike dating, with the "must rise from below" guard
WITH y AS (SELECT id FROM markets WHERE id ~ '^KXTRUMPSAY-' AND result='yes'),
c AS (SELECT c.market_id, c.end_period_ts AS ts,
             coalesce(c.yes_bid_close, c.price_close) AS px,
             min(coalesce(c.yes_bid_close, c.price_close)) OVER (
               PARTITION BY c.market_id ORDER BY c.end_period_ts
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS min_before
      FROM market_candles c JOIN y ON y.id = c.market_id)
SELECT market_id, min(ts) FROM c
WHERE px >= 0.97 AND min_before < 0.85 GROUP BY market_id;
```
