---
name: weekly-review
description: Weekly profitability review for freqpred — analyse the last week's resolved markets across exits, entry gates, signal accuracy, the sizing assessor, and document sources, then propose the top 3 changes with quantified risk and reward. Use when asked to run the weekly review, "what should we improve this week", or invoked as /weekly-review.
---

# Weekly profitability review

Run once a week, after the week's markets have settled. The goal is not a status
report — it is **the two or three changes most likely to make money next week**,
each with an effect size, a confidence interval, a risk, and a revert trigger.

The numbers come from `freqpred metrics weekly-review`, which is deterministic,
makes no LLM calls, writes nothing to the DB, and is free to re-run. Your job is
the judgement on top of it: which findings are real, which are noise, and what to
do about them.

## 1. Pick the window end

**Weeks end Tuesday 00:00 UTC.** Almost everything traded is KXTRUMPSAY, which
closes Monday 10:00 ET (= 14:00 UTC), and settlement lands within 2h of close.
Tuesday 00:00 UTC clears that by ~10 hours and puts exactly one resolution batch
per window, so a week's entries and the settlement that scores them never
straddle the boundary. Always pass it explicitly:

```bash
WEEK_END=$(python3 -c "
from datetime import date, timedelta
t = date.today(); print((t - timedelta(days=(t.weekday()-1) % 7)).isoformat())")
echo "$WEEK_END"    # most recent completed Tuesday
```

Run today only if that Tuesday has passed; the in-progress week has no
resolution event in it yet. Then confirm the week actually settled:

```bash
docker exec freqpred-db-1 psql -U freqpred -d freqpred -c \
  "SELECT count(*) FILTER (WHERE status='finalized') AS resolved,
          count(*) FILTER (WHERE status<>'finalized') AS pending
   FROM markets WHERE close_time >= now() - interval '7 days';"
```

If most of the week's markets have not settled yet, say so and stop — a review
run against unresolved markets silently analyses only the short-dated ones.

## 2. Generate the review

Always run **both** scopes. Paper fills are frictionless and its P&L is not
evidence about live execution; the pooled run is only for sample size on the
signal-level sections, which are mode-independent.

```bash
uv run freqpred metrics weekly-review --weeks 1 --as-of "${WEEK_END}T00:00:00+00:00" --mode live \
  --json-out docs/weekly-review/reports/data/${WEEK_END}-live.json
uv run freqpred metrics weekly-review --weeks 1 --as-of "${WEEK_END}T00:00:00+00:00" \
  --json-out docs/weekly-review/reports/data/${WEEK_END}-all.json
```

**Verify the boundary held: section 1 must report `open=0`.** A non-zero count
means positions were still live at the cutoff, so the window splits a resolution
batch and the P&L is not the week's true result.

Useful variants: `--history-days 0` (all history, for a thin cohort),
`--all-versions` (pool signal prompt cohorts — read the per-version table first,
they are not exchangeable), `--as-of` (analyse a past week).

### Backfilling past weeks

One run per week, walking `--as-of` backwards. Use a short `--history-days` or
every week's sections 2–7 will be near-identical (the default 90-day diagnostic
window makes adjacent weeks share ~92% of their sample):

```bash
mkdir -p docs/weekly-review/backfill
python3 -c "
from datetime import date, timedelta
t = date.today()
d, first = t - timedelta(days=(t.weekday()-1) % 7), date(2026, 3, 24)  # Tuesdays
while d >= first:
    print(d.isoformat()); d -= timedelta(weeks=1)
" | while read -r d; do
  uv run freqpred metrics weekly-review --weeks 1 --history-days 14 \
    --as-of "${d}T00:00:00+00:00" \
    --json-out "docs/weekly-review/backfill/${d}.json" \
    > "docs/weekly-review/backfill/${d}.txt"
done
```

`backfill/` is gitignored — those dumps are derived and re-runnable. The weekly
report and its JSON snapshot are not; see the tracking rule in step 7.

**A backfill is not a reconstruction of what that week's review would have
said.** Outcomes are never rewound: `result` is always as known today, so a
backfilled week sees markets that had not settled at the time. That makes the
numbers better informed than the real review would have been — fine for building
a history, wrong for judging what was knowable then. The signal cohort *is*
rewound (`--as-of` selects the version live at that date, not today's).

Before backfilling, check whether you need it: section 5 already reports the full
weekly accuracy series and the per-prompt-version cohorts from a single run. Only
the ledger and the window-only exit table are genuinely per-week. And the
prediction-scoring loop in step 6 cannot be backfilled at all — it needs reports
that actually made predictions, so it starts from the first real review.

## 3. Read each section — and its trap

A `*` marks a bootstrap CI that excludes zero. **The report emits roughly fifty
of these cells, so at 95% expect two or three false stars every single run.** A
star is a filter, never a finding. Rank by effect size and mechanism, not by
which cell happened to clear zero.

| Section | The question | The trap |
|---|---|---|
| 1. Ledger | What happened | Nothing — these are counts, not inferences |
| 2. Exit effectiveness | Did exiting beat holding to settlement? | The only unbiased counterfactual in the report. `n` is small; check `helped`/`hurt`, not just the total, because one large position can carry the sum |
| 3. Stoploss sweep | Would another threshold have paid? | **Both arms are biased, in opposite directions.** "All positions" has MAE censored at the actual exit, so wide thresholds under-count stops that would still have fired. "Uncensored only" has full-life MAE but its population is conditioned on *not* having stopped, which flatters no-stoploss by construction. Truth is between them; section 2 is the clean read on the current threshold |
| 4. Entry gates | Are we refusing trades that pay? | Marginal — each gate is scored with the others held fixed, so a signal two gates reject counts against neither. Gates showing `blocked n=0` are inert given the rest of the config, not harmless in isolation |
| 5. Signal accuracy | Getting better or worse, and where? | The most recent weeks are incomplete (only settled markets appear), so the last row over-represents short-dated markets. Read the trend from complete weeks. Prompt-version step changes are usually the real story |
| 6. Assessor | Is sizing earning its cost? | Capital tilt near zero means a flat tax, not a discriminator. A rising neutral-fallback rate is an *outage*, not a quality change — check that first |
| 7. Sources | Which sources correlate with profit? | Overlapping populations: a signal retrieving six sources counts under all six, so these do not decompose additively, and a bad-looking source may just co-occur with hard markets |

Two things the report cannot see, which you must not paper over:

- **`min_volume_24h` is not evaluable.** `markets.volume_24h` holds the current
  value, not the value at signal time. Say it is unmeasured; never estimate it.
- **Sample size is `n_markets`, not `n_signals`.** Re-evaluation produces dozens
  of signals per market. Below ~30 markets a slice is a lead, not a result.

## 4. Rank the opportunities

Order candidates by **expected dollars per week × probability the effect is
real**, then break ties toward the cheapest and most reversible change.

Rough hierarchy, from the standing evidence:

1. **Exit and stoploss rules** — largest realised dollar swings, and a config
   change is one line and instantly revertible.
2. **Entry gates** — directly changes which trades exist. Also cheap to revert,
   but changes volume, so re-check the exposure caps.
3. **Signal generation** — the biggest lever and the slowest: a systematic
   direction or cohort bias is worth more than any sizing work, but changing the
   prompt requires the full benchmark in `freqpred/signal/CLAUDE.md`.
4. **Sizing assessor** — smallest lever, most expensive per unit of improvement
   (~$0.04/signal). As of 2026-07-25 no assessor package has beaten a free
   direction×edge-band lookup on incremental AUC. Treat "retire it" as a live
   candidate whenever tilt is flat.
5. **Ingestion sources** — only act on a source that is both unprofitable *and*
   independently confirmable.

## 5. Quantify each recommendation

Every recommendation needs all six fields. A recommendation missing the risk or
the revert trigger is not finished.

```markdown
### R1 — <the change, stated as a diff>

- **Evidence**: section + the specific numbers, including n_markets and the CI.
- **Expected effect**: $X/week (range $A–$B from the CI). Show the arithmetic.
- **Risk**: what breaks if the estimate is wrong, and the worst realistic case.
- **Confidence**: high / medium / low, and what would raise it.
- **Verification**: the exact command or query that will confirm or refute it.
- **Revert trigger**: the observation that means back it out, decided now.
```

Convert per-contract figures into weekly dollars using the actual traded volume
from section 1 — never quote `profit_edge` alone as if it were P&L.

Hard rules on what you may recommend:

- **Never adopt a signal-prompt or assessor change from this review.** Those have
  their own validation gates (`freqpred/signal/CLAUDE.md`,
  `freqpred/metrics/CLAUDE.md`) and both require a run proposed to the user. This
  review may *nominate* such a change and should say which gate it must pass.
- **Never recommend on one week alone.** Require either a CI excluding zero on
  the cumulative window, or the same sign in the previous week's review.
- **Never recommend a change to something marked unevaluable.**
- Cap at **three** recommendations. If a fourth is genuinely strong, drop the
  weakest — a list of eight is a list of none.

## 6. Score last week's calls before writing new ones

Read the most recent report in `docs/weekly-review/reports/`. For every recommendation
that was adopted, state what actually happened versus what you predicted, and
whether any revert trigger fired. A prediction that missed is the most useful
signal in the whole review — record it plainly rather than re-explaining it away.
If a recommendation was made and *not* adopted, either re-make it with the extra
week of evidence or retire it explicitly.

## 7. Write the report

Save to `docs/weekly-review/reports/YYYY-MM-DD.md`. **Commit both the report and the
`.json` snapshot from step 2.** The snapshot is not derived data: outcomes
accumulate, so re-running that week later returns different numbers and the
committed file is the only record of what was known when the call was made —
which is what makes step 6's scoring honest rather than retrofitted. At ~27KB a
week that is a rounding error in the repo. Only `backfill/` is ignored.

Report structure:

```markdown
# Weekly review — YYYY-MM-DD

## Last week's calls
| Recommendation | Adopted? | Predicted | Actual | Verdict |

## This week
- 3–5 lines: P&L, win rate, notable exits, anything anomalous.

## Findings
- Only what survived section 3's traps. State n_markets and the CI for each.

## Recommendations
### R1 ... (the six-field block above)

## Watchlist
- Leads too thin to act on, with the n they need before they qualify.
```

Then give the user the three recommendations in chat, most valuable first, with
the expected weekly dollars and the risk on one line each. Do not apply any
change yourself unless asked — present, then wait.

## Guardrails

- The review is read-only. It never writes to the DB and never places a trade.
- If a section is empty because the week had no activity of that kind, say so.
  Do not fill the gap with the cumulative number presented as this week's.
- If the data contradicts a previous week's conclusion, lead with that.
- Report a null result plainly. "Nothing this week clears the bar; here are the
  two leads and the n they need" is a correct and complete review.
