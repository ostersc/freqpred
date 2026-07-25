# Weekly reviews

Written by the `weekly-review` skill (`.claude/skills/weekly-review/SKILL.md`).

```
reports/         YYYY-MM-DD.md — the readable record, one per week  (committed)
reports/data/    the JSON snapshots and raw .txt renders behind it  (committed)
backfill/        one-time historical dumps                          (gitignored)
```

Reports and their data are separated so `reports/` stays scannable — a year is 52
readable files rather than 52 buried among 200 snapshots.

## Week boundary

Weeks end **Tuesday 00:00 UTC**, not Sunday or Monday. Almost everything traded
is KXTRUMPSAY, which closes **Monday 10:00 ET** (299 of the finalized markets as
of 2026-07-25), and settlement lands within 2h of close (median 0.34h, max
1.97h). A Tuesday 00:00 UTC cutoff clears that by ~10 hours and puts exactly one
resolution batch in each window — so a week's entries and the settlement that
scores them never straddle the boundary.

The check that this is still true: a correctly aligned window reports `open=0`
in section 1. If it does not, markets are resolving on a different schedule and
the boundary needs revisiting.

## reports/

`YYYY-MM-DD.md` — the authored review: findings, recommendations, and the verdict
on the previous week's calls. The date is the week *ending* Tuesday.

`data/YYYY-MM-DD-{all,live}.json` and `.txt` — the run behind it, in both scopes.

Both the report and its JSON are committed. The snapshot looks like derived data
but is not reproducible: outcomes accumulate, so re-running that week months
later returns different numbers. The committed file is the only record of what
was known at the moment a recommendation was made — without it, scoring last
week's predictions turns into retrofitting them.

## backfill/

Gitignored. One-time orientation dumps produced by walking `--as-of` backwards;
re-runnable, and none of them was ever the basis of a decision. They also cannot
reconstruct what a past review *would* have said, because outcomes are never
rewound — see the skill's backfill section.
