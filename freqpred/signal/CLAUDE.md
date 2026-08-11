# Signal LLM — change workflow

**For any change to the signal LLM or its inputs — `SYSTEM_PROMPT`, `build_prompt`, `PROMPT_VERSION` (`freqpred/signal/llm.py`), the signal model (`signal.model` config / `_DEFAULT_MODEL`), or any new/changed data block fed into the signal prompt:**

- Follow the full workflow in README → "Changing the signal prompt — the standard workflow". Non-negotiable steps: scope the edit to written-down findings, bump `PROMPT_VERSION`, regenerate the committed replay fixtures (`uv run freqpred fixtures replay --update`), and benchmark the new version with `scripts/benchmark_signals.py` (prompt mode for prompt/data changes; model mode with `--candidate-model` for model swaps) before treating the change as adopted. Do not adopt on inspection alone — propose the benchmark run to the user as the required validation step.
- Never regenerate `benchmarks/prompt_bank/` after a version bump — it is the frozen control baseline for the experiment; `record-bank` filters on the current version and would empty it.
- One axis per experiment: a prompt change and a model change are separate benchmark decisions, never bundled.

## Running a benchmark — the spend and scope checklist

Benchmark runs cost real API dollars against a cap **shared with live trading**, so exhausting it does not merely fail the experiment, it blocks live signal analysis until the UTC day rolls over.

**Before starting:**

1. Run `--estimate-only` first and **write down the projection**. Note that it covers candidate calls only — extraction (T101) is additional, and `--estimate-only` deliberately does not extract because that would itself cost money.
2. Check today's spend against the cap, and confirm the **live pipeline still has headroom afterwards**. A run that technically fits but leaves the trading loop at zero has taken the day's budget from production.
3. Confirm the scenario and market counts match the `--limit` you intended. `--fixtures` defaults to `tests/fixtures/replay` (8 committed regression fixtures), **not** `benchmarks/prompt_bank` — omitting it silently prices 8 scenarios instead of hundreds.
4. Confirm the run with the user. Never start one on your own initiative.

**Within the first two minutes of any run — not at the end:**

Compare a concrete counter against the scope you requested. Distinct markets touched, calls made, spend so far. **A counter that can exceed the requested scope IS the bug**, and it is visible immediately or not at all.

This is not hypothetical. On 2026-08-11 extraction ran inside `build_fixture_scenarios`, before `sample_markets`, so `--limit 50` still extracted every market in the bank: $3.17 across 944 calls on 66 markets that would never be sampled. The tell — 83 markets extracted against a limit of 50 — was visible two minutes in and was not looked at for fourteen. Serial execution was the only reason it stayed that cheap; T103 removes that accidental brake, which is why the check has to be deliberate.

When watching a run, make the watch itself falsifiable: measure the delta from a baseline captured *before* launch, never a rolling time window. A window that sweeps in a previous run's rows will cry wolf and get a healthy run killed — which also happened that day.
