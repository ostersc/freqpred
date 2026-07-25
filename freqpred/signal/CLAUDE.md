# Signal LLM — change workflow

**For any change to the signal LLM or its inputs — `SYSTEM_PROMPT`, `build_prompt`, `PROMPT_VERSION` (`freqpred/signal/llm.py`), the signal model (`signal.model` config / `_DEFAULT_MODEL`), or any new/changed data block fed into the signal prompt:**

- Follow the full workflow in README → "Changing the signal prompt — the standard workflow". Non-negotiable steps: scope the edit to written-down findings, bump `PROMPT_VERSION`, regenerate the committed replay fixtures (`uv run freqpred fixtures replay --update`), and benchmark the new version with `scripts/benchmark_signals.py` (prompt mode for prompt/data changes; model mode with `--candidate-model` for model swaps) before treating the change as adopted. Do not adopt on inspection alone — propose the benchmark run to the user as the required validation step.
- Never regenerate `benchmarks/prompt_bank/` after a version bump — it is the frozen control baseline for the experiment; `record-bank` filters on the current version and would empty it.
- Before any benchmark run, check today's LLM spend against the daily cap (it is shared with the live pipeline; exhausting it blocks live signal analysis until the UTC day rolls over) and confirm the run with the user — benchmark runs cost real API dollars.
- One axis per experiment: a prompt change and a model change are separate benchmark decisions, never bundled.
