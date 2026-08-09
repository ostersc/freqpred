# freqpred

**LLM-driven prediction market trading framework — freqtrade for prediction markets.**

freqpred uses retrieval-augmented LLM analysis to estimate the "true" probability of prediction market outcomes, identifies edges against market-implied prices, and executes trades with assessment-aware sizing plus systematic risk controls.

**Status:** Phase 2 complete (paper trading + calibration running). Phase 3 (live trading) in progress.

---

## What it does

1. **Monitors** active markets on Kalshi (Politics, Technology, Economics, ...)
2. **Ingests** targeted news via Tavily, NewsAPI, The Guardian, Reddit, GDELT, TV transcripts + chyrons, and Truth Social (catalyst-driven RAG)
3. **Estimates** event probability using Claude Sonnet with structured output
4. **Identifies** markets where LLM probability diverges meaningfully from market price
5. **Sizes and trades** via a pluggable strategy interface, assessment-aware sizing, and hard risk controls
6. **Tracks** calibration — are our probability estimates actually accurate?

> **Why no backtesting?** LLMs have seen market resolutions in training data, creating unavoidable look-ahead bias. freqpred validates via paper trading + Brier score calibration instead.

For full architecture, data models, strategy interface, and roadmap, see **[SPEC.md](SPEC.md)**.

---

## Setup

**Prerequisites:** Python 3.12+, Docker (for Postgres), `uv`

```bash
# 1. Install dependencies
uv sync

# 2. Configure
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml — or set env vars (see below)

# 3. Start local services
docker-compose up -d db

# 4. Apply database migrations
#    Migrations read DATABASE_URL from the environment ONLY — they do not read
#    config.yaml, and .env is not auto-loaded. Export it first:
export DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred"
uv run freqpred db migrate

# 5. (Optional) Database web UI at http://localhost:8080
docker-compose up -d adminer
```

**Required env vars** (set in `.env` or directly):

```
ANTHROPIC_API_KEY=...     # Claude — signal analysis + catalyst generation
DATABASE_URL=postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred

# Optional — enables news/social fetching (Reddit, GDELT, and TV sources need no key)
TAVILY_API_KEY=...
NEWSAPI_KEY=...
GUARDIAN_API_KEY=...
TRUTHSOCIAL_USERNAME=...
TRUTHSOCIAL_PASSWORD=...

# Optional — enables live Kalshi trading
KALSHI_API_KEY=...
KALSHI_PRIVATE_KEY_PATH=...

# Optional — Telegram + Discord alerts
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK_URL=...
```

---

## Running

```bash
# Signal analysis only — no trades placed
uv run freqpred run --strategy ConservativeDefault --mode signal-only

# Paper trading (default)
uv run freqpred run --strategy ConservativeDefault --mode paper

# Live trading (requires LIVE_TRADING_ENABLED=true)
uv run freqpred run --strategy ConservativeDefault --mode live
```

Point `--strategy` at a `.py` file to load a custom strategy:

```bash
uv run freqpred run --strategy strategies/my_strategy.py --mode paper
```

Bundled strategies: `ConservativeDefault`, `PoliticsEdgeStrategy`, `TechNewsStrategy`, `FreshMarketStrategy` — plus `DemoHarness`, which only runs against the Kalshi demo environment to verify order plumbing (see SPEC.md §14).

---

## Architecture

```mermaid
graph TD
    subgraph Exchange[Kalshi Exchange]
        KREST[REST API]
        KWS[WebSocket]
    end

    subgraph Ingestion[Ingestion]
        MW[Market Watcher]
        MS[Market Selector]
        CG[Catalyst Generator]
        IS[Ingestion Scheduler]
        RS[Realtime Scheduler]
        FB[FactBase Scheduler]
        DS[(Document Store)]
        FBD[(FactBase Phrase DB)]
    end

    subgraph Metrics[Metrics]
        SQS[Source Quality Scheduler]
        SQ[(Source Quality Snapshots)]
    end

    subgraph Trading[Signal and Trading]
        SP[Signal Pipeline]
        SE[Strategy Engine]
        AS[Assessment]
        OM[Order Manager]
    end

    subgraph Runtime[Runtime]
        PW[Position Watcher]
        PM[Position Monitor]
        L[(Ledger and DB)]
        API[Dashboard API]
        UI[Dashboard and Alerts]
    end

    subgraph Plugin[Strategy Plugin]
        STRAT[IPredictionStrategy]
    end

    KREST --> MW
    MW --> MS
    MS --> CG
    CG --> IS
    CG --> RS
    IS --> DS
    RS --> DS
    FB --> FBD
    FBD --> SP
    FBD --> AS
    DS --> SP
    MW -->|prices| SP
    SP --> SE
    SE --> AS
    SQ --> AS
    L --> AS
    AS --> OM
    OM --> KREST
    KWS --> PW
    PW --> PM
    PM --> L
    OM --> L
    L --> API
    API --> UI
    SQS --> SQ
    MS -. is_market_interesting .-> STRAT
    FB -. phrase cache gate .-> STRAT
    SE -. should_trade .-> STRAT
    AS -. assessment-aware position_size .-> STRAT
    PM -. exits and resolution .-> STRAT

    classDef exchange fill:#e8e8e8,stroke:#888,color:#333
    classDef ingestion fill:#d4edda,stroke:#28a745,color:#155724
    classDef metrics fill:#e9ecef,stroke:#6c757d,color:#343a40
    classDef signal fill:#cce5ff,stroke:#0069d9,color:#004085
    classDef position fill:#fff3cd,stroke:#d39e00,color:#533f03
    classDef strategy fill:#f8d7da,stroke:#721c24,stroke-width:3px,color:#721c24

    class KREST,KWS exchange
    class MW,MS,CG,IS,RS,FB,DS,FBD ingestion
    class SQS,SQ metrics
    class SP,SE,AS,OM signal
    class PW,PM,L,API,UI position
    class STRAT strategy

    linkStyle 25 stroke:#721c24,stroke-width:2px
    linkStyle 26 stroke:#721c24,stroke-width:2px
    linkStyle 27 stroke:#721c24,stroke-width:2px
    linkStyle 28 stroke:#721c24,stroke-width:2px
    linkStyle 29 stroke:#721c24,stroke-width:2px
```

Four concurrent subsystems: **ingestion** (catalyst-driven, continuous), **signal and trading** (triggered, RAG + LLM + assessment-aware sizing), **position watcher** (WebSocket, sub-second price + resolution events), and **metrics** (daily source-quality snapshots). Persisted assessment and source-quality data are then surfaced in dashboard detail views.

Red dashed edges show `IPredictionStrategy` callback invocations — the single plugin point for market selection, entry, exit, and resolution logic.

See [SPEC.md §6](SPEC.md) for component responsibilities, [SPEC.md §8](SPEC.md) for the full strategy interface, and [SPEC.md §9](SPEC.md) for the signal pipeline.

### Code layout

Where each subsystem above lives:

```
freqpred/
├── markets/       Kalshi REST client, market watcher, position watcher
├── ingestion/     market selector, catalyst generator, schedulers, fetchers, document store
├── rag/           sentence-transformers embedder + pgvector retriever
├── signal/        signal pipeline + LLM analysis
├── strategy/      IPredictionStrategy interface + bundled strategies
├── trading/       order manager, risk caps, ledger
├── metrics/       calibration, sizing assessment, reporting, series history
├── llm/           Claude client + LLM audit log
├── runtime/       freshness telemetry (per-service heartbeats)
├── alerts/        Telegram + Discord dispatch
├── bench/         benchmark harness internals
├── replay/        frozen fixture replay
├── ops/           operational helpers
└── dashboard/
    ├── api/       FastAPI routes, schemas, app
    └── ui/        React app (Vite) — src/{api,components,hooks,pages}, 10 pages

benchmarks/   scenario banks + eval cache        migrations/   Alembic revisions
config/       config.example.yaml (+ gitignored local config.yaml)
scripts/      audit + eval tooling               strategies/   user strategy files (gitignored)
tests/        unit/ (no DB, mocked) + integration/ (needs Postgres)
```

---

## CLI and alerts

See **[COMMANDS.md](COMMANDS.md)** for the full CLI reference and all Telegram bot commands.

Quick reference:

```bash
uv run freqpred run --strategy ConservativeDefault --mode paper
uv run freqpred markets list
uv run freqpred signal analyze --market-id <KALSHI-TICKER>
uv run freqpred ingestion run --limit 5
uv run freqpred positions list --status open
uv run freqpred metrics calibration
uv run freqpred report digest
uv run freqpred alerts test --channel all
uv run freqpred db migrate
uv run freqpred dashboard  # dev-only Vite launcher (requires freqpred run for API)
```

**Alerts** — Telegram and Discord are independently optional. Missing credentials silently disable that channel. Set `telegram_authorized_users` in `config.yaml` to enable inbound bot commands (status queries, position management, circuit breaker control). See [COMMANDS.md — Telegram bot commands](COMMANDS.md#telegram-bot-commands).

**Dashboard** — the current UI includes Signal Feed, Positions, Decisions, Markets, Calibration, P&L Over Time, Source Quality, LLM Cost, Strategy Config, and System Health. Signal and position detail views also show persisted assessment summaries and deep-link into the LLM audit page.

---

## Development

```bash
# Install dev dependencies + lint hook (one-time)
uv sync --group dev
uv run pre-commit install

# Run all unit tests
uv run pytest tests/unit/

# Run integration tests (requires running Postgres)
uv run pytest tests/

# Lint (also runs in pre-commit and CI)
uv run ruff check .

# Create a new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations (needs DATABASE_URL exported — see Setup step 4)
uv run freqpred db migrate

# Apply/roll back against a non-default database (test, demo) — pass it inline
DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" uv run alembic upgrade head
DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" uv run alembic downgrade base

# Dashboard dev server (frontend hot-reload, proxies /api to freqpred run on 8000)
# Start freqpred run first, then:
uv run freqpred dashboard
# OR manually:
cd freqpred/dashboard/ui && npm install && npm run dev
```

Adminer (database UI): `docker-compose up -d adminer` → `http://localhost:8080` (server: `db`, user/pass/db: `freqpred`)

### Benchmarking a model or prompt change

Before switching the signal model or merging a `PROMPT_VERSION` bump, benchmark the candidate against resolved-market outcomes (supersedes the old `compare_model_signals.py`):

```bash
# Model swap: replay each resolved market's stored prompt verbatim to the candidate
uv run python scripts/benchmark_signals.py --candidate-model claude-sonnet-5 \
    --training-cutoff 2026-03-01 --limit 50 --reps 3 --json-out benchmarks/sonnet5.json

# Prompt change: re-render frozen fixtures through the CURRENT prompt template.
# Build the resolved-market scenario bank first (leakage-free by construction;
# records EVERY LLM-backed signal per market — the benchmark samples from it):
uv run freqpred fixtures record-bank
uv run python scripts/benchmark_signals.py --prompt-mode --fixtures benchmarks/prompt_bank \
    --training-cutoff 2026-03-01 --limit 250

# Sampling: --limit N picks N MARKETS at random (seeded via --seed, so runs are
# reproducible), and --per-market picks which of each market's signals to score:
# spread:3 (default — early/mid/late decision points), all, first (earliest
# signal — the purest entry decision), or last (the final pre-resolution
# signal only; the market has usually converged by then).
# Identical evals (model+thinking+prompt) are served free from
# benchmarks/.eval_cache — re-runs and extensions only pay for new calls;
# --no-cache forces fresh calls.

# Preview call volume and token cost first
uv run python scripts/benchmark_signals.py --candidate-model claude-sonnet-5 \
    --training-cutoff 2026-03-01 --estimate-only

# Pre-4.6 candidates (e.g. Haiku 4.5) reject adaptive thinking — omit it
uv run python scripts/benchmark_signals.py --candidate-model claude-haiku-4-5-20251001 \
    --training-cutoff 2026-03-01 --thinking none
```

**When to use it:** before any signal-model swap, any prompt-template change, or judgment-relevant config changes (thinking settings, max_tokens). Not for regression testing (that's the free, deterministic replay harness — `freqpred fixtures replay`), not for P&L estimation, and not for scheduled monitoring — every run costs real API dollars (audited to `llm_queries`, counted against the daily spend cap).

**The adopt/reject decision rule:**

1. **Adopt only on a significant paired Brier delta** — bootstrap 95% CI excluding zero, or sign test p < 0.05. A better raw mean on a small noisy sample is not evidence. Statistics are **clustered by market**: signals on one market share its outcome, so the bootstrap resamples markets and the sign test votes once per market — 16 correlated snapshots of one market cannot manufacture significance.
2. **Guard: trade decisions must not degrade** — check the would-trade rate, disagreement table, per-trade EV, and the stake-weighted P&L. Confidence scales position size in production (the Kelly blend), so each would-trade is also sized by the benchmark strategy's own `position_size()` from that model's posterior + confidence (`--strategy`, default `PoliticsEdgeStrategy`; the strategy config also supplies the default `min_edge`/`min_confidence` gates) — an overconfident model loses proportionally more when wrong, and a better-calibrated but timid candidate shows up as too small a total stake. The gate applies the signal-level entry filters from the strategy config exactly as `should_trade` does (`max_edge`, `min_mid_price`/`max_mid_price` on the entry side's cost) — without them the numbers include longshot trades production would never take. Known simplifications vs live P&L: entry fills at the frozen ask (production posts resting limits), and positions ride to settlement (production exits via stoploss/signal/force_exit).
3. **Tiebreaker: cost and latency.**
4. **Ambiguous → keep the incumbent**; it has live calibration history, the candidate has none.

`--training-cutoff` is required: markets that closed inside the candidate's training window are excluded, since their outcomes may be memorized rather than forecast.

### Changing the signal prompt — the standard workflow

Every edit to `SYSTEM_PROMPT` or `build_prompt` follows this sequence. The goal: no prompt version reaches live trading without benchmark evidence against real resolved-market outcomes.

1. **Scope the edit.** Change only what a specific, written-down finding justifies — no drive-by rewording. The benchmark measures the whole diff; unrelated edits make a negative result unattributable.
2. **Bump `PROMPT_VERSION`** in `freqpred/signal/llm.py` (`signal-vN` → `signal-vN+1`). The replay harness guards (rendered-prompt snapshot, system-prompt hash, version pin test) fail on any unbumped edit — that's by design.
3. **Regenerate the committed replay fixtures**: `uv run freqpred fixtures replay --update`, then run the full test suite.
4. **Treat `benchmarks/prompt_bank/` as the experiment's control artifact.** `record-bank` gates every fixture on a byte-exact re-render of the stored prompt under the *current* `build_prompt`. A SYSTEM_PROMPT-only change (like v9 → v10) keeps all old signals recordable, but if your edit touches `build_prompt`'s user-prompt template, older signals stop round-tripping — record the bank *before* such a change and don't regenerate it mid-experiment. The eval cache (`benchmarks/.eval_cache`) keys on the full prompt, so re-runs after an accidental regeneration still reuse any evals whose prompts are unchanged.
5. **Check spend headroom first.** Benchmark calls share the daily LLM cap with the live pipeline; a run that exhausts the cap stops early **and blocks live signal analysis until the UTC day rolls over**. A full-bank run costs ~$3.50 typical (`--estimate-only` gives the exact projection). Today's spend:
   ```bash
   docker exec freqpred-db-1 psql -U freqpred -d freqpred -c \
     "SELECT SUM(cost_usd) FROM llm_queries WHERE created_at >= date_trunc('day', now() AT TIME ZONE 'utc');"
   ```
6. **Run the isolation cell** — incumbent model on the new prompt vs its stored old-prompt outputs (change one axis at a time):
   ```bash
   uv run python scripts/benchmark_signals.py --prompt-mode --fixtures benchmarks/prompt_bank \
       --training-cutoff <incumbent cutoff> --limit 250 --json-out benchmarks/<vN>_isolation.json
   ```
7. **Apply the adopt/reject rule above** — the paired-Brier gate, then the degradation guard (would-trade, disagreements, stake-weighted P&L), then the regime split (a favorites-heavy bank can hide upset-side regressions; read both).
8. **Adopt or revert.** Adopt: merge — the new version goes live on the next signal run, and its signals start accruing toward a future bank as their markets resolve. Reject: revert the bump (fixtures regenerate back); the old bank remains the control for the next attempt.
9. **A model swap on top of a new prompt is a separate experiment** (model mode, new prompt as incumbent baseline) — never change both axes in one adoption decision.

### Auditing the sizing assessor

`scripts/audit_assessor_enhancement.py` is the paired A/B harness for the trade-sizing assessor (`assess_signal_context()`), the counterpart of `benchmark_signals.py` for the assessment prompt rather than the signal prompt. Use it whenever you change what the assessor sees or how it is asked to judge — an edit to `_SYSTEM_PROMPT` or `_build_prompt_payload` in `freqpred/metrics/assessment.py`, a new context section (the T94 calibration data went through it), a `judgment_model` swap, or a proposed `assessment_scale_min/max` change. Like signal-prompt changes, an assessment-prompt change should not be adopted on inspection alone: the script replays real historical signals with known outcomes through each competing assessor package, one live judgment-model call per signal per arm, and reports whether corr(trust_score, outcome) actually improved.

How it works and what it protects against:

- **Point-in-time context, not current DB state.** The script rebuilds each signal's assessment context with PIT copies of the production loaders — the assessed market is excluded, and only markets closed / positions exited / source-quality snapshots computed *before the signal's `created_at`* are admitted. This is not optional rigor: on since-resolved markets the production loaders self-leak (the exact-question Brier history contains the assessed market's own outcome — measured at r=0.85 vs the honest 0.43, i.e. the model reads the answer key). Production is immune (an unresolved market has no outcome to leak); every retrospective audit must use the PIT loaders.
- **No production pollution.** Calls go through the real `LLMClient` under `query_type="model_eval"` with the daily spend cap enforced; nothing is written to `signal_assessments`.
- **Arms are packages, not just payloads** (`--arms`, default `current,challenger`): an arm is a system prompt *and* payload shape *and* version string pinned together. `current` = the live production package (production `_SYSTEM_PROMPT`, production section loaders, production `_PROMPT_VERSION`) — zero maintenance, it is whatever shipped. `challenger` = the proposed package, defined per experiment in the CHALLENGER block at the top of the script — version string, system prompt, payload builder, and an optional `CHALLENGER_MODEL` judgment-model override so a model swap (e.g. opus-4-7 → opus-4-8) can be screened as its own single-axis experiment (undefined by default; the run fails loudly if requested without being defined, so you can never accidentally measure current-vs-current). `--reuse-csv <prior run>` carries per-signal results for arms you are not re-running from an earlier same-seed run, so a budget-constrained day pays only for the new arm(s); `--out` sets the merged result CSV.
- **Spend**: ~$0.02 per Opus call, one call per signal per live arm — a full two-arm 30-signal run is ~$1.20. Same rule as benchmark runs: check today's spend against the daily cap first (the cap is shared with the live pipeline) and confirm the run with the user.
- **Read the result like a benchmark**: the paired bootstrap CI on the correlation difference is the adopt/reject gate (`scripts/.audit_output/analyze_audit.py`); the stake-weighted P&L comparison is the degradation guard; the verdict distribution (size_down/neutral/size_up per arm) is the structural-bias check — an assessor that never says size_up isn't discriminating, it's a flat tax. Sampling is seeded (`SEED`/`SAMPLE_N`/`MAX_PER_MARKET` constants) so runs are reproducible and extendable.

#### Frozen eval set (preferred since 2026-07-25)

`_pick_sample`'s reshuffling draw made results unusable: the *identical* v6/opus-4-7 package scored corr +0.529 on one draw and +0.246 on the next, so run-to-run differences were sample composition rather than package quality. `scripts/freeze_assessor_eval_set.py` fixes this and costs **nothing** — it reuses the `freqpred.replay.recorder` premise that everything needed is already in the DB:

- **Frozen sample + frozen payloads.** 76 KXTRUMPSAY / `signal-v11` signals, direction-balanced 38 NO / 38 YES and band-matched (NO is the scarce side at 38 available, so it takes all of them). The fully-rendered PIT payload for each arm is stored verbatim with a hash, so runs are byte-reproducible and a harness change invalidates a cached score instead of silently mixing old scores with new prompts.
- **Direction balance is the point.** NO earns +7.3pp over the price paid while YES loses 7.6pp across 8,410 resolved signals — the strongest per-signal discriminator in the data — and the natural mix (22 YES / 8 NO in the last shuffled draw) was far too thin on NO to resolve it. Accepted trade: absolute AUC is no longer representative of live performance, but the arm-vs-arm contrast is much cleaner, which is what the audit is for.
- **`scripts/run_frozen_eval.py`** replays the stored payloads (no DB reconstruction), is resumable from its output CSV, and reuses already-paid `current`-arm responses — sound because the judge is effectively deterministic (7 signals scored 2–3× by the identical package returned identical trust_scores). After the first pass, `current` is cached and a package screen costs only the challenger arm.
- **Analyse with `scripts/.audit_output/analyze_noninferiority.py`**, which reports **capital tilt** (mean multiplier on winners minus losers — the direct expression of "put more size on winners"), AUC, and **incremental AUC over a free direction×band base-rate prior**. AUC is ~2× better powered than Pearson correlation here (CI width 0.336 vs 0.622) and is scale-invariant, so it separates the ranking question from the calibration question. The incremental-over-prior line is the honest adoption gate: absolute AUC flatters every arm because most of it is reproducible from a lookup table that costs nothing.

Reference runs (result CSVs are gitignored and stay on the machine that paid for the run, so the figures below are the record):

**2026-08-09, judgment-model swap opus-5 → `z-ai/glm-5.2`** (assessment-v8 unchanged, single axis) — adopted. Non-inferior and ahead on every point estimate: capital tilt +0.0908x vs +0.0724x (CI −0.0164..+0.0525), AUC 0.712 vs 0.691, corr +0.354 vs +0.338; 35 `size_up` at an 80.0% hit rate vs 19 at 78.9% (base 53.9%), arms agreeing at r=0.856; cost $0.0109 vs $0.0479 per assessment at comparable latency. Formal verdict printed INCONCLUSIVE only because the correlation CI's lower bound (−0.102) missed the −0.10 tolerance by 0.002. Two things this run establishes for any future model screen: **(a)** a challenger model may need its own `max_tokens` — GLM returned a `tool_use` block with `input={}` at the audited 1024 after spending the whole cap on reasoning tokens, so `CHALLENGER_MAX_TOKENS` now sets the challenger arm's budget independently; **(b)** the `current` arm can be seeded for free from a prior run's `challenger` column once that challenger is production (all 76 payload hashes are identical across arms), which is what made this screen cost $0.83 instead of ~$4.50. Note the frozen set's `cached_current_response` entries were harvested under v6/opus-4-7 and the payload hash did not change at v8, so they must be stripped before reuse or they will silently score a stale package as "current". 2026-07-25, assessment-v8 + opus-5 on the frozen 76-signal set — capital tilt **+0.0751x vs +0.0170x, 95% CI (+0.0161, +0.0988)**, the first significant arm difference across the whole effort; ranking a wash (AUC 0.700 vs 0.674); v6 was inert (sd 0.027, 76/76 size_down) while v8 issued 19 size_up at a 78.9% hit rate vs a 54.1% base. Caveats: v8's edge is almost entirely *between* directions (within-YES AUC 0.294, below random), and neither arm beat the free prior (0.685). 2026-07-11, T94 ([#94](https://github.com/ostersc/freqpred/issues/94)) — prototype payload lifted corr(trust, outcome) 0.432 → 0.623 (95% CI on diff +0.019..+0.42) on 30 signals. 2026-07-12, T95 ([#95](https://github.com/ostersc/freqpred/issues/95)) three-arm re-run on the same 30 signals — control 0.432 / t94-as-shipped 0.492 / t95 0.569; the t95−t94 CI spanned zero at n=30 but t95 led every point estimate and produced the only size_up verdict (on a winner); note t94-as-shipped did not replicate the prototype's significance (+0.060 vs +0.191 against the same control), a caution on trusting any single n=30 run's CI. 2026-07-12, judgment-model screen opus-4-7 → opus-4-8 (v6 package unchanged; baseline = the adoption run's t95 arm via `--reuse-csv`) — wash: corr +0.588 vs +0.569, CI (−0.175, +0.157), same multiplier corr, slightly worse sample ROI, and 4.8 issued zero size_up where 4.7 had one (on a winner); stayed on opus-4-7.

### Weekly profitability review

`freqpred metrics weekly-review` is the recurring counterfactual analysis over resolved markets — the counterpart to the two audit harnesses above, but for the *strategy* rather than a model or prompt. It is deterministic, makes no LLM calls, writes nothing, and is free to re-run, so it can be pointed at any window.

Seven sections: the window ledger; **exit effectiveness** (what every early exit earned versus holding to settlement — the only unbiased counterfactual in the report); an MAE-based **stoploss threshold sweep**; **marginal entry-gate analysis** (for each `StrategyConfig` gate, the realised profit-vs-price of the signals it blocks versus the ones it admits, with every other gate held fixed); **signal accuracy** by week, prompt version, and correlate slice; **assessor capital tilt**; and **per-source realised edge**.

Where candle coverage exists it also runs a **candle-based** stoploss sweep over real price paths — free of both biases in the MAE version, since the path continues past the actual exit and every covered position is included. `freqpred candles backfill` populates it; see COMMANDS.md.

Three things it deliberately refuses to fake: `min_volume_24h` is not evaluable point-in-time (`markets.volume_24h` is the *current* value), a signal whose order book cannot be reconstructed is admitted rather than blocked by the spread gate, and the stoploss sweep reports both a censored and an uncensored arm because neither is unbiased on its own — MAE stops updating at the actual exit, while the uncensored population is conditioned on not having stopped.

`.claude/skills/weekly-review/SKILL.md` (invoked as `/weekly-review`) is the weekly procedure built on it: read the review, discard what the traps explain, and produce at most three changes with an effect size, a confidence interval, a risk, and a revert trigger. Reports land in `docs/weekly-review/reports/` so each week scores the previous week's predictions.

### Utility scripts

Other one-off analysis and maintenance scripts live in `scripts/` — see each script's module docstring for full usage.

---

## Tech stack

- **Python 3.12+**, FastAPI, SQLAlchemy 2.0 async, Alembic
- **PostgreSQL 16 + pgvector** — market/signal/position storage + vector search
- **Pydantic v2** — config validation and all external data models
- **Claude (Anthropic)** — signal analysis (`claude-sonnet-4-6`) + catalyst generation (`claude-haiku-4-5`)
- **sentence-transformers** — local document embeddings (`all-MiniLM-L6-v2`, 384 dims, no API key)
- **Kalshi Trade API v2** — REST `https://api.elections.kalshi.com/trade-api/v2`, WebSocket `wss://api.elections.kalshi.com/trade-api/ws/v2`
- **Tavily + NewsAPI + Reddit + GDELT + Internet Archive TV** — news ingestion, over **httpx** (async)
- **React 18 + TypeScript + Vite + Tailwind CSS** — dashboard frontend (served from FastAPI `/` when built; Vite in dev)
- **TanStack Query v5 + Recharts** — dashboard data fetching/polling and charts
- **pytest + pytest-asyncio** — testing
- **uv** — dependency management

---

## License

MIT
