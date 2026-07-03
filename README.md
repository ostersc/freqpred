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

# Apply migrations
uv run freqpred db migrate

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
# Build the resolved-market scenario bank first (leakage-free by construction):
uv run freqpred fixtures record-bank
uv run python scripts/benchmark_signals.py --prompt-mode --fixtures benchmarks/prompt_bank \
    --training-cutoff 2026-03-01 --limit 250

# Preview call volume and token cost first
uv run python scripts/benchmark_signals.py --candidate-model claude-sonnet-5 \
    --training-cutoff 2026-03-01 --estimate-only

# Pre-4.6 candidates (e.g. Haiku 4.5) reject adaptive thinking — omit it
uv run python scripts/benchmark_signals.py --candidate-model claude-haiku-4-5-20251001 \
    --training-cutoff 2026-03-01 --thinking none
```

**When to use it:** before any signal-model swap, any prompt-template change, or judgment-relevant config changes (thinking settings, max_tokens). Not for regression testing (that's the free, deterministic replay harness — `freqpred fixtures replay`), not for P&L estimation, and not for scheduled monitoring — every run costs real API dollars (audited to `llm_queries`, counted against the daily spend cap).

**The adopt/reject decision rule:**

1. **Adopt only on a significant paired Brier delta** — bootstrap 95% CI excluding zero, or sign test p < 0.05. A better raw mean on a small noisy sample is not evidence.
2. **Guard: trade decisions must not degrade** — check the would-trade rate, disagreement table, per-trade EV, and the stake-weighted P&L. Confidence scales position size in production (the Kelly blend), so each would-trade is also sized by the benchmark strategy's own `position_size()` from that model's posterior + confidence (`--strategy`, default `PoliticsEdgeStrategy`; the strategy config also supplies the default `min_edge`/`min_confidence` gates) — an overconfident model loses proportionally more when wrong, and a better-calibrated but timid candidate shows up as too small a total stake. The gate applies the signal-level entry filters from the strategy config exactly as `should_trade` does (`max_edge`, `min_mid_price`/`max_mid_price` on the entry side's cost) — without them the numbers include longshot trades production would never take. Known simplifications vs live P&L: entry fills at the frozen ask (production posts resting limits), and positions ride to settlement (production exits via stoploss/signal/force_exit).
3. **Tiebreaker: cost and latency.**
4. **Ambiguous → keep the incumbent**; it has live calibration history, the candidate has none.

`--training-cutoff` is required: markets that closed inside the candidate's training window are excluded, since their outcomes may be memorized rather than forecast.

### Changing the signal prompt — the standard workflow

Every edit to `SYSTEM_PROMPT` or `build_prompt` follows this sequence. The goal: no prompt version reaches live trading without benchmark evidence against real resolved-market outcomes.

1. **Scope the edit.** Change only what a specific, written-down finding justifies — no drive-by rewording. The benchmark measures the whole diff; unrelated edits make a negative result unattributable.
2. **Bump `PROMPT_VERSION`** in `freqpred/signal/llm.py` (`signal-vN` → `signal-vN+1`). The replay harness guards (rendered-prompt snapshot, system-prompt hash, version pin test) fail on any unbumped edit — that's by design.
3. **Regenerate the committed replay fixtures**: `uv run freqpred fixtures replay --update`, then run the full test suite.
4. **Do NOT regenerate `benchmarks/prompt_bank/`.** `record-bank` only records signals whose stored prompt version is current, so a post-bump sweep produces an empty bank. The bank recorded under the *previous* version is the frozen baseline the benchmark compares against — treat it as the experiment's control artifact.
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

### Utility scripts

Other one-off analysis and maintenance scripts live in `scripts/` — see each script's module docstring for full usage.

---

## Tech stack

- **Python 3.12+**, FastAPI, SQLAlchemy 2.0 async, Alembic
- **PostgreSQL 16 + pgvector** — market/signal/position storage + vector search
- **Claude (Anthropic)** — signal analysis (`claude-sonnet-4-6`) + catalyst generation (`claude-haiku-4-5`)
- **sentence-transformers** — local document embeddings (`all-MiniLM-L6-v2`, 384 dims, no API key)
- **Tavily + NewsAPI + Reddit + GDELT + Internet Archive TV** — news ingestion
- **React 18 + TypeScript + Vite + Tailwind CSS** — dashboard frontend (served from FastAPI `/` when built; Vite in dev)
- **uv** — dependency management

---

## License

MIT
