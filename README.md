# freqpred

**LLM-driven prediction market trading framework — freqtrade for prediction markets.**

freqpred used retrieval-augmented LLM analysis to estimate the "true" probability of prediction market outcomes, identify edges against market-implied prices, and trade them with assessment-aware sizing plus systematic risk controls.

> ## ⚠️ Retired 2026-08-18 — it did not work
>
> **This project is archived. The core idea was tested and failed. Do not trade with this.**
>
> The premise was that an LLM reasoning over fresh news could estimate event
> probabilities well enough to beat market prices. Measured directly over 61 markets,
> point-in-time, the LLM's estimate scored a **Brier of 0.2407** — worse than a
> **Poisson baseline (0.2093)** fit on the same data, and worse than a
> **constant base rate (0.2349)**.
>
> Separately, the economics never worked: production inference ran **~$0.37/contract**
> against a best-measured edge of **+$0.0107/contract** and **~$0.027/contract** of fee drag.
>
> Final live result: **−$18.32** over 164 closed positions (42.1% win rate) against
> **$314.62** of LLM spend. 95% of that record is a single market series.
>
> **Read [docs/POSTMORTEM.md](docs/POSTMORTEM.md) before anything else here.** It covers
> what was built, what it cost, why it failed, what the measurement apparatus got right,
> and what would have to be true for the idea to be worth revisiting.
>
> The code is left public because the *infrastructure* is sound and reusable — the
> Kalshi client, the LLM audit layer, and especially the counterfactual weekly-review
> methodology in `freqpred/metrics/weekly_review.py`. The *thesis* is not.

**Status:** Retired. Phases 1–3 complete (signal engine, paper trading, live trading + ops hardening); Phase 4 (Polymarket) specified but never built.

---

## What it did

1. **Monitors** active markets on Kalshi (Politics, Technology, Economics, ...)
2. **Ingests** targeted news via Tavily, NewsAPI, The Guardian, Reddit, GDELT, TV transcripts + chyrons, and Truth Social (catalyst-driven RAG)
3. **Estimates** event probability using Claude Sonnet with structured output
4. **Identifies** markets where LLM probability diverges meaningfully from market price
5. **Sizes and trades** via a pluggable strategy interface, assessment-aware sizing, and hard risk controls
6. **Tracks** calibration — are our probability estimates actually accurate?

> **Why no backtesting?** LLMs have seen market resolutions in training data, creating unavoidable look-ahead bias. freqpred validates via paper trading + Brier score calibration instead.

For full architecture, data models, strategy interface, and roadmap, see **[SPEC.md](SPEC.md)**.

---

## What it looked like

The dashboard, served read-only against the final database at retirement. Every
footer reads `git b7c4412` — the retirement commit — so these are the system's
last state, not a staged demo.

### Calibration over time — the whole story in one chart

![Calibration over time](docs/screenshots/calibration-over-time.png)

Daily Brier score for the model (blue bars) against the market (orange line),
with each signal-prompt version marked as a vertical boundary. Twelve prompt
versions between March and August. The model's EMA never durably separates from
the market's, and the red regions — where the model is worse than the price it
is trying to beat — never stop appearing. This is what "the thesis failed"
looks like measured daily rather than argued.

### P&L against inference spend

![P&L history](docs/screenshots/pnl-history.png)

Cumulative P&L and cumulative LLM spend on the same axes: −$18.32 realized
against $314.62 spent to produce it. Bankroll $180.00 → $161.68.

### Signal detail

![Signal detail](docs/screenshots/signal-detail.png)

One signal expanded: the model's probability against the market mid, its
reasoning, the sizing assessor's trust score and verdict, key factors, and
per-source quality. Note the assessor's own conclusion on this trade —
*"Strategy win rate 39.4% and negative mean PnL — this is a net-losing
strategy."* The instrumentation was honest even when the strategy was not
working.

### Signal feed and positions

![Signal feed](docs/screenshots/signal-feed.png)

![Positions](docs/screenshots/positions.png)

### Calibration distribution and system health

![Calibration distribution](docs/screenshots/calibration-distribution.png)

![System health](docs/screenshots/system-health.png)

---

## Documentation

| Document | What's in it |
|---|---|
| **[docs/POSTMORTEM.md](docs/POSTMORTEM.md)** | Why the project stopped: what was built, what it cost, the two findings that ended it, and what would have to be true to revisit the idea. **Start here.** |
| [SPEC.md](SPEC.md) | Full architecture, data models, strategy interface, risk framework, development phases |
| [COMMANDS.md](COMMANDS.md) | CLI and Telegram command reference |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Benchmarking, signal-prompt changes, assessor audits, the weekly review, utility scripts |
| [docs/runbook.md](docs/runbook.md) | Incident runbook — circuit breaker alerts and how to respond |
| [docs/weekly-review/reports/](docs/weekly-review/reports/) | The five weekly profitability reviews, with their JSON snapshots |

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

### Deeper workflows

Benchmarking a model or prompt change, the signal-prompt change workflow, the
sizing-assessor audit, the weekly profitability review, and the utility scripts
are documented in **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**.

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
