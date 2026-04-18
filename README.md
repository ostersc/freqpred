# freqpred

**LLM-driven prediction market trading framework — freqtrade for prediction markets.**

freqpred uses retrieval-augmented LLM analysis to estimate the "true" probability of prediction market outcomes, identifies edges against market-implied prices, and executes trades with systematic risk controls.

**Status:** Phase 2 complete (paper trading + calibration running). Phase 3 (live trading) in progress.

---

## What it does

1. **Monitors** active markets on Kalshi (Politics, Technology, Economics, ...)
2. **Ingests** targeted news via Tavily, NewsAPI, Reddit, GDELT, TV archives (catalyst-driven RAG)
3. **Estimates** event probability using Claude Sonnet with structured output
4. **Identifies** markets where LLM probability diverges meaningfully from market price
5. **Trades** via a pluggable strategy interface with hard risk controls
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

# Optional — enables news fetching
TAVILY_API_KEY=...
NEWSAPI_KEY=...

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

Bundled strategies: `ConservativeDefault`, `PoliticsEdgeStrategy`, `TechNewsStrategy`

---

## Architecture

```mermaid
graph TD
    subgraph Exchange[Kalshi Exchange]
        KREST[REST - orders and poll]
        KWS[WebSocket - ticker and lifecycle]
    end

    subgraph Ingestion[Ingestion Pipeline]
        MW[Market Watcher - 5min poll]
        MS[Market Selector]
        CG[Catalyst Generator - Haiku]
        IS[Ingestion Scheduler - 30min]
        RS[Realtime Scheduler - 5min]
        DS[(Document Store - pgvector)]
    end

    subgraph Sources[Third-party Sources]
        IS_SRC[Tavily · NewsAPI · Reddit · GDELT · TV Transcripts]
        RS_SRC[TV Chyrons · Truth Social account feeds]
    end

    subgraph Signal[Signal Pipeline]
        SP[Signal Pipeline - RAG + Sonnet]
        SE[Strategy Engine]
        OM[Order Manager - paper / live]
    end

    subgraph Positions[Position Tracking]
        PW[Position Watcher - WebSocket]
        PM[Position Monitor]
        L[(Ledger - Postgres)]
        DA[Dashboard + Alerts]
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
    DS --> SP
    MW -->|prices| SP
    SP --> SE
    SE --> OM
    OM --> KREST
    KWS --> PW
    PW -->|price ticks| PM
    PW -->|determined / settled| PM
    OM --> PM
    PM --> KREST
    PM --> L
    L --> DA
    IS --> IS_SRC
    RS --> RS_SRC
    MS -. is_market_interesting .-> STRAT
    SE -. "should_trade / position_size" .-> STRAT
    PM -. "force_exit every tick" .-> STRAT
    PM -. "custom_exit / should_exit on signal" .-> STRAT
    PM -. on_resolution .-> STRAT

    classDef exchange fill:#e8e8e8,stroke:#888,color:#333
    classDef ingestion fill:#d4edda,stroke:#28a745,color:#155724
    classDef signal fill:#cce5ff,stroke:#0069d9,color:#004085
    classDef position fill:#fff3cd,stroke:#d39e00,color:#533f03
    classDef strategy fill:#f8d7da,stroke:#721c24,stroke-width:3px,color:#721c24
    classDef sources fill:#e2d9f3,stroke:#7048a8,color:#3d1e6d

    class KREST,KWS exchange
    class MW,MS,CG,IS,RS,DS ingestion
    class SP,SE,OM signal
    class PW,PM,L,DA position
    class STRAT strategy
    class IS_SRC,RS_SRC sources

    linkStyle 21 stroke:#721c24,stroke-width:2px
    linkStyle 22 stroke:#721c24,stroke-width:2px
    linkStyle 23 stroke:#721c24,stroke-width:2px
    linkStyle 24 stroke:#721c24,stroke-width:2px
    linkStyle 25 stroke:#721c24,stroke-width:2px
```

Three concurrent components: **ingestion** (catalyst-driven, continuous), **signal** (triggered, RAG + LLM), and **position watcher** (WebSocket, sub-second price + resolution events). All communicate through Postgres — no shared state.

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

---

## Development

```bash
# Run all unit tests
uv run pytest tests/unit/

# Run integration tests (requires running Postgres)
uv run pytest tests/

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

---

## Tech stack

- **Python 3.12+**, FastAPI, SQLAlchemy 2.0 async, Alembic
- **PostgreSQL 16 + pgvector** — market/signal/position storage + vector search
- **Claude (Anthropic)** — signal analysis (`claude-sonnet-4-6`) + catalyst generation (`claude-haiku-4-5`)
- **sentence-transformers** — local document embeddings (`all-MiniLM-L6-v2`, 384 dims, no API key)
- **Tavily + NewsAPI + Reddit + GDELT + Internet Archive TV** — news ingestion
- **React 18 + TypeScript + Vite + Tailwind CSS** — dashboard frontend (served from FastAPI `/`)
- **uv** — dependency management

---

## License

MIT
