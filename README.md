# freqpred

**LLM-driven prediction market trading framework — freqtrade for prediction markets.**

freqpred uses retrieval-augmented LLM analysis to estimate the "true" probability of prediction market outcomes, identifies edges against market-implied prices, and executes trades with systematic risk controls.

---

## What it does

1. **Monitors** active markets on Kalshi (Politics, Technology, Economics, ...)
2. **Retrieves** relevant news context via Tavily, NewsAPI, and Reddit (RAG)
3. **Estimates** event probability using Claude with structured output
4. **Identifies** markets where LLM probability diverges from market price
5. **Trades** via a pluggable strategy interface with hard risk controls
6. **Tracks** calibration — are our probability estimates actually accurate?

## Philosophy

Backtesting prediction markets is unreliable: LLMs have seen market resolutions in training data, creating unavoidable look-ahead bias. Instead, freqpred validates strategies through **paper trading + calibration tracking** — simulating trades against real market prices and measuring Brier score over resolved markets.

---

## Setup

**Prerequisites:** Python 3.12+, Docker (for Postgres + Redis), `uv`

```bash
# 1. Install dependencies
uv sync

# 2. Configure
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml — or set env vars (see below)

# 3. Start local services
docker-compose up -d db redis

# 4. Apply database migrations
uv run freqpred db migrate

# 5. (Optional) Start Adminer — database web UI at http://localhost:8080
docker-compose up -d adminer
```

**Required environment variables** (set in `.env` or directly):

```
ANTHROPIC_API_KEY=...     # Claude — signal analysis + catalyst generation
DATABASE_URL=postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred
REDIS_URL=redis://localhost:6379

# Optional — enables news fetching
TAVILY_API_KEY=...
NEWSAPI_KEY=...

# Optional — enables live Kalshi trading
KALSHI_API_KEY=...
KALSHI_PRIVATE_KEY_PATH=...
```

---

## Running

### Signal-only mode

Runs the market watcher, ingestion scheduler, and signal pipeline concurrently. Prints each new signal to stdout. No trades are placed.

```bash
uv run freqpred run --strategy ConservativeDefault --mode signal-only
```

Press **Ctrl+C** to stop cleanly.

### Custom strategies

Point `--strategy` at a `.py` file containing a class that subclasses `IPredictionStrategy`:

```bash
uv run freqpred run --strategy strategies/my_strategy.py --mode signal-only
```

---

## CLI reference

```bash
# Run the full pipeline
uv run freqpred run --strategy <name|path> --mode <signal-only|paper|live>

# Fetch active Kalshi markets and write to DB
uv run freqpred markets list
uv run freqpred markets list --category politics

# One-shot signal analysis for a specific market
uv run freqpred signal analyze --market-id <KALSHI-TICKER>

# Run ingestion pipeline manually (fetch news for catalyst queries)
uv run freqpred ingestion run --limit 5
uv run freqpred ingestion run --category politics --dry-run

# Apply database migrations
uv run freqpred db migrate
```

---

## Architecture

```
Market Watcher ──────────────────────────────────────────────────────────────┐
  (polls Kalshi, upserts prices, enqueues price-move signal triggers)        │
                                                                             ▼
Ingestion Scheduler                                          Signal Pipeline (RAG + LLM)
  Catalyst Generator → search queries per market              embed question → vector search
  Fetchers: Tavily + NewsAPI + Reddit                         → Claude probability estimate
  Local embeddings (sentence-transformers) → Postgres pgvector → Signal written to DB
                                                                             │
                                                                             ▼
                                                               Strategy Engine
                                                                 should_trade? position_size?
                                                                             │
                                                                             ▼
                                                               Order Manager → Paper/Live Orders
                                                                             │
                                                                             ▼
                                                               Dashboard + Alerts + Calibration
```

See [SPEC.md](SPEC.md) for the full architecture, data models, strategy interface, and development roadmap.

---

## Development

**Database UI:** Adminer runs at `http://localhost:8080` — use server `db`, username/password/database all `freqpred`.

```bash
docker-compose up -d adminer
```

```bash
# Run all unit tests
uv run pytest tests/unit/

# Run integration tests (requires running Postgres + Redis)
uv run pytest tests/

# Create a new database migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head
```

Built-in strategies: `ConservativeDefault`, `PoliticsEdgeStrategy`, `TechNewsStrategy`

## Platforms

| Platform | Status |
|---|---|
| Kalshi | v1 — primary |
| Interactive Brokers (event contracts) | v2 — planned |

## Tech Stack

- **Python 3.12+**, FastAPI, SQLAlchemy 2.0 async, Alembic
- **PostgreSQL 16 + pgvector** — market/signal/position storage + vector search
- **Redis** — signal trigger queue, ingestion dedup
- **Claude (Anthropic)** — signal analysis + catalyst generation
- **sentence-transformers** — local document embeddings (`all-MiniLM-L6-v2`, 384 dims, no API key required; Voyage AI `voyage-3` is a possible future enhancement for higher-quality retrieval)
- **Tavily + NewsAPI + Reddit** — news ingestion
- **uv** — dependency management

## License

MIT
