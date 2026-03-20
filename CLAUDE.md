# freqpred — Claude Code Guide

## What this project is

freqpred is a prediction market trading framework for Kalshi. It uses RAG + LLM sentiment analysis (Claude) to estimate event probabilities, compares them to market-implied prices, and trades the edge. Think freqtrade but for prediction markets.

Full architecture in [SPEC.md](SPEC.md). Read it before making structural decisions.

---

## Architecture in one paragraph

The system has two async pipelines. The **ingestion pipeline** runs continuously and is catalyst-driven: the **Market Selector** reads active markets from the DB and asks each registered strategy `is_market_interesting(market)` to filter down to markets worth monitoring; the **Catalyst Generator** calls Claude Haiku per selected market to derive 3–5 targeted search queries (catalysts) representing events that could shift the probability — these are stored as `CatalystRun`/`CatalystQuery` DB rows and refreshed daily using RAG context; the **Ingestion Scheduler** reads the latest catalyst queries and runs Tavily, NewsAPI, and Reddit fetchers against them, deduplicates by URL, generates local embeddings (sentence-transformers), and stores results in the `documents` table. The **signal pipeline** is triggered: it embeds a market question, does semantic search against the document store (RAG), checks if the retrieval result is new (hash check), then calls Claude Sonnet for a probability estimate. Signals are written to Postgres and used by strategy plugins to decide whether to place paper or live trades on Kalshi.

---

## Repo structure

```
freqpred/
├── freqpred/
│   ├── cli.py              # entry point
│   ├── config.py           # config loading
│   ├── markets/            # Kalshi API client + watcher
│   ├── ingestion/          # selector, catalyst_generator, scheduler, fetchers, store
│   ├── rag/                # local sentence-transformers embedder + pgvector retriever
│   ├── signal/             # signal pipeline + LLM analysis
│   ├── strategy/           # IPredictionStrategy interface + bundled strategies
│   ├── trading/            # order manager, risk, ledger
│   ├── llm/                # Claude client + LLM audit logging
│   ├── metrics/            # calibration, reporting
│   ├── dashboard/          # FastAPI backend + React frontend
│   └── alerts/             # Telegram, Discord
├── strategies/             # user strategy files (gitignored)
├── config/                 # config.example.yaml + local config.yaml (gitignored)
├── tests/
├── migrations/             # Alembic migration files
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## Tech stack

- **Python 3.12+**
- **FastAPI** — dashboard API
- **SQLAlchemy 2.0** (async) + **Alembic** — ORM + migrations
- **PostgreSQL 16 + pgvector** — main DB + vector search
- **Redis** — signal cache, ingestion dedup
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim) — local document embeddings, no API key required
- **Anthropic SDK** — Claude for signal analysis + social pre-summarizer
- **PRAW** — Reddit API client
- **Tavily SDK** — web search
- **pytest + pytest-asyncio** — testing
- **Pydantic v2** — config validation and data models
- **uv** — dependency management (not pip)

---

## Environment setup

```bash
# Install dependencies
uv sync

# Copy and fill in config
cp config/config.example.yaml config/config.yaml

# Start local services (Postgres)
docker-compose up -d db

# Run migrations
uv run alembic upgrade head

# Run the signal pipeline (paper mode, signal-only)
uv run freqpred run --strategy ConservativeDefault --mode signal-only
```

Required environment variables (set in `.env` or AWS Secrets Manager in prod):
```
KALSHI_API_KEY=
ANTHROPIC_API_KEY=
TAVILY_API_KEY=
NEWSAPI_KEY=
# Reddit uses the public JSON API — no credentials required
DATABASE_URL=postgresql+asyncpg://...
```

---

## Common commands

```bash
uv run pytest                          # run all tests
uv run pytest tests/unit/              # unit tests only
uv run pytest tests/integration/       # integration tests (requires DB)
uv run alembic revision --autogenerate -m "description"  # create migration
uv run alembic upgrade head            # apply migrations
uv run freqpred run --help             # CLI help
uv run freqpred markets list           # list active Kalshi markets
uv run freqpred signal analyze --market-id <id>  # analyze a specific market
```

---

## Hard constraints — never violate these

### 1. Signals are immutable and append-only
**Never UPDATE a signal record.** Every re-evaluation creates a new Signal row. The market's `current_signal_id` field is updated to point to the latest signal. This is architectural — calibration analysis depends on the full signal history.

### 2. Every LLM call must be logged
Every call to any LLM (Claude, or the cheap Haiku pre-summarizer) must write a row to the `llm_queries` table via `llm/audit.py` **before returning**. Even failed calls get logged with `success=False`. The audit log is non-negotiable — it's the cost tracking and decision audit trail.

### 3. Hard risk caps cannot be bypassed
`trading/risk.py` enforces position size limits, daily loss limits, and drawdown limits. Strategy code calls `risk.py`; `risk.py` has final say. Strategy `position_size()` output is always passed through risk checks before any order is submitted.

### 4. No secrets in code or config files
All secrets go through environment variables or AWS Secrets Manager. `config.yaml` contains structure and defaults only — never actual keys. The `config/` directory (except `config.example.yaml`) is gitignored.

### 5. Ingestion and signal pipelines are separate async processes
The ingestion pipeline (fetch → embed → store) runs on its own schedule and writes to the document store. The signal pipeline reads from the document store. They communicate through the DB — not through direct function calls or shared state. This separation allows ingestion to run continuously without blocking signal analysis.

### 6. Paper mode must be the default
All trading defaults to `mode="paper"`. Live trading requires explicit `--mode live` flag AND the `LIVE_TRADING_ENABLED=true` environment variable. Never submit real orders unless both conditions are true.

---

## Key data model rules

- `Market.last_fetched_at` — updated every poll, even if nothing changed
- `Market.price_updated_at` — updated ONLY when bid/ask/mid actually changes
- `Market.current_signal_id` — updated atomically when a new Signal is created (use a transaction)
- `Signal.retrieval_hash` — hash of the sorted list of Document IDs returned by vector search, not hash of content
- `Position.signal_confidence/edge/estimated_prob` — snapshotted from Signal at entry time, never updated
- `Document.source_url` — unique constraint; use `INSERT ... ON CONFLICT (source_url) DO UPDATE` for upserts

---

## Testing conventions

- Unit tests in `tests/unit/` — no DB, no external APIs. Mock everything external.
- Integration tests in `tests/integration/` — require running Postgres + Redis (via docker-compose). Use a test database.
- Never use the production DB in tests. `DATABASE_URL` for tests should point to `freqpred_test`.
- Fixtures for common objects (Market, Signal, Document) live in `tests/conftest.py`.
- Every new module gets a corresponding test file. Aim for coverage on all public functions.
- LLM calls in tests should always be mocked — never make real API calls in tests.

**There is no such thing as a pre-existing failure or a flaky test that can be ignored.** If any test is failing — for any reason, in any file — it must be fixed before the task is considered done. Never declare work complete while tests are red. Never attribute a failure to "pre-existing" state and move on; investigate and fix it.

---

## Database conventions

- Use SQLAlchemy 2.0 async style throughout (`async with session` pattern)
- All migrations via Alembic — never alter schema manually
- Table names are snake_case plural: `markets`, `signals`, `positions`, `documents`, `llm_queries`
- All tables have `created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()`
- UUIDs for all primary keys except `llm_queries` (auto-increment int)
- pgvector column: `embedding VECTOR(384)` (all-MiniLM-L6-v2 dimension)

---

## Definition of done

Before declaring any task complete, verify every acceptance criterion listed in the GitHub issue was actually met — not just that the code was written.

**For any task that touches the database or migrations:**
- Run `docker-compose up -d db` if not already running
- Run migration verification against `freqpred_test` (never `freqpred`) so local dev data is not destroyed:
  ```bash
  DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" uv run alembic upgrade head
  DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" uv run alembic downgrade base
  ```
- Run `uv run pytest tests/unit/` and confirm all tests pass

**For any task that adds code:**
- Always run the full test suite (unit + integration) when DB is available:
  ```bash
  uv run pytest tests/ -v
  ```
- If DB is not available, run unit tests at minimum: `uv run pytest tests/unit/`
- All tests must pass before declaring the task done

**Never say a task is done based on code compiling or unit tests alone when the acceptance criteria include runtime behavior** (e.g. "runs cleanly against a fresh DB").

**For any task that adds, removes, or changes a CLI command or Telegram bot command:**
- Update [COMMANDS.md](COMMANDS.md) to reflect the change before declaring the task done.
- This includes: new commands, removed commands, changed option names/defaults, new Telegram bot commands registered via `TelegramCommandHandler.register()`.

---

## Code style

- Type hints everywhere — no untyped functions
- Pydantic models for all external data (API responses, config)
- SQLAlchemy models in `*/models.py` within each module
- Dataclasses only for internal data transfer objects
- Async/await throughout — no blocking I/O on the main thread
- Log with structured logging (`structlog`) — no bare `print()`
- Errors: raise specific exceptions, never swallow silently
