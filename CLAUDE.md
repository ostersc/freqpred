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
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim) — local document embeddings, no API key required
- **Anthropic SDK** — Claude for signal analysis + social pre-summarizer
- **httpx** — async HTTP client; Reddit fetcher uses the public JSON API directly (no PRAW)
- **Tavily SDK** — web search
- **pytest + pytest-asyncio** — testing
- **Pydantic v2** — config validation and data models
- **uv** — dependency management (not pip)
- **Kalshi Trade API v2** — REST base `https://api.elections.kalshi.com/trade-api/v2`; WebSocket `wss://api.elections.kalshi.com/trade-api/ws/v2`

### Kalshi WS v2 channel names (gotcha)
The WebSocket uses **`market_lifecycle_v2`** (not `market_lifecycle`). Valid channels: `ticker`, `market_lifecycle_v2`, `orderbook_delta`, `trade`, `fill`, `market_positions`, `user_orders`.

**`market_lifecycle_v2` is a global broadcast — `market_ticker` filters are explicitly NOT supported.** All market lifecycle events are received regardless of subscription. Filter in code.

**Settlement result lives in `determined`, not `settled`:**
- `event_type: "determined"` → carries `settlement_value` (`"1.0000"` = YES wins, `"0.0000"` = NO wins). Act here.
- `event_type: "settled"` → only carries `settled_ts`. No result. Just unsubscribe — `MarketWatcher._resolve_settled_live_positions` handles any missed determinations on the next poll cycle.

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

Required environment variables (set in `.env` or AWS Secrets Manager in prod): see `config.py` — it defines a full `_ENV_OVERRIDES` map of every env var the system reads and which config key it maps to.

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
- Integration tests in `tests/integration/` — require running Postgres (via docker-compose). Use a test database.
- Never use the production DB in tests. `DATABASE_URL` for tests should point to `freqpred_test`.
- Fixtures for common objects (Market, Signal, Document) live in `tests/conftest.py`.
- Every new module gets a corresponding test file. Aim for coverage on all public functions.
- LLM calls in tests should always be mocked — never make real API calls in tests.

**There is no such thing as a pre-existing failure or a flaky test that can be ignored.** If any test is failing — for any reason, in any file — it must be fixed before the task is considered done. Never declare work complete while tests are red. Never attribute a failure to "pre-existing" state and move on; investigate and fix it.

---

## Database conventions

- Use SQLAlchemy 2.0 async style throughout (`async with session` pattern)
- All migrations via Alembic — never alter schema manually
- Migration files must be named `NNNN_short_description.py` with a zero-padded 4-digit sequence number matching the `revision` string inside (e.g. `0020_documents_published_at_nullable.py`, `revision = '0020'`). Never use autogenerate's default hex IDs as filenames or revision IDs.
- Table names are snake_case plural: `markets`, `signals`, `positions`, `documents`, `llm_queries`
- All tables have `created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()`
- UUIDs for all primary keys except `llm_queries` (auto-increment int)
- pgvector column: `embedding VECTOR(384)` (all-MiniLM-L6-v2 dimension)

---

## GitHub issues and SPEC

**SPEC.md is the source of truth for what will be built.** GitHub issues contain the implementation detail for each task. The two must stay in sync.

### Task numbering
Every planned task has a `T{N}` identifier in SPEC.md that matches its GitHub issue number. For example, T37 = issue #37. Never create a new task without creating a matching issue, and never create an issue without a matching SPEC entry. If a task is added mid-phase (e.g. discovered scope), pick the next available issue number and use that as the task ID.

### Issue format
Each issue must follow this structure (see issues #37–#48 as reference):
- **`## Context`** — what problem this solves and why it is needed; note any `**Depends on:**` tasks
- **`## Implementation scope`** — specific files to change, key function signatures, code sketches
- **`## New tests`** — named test cases with what each verifies
- **`## Acceptance criteria`** — checkbox list; these are the gates that must pass before closing the issue

### Determining what to work on next
When asked "what's next", "what should we work on", or told to start the next task:
1. Read the Phase 3 checkbox list in SPEC.md — identify all unchecked tasks
2. Run `gh issue list --state open` and compare against SPEC.md checkboxes — flag any discrepancy (e.g. SPEC says done but issue still open, or vice versa) and resolve it before proceeding
3. From the unchecked tasks, identify which are unblocked (all dependencies are checked in SPEC.md)
4. Recommend the highest-priority unblocked task, stating its T-number, title, and why it's next in dependency order
5. **Ask for confirmation before starting any implementation**

### Workflow
- When starting a task, read the GitHub issue before writing any code
- Check off acceptance criteria in the issue as they are met
- If scope changes during implementation, update both the issue body and SPEC.md before proceeding
- When adding a new Phase 3 task: add a checkbox entry to the Phase 3 list in SPEC.md (with issue link), create the issue, then implement

### Keeping SPEC.md current
**Update SPEC.md immediately whenever any of the following change:**
- `StrategyConfig` fields (add, remove, rename, change default, change semantics)
- `IPredictionStrategy` interface methods (signatures, defaults, contracts)
- Core pipeline behaviour (market selection, ingestion, signal generation, exit priority)
- Data model fields on `Market`, `Signal`, `Position`, `Document`, `CatalystRun`, or `LLMQuery`
- Risk rules, circuit breakers, or hard caps
- Calibration methodology (what is scored, how, against what baseline)
- Any invariant documented in §7–§10 of SPEC.md

This applies to **ad-hoc improvements** (outside of formal tasks) just as much as to tracked tasks. If code changes but SPEC.md doesn't, the spec becomes fiction. Update the `Last updated` date at the top of SPEC.md on every edit.

### Completing a task
When all acceptance criteria are met and tests pass, in this order:
1. **Summarize** what was implemented (files changed, key design decisions) and list the manual validation steps from the issue's acceptance criteria that require runtime verification (e.g. "run with --mode live and observe X"). Present this to the user and wait for them to confirm manual validation is done before proceeding.
2. Mark the task's checkbox in the Phase 3 list in SPEC.md as `[x]`
3. Commit and push the code
4. Close the GitHub issue (`gh issue close <N>`) — always **after** pushing, never before

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

## Diagrams

Always use Mermaid diagrams instead of ASCII art. Wrap them in ` ```mermaid ` fences. Keep node labels concise — `\n` does not work in Mermaid; use plain text labels and let surrounding prose carry the detail. Never draw boxes with `┌─┐│└┘` characters. This applies to SPEC.md, GitHub issues, and any other documentation.

---

## Code style

- Type hints everywhere — no untyped functions
- Pydantic models for all external data (API responses, config)
- SQLAlchemy models in `*/models.py` within each module
- Dataclasses only for internal data transfer objects
- Async/await throughout — no blocking I/O on the main thread
- Log with structured logging (`structlog`) — no bare `print()`
- Errors: raise specific exceptions, never swallow silently
