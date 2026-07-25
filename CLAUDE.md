# freqpred — Claude Code Guide

## What this project is

freqpred is a prediction market trading framework for Kalshi. It uses RAG + LLM sentiment analysis (Claude) to estimate event probabilities, compares them to market-implied prices, and trades the edge with assessment-aware sizing. Think freqtrade but for prediction markets.

Full architecture in [SPEC.md](SPEC.md). Read it before making structural decisions.

---

## Architecture in one paragraph

The system has several async subsystems. The **ingestion pipeline** runs continuously and is catalyst-driven: the **Market Selector** reads active markets from the DB and asks each registered strategy `is_market_interesting(market)` to filter down to markets worth monitoring; the **Catalyst Generator** calls Claude Haiku per selected market to derive 3–5 targeted search queries (catalysts) representing events that could shift the probability — these are stored as `CatalystRun`/`CatalystQuery` DB rows and refreshed daily using RAG context; the **Ingestion Scheduler** reads the latest catalyst queries and runs Tavily, NewsAPI, Guardian, Reddit, GDELT, and TV archive fetchers against them, while the **Realtime Scheduler** polls TV chyron and Truth Social sources on a faster cadence. The **signal pipeline** embeds a market question, does semantic search against the document store (RAG), then decides whether to invoke Claude Sonnet. For non-scheduled triggers the LLM is skipped when the retrieval hash is unchanged (same docs → same output). For scheduled triggers (which run every 30 min via `signal.interval_seconds`) the LLM fires when **any** of three conditions is true: the retrieval hash changed, FactBase data was refreshed since the last scheduled run, or `signal.max_scheduled_interval_hours` (default 24h) have elapsed — guaranteeing at least one temporal-reasoning re-run per day while reacting immediately to new evidence. Before the LLM call, a **pre-signal risk gate** (enabled by default via `StrategyConfig.pre_signal_risk_gate`) skips analysis for new-entry markets where risk caps are already hit (max positions/exposure full, spread too wide, or stoploss re-entry blocked) — markets with open positions always bypass this gate so exit signals fire regardless. Before final position sizing, `assess_signal_context()` can add source-quality and similar-market trust context; persisted assessments and source-quality snapshots are then visible in the dashboard.

---

## External API gotchas

- **httpx / Reddit** — the Reddit fetcher uses the public Atom/RSS search feeds (`/r/{sub}/search.rss`). Reddit shut down unauthenticated JSON API access in June 2026 and gated OAuth behind pre-approval ("Responsible Builder Policy"). RSS is a stopgap Reddit has said it may also close; a blanket failure raises `RedditBlockedError` and surfaces via the `fetcher_reddit` telemetry heartbeat.
- **Kalshi Trade API v2** — REST base `https://api.elections.kalshi.com/trade-api/v2`; WebSocket `wss://api.elections.kalshi.com/trade-api/ws/v2`.

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
uv sync --group dev

# One-time: wire up the ruff pre-commit hook
uv run pre-commit install

# Copy and fill in config
cp config/config.example.yaml config/config.yaml

# Start local services (Postgres)
docker-compose up -d db

# Run migrations (DATABASE_URL must be exported — migrations read the env only,
# never config.yaml, and .env is not auto-loaded)
uv run freqpred db migrate

# Run the signal pipeline (paper mode, signal-only)
uv run freqpred run --strategy ConservativeDefault --mode signal-only
```

Required environment variables (set in `.env` or AWS Secrets Manager in prod): see `config.py` — it defines a full `_ENV_OVERRIDES` map of every env var the system reads and which config key it maps to.

---

## Common commands

```bash
uv run freqpred run --help             # CLI help
uv run freqpred markets list           # list active Kalshi markets
uv run freqpred signal analyze --market-id <id>  # analyze a specific market
```

### Linting (ruff)

Rule set and ignores are configured in `pyproject.toml` under `[tool.ruff]` / `[tool.ruff.lint]`; each ignored rule has an inline comment explaining why (e.g. `UP037` is disabled because unquoting an `Annotated[...]` forward reference to a `TYPE_CHECKING`-only import once broke FastAPI request parsing silently — see that comment before re-enabling it). When fixing lint violations by hand (not just `--fix`), treat every change as a potential behavior change, not pure style — re-run tests, and for anything touching `Annotated[...]`/`Depends(...)` or enum base classes, double-check runtime behavior didn't shift.

### Querying the database

`psql` is not installed locally. Always use `docker exec` without the `-it` flag:

```bash
docker exec freqpred-db-1 psql -U freqpred -d freqpred -c "SELECT ..."
```

Never use `psql` directly and never add `-it` to `docker exec` — it fails when stdin is not a terminal.

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

### 5. Ingestion and signal pipelines are separate async subsystems
The ingestion pipeline (fetch → embed → store) runs on its own schedule and writes to the document store. The signal pipeline reads from the document store. They communicate through the DB boundary — not through direct cross-pipeline calls or shared in-memory state. This separation allows ingestion to run continuously without blocking signal analysis.

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

**Wiring tests for pipeline changes.** When changing how two pipeline components interact — e.g. the signal loop calling the position monitor, the order manager calling risk, ingestion feeding the signal pipeline — always add a test that verifies the *wiring*, not just the *component*. A unit test that calls a function in isolation with the right arguments does not prove that anything ever calls it with those arguments in production. The wiring test must exercise the caller (e.g. the signal loop in `_run_main`) and assert that the callee (e.g. `position_monitor.check_all_positions`) receives the expected arguments, in the expected order, relative to any subsequent calls. If you only test the component in isolation and skip the wiring test, bugs like "should_exit() is correct but is never called" will not be caught.

**There is no such thing as a pre-existing failure or a flaky test that can be ignored.** If any test is failing — for any reason, in any file — it must be fixed before the task is considered done. Never declare work complete while tests are red. Never attribute a failure to "pre-existing" state and move on; investigate and fix it.

**Never hardcode dates or timestamps in tests.** Any test that makes assertions about time windows (7d, 30d, 365d, age in hours, etc.) must control the clock — either by accepting a `_now: datetime | None = None` parameter in the production function (defaulting to `datetime.now(UTC)`) and passing a fixed value in the test, or by using `freezegun`. A hardcoded `_NOW = datetime(2026, 5, 16, ...)` combined with a function that calls `datetime.now()` internally will silently become incorrect as calendar time advances. This has happened repeatedly. The rule: if a function uses `datetime.now()` and a test makes time-relative assertions, the function must accept an injectable clock.

**Always test both YES and NO position directions.** Any test that covers position entry, exit, P&L calculation, risk checks, order sizing, or signal-driven behavior must include cases for both `side="yes"` and `side="no"`. Kalshi NO positions have inverted payout logic (cost = `(1 - price) * contracts`), and bugs that only affect one direction are common. A test suite that only exercises YES positions provides false confidence. If a function's behavior is genuinely symmetric and the test already proves symmetry (e.g. by parameterizing over both sides), one parameterized test is sufficient — but both directions must be covered.

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

The task-numbering scheme, required issue body format, "what's next" procedure, and the ordered steps for completing a task live in the **`spec-and-issues` skill** — invoke it when starting, picking, or finishing a tracked task.

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

---

## Kalshi changelog review

The system monitors the Kalshi API changelog daily via `freqpred/ingestion/kalshi_changelog.py`. Unreviewed entries surface as alerts and in the system health UI. When asked to review the changelog — or when `unreviewed_count > 0` is noticed — invoke the **`kalshi-changelog-review` skill**, which carries the full audit-and-mark-reviewed procedure.

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

**For any task that adds, removes, renames, or changes the props of a component in `freqpred/dashboard/ui/src/components/ui.tsx`, `AssessmentCard.tsx`, `PriceTimeline.tsx`, or `DocLinkItem.tsx`:**
- These 19 components are synced to a Claude Design project ("freqpred Dashboard") for prototyping — see `.design-sync/` (config, notes, conventions, authored previews). Re-run `/design-sync` before declaring the task done so the synced project doesn't silently drift from the real component API.
- Data-fetching components (`NavBar`, `Footer`, `PositionDetail`, `SignalDetail`, `AnalyzeButton`) are intentionally excluded from the sync — no action needed if only those change.

**For any task that adds, removes, or changes a scheduled background task:**
- Add or update a `SERVICE_*` constant and a `FreshnessSpec` entry in `freqpred/runtime/telemetry.py:build_freshness_specs()` before declaring the task done.
- Wire `telemetry.mark_success()` and `telemetry.mark_error()` calls into the scheduler loop for that service — each scheduled task must report its own heartbeat independently, even if it runs inside another scheduler's loop.
- The purpose of telemetry is to surface stale data sources. Two conceptually unrelated tasks must never share a heartbeat even if they share a scheduler loop.

**For any change to the signal LLM or the sizing assessor and their inputs** — the full change-and-validation workflows (benchmark gates, frozen eval set, adoption criteria, spend checks) live next to the code they govern, in `freqpred/signal/CLAUDE.md` and `freqpred/metrics/CLAUDE.md`. They load automatically when working in those directories. Neither change may be adopted on inspection alone; both require a validation run proposed to the user.

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
