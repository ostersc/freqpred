# freqpred — Claude Code Guide

## What this project is

freqpred is a prediction market trading framework for Kalshi. It uses RAG + LLM sentiment analysis (Claude) to estimate event probabilities, compares them to market-implied prices, and trades the edge with assessment-aware sizing. Think freqtrade but for prediction markets.

Full architecture in [SPEC.md](SPEC.md). Read it before making structural decisions.

---

## Architecture in one paragraph

The system has several async subsystems. The **ingestion pipeline** runs continuously and is catalyst-driven: the **Market Selector** reads active markets from the DB and asks each registered strategy `is_market_interesting(market)` to filter down to markets worth monitoring; the **Catalyst Generator** calls Claude Haiku per selected market to derive 3–5 targeted search queries (catalysts) representing events that could shift the probability — these are stored as `CatalystRun`/`CatalystQuery` DB rows and refreshed daily using RAG context; the **Ingestion Scheduler** reads the latest catalyst queries and runs Tavily, NewsAPI, Guardian, Reddit, GDELT, and TV archive fetchers against them, while the **Realtime Scheduler** polls TV chyron and Truth Social sources on a faster cadence. The **signal pipeline** embeds a market question, does semantic search against the document store (RAG), then decides whether to invoke Claude Sonnet. For non-scheduled triggers the LLM is skipped when the retrieval hash is unchanged (same docs → same output). For scheduled triggers (which run every 30 min via `signal.interval_seconds`) the LLM fires when **any** of three conditions is true: the retrieval hash changed, FactBase data was refreshed since the last scheduled run, or `signal.max_scheduled_interval_hours` (default 24h) have elapsed — guaranteeing at least one temporal-reasoning re-run per day while reacting immediately to new evidence. Before the LLM call, a **pre-signal risk gate** (enabled by default via `StrategyConfig.pre_signal_risk_gate`) skips analysis for new-entry markets where risk caps are already hit (max positions/exposure full, spread too wide, or stoploss re-entry blocked) — markets with open positions always bypass this gate so exit signals fire regardless. Before final position sizing, `assess_signal_context()` can add source-quality and similar-market trust context; persisted assessments and source-quality snapshots are then visible in the dashboard.

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
│   ├── metrics/            # calibration, assessment, reporting, series history
│   ├── runtime/            # freshness telemetry (heartbeats per scheduled service)
│   ├── dashboard/          # FastAPI backend + React frontend
│   │   ├── api/            # routes.py, schemas.py, app.py
│   │   └── ui/             # React app (Vite, Tailwind, TanStack Query)
│   │       ├── package.json
│   │       ├── vite.config.ts
│   │       └── src/
│   │           ├── api/    # typed fetch wrappers
│   │           ├── components/
│   │           └── pages/  # 10 dashboard pages
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
- **httpx** — async HTTP client; Reddit fetcher uses the public Atom/RSS search feeds (`/r/{sub}/search.rss`) — Reddit shut down unauthenticated JSON API access in June 2026 and gated OAuth behind pre-approval ("Responsible Builder Policy"). RSS is a stopgap Reddit has said it may also close; a blanket failure raises `RedditBlockedError` and surfaces via the `fetcher_reddit` telemetry heartbeat
- **Tavily SDK** — web search
- **pytest + pytest-asyncio** — testing
- **Pydantic v2** — config validation and data models
- **uv** — dependency management (not pip)
- **Kalshi Trade API v2** — REST base `https://api.elections.kalshi.com/trade-api/v2`; WebSocket `wss://api.elections.kalshi.com/trade-api/ws/v2`
- **React 18 + TypeScript + Vite** — dashboard frontend
- **TanStack Query v5** — data fetching + auto-polling in the dashboard
- **Recharts** — calibration curve + LLM cost charts
- **Tailwind CSS v3** — utility-first styling for the dashboard UI

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
uv run ruff check .                    # lint (also runs in CI and pre-commit)
uv run ruff check --fix .              # lint + auto-fix safe violations
uv run alembic revision --autogenerate -m "description"  # create migration
uv run alembic upgrade head            # apply migrations
uv run freqpred run --help             # CLI help
uv run freqpred markets list           # list active Kalshi markets
uv run freqpred signal analyze --market-id <id>  # analyze a specific market
```

### Linting (ruff)

`ruff` is enforced at three points, in this order of how early it catches a problem:

1. **Pre-commit hook** — configured in `.pre-commit-config.yaml`, installed via `uv run pre-commit install` (one-time, see Environment setup). Runs `ruff check --fix` on staged files before each commit; the hook fails the commit if it had to auto-fix anything, so the fix gets reviewed and re-staged rather than silently included.
2. **CI** — the `lint` job in `.github/workflows/test.yml` runs `uv run ruff check .` on every push/PR, independent of the pre-commit hook (covers contributors who haven't installed it, and catches anything the hook was bypassed for).
3. **Manual** — `uv run ruff check .` / `uv run ruff check --fix .` any time.

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

## Kalshi changelog review

The system monitors the Kalshi API changelog daily via `freqpred/ingestion/kalshi_changelog.py`. Unreviewed entries surface as alerts and in the system health UI. When asked to review the changelog — or when `unreviewed_count > 0` is noticed — follow this process:

1. **Check the DB for unreviewed entries:**
   ```bash
   docker exec freqpred-db-1 psql -U freqpred -d freqpred -c \
     "SELECT last_reviewed_at, unreviewed_count, has_unreviewed_breaking_change, last_checked_at FROM kalshi_changelog_state WHERE id = 1;"
   ```

2. **Read the live changelog** at `https://docs.kalshi.com/changelog` — fetch all entries published after `last_reviewed_at`.

3. **Audit each unreviewed entry against the codebase.** For each entry, determine whether it requires a code change. Pay particular attention to:
   - Field renames or removals on order, market, or fill responses (the Kalshi client in `freqpred/markets/` parses these)
   - WebSocket channel or event type changes (`freqpred/markets/watcher.py`, `position_watcher.py`)
   - Endpoint URL or base host changes (CLAUDE.md Tech stack section + any hardcoded URLs)
   - Rate limit scheme changes (retry/backoff logic)
   - Breaking changes flagged in the RSS `<category>` tags
   If a code change is required, implement and test it before proceeding.

4. **Mark entries as reviewed** by creating an Alembic migration that updates all three fields atomically:
   ```bash
   uv run alembic revision -m "mark_kalshi_changelog_reviewed_YYYY_MM_DD"
   ```
   In the migration's `upgrade()`:
   ```python
   op.execute(
       "UPDATE kalshi_changelog_state"
       " SET last_reviewed_at = 'YYYY-MM-DD',"  # literal review date, NOT CURRENT_DATE
       "     unreviewed_count = 0,"
       "     last_checked_at = NULL"
       " WHERE id = 1"
   )
   ```
   **Use the literal review date** (e.g. `'2026-06-04'`), never `CURRENT_DATE`. Migrations are replayed by future developers and CI — `CURRENT_DATE` would stamp the wrong date on any replay, making `last_reviewed_at` meaningless as an audit trail.

   Nulling `last_checked_at` forces the monitor to re-run immediately on next startup and verify the count from RSS. Setting `unreviewed_count = 0` makes the dashboard correct right away without waiting for that re-run.

   Then apply:
   ```bash
   DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred" uv run alembic upgrade head
   ```

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

**For any change to the signal LLM or its inputs — `SYSTEM_PROMPT`, `build_prompt`, `PROMPT_VERSION` (`freqpred/signal/llm.py`), the signal model (`signal.model` config / `_DEFAULT_MODEL`), or any new/changed data block fed into the signal prompt:**
- Follow the full workflow in README → "Changing the signal prompt — the standard workflow". Non-negotiable steps: scope the edit to written-down findings, bump `PROMPT_VERSION`, regenerate the committed replay fixtures (`uv run freqpred fixtures replay --update`), and benchmark the new version with `scripts/benchmark_signals.py` (prompt mode for prompt/data changes; model mode with `--candidate-model` for model swaps) before treating the change as adopted. Do not adopt on inspection alone — propose the benchmark run to the user as the required validation step.
- Never regenerate `benchmarks/prompt_bank/` after a version bump — it is the frozen control baseline for the experiment; `record-bank` filters on the current version and would empty it.
- Before any benchmark run, check today's LLM spend against the daily cap (it is shared with the live pipeline; exhausting it blocks live signal analysis until the UTC day rolls over) and confirm the run with the user — benchmark runs cost real API dollars.
- One axis per experiment: a prompt change and a model change are separate benchmark decisions, never bundled.

**For any change to the sizing assessor's LLM or its inputs — `_SYSTEM_PROMPT`, `_build_prompt_payload`, `_PROMPT_VERSION` (`freqpred/metrics/assessment.py`), the judgment model (`anthropic.judgment_model`), `assessment_scale_min/max`, or any new/changed data section in the assessment payload:**
- Follow README → "Auditing the sizing assessor". Validate with the paired two-arm screen: `current` = the live production package (needs no setup, it is whatever shipped), `challenger` = the proposed package, defined in the CHALLENGER block at the top of `scripts/audit_assessor_enhancement.py`. Do not adopt on inspection alone — propose the run to the user as the required validation step.
- **Use the frozen eval set, not the reshuffling sample.** Score with `scripts/run_frozen_eval.py` against `scripts/.audit_output/frozen_eval_set.json` (built by `scripts/freeze_assessor_eval_set.py`). The old `_pick_sample` path reshuffles over a pool that grows as assessments accrue, and it is not fit for adoption decisions: the *identical* v6/opus-4-7 package scored corr +0.529 on one draw and +0.246 on the next, so run-to-run differences were sample composition rather than package quality. The frozen set stores rendered payloads plus a hash, so runs are byte-reproducible and harness drift invalidates a cached score instead of silently mixing old scores with new prompts. Regenerate the set only deliberately — it is the frozen baseline, and regenerating it discards every cached `current` score.
- **Adoption gate: capital tilt and incremental AUC over the free prior** (`scripts/.audit_output/analyze_noninferiority.py`), not raw correlation. Capital tilt (mean size multiplier on winners minus losers) is the direct expression of "put more size on winners". AUC is ~2× better powered than Pearson correlation on this data (CI width 0.336 vs 0.622) and is scale-invariant, so it separates ranking quality from calibration. Always read the **incremental-over-prior** line: absolute AUC flatters every arm because most of it is reproducible from a free direction×edge-band lookup — as of 2026-07-25 no assessor package has beaten that lookup, so an arm that merely matches it is not earning its ~$0.04/signal. Also check the verdict distribution: an assessor that never says size_up is a flat tax, not a discriminator.
- Keep `max_tokens` at or above the audited value (currently 1024) and keep `trust_score` emitted first in the tool call. At 768 a wordier package truncated mid-response and could drop `trust_score` entirely, which fails open to neutral 1.0x sizing — a silent loss of the sizing decision rather than a visible error.
- Retrospective audits must use the script's point-in-time loader copies, never the raw production context loaders — on since-resolved markets those self-leak the assessed market's own outcome into the prompt (production is immune; an unresolved market has no outcome to leak).
- Audit calls cost real API dollars (~$0.04/signal, two Opus calls) and share the daily LLM cap with the live pipeline — check today's spend and confirm the run with the user first. Calls are logged as `query_type="model_eval"`; nothing may ever be written to `signal_assessments` from an audit.
- Bump `_PROMPT_VERSION` (`assessment-vN`) on any adopted prompt/payload change so old and new assessments stay distinguishable in `llm_queries` and `signal_assessments`.

---

## Diagrams

Always use Mermaid diagrams instead of ASCII art. Wrap them in ` ```mermaid ` fences. Keep node labels concise — `\n` does not work in Mermaid; use plain text labels and let surrounding prose carry the detail. Never draw boxes with `┌─┐│└┘` characters. This applies to SPEC.md, GitHub issues, and any other documentation.

---

## Dashboard UI conventions

### Text color on colored backgrounds — always use CSS variables, never hardcoded colors

When rendering text inside cells or elements that have a dynamically-colored background (e.g. heatmap cells, badges, status indicators), use `var(--fg-0)` for the text color instead of hardcoded values like `#4ade80` or `#f87171`.

**Why:** The dashboard supports both dark mode and light mode. Hardcoded light colors (e.g. bright green `#4ade80`) are invisible on light-mode backgrounds of the same hue — green text on a light-green cell. CSS variables like `var(--fg-0)` resolve to white in dark mode and near-black in light mode, giving correct contrast in both themes automatically.

**Rule:** If text sits on top of a background whose color is determined at runtime (e.g. from a delta value or status), use `var(--fg-0)`. Direction or severity is already communicated by the background color and the +/− sign; repeating the color in the text is redundant and causes legibility problems.

```tsx
// Wrong — green text on green cell = invisible in light mode
<div style={{ color: delta > 0 ? '#4ade80' : '#f87171' }}>
  {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
</div>

// Correct — CSS variable adapts to theme automatically
<div style={{ color: 'var(--fg-0)', opacity: 0.8 }}>
  {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
</div>
```

This applies to: heatmap cell content, badge labels, inline stat values, and any other text rendered on a programmatically-colored surface.

---

## Code style

- Type hints everywhere — no untyped functions
- Pydantic models for all external data (API responses, config)
- SQLAlchemy models in `*/models.py` within each module
- Dataclasses only for internal data transfer objects
- Async/await throughout — no blocking I/O on the main thread
- Log with structured logging (`structlog`) — no bare `print()`
- Errors: raise specific exceptions, never swallow silently
