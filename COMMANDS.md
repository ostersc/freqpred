# freqpred command reference

## CLI commands

All commands are run via `uv run freqpred <command>`.

---

### `run` — start the full pipeline

```bash
uv run freqpred run --strategy <name|path> --mode <signal-only|paper|live>
```

| Option | Default | Description |
|---|---|---|
| `--strategy` | required | Strategy class name (`ConservativeDefault`) or path to a `.py` file |
| `--mode` | `paper` | `signal-only` — no orders; `paper` — simulated orders; `live` — real orders (requires `LIVE_TRADING_ENABLED=true`) |
| `--bankroll` | _(from config)_ | Override `trading.bankroll_usd` for this run (useful for demo env where balance < prod bankroll) |

Starts the trading loop plus the embedded API server (port 8000 by default, controlled by `dashboard.port` in `config.yaml`). Pending Alembic migrations are applied automatically at startup. The API server runs inside this process so it shares the live `OrderManager` — required for force-exit from the dashboard or Telegram. Press **Ctrl+C** to stop all tasks cleanly.

To disable the embedded API, set `dashboard.api_enabled: false` in `config.yaml`.

---

### `markets list` — fetch active Kalshi markets

```bash
uv run freqpred markets list
uv run freqpred markets list --category Elections
uv run freqpred markets list --min-volume 500 --max-days 30
uv run freqpred markets list --no-db
```

| Option | Default | Description |
|---|---|---|
| `--category` | none | Filter by Kalshi category string, case-sensitive (e.g. `Elections`, `Sports`, `World`, `Science and Technology`) |
| `--min-volume` | none | Only show markets with `volume_24h` >= this value |
| `--max-days` | none | Only show markets closing within this many days |
| `--no-db` | false | Print markets but skip writing to the database |

Fetches live markets from the Kalshi API, writes them to the `markets` table, and prints a summary table with `VOL_24H` and `DAYS` columns.

---

### `signal analyze` — one-shot signal for a market

```bash
uv run freqpred signal analyze --market-id <KALSHI-TICKER>
uv run freqpred signal analyze --market-id <KALSHI-TICKER> --force
```

| Option | Default | Description |
|---|---|---|
| `--market-id` | required | Kalshi market ticker to analyze |
| `--force` | false | Bypass hash deduplication and force a new LLM call |
| `--strategy` | `PoliticsEdgeStrategy` | Strategy to load (determines FactBase allowlist and other config) |

Embeds the market question, retrieves relevant documents via vector search, and calls Claude for a probability estimate. Prints the full signal (probability, edge, confidence, direction, reasoning).

---

### `ingestion run` — manually run the news ingestion pipeline

```bash
uv run freqpred ingestion run
uv run freqpred ingestion run --limit 5
uv run freqpred ingestion run --category Elections --dry-run
uv run freqpred ingestion run --min-volume 500
```

| Option | Default | Description |
|---|---|---|
| `--limit` | `3` | Maximum number of markets to process |
| `--category` | none | Only process markets in this exact Kalshi category string |
| `--min-volume` | `0` | Minimum 24h volume filter |
| `--dry-run` | false | Generate catalysts but skip news fetching |

Generates 3–5 Claude Haiku catalyst queries per market, then runs Tavily + NewsAPI + Reddit fetchers against those queries and stores results with local embeddings.

---

### `positions list` — list trading positions

```bash
uv run freqpred positions list
uv run freqpred positions list --status open
uv run freqpred positions list --status closed --limit 100
uv run freqpred positions list --days 1
uv run freqpred positions list --strategy PoliticsEdgeStrategy --days 7
```

| Option | Default | Description |
|---|---|---|
| `--status` | `all` | `open`, `closed`, or `all` |
| `--limit` | `50` | Maximum rows to display |
| `--strategy` | _(none)_ | Filter by strategy name (exact match) |
| `--days` | _(none)_ | Only show positions entered within the last N days (e.g. `1` = last 24 hours, `0.5` = last 12 hours) |

Columns include HELD (time held: `exit_time - entry_time` for closed, elapsed since entry for open), PNL% (return over capital spent: `pnl / (entry_price × contracts + fee)`, only populated on close), MAE and MFE (signed price deltas; multiply by contracts for dollar impact).

---

### `positions resolve` — close a position against a market resolution

```bash
uv run freqpred positions resolve --position-id <UUID> --resolution yes
uv run freqpred positions resolve --position-id <UUID> --resolution no
```

Calculates P&L based on the resolution outcome and closes the position in the database.

---

### `metrics calibration` — print Brier score and calibration buckets

```bash
uv run freqpred metrics calibration
uv run freqpred metrics calibration --days 30
uv run freqpred metrics calibration --period month
```

Prints overall Brier score over finalized signals, compared against the market baseline (`market_mid_at_signal` vs. final outcome), plus a probability-bucket breakdown. This scores all qualifying signals, not just traded positions.

Options:
- `--days N` — restrict to signals created within the last N days
- `--period day|week|month` — convenience aliases for `--days 1/7/30`

---

### `metrics source-calibration` — weighted Brier score per document source name

```bash
uv run freqpred metrics source-calibration
uv run freqpred metrics source-calibration --days 30
uv run freqpred metrics source-calibration --period week
uv run freqpred metrics source-calibration --min-docs 100
uv run freqpred metrics source-calibration --min-docs 0   # show all sources including long tail
```

For each resolved signal, distributes the Brier loss across its evidence documents proportionally by `source_name` (e.g. `Tavily`, `r/politics`, `The Guardian`). Aggregates into a weighted-average Brier score per source — lower is better. Only signals with at least one linked document are included.

Options:
- `--days N` — restrict to signals created within the last N days
- `--period day|week|month` — convenience aliases for `--days 1/7/30`
- `--min-docs N` — hide sources with fewer than N total document appearances across qualifying signals (default: 50)

---

### `report digest` — generate a daily summary

```bash
uv run freqpred report digest
uv run freqpred report digest --send --mode live
```

| Option | Default | Description |
|---|---|---|
| `--send` | false | Also send the digest via configured Telegram/Discord alert channels |
| `--mode` | `paper` | Trading mode label to display in the digest (`paper`, `live`, `signal-only`) |

Produces the daily digest in two parts: a **deterministic stat header** (run state, open positions vs cap, exposure, unrealized P&L, session P&L with win/loss counts, drawdown, LLM spend vs cap, calibration vs market baseline, 24h signal activity, and per-service health from telemetry heartbeats) followed by a Claude Haiku **analyst take** — 3–5 prioritized bullets flagging only what deserves attention (the model receives per-position P&L, top signals by edge, exit breakdown, and stale-service errors, and is instructed not to restate header numbers). Output goes to stdout.

---

### `alerts test` — verify alert channel credentials

```bash
uv run freqpred alerts test --channel telegram
uv run freqpred alerts test --channel discord
uv run freqpred alerts test --channel all
```

Sends a test message to confirm that the configured credentials work. This command is only meaningful when the relevant channel credentials are configured.

---

### `db migrate` — apply database migrations

```bash
uv run freqpred db migrate
```

Equivalent to `alembic upgrade head` — it shells out to exactly that. Safe to run repeatedly.

**Requires `DATABASE_URL` to be exported.** Migrations resolve the database from the environment only; `migrations/env.py` does not read `config/config.yaml`, and `.env` is not auto-loaded by `uv run`. Without it the command exits with `RuntimeError: DATABASE_URL environment variable is not set`. To target a non-default database (test, demo), pass it inline and call alembic directly:

```bash
DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" uv run alembic upgrade head
```

---

### `dashboard` — start the Vite dev server (UI development only)

```bash
uv run freqpred dashboard
```

Starts the Vite dev server at `http://localhost:5173` for hot-reload UI development. API calls are proxied to `http://localhost:8000` (the API embedded inside `freqpred run`).

**Requires `freqpred run` to be running** — the dashboard command has no database or business logic of its own. If `freqpred run` is not running, the UI will show API connection errors.

In production, `freqpred run` also serves the built React SPA (from `freqpred/dashboard/ui/dist/`) at port 8000. No separate `dashboard` command is needed; build the UI once with `npm run build` inside `freqpred/dashboard/ui/`.

---

### `fixtures record` — record a replay-harness fixture from a real signal

```bash
uv run freqpred fixtures record --signal-id <uuid>
uv run freqpred fixtures record --signal-id <uuid> --strategy PoliticsEdgeStrategy \
    --name my_fixture --description "why this scenario matters"
```

Snapshots one LLM-backed signal into a deterministic replay fixture under `tests/fixtures/replay/` — market state at signal time, retrieved documents with scores, catalyst queries, series-history/FactBase context, the frozen clock, and the verbatim LLM response — plus expectations for every pipeline stage (retrieval hash, rendered prompt, parsed output, edge, and the entry decision through the risk caps). Embedding vectors are deliberately not stored: fixtures freeze retrieval *outcomes*; retrieval-code correctness is covered by targeted tests in `tests/integration/test_retriever_integration.py`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--signal-id` | (required) | UUID of an LLM-backed signal (`llm_query_id` set; not a price-moved clone) |
| `--out` | `tests/fixtures/replay/<name>.json` | Output path |
| `--strategy` | `ConservativeDefault` | Strategy used for the entry-decision expectations |
| `--bankroll` | `1000.0` | Frozen bankroll for sizing/risk expectations |
| `--name` | `<market_id>_<direction>` | Fixture name |
| `--description` | (empty) | Free-text note stored in the fixture |

---

### `fixtures record-bank` — build the prompt-mode scenario bank from resolved markets

```bash
uv run freqpred fixtures record-bank                       # all resolved markets, current prompt version
uv run freqpred fixtures record-bank --out-dir benchmarks/prompt_bank --limit 100
```

Records a **frozen-context** fixture from every LLM-backed signal on finalized binary markets (default `--per-market all`; `--per-market last` records only each market's final pre-resolution signal). Unlike plain `fixtures record`, inputs are parsed from the signal's stored `raw_context` — never from the live tables, which contain the outcome after resolution (`in_market_count` includes the occurrence that resolved the market; the series counts include the market's own settlement; the market row's category/close_time/question drift). Every fixture is verified by re-rendering and requiring **byte-equality** with the stored prompt; signals that fail the round-trip are skipped, never written. The round-trip is also the prompt-version compatibility gate — signals from older prompt versions whose stored prompt still re-renders byte-exactly under the current template are recordable.

The output directory is gitignored. The bank is regenerable from the DB **while `build_prompt`'s user-prompt template is unchanged** — a `PROMPT_VERSION` bump that only edits `SYSTEM_PROMPT` (e.g. v9 → v10) keeps every old signal recordable, but a template change makes older signals fail the round-trip, and a re-sweep then loses them until new-template markets resolve. When benchmarking a prompt-template change, treat the recorded bank as the experiment's frozen baseline: do not delete or regenerate it mid-experiment (see README → "Changing the signal prompt"). Use it as the scenario source for prompt benchmarking:

```bash
uv run python scripts/benchmark_signals.py --prompt-mode --fixtures benchmarks/prompt_bank \
    --training-cutoff 2026-03-01 --limit 250
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--out-dir` | `benchmarks/prompt_bank` | Bank directory (gitignored) |
| `--strategy` | `PoliticsEdgeStrategy` | Strategy for the fixtures' entry-decision expectations |
| `--limit` | all | Max markets |
| `--per-market` | `all` | `all` = every LLM-backed signal per market (benchmark samples at run time); `first` = earliest signal only (pure entry decisions); `last` = final pre-resolution signal only |

---

### `fixtures replay` — replay fixtures and report regressions

```bash
uv run freqpred fixtures replay                    # all of tests/fixtures/replay/
uv run freqpred fixtures replay path/to/one.json   # specific file(s) or dirs
uv run freqpred fixtures replay --update           # regenerate expectations after an intentional change
```

Replays each fixture offline — no network, no LLM calls, no DB — and exits non-zero on any regression (changed retrieval hash, changed rendered prompt without a `PROMPT_VERSION` bump, changed parse/edge/trade decision). The same checks run in CI via `tests/unit/test_replay_harness.py`; `--update` is equivalent to `FREQPRED_UPDATE_FIXTURES=1 uv run pytest tests/unit/test_replay_harness.py`, and the regenerated fixture diff should be reviewed like any snapshot change.

---

## Telegram bot commands

When the bot token is configured and `telegram_authorized_users` is set, `freqpred run` starts an inbound polling loop. The bot accepts `/commands` from authorized users only — unrecognized senders are silently ignored.

### Authorization

Set `telegram_authorized_users` in `config.yaml` to a list of Telegram usernames or numeric user IDs (as strings). Empty list means no one can send commands.

```yaml
alerts:
  telegram_authorized_users:
    - alice          # matched against Telegram username
    - "123456789"   # matched against numeric user ID
```

### Reply formatting

Command replies are sent with Telegram's HTML parse mode: bold section headers, `<pre>` blocks for aligned tables, prices in cents (`43¢`), P&L in signed dollars (`+$1.20`), and ages as compact durations (`2h 15m`). If Telegram rejects a reply's markup, the bot automatically re-sends it as plain text so the content is never lost. Long replies are truncated at a line boundary under the 4096-char message limit; tables drop rows with a `... and N more` footer instead of cutting mid-row.

### Built-in commands

| Command | Description |
|---|---|
| `/help` | List all registered commands, grouped by category (System / Positions / Performance / Diagnostics) with a one-line description each |

### System control commands

| Command | Description |
|---|---|
| `/start` | Set run loop to `running`; new positions resume. Also acknowledges the daily-loss circuit breaker — for the running process's trading mode only (paper and live acknowledgements are independent) |
| `/pause` | Set run loop to `paused`; no new positions, existing positions still managed |
| `/stop` | Halt signal analysis entirely; use `/start` to resume |
| `/shutdown` | Gracefully shut down the freqpred process. Requires inline keyboard confirmation (30 s timeout). Sends a shutdown alert to all configured channels before exiting. |
| `/reset_drawdown` | Reset the drawdown circuit breaker for the running process's trading mode. Stores the current timestamp and net bankroll as that mode's baseline; drawdown is measured only from this point forward. Paper and live baselines are independent — resetting one never touches the other. No-op in signal-only mode. |
| `/show_config` | Show strategy name, mode, min edge, max position size, max open positions, LLM budget |
| `/logs [n] [filter]` | Last *n* log lines (default 20), optionally filtered by logger name segment; rendered in a monospace block |
| `/version` | freqpred version + short git commit hash |

State changes (`/start`, `/pause`, `/stop`) are persisted in the database — a process restart picks up the last state.

### Status query commands

| Command | Description |
|---|---|
| `/status` | Summary header (run state, mode, strategy, open count vs cap, total unrealized P&L, drawdown vs baseline) followed by one block per open position: direction, contracts, ticker, question, entry → current price in cents, unrealized P&L ($ and %), time open. NO positions show NO-side prices. |
| `/status <position_id_or_market_ticker>` | Detailed single-position view: status, entry → current price, unrealized (or realized) P&L, cost basis, signal snapshot at entry (est. prob, edge, confidence), MAE/MFE in dollars, position UUID. Accepts either a UUID or a market ticker (open position preferred). |
| `/trades [n]` | Last *n* closed positions (default 10) with a net P&L header: win/loss icon, P&L ($ and %), direction, exit reason, hold duration, question |
| `/signals [n]` | Last *n* signals (default 10): direction icon, ticker, age, question, est. prob vs market price, edge, confidence, trigger |
| `/health` | Freshness of every scheduled service and fetcher (from runtime telemetry heartbeats): ok/stale/idle/unknown per service with last-success age, stale services first with their last error, plus WebSocket connection state |

`/count` was removed — its `Open: N / Max: M` information is in the `/status` header.

### Metrics and performance commands

| Command | Description |
|---|---|
| `/profit [n]` | P&L summary over the last *n* days (default: all time): trade count, win rate, total P&L ($ and % on invested), best/worst trade, avg hold duration, Brier score |
| `/daily [n]` | Table: date \| trade count \| P&L $ \| P&L % — last *n* days (default 7) |
| `/weekly [n]` | Table: week start \| trade count \| P&L $ \| P&L % — last *n* weeks (default 8) |
| `/monthly [n]` | Table: month \| trade count \| P&L $ \| P&L % — last *n* months (default 6) |
| `/stats` | All-time aggregate stats: total trades, P&L, win rate, best/worst trade, avg hold duration, plus an aligned exit-reason breakdown table |
| `/balance` | Portfolio snapshot: bankroll, all-time P&L, net value, gross/net exposure, unrealized P&L, today's P&L, open position count, contract-weighted portfolio MAE/MFE |
| `/budget` | LLM cost: today vs daily cap (%) with time until reset, per-query-type breakdown table, this week, this month, all-time |
| `/calibration [days]` | Brier score vs market baseline, improvement, sample count, per-probability-bucket breakdown. Optional `days` arg limits to last N days (e.g. `/calibration 30`); omit for all-time. |
| `/source_calibration [days] [min_docs]` | Weighted Brier score per document source name. Optional `days` limits lookback; optional `min_docs` sets minimum doc appearances threshold (default 50). E.g. `/source_calibration 30 100`. |
| `/digest` | On-demand daily digest: deterministic stat header (state, positions vs cap, session P&L with W/L, drawdown, LLM spend vs cap, calibration vs market, 24h signals, service health) + a Claude Haiku analyst take of 3–5 prioritized bullets flagging what deserves attention. Same content as the scheduled morning digest. |

Tabular responses use HTML `<pre>` blocks (monospace, aligned in the Telegram client). Rows are truncated under the 4096-char limit with a `... and N more` footer.

---

### Position management commands

| Command | Description |
|---|---|
| `/forceexit <position_id_or_market_id>` | Force-close a specific open position via `OrderManager.force_exit()`. Paper mode: closes immediately at current mid price with `exit_reason=force_exit:manual`. Live mode: requires inline keyboard confirmation, then submits an IOC sell to the exchange. |
| `/forceexit all` | Force-close all open positions. Always requires inline keyboard confirmation regardless of mode. |
| `/fx <position_id_or_market_id>` | Alias for `/forceexit`. |
| `/delete <position_id>` | Hard-delete a paper position record from the database without placing an order. Requires inline keyboard confirmation. Rejected with an error message in live mode. |

**Confirmation flow** — `/forceexit all`, `/forceexit <id>` in live mode, and `/delete <id>` in paper mode send an inline keyboard with **Confirm** and **Cancel** buttons before executing. If no button is pressed within 30 seconds the action is automatically cancelled and the bot sends a timeout notice. Pending confirmation state is stored in memory and is lost on process restart.

The argument to `/forceexit` accepts either a UUID position ID or a market ticker (e.g. `KXTRUMPSAY-26APR06-AUTO`) — if a market ticker is supplied and an open position exists for it, the position is looked up automatically.

---

### Registering custom handlers (for developers)

```python
handler = TelegramCommandHandler(bot_token=..., authorized_users=[...])

async def my_handler(chat_id: int, args: list[str]) -> str:
    return f"Hello from freqpred! Args: {args}"

handler.register("greet", my_handler, description="Say hello", category="Other")
```

Handler receives `(chat_id: int, args: list[str])` and should return a reply string (or `None` to send no reply). `description` and `category` are optional and drive the grouped `/help` output.

Replies are sent with `parse_mode=HTML` — handlers may use `<b>`, `<code>`, and `<pre>` markup, and **must HTML-escape any dynamic content** (market questions, error messages, LLM output) with `html.escape` / the `_esc` helper in `command_handlers.py`. If Telegram rejects the markup, the reply is automatically re-sent as plain text with tags stripped.
