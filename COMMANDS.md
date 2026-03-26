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

Starts four concurrent async tasks: market watcher, ingestion scheduler, signal pipeline, and Telegram command handler. Press **Ctrl+C** to stop all tasks cleanly.

---

### `markets list` — fetch active Kalshi markets

```bash
uv run freqpred markets list
uv run freqpred markets list --category politics
uv run freqpred markets list --no-db
```

| Option | Default | Description |
|---|---|---|
| `--category` | none | Filter by category (`politics`, `technology`, `economics`, ...) |
| `--no-db` | false | Print markets but skip writing to the database |

Fetches live markets from the Kalshi API, writes them to the `markets` table, and prints a summary table.

---

### `signal analyze` — one-shot signal for a market

```bash
uv run freqpred signal analyze --market-id <KALSHI-TICKER>
```

Embeds the market question, retrieves relevant documents via vector search, and calls Claude for a probability estimate. Prints the full signal (probability, edge, confidence, direction, reasoning).

---

### `ingestion run` — manually run the news ingestion pipeline

```bash
uv run freqpred ingestion run
uv run freqpred ingestion run --limit 5
uv run freqpred ingestion run --category politics --dry-run
uv run freqpred ingestion run --min-volume 500
```

| Option | Default | Description |
|---|---|---|
| `--limit` | `3` | Maximum number of markets to process |
| `--category` | none | Only process markets in this category |
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
```

Prints overall Brier score vs. naive baseline and a probability-bucket breakdown over all resolved positions.

---

### `report digest` — generate a daily summary

```bash
uv run freqpred report digest
```

Uses Claude to produce a natural-language summary of system health: open positions, P&L, LLM spend, and calibration. Output goes to stdout.

---

### `alerts test` — verify alert channel credentials

```bash
uv run freqpred alerts test --channel telegram
uv run freqpred alerts test --channel discord
uv run freqpred alerts test --channel all
```

Sends a test message to confirm that the configured credentials work. Missing credentials are silently skipped.

---

### `db migrate` — apply database migrations

```bash
uv run freqpred db migrate
```

Equivalent to `alembic upgrade head`. Safe to run repeatedly.

---

### `dashboard` — start the read-only API server

```bash
uv run freqpred dashboard
uv run freqpred dashboard --host 127.0.0.1 --port 9000
```

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host to bind |
| `--port` | `8000` | Port to listen on |

Starts the FastAPI dashboard server. Exposes a JSON API for signals, positions, and calibration data.

---

## Telegram bot commands

When the bot token is configured and `authorized_users` is set, `freqpred run` starts an inbound polling loop. The bot accepts `/commands` from authorized users only — unrecognized senders are silently ignored.

### Authorization

Set `telegram_authorized_users` in `config.yaml` to a list of Telegram usernames or numeric user IDs (as strings). Empty list means no one can send commands.

```yaml
alerts:
  telegram_authorized_users:
    - alice          # matched against Telegram username
    - "123456789"   # matched against numeric user ID
```

### Built-in commands

| Command | Description |
|---|---|
| `/help` | List all registered commands |

### System control commands

| Command | Description |
|---|---|
| `/start` | Set run loop to `running`; new positions resume |
| `/pause` | Set run loop to `paused`; no new positions, existing positions still managed |
| `/stop` | Halt signal analysis entirely; use `/start` to resume |
| `/reset_drawdown` | Reset the drawdown circuit breaker. Stores the current timestamp; drawdown is measured only from this point forward. |
| `/show_config` | Show strategy name, mode, min edge, max position size, LLM budget |
| `/logs [n]` | Last *n* log lines (default 20); truncated at 4096 chars |
| `/version` | freqpred version + short git commit hash |

State changes (`/start`, `/pause`, `/stop`) are persisted in the database — a process restart picks up the last state.

### Status query commands

| Command | Description |
|---|---|
| `/status` | List all open positions: market question, direction, entry price, est. prob, unrealized P&L, MAE, MFE |
| `/status <position_id>` | Detailed single-position view: confidence, edge at entry, MAE/MFE (with dollar value), time open |
| `/count` | `Open: N / Max: M` |
| `/trades [n]` | Last *n* resolved positions (default 10): market, exit reason, P&L, hold duration |
| `/signals [n]` | Last *n* signals (default 10): market, our prob, market price, edge, direction |

### Metrics and performance commands

| Command | Description |
|---|---|
| `/profit [n]` | P&L summary over the last *n* days (default: all time): total P&L ($, %), win rate, trade count, avg hold duration, best/worst trade, Brier score |
| `/daily [n]` | Table: date \| trade count \| P&L $ \| P&L % — last *n* days (default 7) |
| `/weekly [n]` | Table: week start \| trade count \| P&L $ \| P&L % — last *n* weeks (default 8) |
| `/monthly [n]` | Table: month \| trade count \| P&L $ \| P&L % — last *n* months (default 6) |
| `/stats` | All-time aggregate stats: total trades, P&L, win rate, best/worst trade, avg hold duration, breakdown by exit reason |
| `/balance` | Portfolio snapshot: bankroll, all-time P&L, net value, gross/net exposure, unrealized P&L, today's P&L, open position count, contract-weighted portfolio MAE/MFE |
| `/budget` | LLM cost breakdown: today vs daily cap (%), per-query-type breakdown, this week, this month, all-time |
| `/calibration [days]` | Brier score vs market baseline, improvement, sample count, per-probability-bucket breakdown. Optional `days` arg limits to last N days (e.g. `/calibration 30`); omit for all-time. |
| `/digest` | On-demand daily digest: Claude Haiku natural-language summary of open positions, P&L, LLM spend, and calibration |

Tabular responses use monospace code blocks. Rows are truncated at 4096 chars with `... and N more` footer.

---

### Position management commands

| Command | Description |
|---|---|
| `/forceexit <position_id>` | Force-close a specific open position immediately. Paper mode: closes at current mid price with `exit_reason=manual_telegram`. Live mode: requires inline keyboard confirmation, then closes the ledger record at current mid price. |
| `/forceexit all` | Force-close all open positions. Always requires inline keyboard confirmation regardless of mode. |
| `/fx <position_id>` | Alias for `/forceexit <position_id>`. |
| `/delete <position_id>` | Hard-delete a paper position record from the database without placing an order. Requires inline keyboard confirmation. Rejected with an error message in live mode. |

**Confirmation flow** — `/forceexit all`, `/forceexit <id>` in live mode, and `/delete <id>` in paper mode send an inline keyboard with **Confirm** and **Cancel** buttons before executing. If no button is pressed within 30 seconds the action is automatically cancelled and the bot sends a timeout notice. Pending confirmation state is stored in memory and is lost on process restart.

---

### Registering custom handlers (for developers)

```python
handler = TelegramCommandHandler(bot_token=..., authorized_users=[...])

async def my_handler(chat_id: int, args: list[str]) -> str:
    return f"Hello from freqpred! Args: {args}"

handler.register("greet", my_handler)
```

Handler receives `(chat_id: int, args: list[str])` and should return a plain-text reply string (or `None` to send no reply).
