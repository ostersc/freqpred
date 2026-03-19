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
```

| Option | Default | Description |
|---|---|---|
| `--status` | `all` | `open`, `closed`, or `all` |
| `--limit` | `50` | Maximum rows to display |

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

Additional management commands (pause/resume, status, force-signal) will be added in subsequent tickets and appear in `/help` automatically once registered.

### Registering custom handlers (for developers)

```python
handler = TelegramCommandHandler(bot_token=..., authorized_users=[...])

async def my_handler(chat_id: int, args: list[str]) -> str:
    return f"Hello from freqpred! Args: {args}"

handler.register("greet", my_handler)
```

Handler receives `(chat_id: int, args: list[str])` and should return a plain-text reply string (or `None` to send no reply).
