---
name: run-freqpred
description: Build, run, and drive the freqpred dashboard (React/Vite SPA backed by a FastAPI API embedded in `freqpred run`). Use when asked to start freqpred, run the dashboard, take a screenshot of the dashboard UI, verify a UI change, or interact with the running app in a browser.
---

freqpred's only browser-drivable surface is the dashboard: a Vite-served
React SPA (`freqpred/dashboard/ui/`) that talks to a FastAPI API embedded
inside the `freqpred run` process. `chromium-cli` is not installed in this
environment, so drive it via the Playwright REPL at
`.claude/skills/run-freqpred/driver.mjs` instead — it exposes the same kind
of `nav` / `click` / `screenshot` command vocabulary.

All paths below are relative to the repo root.

## Prerequisites

```bash
docker-compose up -d db      # Postgres 16 + pgvector, idempotent if already up
uv sync                      # Python deps via uv
cd .claude/skills/run-freqpred && npm install   # installs Playwright for the driver
npx playwright install chromium   # only if not already cached (~93MB, one-time;
                                   # cache: ~/Library/Caches/ms-playwright on macOS,
                                   # ~/.cache/ms-playwright on Linux)
```

## Setup

Migrations: this repo's own convention (see root `CLAUDE.md`) is to verify
`alembic upgrade head` against `freqpred_test`, **never** the `freqpred` dev
database directly — the running dashboard's data lives there and schema
commands are not something to run against it casually.

```bash
DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test" uv run alembic upgrade head
```

Real Kalshi/Anthropic/Tavily/etc. API credentials are required to start
`freqpred run` itself (see `.env`, gitignored) — see **Gotchas** below before
trying to launch a fresh instance.

## Run (agent path)

**Check first whether the app is already running** — in this persistent dev
container it usually is (the dashboard backend costs real money to start
fresh; see Gotchas):

```bash
lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | grep -E ':5173|:8000'
```

If both `5173` (Vite) and `8000` (API, proxied through Vite) are listening,
skip straight to driving it. Otherwise see **Run (human path)** below, and
confirm with the user before starting `freqpred run` — it makes real
external API calls.

Drive it with the REPL driver, piping commands via heredoc:

```bash
cd .claude/skills/run-freqpred
node driver.mjs <<'EOF'
launch
nav http://localhost:5173/positions
wait-for text=Positions
screenshot 01-positions
console --errors
quit
EOF
```

Screenshots land in `.claude/skills/run-freqpred/screenshots/` (override via
`SCREENSHOT_DIR`). For iterative work, run the driver interactively (just
`node driver.mjs`, type commands at the `driver>` prompt) or under tmux with
`send-keys`/`capture-pane`.

### Commands

| command | what it does |
|---|---|
| `launch` | launch headless Chromium |
| `nav <url>` | navigate, waits for network idle |
| `wait-for <selector>` | wait up to 15s for a selector (supports `text=...`) |
| `click <selector>` | click via Playwright locator (auto-waits) |
| `click-text <text>` | click first element whose text **contains** the string — see Gotchas, prefer `click` with a scoped CSS selector when the text is short/generic |
| `fill <selector> <text>` | fill a form field |
| `press <key>` | keyboard press (e.g. `Enter`) |
| `sleep <ms>` | fixed delay — last resort; prefer `wait-for` |
| `screenshot [name]` | full-page screenshot |
| `screenshot-element <selector> [name]` | crop to one element |
| `eval <js-expr>` | `page.evaluate`, prints JSON |
| `text [selector]` | print `innerText` (body if no selector) |
| `console --errors` | print collected `pageerror`/`console.error` events since launch |
| `quit` | close the browser |

## Run (human path)

The API server is embedded in the trading loop, not standalone:

```bash
uv run freqpred run --strategy <StrategyClassName> --mode signal-only   # starts API on :8000
uv run freqpred dashboard                                               # separate terminal, starts Vite on :5173
```

`--mode signal-only` avoids placing any orders but still polls Kalshi,
runs ingestion fetchers, and calls the Claude API — it is not free or
side-effect-free. Don't start this without checking with the user first
(see Gotchas).

## Test

```bash
uv run pytest tests/unit/ -q   # 1179 passed, no DB/API needed
```

## Direct invocation (non-UI changes)

Most PRs in this repo touch the Python pipeline, not the dashboard — for
those, `pytest` above is the right verification, not the browser driver.
For a quick CLI sanity check with no credentials required:

```bash
uv run freqpred --help
```

## Gotchas

- **Don't start `freqpred run` casually.** Unlike a typical disposable
  container, this repo's `freqpred run` hits real, metered external APIs
  (Anthropic Claude for signal analysis, Kalshi market data, Tavily/NewsAPI/
  Guardian/Reddit/GDELT for ingestion) using real credentials already
  configured in `.env`. Spinning up a second instance also writes to the
  same `freqpred` Postgres database as any already-running instance,
  causing duplicate ingestion/LLM calls and duplicate cost. In this
  persistent dev environment, `freqpred run` + `freqpred dashboard` are
  usually already running from the user's own workflow — check with `lsof`
  (see Run agent path) before assuming you need to launch anything.

- **`alembic upgrade head` needs `DATABASE_URL` set explicitly** — it is
  not read from `config.yaml`/env-mapped config the way the app itself
  resolves it. Bare `uv run alembic upgrade head` fails with
  `RuntimeError: DATABASE_URL environment variable is not set`.

- **`click-text` does substring matching, not exact.** `click-text all`
  intended to hit the Positions page's `open`/`closed`/`all` status filter
  matched the top-nav "all-time" P&L chip instead (DOM order: nav comes
  before the filter control, and `"all"` is a substring of `"all-time"`).
  The click silently landed on the wrong element with no error. Use a
  scoped selector instead, e.g. `click .seg-item:text-is("all")`
  (Playwright's `:text-is()` CSS pseudo-class requires an exact match).

- **Piped heredoc input races the driver's async command handlers.**
  `readline`'s `line` events for a heredoc all fire in the same tick
  (unlike real typing/tmux `send-keys`, which has natural delay between
  lines), so without an explicit queue, `nav` right after `launch` would
  run before `launch`'s `chromium.launch()` had resolved and fail with a
  bogus `ERROR: launch first`. `driver.mjs` serializes commands through a
  promise chain (`chain = chain.then(...)`) and the `close` handler
  `await`s that chain before exiting — both are required; either one
  missing reproduces the race.

## Troubleshooting

- **`ERROR: launch first` on the very first `nav`/`click-text`/etc. even
  though `launch` was the previous line:** you're running a driver build
  without the promise-chain fix described above — check `driver.mjs` still
  has `chain = chain.then(...)` in the `line` handler and `await chain` in
  the `close` handler.
- **`RuntimeError: DATABASE_URL environment variable is not set`** running
  alembic: prefix the command with
  `DATABASE_URL="postgresql+asyncpg://freqpred:freqpred@localhost:5432/freqpred_test"`.
- **Chromium download during `npm install`/first launch:** expected once
  per machine (~93MB); subsequent runs reuse the Playwright cache dir.
