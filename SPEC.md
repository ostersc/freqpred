# freqpred — Project Specification

> A framework for LLM-driven prediction market trading, modeled on freqtrade's architecture.

**Version:** 0.1-draft
**Last updated:** 2026-06-03 (Polymarket §6b + Phase 4)
**Status:** Phase 2 complete — paper trading running; Phase 3 (live trading + ops hardening) in progress

---

## 1. Vision

freqpred is an extensible, strategy-driven framework for trading prediction markets using LLM sentiment analysis. The core thesis: LLMs can estimate the "true" probability of future events by reasoning over current news and context. Where that estimate diverges meaningfully from a market's implied probability, there is a tradeable edge.

The framework is:
- **Personal first** — optimized for a single operator running their own strategies
- **Open source ready** — architecturally clean enough to publish and extend
- **Honest about signal quality** — paper trading and calibration tracking before live capital

freqpred is to prediction markets what [freqtrade](https://github.com/freqtrade/freqtrade) is to crypto trading.

---

## 2. Goals

- [x] Fetch active prediction markets from Kalshi and score them with an LLM signal pipeline
- [x] Implement a code-driven strategy plugin interface (`IPredictionStrategy`)
- [x] Track paper trades and measure real-world calibration over time
- [ ] Execute live trades on Kalshi with hard risk controls
- [x] Provide a web dashboard and Telegram/Discord alerts
- [ ] Run continuously on AWS as an always-on service
- [ ] Enrich signals and entry/exit decisions with cross-platform intelligence from Polymarket (price comparison, order-book depth, on-chain whale tracking)

## 3. Non-Goals (v1)

- **No backtesting engine** — LLM training data contamination makes historical backtests unreliable; paper trading is the validation approach
- **No non-US markets** — Kalshi is the only regulated trading platform in scope; Polymarket is geo-blocked for US users for trading, but its public APIs and the Polygon blockchain are used as read-only intelligence sources (price comparison, whale tracking — see §6b)
- **No portfolio optimization** — Kelly sizing per market is sufficient; no cross-market correlation modeling
- **No options/spreads** — Binary yes/no positions only in v1

---

## 4. Platform Scope

### v1: Kalshi (primary)
- CFTC-regulated, US-legal real-money prediction market exchange
- REST API with WebSocket for live data
- Supports market lookup, order placement, position management
- Full historical resolved market data available

### v2 (future): Interactive Brokers
- Event contracts available via IBKR API
- Adds broader market coverage without regulatory concerns
- Requires separate adapter implementation

### Intelligence sources (read-only, no trading)

#### Polymarket
- Crypto-based prediction market running on the Polygon blockchain
- US users cannot trade on Polymarket (geo-blocked), but public data APIs require no authentication for reads
- Used purely as a signal source: cross-platform implied probability comparison, order-book depth, and on-chain whale trade tracking
- Three data feeds: Gamma API (market metadata + prices), CLOB API (order book + recent trades), Polygon blockchain (on-chain event history)
- No freqpred capital is ever placed on Polymarket

### Architecture note
All market interactions go through an abstract `IMarketClient` interface. Kalshi is the first concrete implementation. Adding IBKR or any future platform requires only a new adapter — zero changes to strategy or signal logic.

---

## 5. Market Selection

Rather than fetching news for broad categories, freqpred focuses ingestion on the specific markets that registered strategies care about. This produces a tighter, higher-quality document store.

### Strategy-driven market selection

The `IPredictionStrategy` interface exposes an `is_market_interesting(market) → bool` method. The **Market Selector** (a component between the market watcher and the catalyst generator) queries all active markets from the DB and calls every registered strategy's `is_market_interesting()`. A market is included if *any* strategy returns `True`. Markets not selected by any strategy are excluded from catalyst generation and ingestion.

This means the ingestion pipeline adapts automatically as strategies are added, removed, or reconfigured — no separate category config list is needed.

### Default strategy focus (v1)

The bundled strategies cover:

1. **US Politics & Elections** — High news coverage, strong LLM training signal, structured event cycle (primaries, votes, appointments)
2. **Technology & Fintech** — Product launches, earnings beats, regulatory approvals, M&A deals

These categories are selected because:
- News signal is dense and temporally predictable
- LLM knowledge of the domain is deep
- Market liquidity on Kalshi is comparatively high

Other categories (macro/Fed, geopolitics, sports) are supported by the architecture but excluded from default strategy configurations in v1.

---

## 6. System Architecture

```mermaid
graph TD
    subgraph Ingestion[Ingestion]
        MW[Market Watcher]
        MS[Market Selector]
        CG[Catalyst Generator]
        IS[Ingestion Scheduler]
        RS[Realtime Scheduler]
        FB[FactBase Scheduler]
        DS[(Document Store)]
        FBD[(FactBase Phrase DB)]
    end

    subgraph Trading[Signal and Trading]
        SP[Signal Pipeline]
        SE[Strategy Engine]
        AS[Assessment]
        OM[Order Manager]
    end

    subgraph Metrics[Metrics]
        SQS[Source Quality Scheduler]
        SQ[(Source Quality Snapshots)]
    end

    subgraph Runtime[Runtime]
        PW[Position Watcher]
        PM[Position Monitor]
        L[(Ledger and DB)]
        API[Dashboard API]
        DA[Dashboard and Alerts]
    end

    IMC[IMarketClient - Kalshi]

    MW --> MS
    MS --> CG
    CG --> IS
    CG --> RS
    IS --> DS
    RS --> DS
    FB --> FBD
    FBD --> SP
    FBD --> AS
    MW --> SP
    DS --> SP
    SP --> SE
    SE --> AS
    SQ --> AS
    L --> AS
    AS --> OM
    OM -->|orders| IMC
    OM --> L
    SQS --> SQ
    IMC --> PW
    PW --> PM
    PM --> L
    L --> API
    API --> DA
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| **Market Watcher** | Polls Kalshi for active markets, upserts into DB |
| **Market Selector** | Reads active markets from DB; calls `strategy.is_market_interesting()` on each registered strategy; passes selected markets to Catalyst Generator |
| **Catalyst Generator** | LLM call (Haiku) per selected market: derives 3–5 specific search queries (catalysts) representing events that could materially shift probability. Stored as first-class DB entities. Re-runs daily, RAG-informed on subsequent passes. |
| **Position Watcher** | Streams live price updates via Kalshi WebSocket for markets with open positions |
| **Ingestion Scheduler** | Reads the latest active catalyst queries per market from DB; runs Tavily + NewsAPI + Guardian + Reddit + GDELT + TV Archive fetchers against those queries (every 30 min); upserts results into Document store |
| **Realtime Scheduler** | Polls cursor-based near-real-time sources on a faster cadence (default 5 min): TV chyrons via Internet Archive Third Eye API; Truth Social account feeds. Uses `fetcher_cursors` for dedup so frequent polling does not double-process. |
| **Signal Pipeline** | Retrieves news context via RAG, runs LLM analysis, returns probability estimate |
| **Strategy Engine** | Applies `IPredictionStrategy` plugins to signal output, decides trade/skip, and provides the base position-sizing target |
| **Assessment** | Builds source-quality and similar-market context, calls the judgment model when useful, and persists a sizing-only `SignalAssessment` before final position sizing |
| **Source Quality Scheduler** | Refreshes rolling `source_quality_scores` snapshots daily so assessment and dashboard views have fresh source-level calibration data |
| **Series History Scheduler** | Fetches all settled markets per active series from the Kalshi API and upserts per-option YES/NO counts plus an aggregate row into `series_option_history`; runs daily (07:00 ET) + at startup; per-series 6-hour skip guard prevents redundant fetches |
| **FactBase Scheduler** | For KXTRUMPSAY markets: uses Haiku (once per market lifetime) to extract quoted phrase and all plural/possessive variants; queries FactBase API for occurrence counts (`in_market_count`, `count_7d`, `count_30d`, `count_365d`) plus top Trump quotes; persists to `factbase_phrase_frequency`; populates in-process `FactbasePhraseCache` so `is_market_interesting()` gates KXTRUMPSAY markets until data is ready; refreshes every 5 min, re-fetches counts every 24 h |
| **IMarketClient** | Abstract interface over Kalshi (and future platforms); handles orders, positions, balance |
| **Order Manager** | Executes paper or live trades; enforces hard risk caps before any order; passes optional persisted assessment into strategy sizing |
| **Ledger** | Immutable trade log; records every signal, position, and resolution outcome |
| **Dashboard** | Web UI for monitoring signals, positions, decisions, markets, source quality, LLM audit/costs, strategy config, and system health; Telegram/Discord for push alerts |

---

## 6a. Position Watcher — WebSocket Price Tracking

Markets with open positions need tighter price monitoring than the 5-minute REST poll:
- A price move of ±5 cents on a held position can materially change the exit decision.
- Resolution events (market settled, determined) need to be caught quickly so P&L can be recorded.

The **Position Watcher** maintains a persistent Kalshi WebSocket connection and subscribes to the `ticker` channel for every market where freqpred holds at least one open position (`status = "open"`).

### WebSocket channels used

| Channel | Scope | Payload | Action |
|---|---|---|---|
| `ticker` | market-filtered | Real-time best bid/ask update | Update `MarketRow` price fields + `price_updated_at`; emit `price_moved` signal trigger if Δmid ≥ threshold |
| `market_lifecycle_v2` | global broadcast (filter not supported) | `determined` carries `settlement_value`; `settled` does not. | On `determined`: close positions at $1/$0; on `settled` with no cached result: REST fallback |
| `user_orders` | user-scoped | Order state change for an order this account placed (resting → executed/canceled/partial) | Look up the matching `PositionRow.exchange_order_id`, route through `OrderManager.apply_ws_event` (shared status mapping) |
| `fill` | user-scoped | Per-fill notification on one of this account's orders | Same handler as `user_orders` — re-apply exchange state to the matching row |

`user_orders` and `fill` are user-scoped (no `market_tickers` filter) and are re-subscribed on every reconnect alongside the market-scoped channels.

### Connection lifecycle

```
On startup / position opened:
  - Build subscription set: {market_id for position in open_positions}
  - Connect to wss://api.elections.kalshi.com/trade-api/ws/v2
  - Authenticate (same RSA-PSS headers as REST, passed in connect message)
  - Subscribe to ticker + market_lifecycle_v2 for each market in the set

While connected:
  - On ticker update: upsert price in DB; emit price_moved event if threshold crossed
  - On market_lifecycle_v2 → determined: settlement_value present ("1.0000"=YES, "0.0000"=NO) → resolve position, record P&L, unsubscribe
  - On market_lifecycle_v2 → settled: no settlement_value → REST fallback if determined was missed
  (channel is global broadcast; market_ticker filters not supported — filter in code)

On position closed / market resolved:
  - Remove market from subscription set

On disconnect:
  - Exponential backoff reconnect (1s → 2s → 4s → … → 60s max)
  - Re-subscribe to current open-position set on reconnect
```

### Relationship to REST polling

REST polling (Market Watcher, every 5 min) continues for **all** markets regardless of whether WebSocket is active. This provides a fallback: if the WebSocket drops and reconnect is in progress, the REST poll ensures prices don't go stale beyond the polling interval × 3 staleness threshold.

Markets **without** open positions are REST-only. The WebSocket subscription set is strictly scoped to open positions to minimise connection overhead.

### Implementation notes

- Lives in `freqpred/markets/watcher.py` alongside the REST polling loop, as an independent async task.
- Shares the same `AsyncSession` factory and `MarketRow` upsert logic as the REST watcher.
- Auth token for the WebSocket handshake uses the same RSA-PSS signing as REST (`KalshiClient._make_auth_headers`).
- Uses the `websockets` dependency for the Kalshi WebSocket client.
- In paper mode, the WebSocket is still useful for accurate price tracking even though no real orders are submitted.

### Kalshi ↔ DB reconciliation — split responsibilities

Reconciliation lives in two single-responsibility components with **disjoint row scopes**: they never both touch the same row.

| Component | Owns | Row scope | Mechanism | Triggers |
|---|---|---|---|---|
| `OrderManager.reconcile_pending_orders` | Order lifecycle — "did the order we placed fill?" | `status='pending'` AND `mode='live'` AND `exchange_order_id IS NOT NULL` | `get_order(exchange_order_id)` per row | Startup, WS reconnect, periodic (30s default), WS `user_orders`/`fill` event |
| `PositionWatcher._detect_external_drift` | External-trade drift — "did anyone touch our position outside freqpred?" | `status='open'` AND `mode='live'` | `get_positions()` net-position diff | Startup, WS reconnect |

Reconciliation always runs pending first (so rows reach a terminal state) before external drift detection surveys open-only rows.

**Pending-order reconciliation (`OrderManager`):** uses `SELECT … FOR UPDATE SKIP LOCKED` so concurrent calls (e.g. WS event landing while the periodic sweep runs) don't double-process the same row. Every code path that reads exchange order state — REST polling, WS events, place_order response — funnels through the same status-mapping helper:

| Kalshi status | Fills | DB outcome |
|---|---|---|
| `executed` | any | `status='open'`, `contracts = filled` |
| `resting` / `partial` | 0 fills | stay `pending` |
| `resting` / `partial` | partial | `status='open'`, `contracts = filled` (partial substate: `requested_contracts > contracts`) |
| `canceled` | 0 fills | `status='cancelled'` |
| `canceled` | with fills | `status='open'`, `contracts = filled` (partial substate) |

Rows still `pending` after status check past `StrategyConfig.pending_order_timeout_seconds` (default 900s) are cancelled via `cancel_order` and re-polled.

**External drift detection (`PositionWatcher`):** the operator may place manual trades on Kalshi outside freqpred. Kalshi returns **net** contracts per ticker from `get_positions()` — manually-added contracts are indistinguishable from freqpred's. The detector accepts this:

| DB (open live position) | Kalshi net | Action |
|---|---|---|
| `contracts = N` | `position = M`, M ≠ N | Update `PositionRow.contracts` to M and log |
| `contracts = N` | not present / 0 | Auto-close position in DB at current mid price; log warning |
| not present | `position = M` | Log info and skip — manual-only trade, no DB record to manage |

**At exit time:** the exit order is submitted for the Kalshi net position size (from the most recent reconciliation snapshot), not the DB `contracts` value. This ensures a manually-augmented position is fully closed.

**P&L note:** entry price is taken from the DB (freqpred's original entry). If the operator manually added contracts at a different price, the average entry will be slightly wrong. This is accepted — the DB is not a full order blotter, just a position tracker.

**Orphan order safety:** if `place_order` succeeds but the DB write fails inside `_submit_live`, the order is auto-cancelled on the exchange before the exception is re-raised — preventing a permanently orphaned resting order.

---

## 6b. Cross-Platform Intelligence — Polymarket

### Overview

Polymarket is the largest global prediction market by volume, operating on the Polygon blockchain. While US-based trading is blocked, the platform's market data is publicly accessible without authentication. freqpred uses three Polymarket data feeds as **read-only intelligence**:

| Feed | Source | Cadence | Use |
|---|---|---|---|
| Market prices | CLOB API (`https://clob.polymarket.com`) | 5 min | Assessment enrichment, entry/exit gates |
| Market metadata | Gamma API (`https://gamma-api.polymarket.com`) | Daily | Market matching |
| Trade history | CLOB API `/trades` endpoint (MVP); direct Polygon RPC deferred | 5 min | Whale tracking, smart-money alerts |

### Polymarket API overview

**Gamma API** — no auth required:
- `GET /markets?active=true&closed=false&limit=100` — paginated market list; returns `condition_id`, `question`, `endDate`, `outcomePrices` (e.g. `["0.65", "0.35"]`), `volume`, `liquidity`
- `GET /markets/{condition_id}` — single market detail

**CLOB API** — no auth required for reads:
- `GET /markets` — market list with YES/NO token IDs and mid prices
- `GET /order-book/{token_id}` — full bid/ask order book for one outcome token
- `GET /trades?market={condition_id}&limit=100&before={cursor}` — recent fill history with taker wallet address, outcome, price, size (USDC)

**Polygon blockchain (future enhancement)**:
- The CTF Exchange contract (`0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`) emits `OrderFilled` events on-chain
- More complete than the CLOB API — catches direct contract interactions that bypass the CLOB
- Queryable via Polygon RPC or The Graph subgraph; deferred to post-MVP

### Market matching algorithm

Mapping Kalshi markets to Polymarket markets is the core challenge (different question phrasing, different resolution criteria). The matching pipeline:

1. **Daily batch fetch**: pull all active Polymarket markets via Gamma API
2. **Embed Polymarket questions**: run through the existing local sentence-transformers embedder (`all-MiniLM-L6-v2`) — reuses `rag/embedder.py`
3. **Determine the candidate Kalshi market set** using `config.polymarket.llm_match_scope` (see below)
4. **Cosine similarity** between candidate Kalshi market questions and all Polymarket market questions:
   - Score ≥ `config.polymarket.match_auto_threshold` (default 0.85) → auto-confirm
   - Score in `[match_llm_threshold, match_auto_threshold)` (default 0.70–0.85) → LLM confirmation gate (see below)
   - Score < `match_llm_threshold` → skip
5. **Manual override**: `polymarket_market_links.match_method = 'manual'` is never overwritten by the pipeline
6. Confirmed matches stored in `polymarket_market_links`; re-run daily (new markets added, expired/closed markets deactivated)

Haiku LLM call for borderline matches is logged to `llm_queries` per the hard constraint.

#### LLM confirmation scope — `config.polymarket.llm_match_scope`

Controls which Kalshi markets get the Haiku confirmation step for borderline (0.70–0.85 similarity) candidates. Embedding-only auto-confirms (≥ 0.85) always run regardless of this setting.

| Value | Behaviour |
|---|---|
| `"never"` | Skip LLM confirmation entirely — only auto-confirmed matches are stored |
| `"always"` | Run Haiku confirmation for every borderline candidate |
| `"interesting"` | Confirm only markets currently selected by at least one active strategy via `is_market_interesting()` |
| `"categories"` | Confirm only markets whose `category` is in `config.polymarket.llm_match_categories` |

Default: `"interesting"` — LLM cost proportional to markets the system is already watching.

`config.polymarket.llm_match_categories` (list of strings, used when scope = `"categories"`) defaults to `[]`. An empty list with scope `"categories"` is treated as `"never"` and logs a startup warning.

### Whale identification and tracking

**Data source**: Polymarket CLOB API `/trades?market={condition_id}&limit=100&before={cursor}` — polled every 5 min per matched market in `realtime_scheduler.py` alongside price collection. Cursor-based dedup via `fetcher_cursors` (reuses existing mechanism). Covers the large majority of Polymarket volume.

**Trade qualification** — a trade is a whale if it meets EITHER threshold (hybrid USD floor + volume % ceiling self-normalizes across market sizes):
```
is_whale = (size_usd >= whale_alert_min_usd) OR (size_usd >= whale_alert_pct_volume × market.volume_24h_usd)
```
Only qualifying trades are persisted to `polymarket_whale_trades`.

**Toxic flow**: A qualifying trade's impact on available market depth. `pct_of_liquidity = size_usd / market.liquidity_usd` — a trade that exceeds 20–25% of available liquidity will materially move the price ("toxic"). Stored on `PolymarketWhaleTrade`; surfaced in the assessment prompt and dashboard.

**Wallet profiling** — `polymarket_whale_wallets` table:
- Upsert running totals on each qualifying trade: `total_volume_usd`, `trade_count`, `markets_active`
- Win/loss tracking: when a Polymarket market resolves (detected from `is_active` state in Gamma API daily refresh), batch-score all whale trades in that market — if the taker's outcome won, increment `win_count`
- Per-category win rates stored as JSONB (`{"politics": {"wins": 8, "losses": 3}, ...}`) — accuracy is category-specific (a sharp crypto trader may be unreliable on politics)
- A wallet earns **"known sharp"** status when `win_rate > 0.60 AND resolved_count >= 15`; **"known whale"** when `total_volume_usd >= 100_000 OR trade_count >= 20`

A known-sharp wallet buying against our signal is a much stronger exit/block trigger than an unknown first-time wallet making the same trade.

**Wallet cluster detection**: Full cluster analysis (graph of shared funding sources, coordinated entry timing) is out of scope for MVP. The `polymarket_whale_wallets` table accommodates a future `cluster_id` FK.

### Data models

```python
@dataclass
class PolymarketMarketLink:
    id: UUID
    kalshi_market_ticker: str        # FK → markets.ticker
    condition_id: str                # hex string, Polymarket primary key
    yes_token_id: str                # ERC-1155 token ID for YES outcome
    no_token_id: str                 # ERC-1155 token ID for NO outcome
    question_text: str               # Polymarket question (stored for reference)
    similarity_score: float          # cosine sim at match time
    match_method: str                # 'auto' | 'llm' | 'manual'
    is_active: bool                  # False when Polymarket market closes
    last_verified_at: datetime       # updated on each daily re-run
    created_at: datetime

@dataclass
class PolymarketPrice:
    id: int                          # bigserial (high-frequency writes)
    condition_id: str
    yes_price: float                 # mid price for YES outcome (0–1)
    spread: float                    # yes_ask - yes_bid
    liquidity_usd: float             # total open interest
    volume_24h_usd: float
    sampled_at: datetime

@dataclass
class PolymarketWhaleTrade:
    id: UUID
    condition_id: str
    wallet_address: str              # checksummed Polygon address
    outcome: str                     # 'YES' | 'NO'
    size_usd: float                  # position value (USDC) at execution
    pct_of_24h_volume: float         # size_usd / volume_24h_usd at poll time
    pct_of_liquidity: float          # size_usd / liquidity_usd — toxic flow severity
    price: float                     # contract price (0–1)
    clob_trade_id: str               # UNIQUE — prevents re-processing
    trade_ts: datetime
    created_at: datetime

@dataclass
class PolymarketWhaleWallet:
    wallet_address: str              # PK — checksummed Polygon address
    first_seen_at: datetime
    last_active_at: datetime
    total_volume_usd: float
    trade_count: int
    win_count: int                   # resolved markets where this wallet's outcome won
    loss_count: int
    win_rate: float | None           # None until >= 3 resolved trades
    markets_active: int              # distinct condition_ids ever traded
    category_stats: dict             # JSONB: {"politics": {"wins": 8, "losses": 3}, ...}
    created_at: datetime
```

DB tables: `polymarket_market_links`, `polymarket_prices` (bigserial), `polymarket_whale_trades`, `polymarket_whale_wallets`.

### Integration points

#### Assessment prompt enrichment (T79)

The signal pipeline intentionally excludes all market price data from the LLM prompt to prevent anchoring on market consensus — this applies equally to Polymarket. Polymarket prices are **not** injected into `signal/llm.py:build_prompt()`.

Polymarket context IS included in the **assessment prompt** in `metrics/assessment.py:assess_signal_context()`, which already uses market-context data and runs after the signal is formed. When a `PolymarketPrice` row < 30 min old exists, the assessor payload gains a context block:

```
CROSS-PLATFORM CONTEXT
Polymarket implied probability: 67% YES
Kalshi mid: 55% YES  |  Cross-platform divergence: +12pp
Polymarket 24h volume: $142,000  |  Spread: 2.1pp
Recent whale activity: Known sharp (68% win rate, 22 resolved markets) bought YES $18k
  4h ago — 12% of 24h volume, 8% of liquidity (toxic flow)
```

The assessment LLM (`judgment_model`) factors this into `trust_score` and `verdict`. The `signal_assessments.warnings` array can include `"polymarket_divergence_high"` or `"whale_opposing_known_sharp"`. `SignalAssessment` gains `polymarket_yes_price: float | None` (snapshotted at assessment time).

Bumps assessment prompt version (independent of signal prompt version).

#### Strategy hooks — `StrategyConfig` additions (T80)

All fields default to disabled. All gates are **fail-open**: if no Polymarket data exists for a market, the gate passes.

```python
# Price divergence gates
polymarket_signal_gate: bool = False
    # Block new entries when |signal.estimated_probability - polymarket_yes_price| > polymarket_max_divergence_entry
polymarket_max_divergence_entry: float = 0.10
    # Max cross-platform divergence (absolute, 0–1) allowed for entry gate
polymarket_exit_divergence_threshold: float = 0.0
    # If > 0: flag for exit review when Polymarket price moves this many pp against our position direction

# Whale alert thresholds — hybrid: qualifies as whale if meets EITHER floor OR pct
whale_alert_min_usd: float = 0.0
whale_alert_pct_volume: float = 0.0
    # Single trade qualifies as whale if size_usd >= whale_alert_min_usd OR >= whale_alert_pct_volume × volume_24h

# Whale exit review triggers (single opposing trade)
whale_exit_min_usd: float = 0.0
whale_exit_pct_volume: float = 0.0
    # Trigger exit review if opposing whale trade meets EITHER threshold; never auto-exits

# Whale entry block (aggregate opposing volume in window)
whale_entry_block: bool = False
whale_entry_block_min_usd: float = 10000.0
whale_entry_block_pct_volume: float = 0.05
    # Block entry if aggregate opposing whale volume in window meets EITHER threshold
whale_entry_block_lookback_hours: float = 4.0
```

#### PositionMonitor integration (T80)

`PositionMonitor.evaluate_exit()` gains a check after trailing-stop, before force-exit:
- If `polymarket_exit_divergence_threshold > 0` and the latest Polymarket price diverges against our position direction by ≥ threshold → log, alert, return `("polymarket_divergence_flag", current_price)`. This is a flag, not a hard exit — `should_exit()` is still called. The `polymarket_` prefix on `exit_reason` enables dashboard filtering.

#### Entry gate (T80)

`OrderManager.submit()` checks before `should_trade()`:
- **Price gate**: if `polymarket_signal_gate=True` and a fresh Polymarket price exists and `|signal.estimated_probability − polymarket_yes_price| > polymarket_max_divergence_entry` → skip entry (not logged as a risk reject; it's a gate)
- **Whale block**: if `whale_entry_block=True` and aggregate opposing whale volume in the lookback window meets EITHER threshold → skip entry with `log.info("whale_entry_block")`

### Arbitrage note

When Kalshi and Polymarket diverge above a configurable threshold, freqpred logs the opportunity for dashboard visibility. **freqpred does not execute cross-platform arbitrage** — US users cannot trade on Polymarket, and simultaneous two-platform execution is out of scope. Divergence is used as directional confirmation or an entry/exit gate only.

---

## 7. Core Data Models

### Market
```python
@dataclass
class Market:
    # --- Identity (never changes after creation) ---
    id: str                          # Kalshi market ID
    platform: str                    # "kalshi"
    question: str                    # "Will X happen by Y?"
    category: str                    # Kalshi API string (e.g. "Elections", "Sports", "World")
    close_time: datetime             # when market resolves

    # --- Price snapshot (changes frequently) ---
    yes_bid: float                   # current best bid for YES (0.0-1.0)
    yes_ask: float                   # current best ask for YES (0.0-1.0)
    mid_price: float                 # (bid + ask) / 2
    volume_24h: float                # 24-hour volume (liquidity proxy)
    volume_total: float              # total lifetime volume (from Kalshi volume_fp)
    open_interest: float
    series_ticker: str | None        # Kalshi series identifier (e.g. "KXPRES")

    # --- Cache control ---
    last_fetched_at: datetime        # last time we polled Kalshi for this market
    price_updated_at: datetime       # last time price data actually changed
    metadata_fetched_at: datetime    # last time we refreshed metadata (question, close_time, etc.)

    # --- Signal linkage ---
    current_signal_id: str | None    # FK → latest Signal for this market

    metadata: dict                   # raw platform data
```

**Cache refresh rules:**
- **Price data** (`yes_bid`, `yes_ask`, `mid_price`, `volume_24h`): refresh every polling interval (default: 5 minutes). A market watcher loop updates these fields and sets `last_fetched_at`.
- **Metadata** (question, `close_time`, category): refresh once on creation, then daily. These rarely change; `metadata_fetched_at` tracks staleness.
- **`price_updated_at`** is set only when price actually changes, not on every poll. This lets us detect when a market has gone stale (no price movement in 24h+ = potentially illiquid, flag it).
- A market is considered **stale** and skipped for signal analysis if `last_fetched_at` is older than the configured polling interval × 3.
- **Markets with open positions** bypass the polling interval and receive real-time price updates via the Kalshi WebSocket `ticker` channel (see §6a below). `last_fetched_at` and `price_updated_at` are still updated on each WebSocket tick so the DB stays current.

### Signal

**Signals are immutable and append-only.** Every re-evaluation of a market creates a new Signal record — existing signals are never updated. `Market.current_signal_id` is updated to point to the latest. This preserves the full history of how estimates evolved as evidence accumulated, which is valuable for calibration analysis (e.g., do early signals or late signals predict resolution better?).

```python
@dataclass
class Signal:
    id: str
    market_id: str                   # FK → Market

    # --- Estimate ---
    estimated_probability: float     # LLM's estimate (0.0-1.0)
    confidence: float                # LLM self-reported confidence (0.0-1.0)
    edge: float                      # direction-adjusted edge at signal time:
                                     #   YES: estimated_probability - market.mid_price
                                     #   NO:  market.mid_price - estimated_probability
                                     # (positive = we believe the contract is underpriced)
    market_mid_at_signal: float      # snapshot of market price when signal was created
    direction: str                   # "YES" | "NO" | "SKIP"

    # --- Context ---
    reasoning: str                   # LLM explanation (logged, not traded on)
    sources: list[str]               # URLs used in RAG context
    social_sentiment_summary: str | None  # pre-summarized social signal (nullable)
    retrieval_hash: str              # hash of retrieved Document IDs — same hash = no new evidence in store

    # --- Provenance ---
    model_used: str                  # e.g. "claude-sonnet-4-6"
    prompt_version: str              # e.g. "politics-v2" — for tracking prompt changes
    trigger: str                     # "scheduled" | "price_moved" | "new_evidence" | "manual"
    created_at: datetime
    raw_context: str                 # full retrieved context (for debugging)
```

**Signal refresh triggers** (any of these causes a new Signal to be created):
1. **Scheduled** — the signal loop runs every `signal.interval_seconds` (default 30 min) and fires an LLM call when **any** of the following is true: the retrieval hash has changed since the last scheduled signal (new docs), FactBase data was refreshed since the last scheduled signal, or `signal.max_scheduled_interval_hours` (default 24h) have elapsed (temporal reasoning). All three must be false to skip the LLM call.
2. **Price moved** — when the LLM call is skipped but the market mid has shifted by more than the price-move threshold (default 5¢) since the last signal, a clone of the current signal is written at the new price without an LLM call, keeping edge calculations current.
3. **Manual** — operator triggers re-analysis via CLI (`freqpred signal analyze`) or the dashboard Analyze button; bypasses all dedup and cooldown checks (`force=True`).

**What does NOT trigger a scheduled LLM call:** a 30-minute poll where the retrieval hash is unchanged, FactBase has not been refreshed, and the last scheduled LLM call is less than `signal.max_scheduled_interval_hours` old. The loop still runs; it just skips the LLM and optionally creates a price clone if the market mid moved.

### Position
```python
@dataclass
class Position:
    id: str
    market_id: str                   # FK → Market
    signal_id: str                   # FK → Signal that triggered this position

    # --- Strategy attribution ---
    strategy_name: str               # e.g. "PoliticsEdgeStrategy"
    strategy_version: str            # e.g. "1.2.0" — for P&L attribution across versions

    # --- Signal snapshot at time of trade ---
    # These are snapshotted from the Signal at entry time because the signal
    # may be superseded by a newer one before the market resolves.
    signal_confidence: float         # confidence score that drove the trade decision
    signal_edge: float               # edge at time of trade
    signal_estimated_prob: float     # LLM probability estimate at time of trade

    # --- Order details ---
    direction: str                   # "YES" | "NO"
    contracts: int
    entry_price: float
    entry_time: datetime
    mode: str                        # "paper" | "live"

    # --- Lifecycle ---
    status: str                      # "pending" | "open" | "closed" | "cancelled"
    # Live-mode lifecycle: pending → open → closed | cancelled
    # Partial-fill substate: status='open' AND requested_contracts > contracts
    # Paper-mode lifecycle: positions enter directly as 'open' (no pending phase).
    # pending:   order submitted, awaiting fill confirmation from Kalshi
    # open:      position filled and active (fully or partially)
    # closed:    market resolved or manually exited
    # cancelled: order submitted but cancelled before any fill

    # --- Exchange-confirmed order state (live mode only; NULL for paper / legacy rows) ---
    exchange_order_id: str | None
    requested_contracts: int | None      # original request size; > contracts during partial fills
    exchange_order_status: str | None    # raw Kalshi status (executed/resting/partial/canceled/...)
    last_exchange_sync_at: datetime | None

    # --- Exit-side order state (live mode only; NULL/0 for paper or pre-T76 rows) ---
    exit_order_id: str | None            # Kalshi order ID of the sell order
    exit_fee_usd: float                  # cumulative fees paid across all exit orders (default 0)
    exit_requested_contracts: int | None # contracts the latest exit order asked to fill
    exit_filled_contracts: int | None    # cumulative contracts closed via exit orders so far
    realized_pnl_accumulator: float      # Σ (fill_price_i − entry_price) × contracts_i; used to
                                         # derive weighted-avg exit_price on final close (default 0)

    # --- Filled after resolution ---
    exit_price: float | None             # weighted-avg exit price (set on full close)
    exit_time: datetime | None
    resolution: int | None           # 1 = YES won, 0 = NO won
    pnl: float | None                    # net P&L after all entry + exit fees
    pnl_pct: float | None
```

**Mid-exit substate:** a live position is *mid-exit* when `exit_requested_contracts > exit_filled_contracts` and `status = 'open'`. This means an IOC sell partially filled; the residual contracts remain open and the position monitor will re-attempt the exit on the next tick if the original exit-trigger condition still holds.

**On snapshotting signal fields into Position:** The signal that triggered a trade may be superseded before the market resolves — a `price_moved` trigger could create a new Signal with a different estimate. Snapshotting `confidence`, `edge`, and `estimated_prob` at entry time means the Position record is a self-contained record of *why* the trade was placed, independent of subsequent re-evaluations. This is essential for honest P&L attribution: did the trades placed at high confidence actually outperform low confidence trades?

### StrategyConfig
```python
@dataclass
class StrategyConfig:
    name: str
    min_edge: float                  # minimum edge to trade (e.g. 0.15)
    min_confidence: float            # LLM confidence threshold (e.g. 0.70)
    max_exposure_per_market: float   # % of bankroll (e.g. 0.05)
    kelly_fraction: float            # fractional Kelly multiplier (e.g. 0.25)
    categories: list[str]            # Kalshi category strings to trade (e.g. ["Elections", "Sports"])
    min_volume_24h: float            # liquidity filter
    max_days_to_close: int           # don't trade markets closing too soon/late
    min_days_to_close: int

    # --- Exit management (freqtrade-style) ---
    stoploss: float = -0.20
    # Trigger exit when position loses this fraction from entry price.
    # e.g. -0.20 = exit if unrealized loss exceeds 20%.
    # Enforced by the position monitor on every price poll — strategy cannot override.
    # All stoploss/trailing/ROI thresholds operate on the *effective contract price*:
    #   YES positions: effective_price = market.mid_price
    #   NO  positions: effective_price = 1.0 - market.mid_price
    # The position monitor converts automatically — strategy configs use the same
    # threshold values regardless of direction.

    trailing_stop: bool = False
    # If True, stoploss trails from the best effective contract price achieved since entry
    # (i.e. the stop floor rises as the position profits, locking in gains).
    # Peak tracking uses the same direction-corrected effective price as stoploss.

    trailing_stop_positive: float | None = None
    # Once unrealized P&L crosses this threshold (e.g. 0.10 = 10% profit),
    # switch to a tighter trailing stop equal to trailing_stop_positive_offset
    # below the peak effective price. Encourages letting winners run while protecting profit.

    trailing_stop_positive_offset: float = 0.02
    # Tight trail applied once trailing_stop_positive is crossed.
    # e.g. 0.02 = trail 2% below the peak effective price once in profit.

    # --- Price range filter ---
    min_mid_price: float | None = 0.05
    max_mid_price: float | None = 0.95
    # Skip markets whose current mid_price is outside this range.
    # Markets priced below 0.05 or above 0.95 are effectively decided by the
    # market — the LLM has no edge and generating signals on them produces noise.
    # Applied in is_market_interesting() so it gates both ingestion and signal
    # generation. Set either bound to None to disable that side of the filter.

    # --- Stoploss re-entry guards ---
    block_reentry_after_stoploss: bool = False
    # If True, permanently block re-entry into any market that has ever had a
    # stoploss or trailing_stop exit. Takes precedence over stoploss_cooldown_hours.

    stoploss_cooldown_hours: float = 4.0
    # Block re-entry into a market for this many hours after a stoploss or
    # trailing_stop exit. Set to 0.0 to disable. Ignored if
    # block_reentry_after_stoploss is True.

    # --- Pre-signal risk gate ---
    pre_signal_risk_gate: bool = True
    # Skip LLM analysis for new-entry markets where risk would block the resulting
    # trade (global capacity caps reached, spread too wide, stoploss re-entry blocked).
    # Set to False to always generate fresh signals — useful when signals are needed
    # for calibration or analytics even when trading is constrained.
    # Has no effect in signal-only mode (order manager is not active).

    # --- Assessment-driven sizing controls ---
    assessment_scale_min: float = 0.80
    assessment_scale_max: float = 1.20
    similar_market_min_signals: int = 10
    similar_market_min_trades: int = 5
    # T57 maps a trust_score from the judgment model into this multiplier range.
    # Similar-market history is only considered available once at least one of
    # these minimums is met for the matched market family.

    # --- Live-mode pending-order timeout ---
    pending_order_timeout_seconds: float = 900.0
    # Live mode only. After this many seconds in 'pending', reconcile sweeps
    # call cancel_order on the exchange order so we don't sit forever on a
    # resting order that never fills. T47 added limit_order_timeout_hours for
    # paper-mode resting entries; this knob governs live exchange-side timeouts.
```

### Document (RAG Store)

Every fetched news article and social post is stored here. Documents are the persistent foundation of the RAG system — fetched once, reused across many market analyses. A document fetched while analyzing one market may be highly relevant to another market in the same category days later.

```python
@dataclass
class Document:
    id: str                          # UUID, generated on insert

    # --- Identity & deduplication ---
    source_url: str                  # canonical URL (unique — prevents duplicate storage)
    content_hash: str                # hash of cleaned content body — detect if article updated

    # --- Content ---
    title: str
    body: str                        # cleaned full text (HTML stripped)
    summary: str | None              # LLM-generated summary (populated lazily on first use)
    source_type: str                 # "news" | "reddit" | "social" | "twitter" | "kalshi_comment" | "manifold"
                                     # "social" is used for Truth Social posts
    source_name: str                 # e.g. "Reuters", "r/politics", "Kalshi"

    # --- Classification ---
    category: str                    # "politics" | "technology" | "fintech" | ...
    tags: list[str]                  # extracted keywords/entities for coarse filtering

    # --- Temporal ---
    published_at: datetime           # when the article/post was published (source timestamp)
    fetched_at: datetime             # when we first stored it

    # --- Vector search ---
    embedding: list[float]           # dense vector (pgvector) — generated on insert
    embedding_model: str             # e.g. "all-MiniLM-L6-v2" — track model version for re-embedding
```

**Deduplication:** Documents are inserted with `ON CONFLICT (source_url) DO UPDATE` — if a URL is fetched again, we update `content_hash` and `fetched_at` only if the content changed. The embedding is regenerated only when content changes.

**Embedding model:** `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) — local CPU-based embeddings, no API key required. Stored via the **pgvector** extension on RDS Postgres. No separate vector database needed. Voyage AI (`voyage-3`, 1024-dim) is a possible future enhancement for higher-quality retrieval.

**Full-text search index:** A GIN index on `to_tsvector('english', title || ' ' || body)` supports BM25 keyword scoring via `ts_rank`. Used in hybrid retrieval alongside cosine similarity.

**Retrieval approach — slot-budgeted hybrid:**

The retriever (`freqpred/rag/retriever.py`) combines two complementary selection strategies to maximise both relevance and thesis coverage.

**When no active catalyst queries exist** (e.g. first analysis before catalyst generation has run): embed the market question, score all linked candidates with `0.7 × norm(cosine_sim) + 0.3 × norm(BM25)`, return the top `top_k` by blended score. No slot budget applies.

**When catalyst queries exist:** start from the full market-question ranked list and apply a slot budget:

1. **Market-question core (≥ `top_k // 2` slots guaranteed):** Take the top `top_k // 2` by blended score. These slots cannot be displaced by catalyst docs.

2. **Catalyst-query supplemental (up to `top_k // 2` slots):** For each active `CatalystQuery` text, embed it in parallel via `asyncio.gather` and find the highest-cosine-similarity candidate not already in the core set. De-duplicate across queries (a doc that matches multiple catalyst queries is kept once at the highest similarity). Rank-select the top `top_k // 2` supplemental candidates by cosine similarity.

3. **Back-fill:** Any unused supplemental slots (fewer catalyst docs than slots) are filled from the next-best market-question-ranked docs, so the total always reaches `top_k` when enough candidates exist.

This design prevents a single marginal document swap from collapsing an otherwise good signal: the core set anchors the LLM to the market question, while catalyst supplements guarantee that evidence surfaced by hypothesis-driven queries is always represented without crowding out core context.

### CatalystRun + CatalystQuery

Each time the Catalyst Generator runs for a market it creates one `CatalystRun` (the generation event) and N `CatalystQuery` rows (the actual search strings). The ingestion scheduler always reads from the latest active run per market.

```python
@dataclass
class CatalystRun:
    id: str                      # UUID
    market_id: str               # FK → Market
    generation: int              # monotonically increasing per market (1, 2, 3 ...)
    llm_query_id: int | None     # FK → LLMQuery — audit trail for the catalyst LLM call (nullable)
    is_active: bool              # False when market closed or no strategy is interested
    created_at: datetime


@dataclass
class CatalystQuery:
    id: str                      # UUID
    run_id: str                  # FK → CatalystRun
    query_text: str              # natural-language web search string, e.g. "February CPI release 2026"
    tv_query: str | None         # Solr/Lucene boolean query for TV archive search, e.g. 'trump AND ("communist" OR "communism")'; None if not applicable
    created_at: datetime
```

**Lifecycle rules:**
- A `CatalystRun` is created when a market is first selected (generation=1) and then daily (generation increments).
- On each new run, the previous run's `is_active` flag is left as-is; only the latest run is used for scheduling.
- `CatalystRun.is_active` is set to `False` when: (a) the market's `close_time` has passed, or (b) all registered strategies return `False` from `is_market_interesting()` for that market AND the market has no open positions. Markets with open positions are protected from deactivation so the ingestion pipeline continues supplying fresh context for exit decisions.
- Ingestion scheduler query: `SELECT cq.query_text, cq.tv_query FROM catalyst_queries cq JOIN catalyst_runs cr ON cr.id = cq.run_id WHERE cr.is_active = TRUE AND cr.id IN (SELECT MAX(id)... per market)`.

**Catalyst generation context (LLM prompt inputs):**
- **Generation 1:** market question + market metadata (close_time, category, description from Kalshi)
- **Generation 2+:** same as above, plus the top-K documents most recently retrieved for this market's existing catalyst queries (RAG pull). This lets the LLM refine or add catalysts based on what has actually been appearing in the news.

**Catalyst generation model:** use the configured `cheap_model` lane (default: Claude Haiku). This is a reasoning task, not primary signal analysis. Logged to `llm_queries` with `query_type="catalyst_generation"`.

**Dual-format query generation:** The catalyst generator prompt asks Haiku to produce both a `query_text` (natural-language web search string for Tavily/NewsAPI/GDELT/Reddit) and a `tv_query` (Solr/Lucene boolean syntax for the Internet Archive TV search). The LLM response is a JSON array of objects with both fields. `tv_query` may be `null` for catalysts where TV transcripts are not a useful signal. The TV query uses AND/OR/phrase syntax to precisely target what needs to be *said* on air, not just discussed in text — particularly valuable for word-mention markets and markets about public statements by named individuals.

### DocumentMarketLink (join table)

Links documents to markets at two distinct points in time:

1. **Ingestion time** (`signal_id=None`) — written by the ingestion scheduler immediately after a document is upserted. Records which market's catalyst query caused this document to be fetched. This scopes retrieval: the retriever only searches documents pre-linked to the target market, not the entire category.

2. **Signal time** (`signal_id=<uuid>`) — written by the signal pipeline after retrieval. Records which documents were actually used in a specific signal analysis, with the blended relevance score.

This design solves a real failure mode: without ingestion-time links, the retriever searches all documents in a category and cosine similarity may miss documents with exact keyword matches (e.g. a document titled "Epic Fury" being retrieved for a market asking whether Trump says "Epic Fury"). By restricting retrieval to documents fetched for that specific market, the search corpus is both smaller and more relevant.

```python
@dataclass
class DocumentMarketLink:
    document_id: str                 # FK → Document
    market_id: str                   # FK → Market
    signal_id: str | None            # FK → Signal (None at ingestion time)
    relevance_score: float           # blended hybrid score at signal time; 0.0 at ingestion time
    linked_at: datetime
```

### LLMQuery (Audit Log)

Every call to any LLM is recorded here — both the primary reasoning model and the cheap pre-summarizer pass. This serves two purposes: cost tracking and full auditability of every decision the system made.

```python
@dataclass
class LLMQuery:
    id: int                          # auto-increment PK

    # --- When & why ---
    timestamp: datetime              # when the query was made
    strategy: str                    # which strategy triggered it (or "system" for non-strategy calls)
    query_type: str                  # see query types below
    market_id: str | None            # market being analyzed (null for non-market queries)
    signal_id: str | None            # FK → Signal if this query produced one

    # --- Full request/response (immutable audit record) ---
    model_used: str                  # e.g. "claude-sonnet-4-6"
    prompt_version: str              # versioned prompt template identifier
    prompt: str                      # full prompt sent to LLM
    response: str                    # full raw LLM response

    # --- Cost ---
    tokens_input: int                # input (prompt) token count
    tokens_output: int               # output (completion) token count
    tokens_total: int                # tokens_input + tokens_output
    cost_usd: float                  # dollar cost of this query

    # --- Extracted outputs ---
    confidence_extracted: float | None   # confidence score parsed from response
    decision_extracted: str | None       # "BUY" | "SELL" | "SKIP" parsed from response
    latency_ms: int                      # wall-clock response time in milliseconds

    # --- Error handling ---
    success: bool                    # False if LLM call failed or response was unparseable
    error_message: str | None        # populated if success=False
```

**Query types:**

| `query_type` | Description | Model tier |
|---|---|---|
| `market_analysis` | Primary signal generation for a market | Primary (Sonnet) |
| `catalyst_generation` | Generate 3–5 targeted search queries for a market | Cheap (Haiku) |
| `social_summarization` | Pre-summarizer pass on raw Reddit/Twitter posts | Cheap (Haiku) |
| `movement_prediction` | Optional: predict short-term price movement | Primary (Sonnet) |
| `daily_digest` | Generate natural language daily summary for alerts | Cheap (Haiku) |

**Cost tracking views (to be implemented as DB views or dashboard queries):**

```sql
-- Daily LLM spend
SELECT DATE(timestamp), SUM(cost_usd), COUNT(*) FROM llm_queries GROUP BY DATE(timestamp);

-- Cost by query type
SELECT query_type, SUM(cost_usd), AVG(cost_usd), COUNT(*) FROM llm_queries GROUP BY query_type;

-- Cost by strategy
SELECT strategy, SUM(cost_usd) FROM llm_queries GROUP BY strategy;

-- Most expensive markets (to identify if certain categories cost more to analyze)
SELECT market_id, SUM(cost_usd) FROM llm_queries WHERE market_id IS NOT NULL
GROUP BY market_id ORDER BY SUM(cost_usd) DESC LIMIT 20;
```

A **cost budget circuit breaker** enforces a configurable daily LLM spend cap (default: $10/day). If crossed, the signal pipeline halts and alerts via Telegram before incurring further cost.

---

### SeriesOptionHistory

Caches YES/NO settlement counts per `(series_ticker, option_code)` pair so the signal pipeline can inject historical base rates into the LLM prompt without hitting the Kalshi API on the critical path.

```python
@dataclass
class SeriesOptionHistory:
    series_ticker: str         # e.g. "KXTRUMPSAY"
    option_code: str           # last dash-segment of ticker, or "__series__" for the aggregate row
    option_label: str          # yes_sub_title for option rows; series_ticker for the aggregate row
    yes_count: int             # settled YES count
    no_count: int              # settled NO count
    last_fetched_at: datetime  # when this row was last refreshed from the Kalshi API
    created_at: datetime
```

**Key rules:**
- Primary key is `(series_ticker, option_code)` — one row per option, plus one `option_code = "__series__"` aggregate row per series that accumulates counts across all options and all weeks.
- Populated by `freqpred/metrics/series_history.py:refresh_series_history()` via `GET /markets?status=settled&series_ticker=X`.
- The signal pipeline reads this table immediately before building the LLM prompt (`get_series_history_for_market()`). When a matching row exists, a `=== HISTORICAL BASE RATE ===` block is injected between `=== MARKET CONTEXT ===` and `=== EVIDENCE ===`.
- `MIN_SAMPLE = 3`: option rows with `yes_count + no_count < 3` are shown with a "small sample, treat as weak signal" note; the series aggregate is always shown when available.
- **Type A series** (e.g. `KXTRUMPSAY`, `KXTRUMPACT`): option code repeats across weekly events, so per-option rates are meaningful.
- **Type B series** (e.g. `KXTRUMPPHOTO`): option code encodes the event date — each code is unique, so per-option rows have `n=1`. The series aggregate still applies; the option block is omitted from the prompt when below `MIN_SAMPLE`.

---

## 8. Strategy Plugin Interface

Strategies are Python classes that implement `IPredictionStrategy`. The design mirrors freqtrade's `IStrategy` — entry signals, exit signals, stoploss, and trailing stops are all strategy-owned. The framework enforces hard caps on top; strategy logic defines the alpha.

| freqtrade concept | freqpred equivalent |
|---|---|
| `populate_entry_trend()` | `should_trade(signal, market) -> bool` |
| `populate_exit_trend()` | `should_exit(position, signal, market) -> bool` |
| `stoploss = -0.10` | `config.stoploss = -0.20` |
| `trailing_stop = True` | `config.trailing_stop = True` |
| `custom_exit()` | `custom_exit(position, signal, market) -> str \| None` |
| post-trade hook | `on_resolution(position)` |

```python
from abc import ABC, abstractmethod

class IPredictionStrategy(ABC):
    """
    Base class for all freqpred trading strategies.

    Users subclass this, implement the required methods, and point
    freqpred at their strategy file via config.

    Example:
        class MyPoliticsStrategy(IPredictionStrategy):
            config = StrategyConfig(
                name="my_politics_v1",
                min_edge=0.18,
                min_confidence=0.72,
                kelly_fraction=0.25,
                categories=["Elections"],
                stoploss=-0.20,
                trailing_stop=True,
                ...
            )

            def should_trade(self, signal: Signal, market: Market) -> bool:
                return signal.edge >= self.config.min_edge

            # position_size uses the default confidence-blended Kelly from IPredictionStrategy.
            # Override only if custom sizing logic is needed.

            def should_exit(self, position: Position, signal: Signal, market: Market) -> bool:
                # Signal-driven exit: LLM now disagrees with the position direction
                return (
                    signal.direction != position.direction
                    and signal.confidence >= self.config.min_confidence
                )
    """

    config: StrategyConfig

    # -------------------------------------------------------------------------
    # Entry interface (required)
    # -------------------------------------------------------------------------

    @abstractmethod
    def should_trade(self, signal: Signal, market: Market) -> bool:
        """Return True if this signal warrants opening a new position."""
        ...

    def position_size(
        self,
        signal: Signal,
        bankroll: float,
        existing_market_exposure: float = 0.0,
        assessment: SignalAssessment | None = None,
    ) -> float:
        """Return the incremental dollar exposure to add (before risk capping).

        order_manager passes current open exposure for the market and, when
        available, a T57 SignalAssessment produced after should_trade().
        The default implementation computes the ideal total Kelly exposure,
        applies assessment.size_multiplier to that target, then subtracts
        existing_market_exposure and floors at 0.0. Legacy overrides that only
        accept 2 or 3 arguments continue to work unchanged.
        """
        ...

    # -------------------------------------------------------------------------
    # Exit interface (optional overrides — defaults handle the common cases)
    # -------------------------------------------------------------------------

    def should_exit(self, position: Position, signal: Signal, market: Market) -> bool:
        """
        Signal-driven exit. Called after every LLM re-analysis of a market
        with an open position (triggered by price move or scheduled refresh).

        Return True to trigger an exit at current market price.

        Default: exit if the new signal direction is opposite to the position
        direction AND the signal confidence meets the strategy threshold.
        Override for custom logic (e.g. exit when estimated probability
        drops below a floor regardless of direction flip).
        """
        return (
            signal.direction not in ("SKIP", position.direction)
            and signal.confidence >= self.config.min_confidence
        )

    def custom_exit(
        self,
        position: Position,
        signal: Signal,
        market: Market,
    ) -> str | None:
        """
        Custom exit hook. Called by the position monitor on every price poll,
        after stoploss/ROI checks but before should_exit().

        Return a non-None string (the exit reason tag) to force an immediate
        exit. Return None to let normal exit logic proceed.

        Use this for market-specific conditions that don't fit the standard
        stoploss/ROI/signal framework — e.g. exit all positions 48h before
        a market closes regardless of P&L.
        """
        return None

    # -------------------------------------------------------------------------
    # Market selection (optional override)
    # -------------------------------------------------------------------------

    def is_market_interesting(self, market: Market) -> bool:
        """
        Return True if this strategy wants the ingestion pipeline to monitor
        this market (i.e. generate catalysts and fetch targeted news for it).

        The Market Selector calls this on all registered strategies. A market
        is selected for catalyst generation if *any* strategy returns True.
        When all strategies return False for a market, its catalysts are
        marked inactive and ingestion stops.

        Default implementation applies StrategyConfig filters (category,
        volume, days-to-close, and mid_price range). Override for custom
        market selection logic.

        Note: markets with open positions are NEVER deactivated by the Market
        Selector or excluded from signal analysis, regardless of whether
        is_market_interesting() returns False. The price filter and other
        filters only govern entry into new markets.
        """
        if self.config.min_mid_price is not None and market.mid_price < self.config.min_mid_price:
            return False
        if self.config.max_mid_price is not None and market.mid_price > self.config.max_mid_price:
            return False
        days_to_close = (market.close_time - datetime.utcnow()).days
        return (
            market.category in self.config.categories
            and market.volume_24h >= self.config.min_volume_24h
            and self.config.min_days_to_close
                <= days_to_close
                <= self.config.max_days_to_close
        )

    def filter_markets(self, markets: list[Market]) -> list[Market]:
        """Pre-filter markets before signal analysis."""
        return [m for m in markets if self.is_market_interesting(m)]

    # -------------------------------------------------------------------------
    # Lifecycle hooks (optional)
    # -------------------------------------------------------------------------

    def on_resolution(self, position: Position) -> None:
        """
        Called when a market resolves (position closed by resolution, not exit).
        Use for logging, alerting, or adaptive strategy logic.
        """
        pass
```

### Exit Priority Order

The position monitor evaluates exit conditions in this order on every price poll:

1. **Hard stoploss** (`config.stoploss`) — framework-enforced, cannot be overridden
2. **Trailing stoploss** (`config.trailing_stop`) — trails from the best price since entry
3. **Force exit** (`strategy.force_exit()`) — strategy's own initiative, signal-independent, every tick
4. **Custom exit** (`strategy.custom_exit()`) — strategy-defined, requires fresh signal
5. **Signal exit** (`strategy.should_exit()`) — called only after LLM re-analysis (on price-triggered signal refreshes, not every poll)
6. **Market resolution** — market closes, position settled at $1.00 or $0.00

If none of these conditions fire, the position is held.

**Partial-fill detection and residual sizing:** before evaluating any exit condition, the position's `contracts` field already reflects only the residual (unfilled) contracts. When an IOC exit partially fills, `ledger.partial_close_position` decrements `contracts` to the residual count and leaves `status = 'open'`. On the next tick, stoploss/trailing/ROI/force-exit thresholds are evaluated against the residual size — not the original position size. If the original exit-trigger condition still holds, the monitor submits a new IOC sell for the residual; if it no longer holds (e.g. price recovered above stoploss), the residual remains open and continues to be managed normally.

### Exit Reason Tagging

Every closed position records an `exit_reason` string for analysis:

| Exit reason | Source |
|---|---|
| `"stoploss"` | Hard stoploss hit |
| `"trailing_stop"` | Trailing stoploss hit |
| `"custom_exit:<tag>"` | `custom_exit()` returned a tag |
| `"signal"` | `should_exit()` returned True |
| `"market_resolved"` | Market paid out at resolution |
| `"force_exit:<tag>"` | Operator-initiated manual close (e.g. `"force_exit:manual"` via dashboard or Telegram) |

### Bundled Strategies

| Strategy | Description |
|---|---|
| `PoliticsEdgeStrategy` | US politics markets, min edge 0.18, conservative Kelly 0.25x |
| `TechNewsStrategy` | Technology/fintech markets, skewed toward shorter-dated markets |
| `ConservativeDefault` | High-confidence only (0.80+), tiny sizing — good starting point |

---

## 9. LLM Signal Pipeline

The pipeline has two distinct phases that run on different schedules: **ingestion** (continuous, cheap) and **analysis** (triggered, expensive).

### Phase 1: Document Ingestion (continuous background job)

Runs on a schedule independent of signal generation. Fetches new content and stores it — the only LLM calls here are the cheap catalyst generator and social pre-summarizer.

Ingestion is **catalyst-driven**, not category-driven. Instead of broad keyword searches per category, the scheduler fetches news and social content targeted at the specific events and hypotheses that matter for each selected market.

```mermaid
flowchart TD
    DB([Market Watcher upserts active markets into DB])
    DB --> MS[Market Selector]
    MS --> CG[Catalyst Generator - LLM Haiku]
    CG --> IS[Ingestion Scheduler - every 30 min]
    CG --> RS[Realtime Scheduler - every 5 min]
    IS --> DS[Dedup and Store]
    RS --> DS
    DS --> SS[Social Pre-summarizer - Reddit only]
```

### Phase 2: Signal Analysis (triggered)

Runs when a signal refresh trigger fires (scheduled, price moved, new evidence, manual). This is where the expensive LLM call happens.

```mermaid
flowchart TD
    T([Signal trigger fires for a market])
    T --> HOP{Has open position?}
    HOP -->|yes| CQ
    HOP -->|no| RG{Pre-signal risk gate enabled?}
    RG -->|no - gate disabled| CQ
    RG -->|yes| GC{Global capacity full or per-market blocked?}
    GC -->|spread too wide or stoploss block or caps full| SKIP2([Skip - no LLM call])
    GC -->|pass| CQ[Load active catalyst queries for market]
    CQ --> HS[Hybrid Search - market question core set]
    HS --> CS[Catalyst supplemental - top-1 per query not in core]
    CS --> TRG{Trigger type}
    TRG -->|non-scheduled| HC{Hash changed?}
    HC -->|no change| SKIP([Skip - price clone only])
    HC -->|changed| BR
    TRG -->|scheduled| SG{New docs, FactBase updated, or min interval elapsed?}
    SG -->|none of the above| SKIP
    SG -->|yes| BR[Load series base-rate history]
    BR --> LLM[LLM Analysis - Claude Sonnet]
    LLM --> SC[Signal Creation]
```

**Pre-signal risk gate:** when running in trading mode (`order_manager` active) with `StrategyConfig.pre_signal_risk_gate=True` (default), the signal loop evaluates two gate layers for each new-entry market before invoking the LLM pipeline. First, a single cycle-level check (`check_entry_capacity`) determines whether global caps (max open positions, total exposure ceiling) are already full — if so, every new-entry market in that cycle is skipped. Then, per-market checks (`pre_signal_gate`) verify that the spread is within limits and the stoploss re-entry policy allows entry. Markets with existing open positions bypass both gates — exit signals must always fire. Set `pre_signal_risk_gate=False` in your strategy to always generate signals regardless of risk state (useful for calibration or analytics runs).

**Base-rate prompt enrichment:** immediately before building the LLM prompt, the pipeline calls `get_series_history_for_market(session, series_ticker, option_code)` for markets that have a `series_ticker`. When data exists, a `=== HISTORICAL BASE RATE ===` block is injected into the prompt between `=== MARKET CONTEXT ===` and `=== EVIDENCE ===`. See `SeriesOptionHistory` in §7 for the data model and Type A / Type B series semantics. Prompt version: `signal-v7`.

### Retrieval Sources

#### Structured News
| Source | Use Case | Priority |
|---|---|---|
| **Tavily Search API** | Fresh web search per market question | Primary |
| **NewsAPI** | Structured article archive for less-breaking topics | Secondary |
| **The Guardian API** | Full article body text; real-time; free developer key (500 req/day); strong on politics, economics, international news | Secondary |
| **Kalshi market metadata** | Market description + linked sources from the exchange | Always included |
| **GDELT** | High-volume global news index; free, no key required | Supplementary |
| **Internet Archive TV News Archive** | Closed-caption transcripts from 163+ U.S. TV stations; current to present day; free, no key required | Supplementary — especially valuable for word-mention markets and markets about public statements |
| **Internet Archive Third Eye (TV chyrons)** | OCR-extracted lower-third ticker text from live US TV (CNN, Fox News, MSNBC, BBC); near-real-time; free, no key required | High-signal for breaking news markets — a chyron like `FED CUTS RATES` often appears minutes before full transcripts. Runs in realtime_scheduler every 5 min. |

**Adaptive per-market fetch intervals (Tavily, NewsAPI, Guardian):**

All three rate-limited news fetchers use the same adaptive interval pattern to stay within their daily budgets regardless of how many markets are being monitored:

- **Interval formula:** `max(min_interval, min(24h, total_active_queries × 24h / daily_cap))`
- Each fetcher is checked once per market (cursor lookup before the query loop) and set once per market (after all queries complete). Every catalyst query for a due market fires — no single-query sampling.
- Per-market last-fetch time tracked in `fetcher_cursors` keyed `(fetcher, market_id)`. Cursors are deleted when markets close or lose strategy interest (via `delete_cursors` called after `deactivate_stale_catalysts`).
- Daily usage tracked in `api_daily_counters` (same table as NewsAPI window tracking); incremented per successful fetch.

| Fetcher | Budget | Daily cap default | Notes |
|---|---|---|---|
| **Tavily** | 1,000 credits/month | 33/day (÷30) | Hard cap via `api_daily_counters`; plan `UsageLimitExceededError` also triggers backoff |
| **NewsAPI** | 100 req/day | derived from `max_window_requests × 2` | Window cap (45/12h) remains the hard backstop; adaptive interval spreads load across markets |
| **Guardian** | 500 req/day | 490 | Hard cap via `api_daily_counters`; HTTP 429 triggers backoff |

Config keys: `tavily.daily_cap`, `tavily.min_fetch_interval_hours`; `newsapi.min_fetch_interval_hours`; `guardian.daily_cap`, `guardian.min_fetch_interval_hours`. All default to `min_fetch_interval_hours=1.0h`.

**Guardian API implementation:**
- Query endpoint: `https://content.guardianapis.com/search` with `show-fields=body,headline`
- Uses `tv_query` (Solr/Lucene boolean syntax) when available; falls back to `query_text`
- `from-date=YYYY-MM-DD` set to 7-day lookback (`_GUARDIAN_LOOKBACK_DAYS`)
- Response body field is HTML — stripped of tags and entity-unescaped before storage
- Rate limit: 1 req/sec; uses fixed `asyncio.sleep(1.0)` before each call
- `source_type="news"`, `source_name="The Guardian"`

**GDELT implementation:**
- Query the GDELT Doc API (`api.gdeltproject.org/api/v2/doc/doc`) with the catalyst query text and a `timespan=1d` parameter per cycle
- Response returns article URLs + titles only — no body text
- Fetch article bodies in parallel (`asyncio.gather`) with a per-URL timeout of 10s; skip paywalled/failed fetches silently
- Wired into the per-query loop in the scheduler alongside Tavily/NewsAPI
- No API key required; no quota tracking needed
- `source_type="news"`, `source_name="GDELT"`

**TV Chyron (Third Eye) implementation:**
- Bulk-pull endpoint: `GET https://archive.org/services/third-eye.php?last=1` (whole-number hours only)
- Response: tab-separated TSV with columns `date_time_(UTC)`, `channel`, `duration`, `identifier_path`, `text`
- Architecture: **bulk-pull + local-filter** — fetch all chyrons once per realtime cycle, then distribute matches to each market using its `tv_query` AND-groups
- `parse_and_groups(tv_query)` splits Solr/Lucene boolean syntax into AND-groups of OR'd terms; a chyron matches if every group has at least one term (case-insensitive substring)
- Deduplication via `fetcher_cursors` row `('tv_chyron', 'global')` — only chyrons with `dt > last_cursor` are stored
- `source_type="tv_chyron"`, `source_name="TVThirdEye"`
- Controlled by `ingestion.tv_chyron_enabled` config flag (default: `true`)

#### Social & Community Signals
| Source | Use Case | Notes |
|---|---|---|
| **Reddit** | Subreddit sentiment for relevant communities | No credentials required — uses public JSON API (`reddit.com/r/{sub}/search.json`); target subs per category (see below) |
| **Truth Social** | Real-time posts from key political/market-moving accounts | Requires a Truth Social account (`TRUTHSOCIAL_USERNAME` / `TRUTHSOCIAL_PASSWORD`); uses `truthbrush` library; account feed mode only (runs in realtime_scheduler) |
| **Twitter/X API** | Real-time public sentiment on market topics | Expensive ($100–$5000/mo tier); treat as optional enrichment |
| **Kalshi market comments** | Crowd reasoning directly on the market in question | Already fetched with market metadata |
| **Manifold Markets** | Community probability estimates on overlapping questions | Free API; useful as an independent signal cross-check |

**Truth Social implementation (account feed mode, runs in realtime_scheduler every 5 min):**
- Calls `api.pull_statuses(username, created_after=last_run)` for each configured account
- `last_run` tracked in Postgres (`fetcher_cursors` table, keyed by `(fetcher, key)`)
- Not tied to a specific market — broad ingestion, all categories benefit
- Configured via `ingestion.truthsocial.accounts: [realDonaldTrump, ...]` in `config.yaml`
- Runs in the realtime scheduler (not the main 30-min scheduler) so breaking posts are ingested within minutes

**Truth Social error handling:**
- `LoginErrorException` → log error + disable Truth Social for the rest of the cycle (circuit-breaker, same pattern as Tavily plan limit)
- Rate limiting handled internally by `truthbrush` (auto-sleep on approach)
- `ingestion.truthsocial.enabled` flag in config to disable entirely (same pattern as `newsapi.enabled`)

**Reddit subreddit targets by category:**

| Category | Subreddits |
|---|---|
| US Politics | r/politics, r/PoliticalDiscussion, r/neutralpolitics |
| Technology | r/technology, r/MachineLearning, r/singularity |
| Fintech | r/investing, r/wallstreetbets, r/stocks, r/fintech |
| Prediction markets | r/predictionmarkets, r/Kalshi |

#### Social Signal Handling

Social content is noisier than structured news and requires preprocessing before it reaches the LLM:

```mermaid
flowchart TD
    RP([Raw posts - Reddit / Twitter])
    RP --> AGG[Aggregator]
    AGG --> PS[Pre-summarizer - Haiku]
    PS --> CTX([One context block in main LLM analysis])
```

This two-pass approach keeps social signal cost-efficient: a cheap summarization pass (haiku/mini) collapses noisy social data before it hits the primary reasoning model. The main LLM sees a structured social summary, not raw posts.

**Social signal weight:** The LLM prompt explicitly instructs the model to treat social sentiment as weak/corroborating evidence, not primary evidence. Crowd sentiment without corroborating news should not be sufficient to cross a trade threshold alone.

### LLM Configuration

- **primary_model:** market probability analysis (default `claude-sonnet-4-6`)
- **cheap_model:** catalyst generation, body/social summarization, and daily digests (default `claude-haiku-4-5-20251001`)
- **judgment_model:** trade sizing and future trade-override judgment tasks (default `claude-opus-4-6`)
- **Output format:** Structured JSON via tool use (not free-form text parsing)
- **Prompt versioning:** Prompts are versioned and stored; every signal logs the prompt version used
- **Deduplication:** Non-scheduled triggers skip the LLM when the retrieval hash matches the last signal (same docs → same output). Scheduled triggers use a three-part gate: skip only when hash is unchanged AND FactBase has not been refreshed AND fewer than `signal.max_scheduled_interval_hours` (default 24h) have passed since the last scheduled LLM call. This guarantees at least one temporal-reasoning re-run per day while reacting immediately to new evidence or new FactBase data.

### Structured Output Schema (LLM response)

```json
{
  "probability": 0.71,
  "confidence": 0.78,
  "direction": "YES",
  "supporting_evidence": ["...", "..."],
  "counter_evidence": ["...", "..."],
  "reasoning": "...",
  "data_quality": "high | medium | low",
  "skip_reason": null
}
```

If `data_quality` is `"low"` (insufficient news context), the signal is discarded regardless of probability estimate.

---

## 10. Risk Framework

### Hard Caps (enforced by Order Manager, not overridable by strategy)

| Rule | Default | Notes |
|---|---|---|
| Max position size | 5% of bankroll | Per market |
| Max daily loss | 15% of bankroll | Triggers circuit breaker |
| Max total exposure | 30% of bankroll | Sum of all active positions |
| Min edge to trade | 10% | Absolute floor; strategy can raise, not lower |
| Max open positions | 20 | Prevents overextension; counts active positions |

**Pending orders count toward all capacity caps.** All three sites (max open positions, total exposure, per-market exposure) include rows in `status IN ('open', 'pending')`, and exposure sums use `COALESCE(requested_contracts, contracts) * entry_price` so a resting limit order reserves the capital it asked for — preventing a stack of in-flight orders from blowing past a cap before any fill. Log payloads (`risk.max_open_positions_reached`, `risk.total_exposure_exceeded`, `risk.market_exposure_exceeded`) split counts/exposure between `open` and `pending` for diagnostics.

### Bankroll vs. Kalshi Account Balance

`trading.bankroll_usd` in config is the system's self-imposed deployment budget — it can be less than the actual Kalshi account balance, allowing the operator to reserve funds for manual trading. The system enforces the following at startup in live mode:

- **Startup guard**: `get_balance()` is called against Kalshi. If the account balance is less than the configured bankroll, startup aborts with an error. If balance ≥ bankroll, both values are logged (structlog INFO).
- **Kelly sizing uses `bankroll_usd`**, not the Kalshi account balance. The operator controls deployment size through config.

### Position Sizing

Default: **confidence-blended Kelly criterion** (`IPredictionStrategy.position_size`)

```
p_market = mid-price at signal time (recovered from signal fields)
B        = (1 - p_market) / p_market          # net payout odds
p_adj    = confidence × p_est + (1 - confidence) × p_market
f*       = (B × p_adj - (1 - p_adj)) / B      # Kelly fraction

market_budget = max_exposure_per_market × bankroll
position_size = f* × kelly_fraction × market_budget
```

Key design properties:
- `max_exposure_per_market` is the **per-market budget** — Kelly scales within it, not against the full bankroll. Maximum possible position = `kelly_fraction × max_exposure_per_market × bankroll`.
- `confidence` blends the LLM estimate toward the market price. At confidence = 0 the edge collapses to zero and no position is taken. At confidence = 1 the raw estimate is used directly.
- `kelly_fraction` controls overall aggression (quarter-Kelly = 0.25 is default). Lowering it shrinks all positions proportionally.
- If `f* ≤ 0` (no edge after confidence blending), `position_size` returns 0.

**Trade sizing judgment** (T57): after `should_trade` passes, `order_manager.submit()` calls `assess_signal_context()` before sizing. The assessment combines two inputs:

- latest `source_quality_scores` snapshot rows for the sources used in the signal's evidence documents
- similar-market history for the market family (`series_ticker` match, with exact first-line question matches tracked as a stricter subset)

The structured summary is sent to the configured **judgment model** (default: Claude Opus). The model returns a `SignalAssessment` containing `trust_score`, `verdict`, and reasoning; framework code maps `trust_score` linearly into `[assessment_scale_min, assessment_scale_max]` and clamps it:

```
adjusted_ideal_total = base_ideal_total × assessment.size_multiplier
incremental_size     = max(adjusted_ideal_total - existing_market_exposure, 0)
```

This is sizing-only. The probability estimate and trade direction are never modified. When neither source-quality nor similar-market history is available, the LLM call is skipped and a neutral assessment is persisted with `trust_score=0.5` and `size_multiplier=1.0`. `source_quality_scores` is refreshed on its own daily scheduler; similar-market history is computed on demand at assessment time.

### Circuit Breakers

- Daily loss > 15% (`max_daily_loss_pct`): halt all new positions, alert via Telegram
- Total drawdown > 30% from reset baseline: halt all trading, require manual restart
- LLM API errors > `max_consecutive_llm_errors` consecutive (default 3): halt signal pipeline, alert
- LLM daily spend ≥ `max_daily_llm_spend_usd` (default $10): halt signal pipeline, alert

All circuit breaker alerts use a standard format (🚨 CIRCUIT BREAKER TRIPPED, type, reason, resume instructions). See `docs/runbook.md` for response procedures.

---

## 11. Infrastructure

### Cloud: AWS

| Service | Use |
|---|---|
| **ECS Fargate** | Runs the main freqpred service (containerized, always-on) |
| **RDS (Postgres + pgvector)** | Ledger, positions, signals, market history, Document store + embeddings, signal cache (retrieval hash), per-service daily request quotas, ingestion dedup |
| **CloudWatch** | Logs, metrics, alarms |
| **Secrets Manager** | API keys (Kalshi, LLM providers, news APIs) |
| **ECR** | Docker image registry |

### Deployment

- Docker-based, single `docker-compose.yml` for local development
- ECS task definition mirrors compose config for production parity
- GitHub Actions CI: lint → test → build → push to ECR → deploy to ECS
- Secrets injected at runtime via AWS Secrets Manager (never in code or config files)

### Environments

| Environment | Mode | Notes |
|---|---|---|
| `local` | Paper trading | Uses Kalshi sandbox API if available |
| `staging` | Paper trading | AWS ECS, real Kalshi data, no real orders |
| `production` | Live trading | AWS ECS, real orders, full risk controls active |

---

## 12. Dashboard & Alerts

### Web Dashboard

Built with **FastAPI** (backend) + **React 18 + TypeScript** (frontend), served via ECS.

**Frontend stack:** React 18, TypeScript, Vite, TanStack Query v5 (data fetching + polling), React Router v6, Recharts (calibration + cost charts), Tailwind CSS v3.

**API server:** Embedded inside `freqpred run` so the dashboard shares the live `OrderManager` instance — required for force-exit and any other write operations. Controlled by `dashboard.api_enabled` in `config.yaml` (default: `true`). The embedded server listens on `dashboard.host`:`dashboard.port` (default `0.0.0.0:8000`).

**`freqpred dashboard` command:** Dev-only Vite launcher — starts the Vite dev server at `http://localhost:5173` for hot-reload UI development. Has no DB or business logic of its own; API calls are proxied to `freqpred run` on port 8000. In production, `freqpred run` serves both the API and the built React SPA (from `freqpred/dashboard/ui/dist/`) — no separate `dashboard` command needed.

**Serve path:** `freqpred/dashboard/ui/` contains the React app. `npm run build` produces `freqpred/dashboard/ui/dist/`. When that directory exists, FastAPI serves the built SPA from `/` and `/assets`; otherwise `/` redirects to Swagger UI so the API remains usable in dev. In dev, Vite proxies `/api` to `localhost:8000`.

**Pages:**

1. **Signal Feed** — live stream of new signals with market question, our probability, market price, edge, direction, and expandable assessment detail
2. **Positions** — current paper/live positions with unrealized P&L, strategy filters, and force-exit controls
3. **Decisions** — strategy entry/exit decision analysis, including exit counterfactuals and prior-signal comparisons
4. **Markets** — searchable market browser with current signal detail and manual analyze-now actions
5. **Calibration** — scatter plot of estimated probability vs. resolution rate; Brier score trend. Brier score is computed per-signal across all analyzed markets — not just traded ones. Supports lookback windows and category filtering. Baseline comparison is `market_brier_score` (market mid-price at signal time vs. outcome).
6. **Source Quality** — per-source weighted Brier analysis vs overall baseline, with lookback and category filtering
7. **LLM Cost & Audit** — daily/weekly spend charts, cost by query type and strategy, query log with full prompt/response drilldown, budget burn rate vs. daily cap
8. **Strategy Config** — view/edit active strategy parameters (no code changes needed for threshold tuning)
9. **System Health** — API status, error rates, circuit breaker state, LLM budget circuit breaker status

### Telegram / Discord Alerts

Push notifications sent on key events (outbound only for Discord; Telegram supports inbound commands):

| Event | Alert |
|---|---|
| New signal above threshold | "📊 NEW SIGNAL: [question] — Our prob: 71%, Market: 54%, Edge: +17%" |
| Position opened | "🟢 PAPER TRADE: YES on [question] @ $0.54, size: $X" |
| Market resolved | "✅ WIN: [question] resolved YES. P&L: +$X (+34%)" or "❌ LOSS..." |
| Circuit breaker triggered | "🚨 CIRCUIT BREAKER: Daily loss limit hit. Trading halted." |
| Daily digest | Morning summary: open positions, yesterday's P&L, calibration score |

### Telegram Management Commands

The Telegram bot supports inbound slash commands for monitoring and control. Commands are processed via long-polling. Only users listed in `telegram.authorized_users` may issue commands; all others receive the push alerts but are silently ignored on commands.

**Configuration:**
```yaml
telegram:
  enabled: true
  bot_token: "..."
  chat_id: "..."           # personal or group chat id (include leading - for groups)
  authorized_users:        # list of telegram user IDs allowed to run commands
    - "123456789"
  notification_settings:
    new_signal: "on"       # on | silent | off
    position_opened: "on"
    position_closed: "on"
    circuit_breaker: "on"
    daily_digest: "silent"
```

**System commands:**

| Command | Description |
|---|---|
| `/start` | Resume trading after a pause. Changes run-loop state to `running`. |
| `/pause` | Stop entering new positions. Open positions continue to be managed. |
| `/stop` | Halt the run loop entirely (requires restart to resume). |
| `/show_config` | Print active strategy name, mode (paper/live), key thresholds. |
| `/logs [n]` | Show the last *n* structured log lines (default: 20). |
| `/help` | List all available commands. |
| `/version` | Show freqpred version and git commit hash. |

**Status commands:**

| Command | Description |
|---|---|
| `/status` | List all open positions: market question, direction, entry price, estimated prob, unrealized P&L. |
| `/status <position_id>` | Detailed view of one position including the signal that triggered entry. |
| `/count` | Current open position count vs. configured maximum. |
| `/trades [n]` | List the last *n* resolved/closed positions (default: 10) in a table. |
| `/signals [n]` | List the last *n* signals generated (default: 10): market, our prob, market price, edge. |

**Metrics commands:**

| Command | Description |
|---|---|
| `/profit [n]` | P&L summary over the last *n* days (default: all time): total, win rate, Brier score. |
| `/daily [n]` | P&L broken down by day for the last *n* days (default: 7). |
| `/weekly [n]` | P&L broken down by week for the last *n* weeks (default: 8). |
| `/monthly [n]` | P&L broken down by month for the last *n* months (default: 6). |
| `/stats` | Win/loss counts by exit reason (stoploss, ROI, signal flip, manual). |
| `/balance` | Current Kalshi account balance and available capital. |
| `/budget` | LLM spend today vs. daily cap, breakdown by query type. |
| `/calibration` | Current Brier score and calibration summary (last 30/90/all-time). |

**Position management commands:**

| Command | Description |
|---|---|
| `/forceexit <position_id>` | Force-close a specific open position immediately (submits a sell order). Works in both paper and live mode. |
| `/forceexit all` | Force-close all open positions. Prompts for confirmation before executing. |
| `/fx <position_id>` | Alias for `/forceexit`. |
| `/delete <position_id>` | Delete a paper position from the DB without placing an order. Paper mode only; rejected in live mode. |

**Implementation notes:**
- Command polling runs as a background asyncio task inside the main run loop.
- Each inbound update is checked against `authorized_users` before dispatch; unauthorized senders receive no response.
- `/forceexit` and `/delete` in live mode require a confirmation reply within 30 seconds (inline keyboard with Confirm / Cancel buttons) to prevent accidental execution.
- All command responses are sent as reply messages to the original command.
- The polling task logs every inbound command to `structlog` at INFO level (not to `llm_queries` — these are not LLM calls).

---

## 13. Development Phases

### Phase 1: Signal Engine MVP ✅
*Goal: produce scored signals for active Kalshi markets*

- [x] Kalshi API client (market fetch, no trading yet)
- [x] Document store schema (Postgres + pgvector extension)
- [x] Ingestion pipeline: Tavily + NewsAPI fetchers → dedup → embed (sentence-transformers) → store
- [x] Ingestion pipeline: Reddit fetcher + social pre-summarizer → store
- [x] Guardian API fetcher: full article body via Content API; Solr query support; HTML strip; 429 backoff
- [x] GDELT fetcher: Doc API query → parallel article body fetch → store (T32)
- [x] Truth Social fetcher: account feed mode (per cycle) via `truthbrush` (T33)
- [x] Internet Archive TV fetcher: catalyst `tv_query` → transcript clip search → store
- [ ] Twitter/X fetcher (optional — gated on API cost decision)
- [x] Strategy interface (`IPredictionStrategy`) + `ConservativeDefault`, `PoliticsEdgeStrategy`, `TechNewsStrategy` strategies
- [x] Market Selector: reads active markets, calls `strategy.is_market_interesting()` per registered strategy
- [x] Catalyst Generator: LLM (Haiku) derives 3–5 search queries per selected market; writes `CatalystRun` + `CatalystQuery` to DB; RAG-informed on re-runs; deactivates on market close/deselection
- [x] Ingestion Scheduler: reads latest active `CatalystQuery` rows from DB; drives fetchers; tracks last-run per fetcher in Postgres (`fetcher_cursors` table)
- [x] RAG retriever: vector search against Document store for a given market question
- [x] LLM signal pipeline: retrieve docs → hash check → Claude analysis → Signal creation
- [x] LLM audit logging (LLMQuery table — every call logged with cost)
- [x] LLM budget circuit breaker (configurable daily spend cap — enforced in `LLMClient.complete()`)
- [x] Signal scoring and logging to Postgres
- [x] Basic CLI: `freqpred run --strategy ConservativeDefault --mode signal-only`
- [x] Docker + local dev setup

**Done when:** System runs continuously, produces signals for 20+ markets/day, signals are logged with full context for review.

---

### Phase 2: Paper Trading + Calibration ✅
*Goal: validate signal quality with simulated trades*

**Ingestion improvements (address before running at scale):**
- [x] NewsAPI rate limiting (T15): `config.newsapi.enabled` flag to disable fetcher; daily request quota tracked in `api_daily_counters` Postgres table (default cap: 90/day).

**RAG improvements:**
- [x] `document_market_links.relevance_score` — swapped to actual cosine similarity scores from pgvector (T16).
- [x] Per-catalyst-query supplemental retrieval — signal pipeline embeds each active `CatalystQuery` text in parallel and adds top-1 per query not in the market-question core set; slot-budgeted so core always holds ≥ half of `top_k` slots, supplemental slots rank-selected by cosine similarity.

- [x] Order Manager (paper mode) (T19)
- [x] Ledger (positions, resolutions, P&L) (T18)
- [x] Strategy exit interface — `should_exit()`, `custom_exit()`, `force_exit()`, `stoploss`, `trailing_stop` on `IPredictionStrategy` and `StrategyConfig` (T25)
- [x] Position monitor — background loop checking all open positions on each price poll; simulates paper exits when stoploss/ROI/signal conditions are met; logs `exit_reason` (T26)
- [x] Calibration metrics (Brier score, calibration curve) (T21)
- [x] Web dashboard (signal feed + ledger views) (T24)
- [x] Telegram alerts (T23)

**Done when:** 100+ markets resolved or exited with logged signals. Calibration score measured. Exit behavior observable from ledger. Decision made: is the signal real?

**Go/no-go criteria for Phase 3:**
- Brier score < `market_brier_score` (beat the market's own calibration)
- Positive calibration: 60-70% estimated → 60-70% resolution rate (measured per-signal across all analyzed markets, not just traded ones)
- Positive simulated ROI over 100+ trades

---

### Phase 3: Live Trading
*Goal: real capital, controlled risk*

Each task has a linked GitHub issue (same number) with full implementation scope, test plan, and acceptance criteria.

- [x] **T35** [#35](https://github.com/ostersc/freqpred/issues/35) — Kalshi demo environment: `KALSHI_BASE_URL`, `KALSHI_DEMO_API_KEY`, `KALSHI_DEMO_PRIVATE_KEY_PATH` env var overrides; `demo_api_key`/`demo_private_key_path` fields in `KalshiConfig` for one-off test scripts. No runtime demo mode — Kalshi demo has no real liquidity.
- [x] **T36** [#36](https://github.com/ostersc/freqpred/issues/36) — KalshiClient live order placement: `_post()` transport, `place_order()`, `get_positions()` (uses `market_positions` key), `get_balance()` (cents→USD). `KalshiAPIError` on non-2xx. `Order.exchange_order_id` field added.
- [x] **T37** [#37](https://github.com/ostersc/freqpred/issues/37) — OrderManager live branch: startup balance guard (abort if Kalshi balance < `bankroll_usd`); route to `KalshiClient.place_order()` when `mode=live`; record positions as `pending` with `exchange_order_id`; `reconcile_pending_orders()` stub. Depends on: T36.
- [x] **T38** [#38](https://github.com/ostersc/freqpred/issues/38) — PositionMonitor live exits: submit IOC sell order on exit; close ledger only after exchange confirms; alert on `KalshiAPIError`. Depends on: T36, T37.
- [x] **T39** [#39](https://github.com/ostersc/freqpred/issues/39) — PositionWatcher WebSocket `ticker`: persistent connection per open live position; sub-second price updates; triggers position monitor on each tick; exponential backoff reconnect; Kalshi↔DB position reconciliation on startup and reconnect (sync contracts, auto-close zero-net, skip manual-only). Depends on: T37, T38.
- [x] **T40** [#40](https://github.com/ostersc/freqpred/issues/40) — PositionWatcher `market_lifecycle_v2`: resolution events; settle at $1/$0 on `determined` (`settlement_value` "1.0000"=YES/"0.0000"=NO, `event_type` not `status`); REST poll fallback for missed events on later `settled` events. Depends on: T39.
- [x] **T41** [#41](https://github.com/ostersc/freqpred/issues/41) — Dashboard: Strategy Config + System Health API endpoints; GET/PUT strategy params at runtime; circuit breaker state, WebSocket status, pending order count. Depends on: T36.
- [ ] **T42** [#42](https://github.com/ostersc/freqpred/issues/42) — Production AWS deployment: ECS Fargate, RDS Postgres 16 + pgvector, Secrets Manager, CloudWatch alarms, deployment runbook. Depends on: T36.
- [ ] **T43** [#43](https://github.com/ostersc/freqpred/issues/43) — GitHub Actions CI/CD: lint → test → build Docker → push to ECR → migrate → deploy to ECS. Depends on: T42.
- [x] **T44** [#44](https://github.com/ostersc/freqpred/issues/44) — React dashboard frontend foundation: initial dashboard shell and base monitoring pages, later expanded by T61/T62/T64/T65 into the current Signal Feed, Positions, Decisions, Markets, Calibration, Source Quality, LLM Cost, Strategy Config, and System Health views. Depends on: T41.
- [x] **T45** [#45](https://github.com/ostersc/freqpred/issues/45) — Circuit breaker hardening: drawdown breaker implementation; all four breakers verified; Telegram alert format; incident runbook. Depends on: T36.
- [x] **T47** [#47](https://github.com/ostersc/freqpred/issues/47) — `OrderTypes` config + limit order entry: `OrderTypes` dataclass on `StrategyConfig`; `custom_entry_price()` hook; entry at `estimated_prob - min_edge`; pending position fill-check + timeout cancellation; paper mode only.
- [ ] **T48** [#48](https://github.com/ostersc/freqpred/issues/48) — Limit order exits + exchange-hosted stoploss: `exit=limit` posts resting ROI/trailing targets; `custom_exit_price()` hook; `stoploss_on_exchange` with interval refresh; emergency/circuit-breaker always market. Depends on: T47.
- [x] **T49** [#49](https://github.com/ostersc/freqpred/issues/49) — `IAlgoStrategy`: DataFrame-driven exits via WebSocket tick data; freqtrade-style `populate_indicators()` + `populate_exit_trend()` hooks; OHLC candle buffer per market; `force_exit()` reads `exit_long` column; `PositionMonitor.on_tick()` feeds ticks to algo strategy buffers. OHLC is direction-corrected before being passed to `populate_indicators`/`populate_exit_trend`: NO positions receive inverted candles (`no_high = 1 - yes_low`, `no_low = 1 - yes_high`) so that indicator logic (RSI, EMA crossovers, etc.) operates on contract value from the holder's perspective. Candle cache is keyed by `(market_id, direction)` so YES and NO positions maintain independent OHLC series. Depends on: T39.
- [x] **T57** [#57](https://github.com/ostersc/freqpred/issues/57) — Trade sizing judgment: `source_quality_scores` table (daily rolling snapshot per source name × market category) plus `signal_assessments` table (append-only, one row per assessed signal); `assess_signal_context()` in `freqpred/metrics/assessment.py` runs between `should_trade` and `position_size` in `order_manager.submit()`; assessment combines source-quality history and similar-market family history (`series_ticker`, plus exact-question subset stats), then uses the configured `judgment_model` (default Opus) to return `trust_score`, `verdict`, and reasoning; framework maps trust score into `[assessment_scale_min, assessment_scale_max]`; default `position_size()` applies the multiplier to ideal total exposure before subtracting `existing_market_exposure`; neutral assessment when no usable history exists and the LLM call is skipped. `StrategyConfig` gains `assessment_scale_min/max` and `similar_market_min_signals/trades`. Depends on: T56.
- [ ] **T50** [#50](https://github.com/ostersc/freqpred/issues/50) — LLM-assisted exit analysis: `should_request_llm_exit()` predicate + `llm_exit_check()` async hook on `IAlgoStrategy`; PositionMonitor calls LLM when predicate fires; prompt includes candle metrics + P&L; response logged to `llm_queries`. Depends on: T49.
- [x] **T51** [#51](https://github.com/ostersc/freqpred/issues/51) — TV chyron ingestion via Internet Archive Third Eye API + realtime scheduler: `tv_chyron.py` fetcher (`fetch_all`, `parse_and_groups`, `filter_chyrons`); new `realtime_scheduler.py` runs chyrons and Truth Social account feeds every 5 min (moved from main scheduler); `backoff.py` `tick_and_load` gains `services` filter so each scheduler manages its own counters independently; `ingestion.tv_chyron_enabled` and `ingestion.realtime_interval_seconds` config keys added.
- [x] **T61** [#61](https://github.com/ostersc/freqpred/issues/61) — Dashboard: force-exit positions from Positions page; `POST /api/positions/{id}/force-exit` endpoint; "Force Exit" button in expanded detail panel (open positions only) with confirmation dialog; invalidates TanStack Query cache on success. `OrderManager.force_exit()` centralizes paper/live exit logic; API server embedded in `freqpred run`; `freqpred dashboard` is dev-only Vite launcher.
- [x] **T62** [#62](https://github.com/ostersc/freqpred/issues/62) — Dashboard: market browser page; `GET /api/markets`, `GET /api/markets/{id}`, `POST /api/markets/{id}/analyze`; new Markets page with search, expandable rows showing full market detail + current signal, "Analyze now" button triggers signal pipeline and refreshes panel; 429 cooldown if analyzed within 60 s.
- [x] **T64** [#64](https://github.com/ostersc/freqpred/issues/64) — Dashboard: strategy decision analysis page; `GET /api/strategy-decisions` with filters (strategy, exit_reason prefix, ticker_prefix ILIKE, date_from/to) and pagination; per-row exit counterfactual P&L (`our_side_win_value − entry_price`) and exit Δ vs hold (`exit_price − our_side_win_value`); entry efficiency loss vs best prior signal with `edge > 0` (`best_prior_ask − entry_price`); symmetric for YES/NO via side-specific `signals.market_ask_at_signal`. Extracts `PriceTimeline` + `SignalDetail` + `SelectedSignalPanel` into shared components and adds exit-event reference lines (vertical at `exit_time`, horizontal at exit price, NO-flipped) — benefit flows back to Positions page closed rows. Adds `exit_reason` to `PositionOut`.
- [x] **T65** [#65](https://github.com/ostersc/freqpred/issues/65) — Dashboard: signal assessment visibility for source quality + similar-market trust; expose persisted assessment summary and `llm_query_id` on signal/position detail APIs; add dashboard card showing trust score, implied size effect, source-quality summary, similar-market summary, warnings, and a link to the existing LLM audit detail. Depends on: T57.
- [ ] **T66** [#66](https://github.com/ostersc/freqpred/issues/66) — Deterministic replay/regression harness: record time-locked market/document fixtures and replay signal-generation decisions offline to catch prompt/model/config regressions without introducing a historical backtesting engine. Depends on: T11.
- [x] **T67** [#67](https://github.com/ostersc/freqpred/issues/67) — Live order-state hardening (entry side): `KalshiClient.get_order`/`cancel_order`; `Order` + `PositionRow` carry exchange-confirmed fill metadata (`requested_contracts`, `exchange_order_status`, `last_exchange_sync_at`); `OrderManager.reconcile_pending_orders` rewritten to per-order `get_order` polling with shared status-mapping helper, `SELECT … FOR UPDATE SKIP LOCKED` concurrency guard, configurable `pending_order_timeout_seconds`, and `place_order → ledger` orphan cancel; `PositionWatcher` subscribes to `user_orders`/`fill` WS channels and renames `_reconcile_positions` → `_detect_external_drift` (open-only scope); `position_monitor` drives reconcile every 30s; risk engine counts pending orders as committed exposure (max positions, total exposure, per-market exposure); dashboard exposes new fields + `pending_orders_detail` table. Depends on: T36, T37, T39, T68.
- [x] **T76** [#76](https://github.com/ostersc/freqpred/issues/76) — Live order-state hardening (exit side): `_execute_live_exit` polls `get_order` until terminal; `ledger.partial_close_position` handles partial IOC fills (residual stays open, weighted-avg exit_price on final close); exit-side columns on `PositionRow` (`exit_order_id`, `exit_fee_usd`, `exit_requested_contracts`, `exit_filled_contracts`, `realized_pnl_accumulator`); `force_exit` uses same polling + `partial_close_position`; dashboard `PositionOut` exposes exit fields; Positions page shows "mid-exit" status indicator. Depends on: T67.
- [x] **T68** [#68](https://github.com/ostersc/freqpred/issues/68) — Ops freshness telemetry: persist heartbeat/freshness timestamps for ingestion, signal, source-quality, and WebSocket loops; expose real websocket connectivity + last-message telemetry and stale-loop indicators in System Health; optional alerts when critical loops stop making progress. Depends on: T41.
- [ ] **T69** [#69](https://github.com/ostersc/freqpred/issues/69) — Correlated exposure caps: enforce series/category/event-family risk limits so multiple related markets cannot collectively exceed configured exposure even when per-market limits pass. Depends on: T17.
- [ ] **T70** [#70](https://github.com/ostersc/freqpred/issues/70) — Series option base-rate history: `series_option_history` table keyed by `(series_ticker, option_code)`; background refresh fetches all settled markets per active series from Kalshi API and upserts YES/NO counts + label; signal prompt receives a base-rate context block when `n >= 3`; Type B single-option series degrade gracefully via low counts.
- [x] **T71** [#71](https://github.com/ostersc/freqpred/issues/71) — Pre-signal risk gate: skip LLM analysis for new-entry markets where risk would block the resulting trade (global capacity caps, spread too wide, stoploss re-entry blocked); gate is opt-out via `StrategyConfig.pre_signal_risk_gate` (default `True`); markets with open positions always bypass the gate.
- [x] **T72** [#72](https://github.com/ostersc/freqpred/issues/72) — Dashboard: P&L over time page; `GET /api/pnl/time-series` with 7 filter dimensions (strategy, signal model, prompt version, direction, category, series, market); Recharts `ComposedChart` with daily P&L bars, cumulative P&L line, EMA overlay, dual Y-axis for P&L vs LLM spend; projection tab with CAGR extrapolation, linear LLM spend projection, and "days until broke" countdown (broke = initial_bankroll + projected_trading_pnl − projected_llm_spend ≤ 0).
- [x] **T74** [#74](https://github.com/ostersc/freqpred/issues/74) — Calibration page tabs: Distribution (existing scatter plot), Over Time (daily Brier time-series with EMA + prompt-version flags), By Option (heatmap of `series_ticker × option_code` rows vs prompt-version columns, colored by Brier-vs-market delta, with vs-avg delta, n-bar, market-difficulty bar, and row sparkline).
- [ ] **T75** [#75](https://github.com/ostersc/freqpred/issues/75) — Kalshi changelog monitor: daily RSS fetch of `https://docs.kalshi.com/changelog/rss.xml`, `kalshi_changelog_state` DB singleton tracking `last_reviewed_at` / `unreviewed_count` / `has_unreviewed_breaking_change`, warning/critical alerts on new entries, system health surface, telemetry heartbeat.
- [x] **T73** [#73](https://github.com/ostersc/freqpred/issues/73) — FactBase phrase frequency gate + signal enrichment for KXTRUMPSAY markets: Haiku extracts search terms (slash variants + plurals/possessives) once per market; `factbase_phrase_frequency` table caches window counts (`in_market_count`, 7d, 30d, 365d) + top Trump quotes; `is_market_interesting()` blocks until cache is ready; signal prompt gets a `PHRASE FREQUENCY DATA` block; assessor payload gets `phrase_frequency`; `SERVICE_FACTBASE_SCHEDULER` telemetry heartbeat; bumps signal prompt to `signal-v8`. Depends on: T70, T71.
- [x] **T77** [#77](https://github.com/ostersc/freqpred/issues/77) — Migrate order endpoints from legacy `/portfolio/orders` to V2 `/portfolio/events/orders`: add `event_ticker` to `Order` dataclass; update `place_order`, `get_order`, `cancel_order` in `kalshi.py`; populate `event_ticker` from `market.metadata` at entry time and via `MarketRow` join at exit time; empty-ticker fallback to legacy path with warning log.

**`OrderTypes` interface** (strategy-level, all fields have defaults — existing strategies unchanged):
```python
order_types = OrderTypes(
    entry="market",                        # "limit" | "market"
    exit="market",                         # "limit" | "market"
    emergency_exit="market",               # always market; circuit-breaker / force-exit paths
    stoploss="market",                     # "limit" | "market"
    stoploss_on_exchange=False,            # post resting stoploss on Kalshi exchange
    stoploss_on_exchange_interval=60,      # seconds between refreshing exchange stoploss price
    stoploss_on_exchange_limit_ratio=0.99, # limit price = trigger_price × ratio
)
```

**Price hooks on `IPredictionStrategy`** (optional — default behaviour applies when not overridden):
```python
def custom_entry_price(self, signal: Signal, market: Market) -> float | None:
    """Custom limit entry price. None → default (estimated_probability - min_edge)."""
    return None

def custom_exit_price(self, position: Position, signal: Signal | None, market: Market, exit_reason: str) -> float | None:
    """Custom limit exit price. None → default (ROI / trailing-stop target price)."""
    return None
```

`emergency_exit` is always market — safety constraint, not overridable by strategy.

**Done when:** System executes real trades automatically with enforced risk limits, monitored via dashboard and Telegram.

---

### Phase 4: Cross-Platform Intelligence — Polymarket
*Goal: enrich signals and entry/exit decisions with Polymarket pricing and smart-money flow data*

Each task has a linked GitHub issue with full implementation scope, test plan, and acceptance criteria.

- [ ] **T78** [#78](https://github.com/ostersc/freqpred/issues/78) — Polymarket market matching engine: `PolymarketClient` (Gamma + CLOB APIs, no auth) in `freqpred/markets/polymarket.py`; `polymarket_market_links` DB table + Alembic migration; `freqpred/ingestion/polymarket_matcher.py` daily batch match (embedding cosine sim + Haiku confirm for borderline per scope gate); wire into ingestion scheduler (runs once/day after market selector); config keys: `polymarket.enabled`, `polymarket.match_auto_threshold` (0.85), `polymarket.match_llm_threshold` (0.70), `polymarket.llm_match_scope` (`"never"` | `"always"` | `"interesting"` | `"categories"`, default `"interesting"`), `polymarket.llm_match_categories` (list[str], default `[]`).

- [ ] **T79** [#79](https://github.com/ostersc/freqpred/issues/79) — Polymarket price collection + assessment enrichment: `freqpred/ingestion/fetchers/polymarket_prices.py` CLOB price poller; `polymarket_prices` DB table + migration; wire into `realtime_scheduler.py` (5 min cadence, only matched active markets); inject CROSS-PLATFORM CONTEXT block in `metrics/assessment.py:assess_signal_context()` — NOT the signal prompt (signals exclude market data by design); `SignalAssessment` gains `polymarket_yes_price: float | None`; `GET /api/polymarket/prices/{kalshi_ticker}` dashboard endpoint; bump assessment prompt version. Depends on: T78.

- [ ] **T80** [#80](https://github.com/ostersc/freqpred/issues/80) — Strategy hooks + PositionMonitor divergence gate: 10 new `StrategyConfig` fields (price gates: `polymarket_signal_gate`, `polymarket_max_divergence_entry`, `polymarket_exit_divergence_threshold`; whale fields: `whale_alert_min_usd`, `whale_alert_pct_volume`, `whale_exit_min_usd`, `whale_exit_pct_volume`, `whale_entry_block`, `whale_entry_block_min_usd`, `whale_entry_block_pct_volume`, `whale_entry_block_lookback_hours` — hybrid USD-floor-OR-volume-pct throughout, all default disabled); entry gate in `OrderManager.submit()` (Polymarket divergence check + whale block, both fail-open when no data); exit divergence check in `PositionMonitor.evaluate_exit()` (`polymarket_` prefixed `exit_reason`); `GET /api/polymarket/divergence` endpoint. Depends on: T79, T81.

- [ ] **T81** [#81](https://github.com/ostersc/freqpred/issues/81) — Whale tracking via Polymarket CLOB trades API: `polymarket_whale_trades` + `polymarket_whale_wallets` DB tables + migration; wire into `realtime_scheduler.py` (5 min cadence, cursor-based dedup via `fetcher_cursors`); whale qualification uses hybrid USD-floor-OR-volume-pct threshold; upsert `polymarket_whale_wallets` running totals per trade; daily batch win/loss scoring on resolved Polymarket markets including per-category `category_stats` JSONB; Telegram/Discord alert on qualifying trades for watched markets; `SERVICE_POLYMARKET_WHALE_TRACKER` telemetry heartbeat. Depends on: T78.

- [ ] **T82** [#82](https://github.com/ostersc/freqpred/issues/82) — Cross-platform dashboard page: `GET /api/polymarket/dashboard` summary endpoint; new "Cross-Platform" React page with: divergence table (all matched markets, Kalshi vs Polymarket price, delta column, toxic-flow indicator), price comparison chart for selected market (Kalshi mid vs Polymarket mid, last 24h), whale trade feed (market, wallet short-hash with "Known sharp" / "Known whale" badge, direction, size, % of volume, pct_of_liquidity, age). Depends on: T79, T81.

**Done when:** Polymarket prices are feeding the assessment prompt, matched markets appear in the dashboard, and whale alerts are firing on live markets with open positions.

---

## 14. Open Questions

1. **Kalshi sandbox API** ✅ — Kalshi offers a demo environment at `https://demo-api.kalshi.co/trade-api/v2` with separate credentials. Investigation found the demo API has the same 42k tickers as production but zero real liquidity (synthetic seed prices, no trades). It is not useful as a runtime mode — signals and signal calibration would be meaningless. Decision: demo credentials are stored in config (`demo_api_key`, `demo_private_key_path`) for one-off API smoke tests (e.g. verifying `place_order()` returns a valid response) but freqpred has no `--mode demo` runtime. Testing live order flow is done at small position sizes directly against production.
2. **LLM look-ahead bias mitigation** — Even with time-locked news retrieval, Claude's training data includes past market resolutions. How much does this inflate signal quality in paper trading? Consider running a "naive baseline" strategy (always bet with the market) to measure true alpha.
3. **Rate limits** — Kalshi API rate limits for market data polling. What's the sustainable polling interval?
4. **Strategy versioning** — When a strategy's parameters change, how do we attribute P&L? Track `(strategy_name, strategy_version)` per position.
5. **Multi-resolution markets** — Some Kalshi markets have multiple outcomes (not just binary). Handle in v2.
6. **News freshness** — How often should signals be refreshed for markets that don't close for weeks? Define a refresh schedule (e.g., re-analyze every 24h if market hasn't closed).

---

## 15. Repository Structure

```
freqpred/
├── freqpred/
│   ├── __init__.py
│   ├── cli.py                   # entry point: freqpred run/backtest/etc
│   ├── config.py                # config loading (YAML + env vars)
│   ├── markets/
│   │   ├── base.py              # IMarketClient abstract interface
│   │   ├── kalshi.py            # Kalshi adapter
│   │   ├── polymarket.py        # Polymarket adapter (Gamma + CLOB APIs, read-only; no auth)
│   │   ├── watcher.py           # polling loop: price refresh, staleness detection
│   │   └── models.py            # Market, Order, Position dataclasses
│   ├── ingestion/
│   │   ├── selector.py          # market selector: calls strategy.is_market_interesting()
│   │   ├── catalyst_generator.py# LLM (Haiku) derives catalyst queries per market; manages CatalystRun/CatalystQuery
│   │   ├── polymarket_matcher.py# daily batch: match Kalshi markets to Polymarket via embedding + optional Haiku confirm
│   │   ├── scheduler.py         # main ingestion scheduler (30 min): catalyst queries → Tavily/NewsAPI/Guardian/Reddit/GDELT/TV Archive + Polymarket matching
│   │   ├── realtime_scheduler.py# fast scheduler (5 min): TV chyrons + Truth Social + Polymarket prices + whale trades
│   │   ├── fetchers/
│   │   │   ├── tavily.py        # Tavily Search API fetcher
│   │   │   ├── newsapi.py       # NewsAPI fetcher
│   │   │   ├── guardian.py      # The Guardian Content API fetcher (full body, free tier, Solr query support)
│   │   │   ├── gdelt.py         # GDELT Doc API fetcher + article body fetch
│   │   │   ├── reddit.py        # Reddit API fetcher
│   │   │   ├── tv_archive.py    # Internet Archive TV transcript search fetcher
│   │   │   ├── tv_chyron.py     # Internet Archive Third Eye chyron fetcher (bulk-pull + local-filter)
│   │   │   ├── truthsocial.py   # Truth Social fetcher (search + account feeds via truthbrush)
│   │   │   ├── polymarket_prices.py  # Polymarket CLOB price poller (5 min, matched markets only)
│   │   │   └── twitter.py       # Twitter/X API fetcher (optional)
│   │   ├── store.py             # dedup, embed (sentence-transformers), insert into Document store
│   │   └── social_summarizer.py # cheap LLM pre-summarizer for raw social posts
│   ├── rag/
│   │   ├── embedder.py          # local sentence-transformers embedding client
│   │   ├── retriever.py         # vector search against Document store (pgvector)
│   │   └── models.py            # Document, DocumentMarketLink dataclasses
│   ├── signal/
│   │   ├── pipeline.py          # orchestrates retrieval + LLM analysis
│   │   ├── llm.py               # Claude client, structured output
│   │   ├── cache.py             # retrieval hash check, signal dedup
│   │   └── models.py            # Signal dataclass
│   ├── strategy/
│   │   ├── base.py              # IPredictionStrategy interface
│   │   ├── config.py            # StrategyConfig dataclass
│   │   └── defaults/
│   │       ├── politics.py      # PoliticsEdgeStrategy
│   │       ├── tech.py          # TechNewsStrategy
│   │       └── conservative.py  # ConservativeDefault
│   ├── trading/
│   │   ├── order_manager.py     # paper + live order execution
│   │   ├── risk.py              # hard cap enforcement, circuit breakers
│   │   └── ledger.py            # position tracking, P&L calc
│   ├── llm/
│   │   ├── client.py            # LLM API wrapper (Claude)
│   │   ├── audit.py             # LLMQuery logging, cost tracking, budget circuit breaker
│   │   └── models.py            # LLMQuery dataclass
│   ├── metrics/
│   │   ├── assessment.py        # source-quality + similar-market sizing judgment
│   │   ├── calibration.py       # Brier score, calibration curve, source quality
│   │   ├── models.py            # signal assessment + source-quality ORM models
│   │   ├── reporting.py         # daily digest generation
│   │   └── scheduler.py         # source-quality refresh scheduler
│   ├── dashboard/
│   │   ├── api/                 # FastAPI routes + schemas
│   │   └── ui/                  # React frontend (Vite, Tailwind, TanStack Query)
│   │       ├── package.json
│   │       ├── vite.config.ts
│   │       └── src/
│   │           ├── api/         # typed fetch wrappers per endpoint
│   │           ├── components/  # NavBar, StatusBadge, LoadingSpinner, ErrorBanner
│   │           └── pages/       # 9 dashboard pages
│   └── alerts/
│       ├── telegram.py
│       └── discord.py
├── strategies/                  # user strategy files (gitignored by default)
├── config/
│   ├── config.example.yaml      # template config
│   └── config.yaml              # local config (gitignored)
├── tests/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── SPEC.md                      # this file
└── README.md
```

---

## 16. Key Design Principles

1. **Paper first, live second.** No real money until calibration is proven over 100+ markets.
2. **Calibration over returns.** A 55% win rate with good calibration beats 70% wins with poor edge estimation. Optimize for signal accuracy, not raw P&L.
3. **Hard caps are sacred.** Risk rules in `Order Manager` cannot be overridden by strategy code. Period.
4. **Log everything.** Every signal, every retrieved article, every LLM response, every trade. The value is in the data over time.
5. **Strategy code is user-owned.** The framework provides the plumbing. Users define the alpha.
6. **Open source hygiene from day one.** No hardcoded keys, no personal data in the repo, clean public-facing interfaces even when running privately.
