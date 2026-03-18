# freqpred — Project Specification

> A framework for LLM-driven prediction market trading, modeled on freqtrade's architecture.

**Version:** 0.1-draft
**Last updated:** 2026-03-17
**Status:** Phase 1 complete — signal engine running; Phase 2 (paper trading) in progress

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
- [ ] Track paper trades and measure real-world calibration over time
- [ ] Execute live trades on Kalshi with hard risk controls
- [ ] Provide a web dashboard and Telegram/Discord alerts
- [ ] Run continuously on AWS as an always-on service

## 3. Non-Goals (v1)

- **No backtesting engine** — LLM training data contamination makes historical backtests unreliable; paper trading is the validation approach
- **No non-US markets** — Polymarket is geo-blocked for US users; Kalshi is the only regulated platform in scope
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

```
┌─────────────────────────────────────────────────────────────┐
│                        freqpred                              │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐ │
│  │  Market      │   │  Signal      │   │  Strategy       │ │
│  │  Watcher     │──▶│  Pipeline    │──▶│  Engine         │ │
│  │              │   │  (RAG+LLM)   │   │  (plugins)      │ │
│  └──────┬───────┘   └──────────────┘   └────────┬────────┘ │
│         │                                         │          │
│         ▼                                         ▼          │
│  ┌──────────────┐                       ┌─────────────────┐ │
│  │  IMarketClient│                      │  Order Manager  │ │
│  │  (Kalshi)    │◀──────────────────────│  (paper/live)   │ │
│  └──────────────┘                       └────────┬────────┘ │
│                                                   │          │
│                                         ┌─────────▼────────┐ │
│                                         │  Ledger + DB     │ │
│                                         │  (RDS Postgres)  │ │
│                                         └────────┬────────┘ │
│                                                   │          │
│                                         ┌─────────▼────────┐ │
│                                         │  Dashboard +     │ │
│                                         │  Alerts          │ │
│                                         └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| **Market Watcher** | Polls Kalshi for active markets, upserts into DB |
| **Market Selector** | Reads active markets from DB; calls `strategy.is_market_interesting()` on each registered strategy; passes selected markets to Catalyst Generator |
| **Catalyst Generator** | LLM call (Haiku) per selected market: derives 3–5 specific search queries (catalysts) representing events that could materially shift probability. Stored as first-class DB entities. Re-runs daily, RAG-informed on subsequent passes. |
| **Position Watcher** | Streams live price updates via Kalshi WebSocket for markets with open positions |
| **Ingestion Scheduler** | Reads the latest active catalyst queries per market from DB; runs Tavily + NewsAPI + Reddit fetchers against those queries; upserts results into Document store |
| **Signal Pipeline** | Retrieves news context via RAG, runs LLM analysis, returns probability estimate |
| **Strategy Engine** | Applies `IPredictionStrategy` plugins to signal output, decides trade/size/skip |
| **IMarketClient** | Abstract interface over Kalshi (and future platforms); handles orders, positions, balance |
| **Order Manager** | Executes paper or live trades; enforces hard risk caps before any order |
| **Ledger** | Immutable trade log; records every signal, position, and resolution outcome |
| **Dashboard** | Web UI for monitoring; Telegram/Discord for push alerts |

---

## 6a. Position Watcher — WebSocket Price Tracking

Markets with open positions need tighter price monitoring than the 5-minute REST poll:
- A price move of ±5 cents on a held position can materially change the exit decision.
- Resolution events (market settled, determined) need to be caught quickly so P&L can be recorded.

The **Position Watcher** maintains a persistent Kalshi WebSocket connection and subscribes to the `ticker` channel for every market where freqpred holds at least one open position (`status = "open"`).

### WebSocket channels used

| Channel | Payload | Action |
|---|---|---|
| `ticker` | Real-time best bid/ask update | Update `MarketRow` price fields + `price_updated_at`; emit `price_moved` signal trigger if Δmid ≥ threshold |
| `market_lifecycle` | Status change (active → determined → settled) | Mark position for resolution; trigger P&L calculation |

### Connection lifecycle

```
On startup / position opened:
  - Build subscription set: {market_id for position in open_positions}
  - Connect to wss://trading-api.kalshi.com/trade-api/v2/ws/v2
  - Authenticate (same RSA-PSS headers as REST, passed in connect message)
  - Subscribe to ticker + market_lifecycle for each market in the set

While connected:
  - On ticker update: upsert price in DB; emit price_moved event if threshold crossed
  - On market_lifecycle → settled: resolve position, record P&L, unsubscribe ticker

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
- Requires `websockets` or `httpx-ws` dependency (to be added when implemented — Phase 3).
- In paper mode, the WebSocket is still useful for accurate price tracking even though no real orders are submitted.

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
    category: str                    # "politics" | "technology" | ...
    close_time: datetime             # when market resolves

    # --- Price snapshot (changes frequently) ---
    yes_bid: float                   # current best bid for YES (0.0-1.0)
    yes_ask: float                   # current best ask for YES (0.0-1.0)
    mid_price: float                 # (bid + ask) / 2
    volume_24h: float                # liquidity proxy
    open_interest: float

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
    edge: float                      # estimated_probability - market.mid_price at signal time
    market_mid_at_signal: float      # snapshot of market price when signal was created
    direction: str                   # "YES" | "NO" | "SKIP"

    # --- Context ---
    reasoning: str                   # LLM explanation (logged, not traded on)
    sources: list[str]               # URLs used in RAG context
    social_sentiment_summary: str | None  # pre-summarized social signal (nullable)
    retrieval_hash: str              # hash of retrieved Document IDs — same hash = no new evidence in store

    # --- Provenance ---
    model_used: str                  # e.g. "claude-3-5-sonnet-20241022"
    prompt_version: str              # e.g. "politics-v2" — for tracking prompt changes
    trigger: str                     # "scheduled" | "price_moved" | "new_evidence" | "manual"
    created_at: datetime
    raw_context: str                 # full retrieved context (for debugging)
```

**Signal refresh triggers** (any of these causes a new Signal to be created):
1. **Scheduled** — re-analyze every N hours (configurable per strategy, e.g. every 12h for markets closing in >7 days, every 2h for markets closing in <48h)
2. **Price moved** — market mid-price shifted by more than a configurable threshold (e.g. ±5 cents) since the last signal; our edge may have changed materially
3. **New evidence detected** — retrieval hash differs from the last signal's hash, meaning new articles/posts were found
4. **Manual** — operator triggers re-analysis via CLI or dashboard

**What does NOT trigger a new signal:** a scheduled poll that returns the same retrieval hash as the last signal. If nothing new was retrieved, the LLM would produce the same output — no point calling the API.

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
    # pending:   order submitted, awaiting fill confirmation from Kalshi
    # open:      position filled and active
    # closed:    market resolved or manually exited
    # cancelled: order submitted but cancelled before fill

    # --- Filled after resolution ---
    exit_price: float | None
    exit_time: datetime | None
    resolution: int | None           # 1 = YES won, 0 = NO won
    pnl: float | None
    pnl_pct: float | None
```

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
    categories: list[str]            # which categories this strategy trades
    min_volume_24h: float            # liquidity filter
    max_days_to_close: int           # don't trade markets closing too soon/late
    min_days_to_close: int

    # --- Exit management (freqtrade-style) ---
    stoploss: float = -0.20
    # Trigger exit when position loses this fraction from entry price.
    # e.g. -0.20 = exit if unrealized loss exceeds 20%.
    # Enforced by the position monitor on every price poll — strategy cannot override.

    minimal_roi: dict[str, float] = field(default_factory=lambda: {"0": 0.30, "1440": 0.15, "10080": 0.05})
    # Time-based profit targets. Key = minutes since entry, value = required profit fraction.
    # Exit as soon as unrealized P&L % >= target for the elapsed time tier.
    # e.g. {"0": 0.30, "1440": 0.15, "10080": 0.05}:
    #   - exit immediately at 30% profit
    #   - exit at 15% profit after 1 day (1440 min)
    #   - exit at 5% profit after 1 week (10080 min)
    # Prediction markets move slowly — use hour/day-scale values, not minute-scale.
    # Set to {} to disable.

    trailing_stop: bool = False
    # If True, stoploss trails from the best mid-price achieved since entry
    # (i.e. the stop floor rises as the position profits, locking in gains).

    trailing_stop_positive: float | None = None
    # Once unrealized P&L crosses this threshold (e.g. 0.10 = 10% profit),
    # switch to a tighter trailing stop equal to trailing_stop_positive_offset
    # below the peak price. Encourages letting winners run while protecting profit.

    trailing_stop_positive_offset: float = 0.02
    # Tight trail applied once trailing_stop_positive is crossed.
    # e.g. 0.02 = trail 2% below the peak price once in profit.
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
    source_type: str                 # "news" | "reddit" | "twitter" | "kalshi_comment" | "manifold"
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

### CatalystRun + CatalystQuery

Each time the Catalyst Generator runs for a market it creates one `CatalystRun` (the generation event) and N `CatalystQuery` rows (the actual search strings). The ingestion scheduler always reads from the latest active run per market.

```python
@dataclass
class CatalystRun:
    id: str                      # UUID
    market_id: str               # FK → Market
    generation: int              # monotonically increasing per market (1, 2, 3 ...)
    llm_query_id: int            # FK → LLMQuery — audit trail for the catalyst LLM call
    is_active: bool              # False when market closed or no strategy is interested
    created_at: datetime


@dataclass
class CatalystQuery:
    id: str                      # UUID
    run_id: str                  # FK → CatalystRun
    query_text: str              # the actual search string, e.g. "February CPI release 2026"
    created_at: datetime
```

**Lifecycle rules:**
- A `CatalystRun` is created when a market is first selected (generation=1) and then daily (generation increments).
- On each new run, the previous run's `is_active` flag is left as-is; only the latest run is used for scheduling.
- `CatalystRun.is_active` is set to `False` when: (a) the market's `close_time` has passed, or (b) all registered strategies return `False` from `is_market_interesting()` for that market.
- Ingestion scheduler query: `SELECT cq.query_text FROM catalyst_queries cq JOIN catalyst_runs cr ON cr.id = cq.run_id WHERE cr.is_active = TRUE AND cr.id IN (SELECT MAX(id)... per market)`.

**Catalyst generation context (LLM prompt inputs):**
- **Generation 1:** market question + market metadata (close_time, category, description from Kalshi)
- **Generation 2+:** same as above, plus the top-K documents most recently retrieved for this market's existing catalyst queries (RAG pull). This lets the LLM refine or add catalysts based on what has actually been appearing in the news.

**Catalyst generation model:** Claude Haiku (cheap) — this is a reasoning task, not primary signal analysis. Logged to `llm_queries` with `query_type="catalyst_generation"`.

### DocumentMarketLink (join table)

Tracks which documents were retrieved for which market analysis, enabling retroactive analysis of what evidence was available when a signal was created.

```python
@dataclass
class DocumentMarketLink:
    document_id: str                 # FK → Document
    market_id: str                   # FK → Market
    signal_id: str | None            # FK → Signal this retrieval contributed to
    relevance_score: float           # cosine similarity score from vector search
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
    model_used: str                  # e.g. "claude-3-5-sonnet-20241022"
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

## 8. Strategy Plugin Interface

Strategies are Python classes that implement `IPredictionStrategy`. The design mirrors freqtrade's `IStrategy` — entry signals, exit signals, stoploss, and ROI targets are all strategy-owned. The framework enforces hard caps on top; strategy logic defines the alpha.

| freqtrade concept | freqpred equivalent |
|---|---|
| `populate_entry_trend()` | `should_trade(signal, market) -> bool` |
| `populate_exit_trend()` | `should_exit(position, signal, market) -> bool` |
| `stoploss = -0.10` | `config.stoploss = -0.20` |
| `minimal_roi = {"0": 0.04}` | `config.minimal_roi = {"0": 0.30, "1440": 0.15}` |
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
                categories=["politics"],
                stoploss=-0.20,
                minimal_roi={"0": 0.30, "1440": 0.15},
                trailing_stop=True,
                ...
            )

            def should_trade(self, signal: Signal, market: Market) -> bool:
                return signal.edge >= self.config.min_edge

            def position_size(self, signal: Signal, bankroll: float) -> float:
                kelly = signal.edge / (1 - signal.estimated_probability)
                return bankroll * kelly * self.config.kelly_fraction

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

    @abstractmethod
    def position_size(self, signal: Signal, bankroll: float) -> float:
        """Return dollar amount to risk on this position (before risk capping)."""
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
        volume, days-to-close). Override for custom market selection logic.
        """
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
3. **Minimal ROI** (`config.minimal_roi`) — time-based profit targets
4. **Custom exit** (`strategy.custom_exit()`) — strategy-defined special conditions
5. **Signal exit** (`strategy.should_exit()`) — called only after LLM re-analysis (on price-triggered signal refreshes, not every poll)
6. **Market resolution** — market closes, position settled at $1.00 or $0.00

If none of these conditions fire, the position is held.

### Exit Reason Tagging

Every closed position records an `exit_reason` string for analysis:

| Exit reason | Source |
|---|---|
| `"stoploss"` | Hard stoploss hit |
| `"trailing_stop"` | Trailing stoploss hit |
| `"roi"` | Minimal ROI target met |
| `"custom_exit:<tag>"` | `custom_exit()` returned a tag |
| `"signal"` | `should_exit()` returned True |
| `"market_resolved"` | Market paid out at resolution |

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

```
Market Watcher upserts active markets into DB
      │
      ▼
┌─────────────────┐
│  Market         │  Reads all active markets from DB.
│  Selector       │  Calls strategy.is_market_interesting(market)
│                 │  on each registered strategy. Selects markets
│                 │  where any strategy returns True.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Catalyst       │  For each selected market:
│  Generator      │  - First seen: LLM (Haiku) derives 3-5 search
│                 │    queries from market question + metadata.
│                 │  - Daily re-run: same, but also pulls recent docs
│                 │    from RAG (what has been found so far) so the
│                 │    LLM can refine or add catalysts.
│                 │  Writes CatalystRun + CatalystQuery rows to DB.
│                 │  Logs to llm_queries (query_type=catalyst_generation).
│                 │  Deactivates catalysts for closed/unselected markets.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Ingestion      │  Reads latest active CatalystQuery rows from DB.
│  Scheduler      │  For each query: runs Tavily + NewsAPI + Reddit.
│  (every 30 min) │  Tracks last-run per market in Redis.
│                 │  One fetcher failing does not stop others.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dedup &        │  Check source_url against Document store.
│  Store          │  Skip if URL known + content_hash unchanged.
│                 │  New/changed docs: clean, generate embedding
│                 │  (sentence-transformers), insert into Document store.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Social         │  For Reddit posts only: cheap LLM pass (Haiku)
│  Pre-summarizer │  compresses raw posts into structured sentiment
│  (if social)    │  summary before storing.
└─────────────────┘
```

### Phase 2: Signal Analysis (triggered)

Runs when a signal refresh trigger fires (scheduled, price moved, new evidence, manual). This is where the expensive LLM call happens.

```
Signal trigger fires for a market
      │
      ▼
┌─────────────────┐
│  Vector Search  │  Embed the market question (sentence-transformers).
│  (RAG retrieval)│  Semantic search against Document store:
│                 │  - Filter: category match + published_at recency
│                 │  - Rank: cosine similarity to market question
│                 │  - Select: top-K documents (default K=10)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Retrieval Hash │  Hash the IDs of top-K retrieved documents.
│  Check          │  If hash == last Signal's retrieval_hash:
│                 │  no new evidence → skip, no LLM call needed.
└────────┬────────┘
         │ (hash changed — new evidence exists)
         ▼
┌─────────────────┐
│  LLM Analysis   │  Structured prompt with retrieved docs as context.
│  (Claude Sonnet)│  Asks for:
│                 │  - Probability estimate (0.0-1.0)
│                 │  - Confidence score (0.0-1.0)
│                 │  - Key supporting evidence (with doc citations)
│                 │  - Key counter-evidence
│                 │  - Reasoning summary
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Signal         │  Validate output, compute edge vs market price,
│  Creation       │  write Signal + DocumentMarketLinks, update
│                 │  Market.current_signal_id.
└─────────────────┘
```

### Retrieval Sources

#### Structured News
| Source | Use Case | Priority |
|---|---|---|
| **Tavily Search API** | Fresh web search per market question | Primary |
| **NewsAPI** | Structured article archive for less-breaking topics | Secondary |
| **Kalshi market metadata** | Market description + linked sources from the exchange | Always included |
| **GDELT** | High-volume event archive, especially for politics | Supplementary |

#### Social & Community Signals
| Source | Use Case | Notes |
|---|---|---|
| **Reddit** | Subreddit sentiment for relevant communities | No credentials required — uses public JSON API (`reddit.com/r/{sub}/search.json`); target subs per category (see below) |
| **Twitter/X API** | Real-time public sentiment on market topics | Expensive ($100–$5000/mo tier); treat as optional enrichment |
| **Kalshi market comments** | Crowd reasoning directly on the market in question | Already fetched with market metadata |
| **Manifold Markets** | Community probability estimates on overlapping questions | Free API; useful as an independent signal cross-check |

**Reddit subreddit targets by category:**

| Category | Subreddits |
|---|---|
| US Politics | r/politics, r/PoliticalDiscussion, r/neutralpolitics |
| Technology | r/technology, r/MachineLearning, r/singularity |
| Fintech | r/investing, r/wallstreetbets, r/stocks, r/fintech |
| Prediction markets | r/predictionmarkets, r/Kalshi |

#### Social Signal Handling

Social content is noisier than structured news and requires preprocessing before it reaches the LLM:

```
Raw posts (Reddit/Twitter)
         │
         ▼
┌─────────────────┐
│  Aggregator     │  Filter by recency, upvotes/engagement,
│                 │  deduplicate, remove low-signal content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pre-summarizer │  Lightweight LLM pass to compress 50 posts
│  (cheap model)  │  into a structured sentiment summary:
│                 │  {sentiment, key_claims, notable_threads}
└────────┬────────┘
         │
         ▼
  Included as one context block in main LLM analysis
  (alongside news articles and Kalshi metadata)
```

This two-pass approach keeps social signal cost-efficient: a cheap summarization pass (haiku/mini) collapses noisy social data before it hits the primary reasoning model. The main LLM sees a structured social summary, not raw posts.

**Social signal weight:** The LLM prompt explicitly instructs the model to treat social sentiment as weak/corroborating evidence, not primary evidence. Crowd sentiment without corroborating news should not be sufficient to cross a trade threshold alone.

### LLM Configuration

- **Primary model:** `claude-3-5-sonnet` — best reasoning/cost tradeoff
- **Output format:** Structured JSON via tool use (not free-form text parsing)
- **Prompt versioning:** Prompts are versioned and stored; every signal logs the prompt version used
- **Caching:** Signal results cached by `(market_id, prompt_version, retrieval_hash)` — same market won't be re-analyzed unless new evidence is retrieved

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
| Max total exposure | 40% of bankroll | Sum of all open positions |
| Min edge to trade | 10% | Absolute floor; strategy can raise, not lower |
| Max open positions | 20 | Prevents overextension |

### Position Sizing

Default: **fractional Kelly criterion**

```
kelly_fraction = edge / (1 - p_win)
position_size = bankroll × kelly_fraction × config.kelly_multiplier
```

Where `config.kelly_multiplier` defaults to `0.25` (quarter-Kelly). Full Kelly is mathematically optimal but practically too aggressive; quarter-Kelly is standard for new strategies without long track records.

### Circuit Breakers

- Daily loss > 15%: halt all new positions for 24 hours, alert via Telegram
- Total drawdown > 30%: halt all trading, require manual restart
- LLM API errors > 3 consecutive: halt signal pipeline, alert

---

## 11. Infrastructure

### Cloud: AWS

| Service | Use |
|---|---|
| **ECS Fargate** | Runs the main freqpred service (containerized, always-on) |
| **RDS (Postgres + pgvector)** | Ledger, positions, signals, market history, Document store + embeddings |
| **ElastiCache (Redis)** | Signal cache, rate limiting, ingestion job dedup |
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

Built with **FastAPI** (backend) + **React** (frontend), served via ECS.

**Pages:**

1. **Signal Feed** — live stream of new signals with market question, our probability, market price, edge, direction
2. **Open Positions** — current paper/live positions with unrealized P&L, filterable by strategy
3. **Ledger** — resolved positions, actual P&L, running totals, filterable by strategy/confidence tier
4. **Calibration** — scatter plot of estimated probability vs. resolution rate; Brier score trend
5. **LLM Cost & Audit** — daily/weekly spend charts, cost by query type and strategy, query log with full prompt/response drilldown, budget burn rate vs. daily cap
6. **Strategy Config** — view/edit active strategy parameters (no code changes needed for threshold tuning)
7. **System Health** — API status, error rates, circuit breaker state, LLM budget circuit breaker status

### Telegram / Discord Alerts

| Event | Alert |
|---|---|
| New signal above threshold | "📊 NEW SIGNAL: [question] — Our prob: 71%, Market: 54%, Edge: +17%" |
| Position opened | "🟢 PAPER TRADE: YES on [question] @ $0.54, size: $X" |
| Market resolved | "✅ WIN: [question] resolved YES. P&L: +$X (+34%)" or "❌ LOSS..." |
| Circuit breaker triggered | "🚨 CIRCUIT BREAKER: Daily loss limit hit. Trading halted." |
| Daily digest | Morning summary: open positions, yesterday's P&L, calibration score |

---

## 13. Development Phases

### Phase 1: Signal Engine MVP ✅
*Goal: produce scored signals for active Kalshi markets*

- [x] Kalshi API client (market fetch, no trading yet)
- [x] Document store schema (Postgres + pgvector extension)
- [x] Ingestion pipeline: Tavily + NewsAPI fetchers → dedup → embed (sentence-transformers) → store
- [x] Ingestion pipeline: Reddit fetcher + social pre-summarizer → store
- [ ] Twitter/X fetcher (optional — gated on API cost decision)
- [x] Strategy interface (`IPredictionStrategy`) + `ConservativeDefault`, `PoliticsEdgeStrategy`, `TechNewsStrategy` strategies
- [x] Market Selector: reads active markets, calls `strategy.is_market_interesting()` per registered strategy
- [x] Catalyst Generator: LLM (Haiku) derives 3–5 search queries per selected market; writes `CatalystRun` + `CatalystQuery` to DB; RAG-informed on re-runs; deactivates on market close/deselection
- [x] Ingestion Scheduler: reads latest active `CatalystQuery` rows from DB; drives fetchers; tracks last-run per market in Redis
- [x] RAG retriever: vector search against Document store for a given market question
- [x] LLM signal pipeline: retrieve docs → hash check → Claude analysis → Signal creation
- [x] LLM audit logging (LLMQuery table — every call logged with cost)
- [x] LLM budget circuit breaker (configurable daily spend cap — enforced in `LLMClient.complete()`)
- [x] Signal scoring and logging to Postgres
- [x] Basic CLI: `freqpred run --strategy ConservativeDefault --mode signal-only`
- [x] Docker + local dev setup

**Done when:** System runs continuously, produces signals for 20+ markets/day, signals are logged with full context for review.

---

### Phase 2: Paper Trading + Calibration
*Goal: validate signal quality with simulated trades*

**Ingestion improvements (address before running at scale):**
- [ ] NewsAPI rate limiting: developer accounts are capped at 100 req/24h. Either upgrade to a paid plan, reduce query frequency, or disable NewsAPI and rely on Tavily + Reddit only. Running 5 queries × 3 markets per cycle exhausts the quota in one ingestion pass.

**RAG improvements:**
- [ ] `document_market_links.relevance_score` is currently a rank-based proxy (`1/rank`). Swap to actual cosine similarity scores from pgvector so calibration analysis can weight document influence by true semantic relevance.

- [ ] Order Manager (paper mode)
- [ ] Ledger (positions, resolutions, P&L)
- [ ] Strategy exit interface — `should_exit()`, `custom_exit()`, `stoploss`, `minimal_roi`, `trailing_stop` on `IPredictionStrategy` and `StrategyConfig`
- [ ] Position monitor — background loop checking all open positions on each price poll; simulates paper exits when stoploss/ROI/signal conditions are met; logs `exit_reason`
- [ ] Calibration metrics (Brier score, calibration curve)
- [ ] Web dashboard (signal feed + ledger views)
- [ ] Telegram alerts

**Done when:** 100+ markets resolved or exited with logged signals. Calibration score measured. Exit behavior observable from ledger. Decision made: is the signal real?

**Go/no-go criteria for Phase 3:**
- Brier score < 0.20 (better than naive baseline)
- Positive calibration: 60-70% estimated → 60-70% resolution rate
- Positive simulated ROI over 100+ trades

---

### Phase 3: Live Trading
*Goal: real capital, controlled risk*

- [ ] Kalshi order execution (real API) — entry orders via REST
- [ ] Real exit order execution — when position monitor fires an exit condition, submit a sell order via Kalshi REST API instead of simulating it
- [ ] Hard cap enforcement in Order Manager
- [ ] Position Watcher: Kalshi WebSocket `ticker` subscription drives real-time position monitor for open positions — replaces the poll-based paper monitor with low-latency price feed; same exit logic, real sell orders (see §6a)
- [ ] Position Watcher: `market_lifecycle` subscription for resolution events — closes positions at settlement price
- [ ] Production AWS deployment (ECS, RDS, Secrets Manager)
- [ ] GitHub Actions CI/CD pipeline
- [ ] Full dashboard (all pages)
- [ ] Circuit breakers + runbook for incidents

**Done when:** System executes real trades automatically with enforced risk limits, monitored via dashboard and Telegram.

---

## 14. Open Questions

1. **Kalshi sandbox API** — Does Kalshi offer a paper trading / sandbox environment? If not, paper trading will simulate orders against real prices without submitting to the exchange.
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
│   │   ├── watcher.py           # polling loop: price refresh, staleness detection
│   │   └── models.py            # Market, Order, Position dataclasses
│   ├── ingestion/
│   │   ├── selector.py          # market selector: calls strategy.is_market_interesting()
│   │   ├── catalyst_generator.py# LLM (Haiku) derives catalyst queries per market; manages CatalystRun/CatalystQuery
│   │   ├── scheduler.py         # background ingestion job: reads catalyst queries, drives fetchers
│   │   ├── fetchers/
│   │   │   ├── tavily.py        # Tavily Search API fetcher
│   │   │   ├── newsapi.py       # NewsAPI fetcher
│   │   │   ├── gdelt.py         # GDELT fetcher
│   │   │   ├── reddit.py        # Reddit API fetcher
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
│   │   ├── calibration.py       # Brier score, calibration curve
│   │   └── reporting.py         # daily digest generation
│   ├── dashboard/
│   │   ├── api/                 # FastAPI routes
│   │   └── ui/                  # React frontend
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
