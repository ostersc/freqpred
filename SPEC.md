# freqpred — Project Specification

> A framework for LLM-driven prediction market trading, modeled on freqtrade's architecture.

**Version:** 0.1-draft
**Last updated:** 2026-03-15
**Status:** Pre-development

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

- [ ] Fetch active prediction markets from Kalshi and score them with an LLM signal pipeline
- [ ] Implement a code-driven strategy plugin interface (`IPredictionStrategy`)
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

## 5. Market Categories (v1)

Initial focus:

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
| **Market Watcher** | Polls Kalshi for active markets, filters by category, enqueues for analysis |
| **Signal Pipeline** | Retrieves news context, runs LLM analysis, returns probability estimate |
| **Strategy Engine** | Applies `IPredictionStrategy` plugins to signal output, decides trade/size/skip |
| **IMarketClient** | Abstract interface over Kalshi (and future platforms); handles orders, positions, balance |
| **Order Manager** | Executes paper or live trades; enforces hard risk caps before any order |
| **Ledger** | Immutable trade log; records every signal, position, and resolution outcome |
| **Dashboard** | Web UI for monitoring; Telegram/Discord for push alerts |

---

## 7. Core Data Models

### Market
```python
@dataclass
class Market:
    id: str                          # Kalshi market ID
    platform: str                    # "kalshi"
    question: str                    # "Will X happen by Y?"
    category: str                    # "politics" | "technology" | ...
    close_time: datetime             # when market resolves
    yes_bid: float                   # current best bid for YES (0.0-1.0)
    yes_ask: float                   # current best ask for YES (0.0-1.0)
    mid_price: float                 # (bid + ask) / 2
    volume_24h: float                # liquidity proxy
    open_interest: float
    metadata: dict                   # raw platform data
```

### Signal
```python
@dataclass
class Signal:
    market_id: str
    estimated_probability: float     # LLM's estimate (0.0-1.0)
    confidence: float                # LLM self-reported confidence (0.0-1.0)
    edge: float                      # estimated_probability - market.mid_price
    direction: str                   # "YES" | "NO" | "SKIP"
    reasoning: str                   # LLM explanation (logged, not traded on)
    sources: list[str]               # URLs used in RAG context
    social_sentiment_summary: str | None  # pre-summarized social signal (nullable)
    model_used: str                  # e.g. "claude-3-5-sonnet-20241022"
    created_at: datetime
    raw_context: str                 # full retrieved context (for debugging)
```

### Position
```python
@dataclass
class Position:
    id: str
    market_id: str
    signal_id: str
    direction: str                   # "YES" | "NO"
    contracts: int
    entry_price: float
    entry_time: datetime
    mode: str                        # "paper" | "live"

    # Filled after resolution
    exit_price: float | None
    exit_time: datetime | None
    resolution: int | None           # 1 = YES won, 0 = NO won
    pnl: float | None
    pnl_pct: float | None
```

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
```

---

## 8. Strategy Plugin Interface

Strategies are Python classes that implement `IPredictionStrategy`. This mirrors freqtrade's `IStrategy` design.

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
                ...
            )

            def should_trade(self, signal: Signal, market: Market) -> bool:
                return signal.edge >= self.config.min_edge

            def position_size(self, signal: Signal, bankroll: float) -> float:
                kelly = signal.edge / (1 - signal.estimated_probability)
                return bankroll * kelly * self.config.kelly_fraction
    """

    config: StrategyConfig

    @abstractmethod
    def should_trade(self, signal: Signal, market: Market) -> bool:
        """Return True if this signal warrants opening a position."""
        ...

    @abstractmethod
    def position_size(self, signal: Signal, bankroll: float) -> float:
        """Return dollar amount to risk on this position."""
        ...

    def filter_markets(self, markets: list[Market]) -> list[Market]:
        """
        Pre-filter markets before signal analysis.
        Default implementation applies config filters.
        Override for custom filtering logic.
        """
        return [
            m for m in markets
            if m.category in self.config.categories
            and m.volume_24h >= self.config.min_volume_24h
            and self.config.min_days_to_close
                <= (m.close_time - datetime.utcnow()).days
                <= self.config.max_days_to_close
        ]

    def on_resolution(self, position: Position) -> None:
        """
        Optional hook called when a market resolves.
        Use for logging, alerting, or adaptive logic.
        """
        pass
```

### Bundled Strategies

| Strategy | Description |
|---|---|
| `PoliticsEdgeStrategy` | US politics markets, min edge 0.18, conservative Kelly 0.25x |
| `TechNewsStrategy` | Technology/fintech markets, skewed toward shorter-dated markets |
| `ConservativeDefault` | High-confidence only (0.80+), tiny sizing — good starting point |

---

## 9. LLM Signal Pipeline

### Flow

```
Market question
      │
      ▼
┌─────────────────┐
│  Context        │  Retrieve relevant news articles, Kalshi market
│  Retrieval      │  description, recent resolution history for
│  (RAG)          │  similar markets
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Analysis   │  Structured prompt asking for:
│  (Claude)       │  - Probability estimate (0.0-1.0)
│                 │  - Confidence score (0.0-1.0)
│                 │  - Key supporting evidence
│                 │  - Key counter-evidence
│                 │  - Reasoning summary
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Signal         │  Validate output, compute edge vs market price,
│  Validation     │  cache result, emit Signal object
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
| **Reddit API** | Subreddit sentiment for relevant communities | Free with OAuth; target subs per category (see below) |
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
| **RDS (Postgres)** | Ledger, positions, signals, market history |
| **ElastiCache (Redis)** | Signal cache, rate limiting |
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
2. **Open Positions** — current paper/live positions with unrealized P&L
3. **Ledger** — resolved positions, actual P&L, running totals
4. **Calibration** — scatter plot of estimated probability vs. resolution rate; Brier score trend
5. **Strategy Config** — view/edit active strategy parameters (no code changes needed for threshold tuning)
6. **System Health** — API status, error rates, circuit breaker state

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

### Phase 1: Signal Engine MVP
*Goal: produce scored signals for active Kalshi markets*

- [ ] Kalshi API client (market fetch, no trading yet)
- [ ] Tavily + NewsAPI retrieval layer
- [ ] Reddit social signal fetcher + pre-summarizer (two-pass LLM)
- [ ] Twitter/X signal fetcher (optional — gated on API cost decision)
- [ ] LLM signal pipeline (Claude, structured output)
- [ ] Signal scoring and logging to Postgres
- [ ] Basic CLI: `freqpred run --strategy ConservativeDefault --mode signal-only`
- [ ] Docker + local dev setup

**Done when:** System runs continuously, produces signals for 20+ markets/day, signals are logged with full context for review.

---

### Phase 2: Paper Trading + Calibration
*Goal: validate signal quality with simulated trades*

- [ ] `IPredictionStrategy` plugin interface
- [ ] `PoliticsEdgeStrategy` and `TechNewsStrategy` implementations
- [ ] Order Manager (paper mode)
- [ ] Ledger (positions, resolutions, P&L)
- [ ] Calibration metrics (Brier score, calibration curve)
- [ ] Web dashboard (signal feed + ledger views)
- [ ] Telegram alerts

**Done when:** 100+ markets resolved with logged signals. Calibration score measured. Decision made: is the signal real?

**Go/no-go criteria for Phase 3:**
- Brier score < 0.20 (better than naive baseline)
- Positive calibration: 60-70% estimated → 60-70% resolution rate
- Positive simulated ROI over 100+ trades

---

### Phase 3: Live Trading
*Goal: real capital, controlled risk*

- [ ] Kalshi order execution (real API)
- [ ] Hard cap enforcement in Order Manager
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
│   │   └── models.py            # Market, Order, Position dataclasses
│   ├── signal/
│   │   ├── pipeline.py          # orchestrates retrieval + LLM analysis
│   │   ├── retrieval.py         # Tavily, NewsAPI, GDELT fetchers
│   │   ├── social.py            # Reddit, Twitter/X, Manifold fetchers + pre-summarizer
│   │   ├── llm.py               # Claude client, structured output
│   │   ├── cache.py             # signal result caching
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
