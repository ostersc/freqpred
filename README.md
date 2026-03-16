# freqpred

**LLM-driven prediction market trading framework — freqtrade for prediction markets.**

freqpred uses retrieval-augmented LLM analysis to estimate the "true" probability of prediction market outcomes, identifies edges against market-implied prices, and executes trades with systematic risk controls.

> ⚠️ **Status: Pre-development.** Specification complete, implementation in progress.

---

## What it does

1. **Monitors** active markets on Kalshi (US Politics, Technology/Fintech)
2. **Retrieves** relevant news context via Tavily and NewsAPI (RAG)
3. **Estimates** event probability using Claude with structured output
4. **Identifies** markets where LLM probability diverges from market price
5. **Trades** via a pluggable strategy interface with hard risk controls
6. **Tracks** calibration — are our probability estimates actually accurate?

## Philosophy

Backtesting prediction markets is unreliable: LLMs have seen market resolutions in training data, creating unavoidable look-ahead bias. Instead, freqpred validates strategies through **paper trading + calibration tracking** — simulating trades against real market prices and measuring Brier score over resolved markets.

## Architecture

```
Market Watcher → Signal Pipeline (RAG + LLM) → Strategy Engine → Order Manager → Ledger
                                                                         ↕
                                                              Dashboard + Alerts
```

See [SPEC.md](SPEC.md) for full architecture, data models, strategy interface, and development roadmap.

## Platforms

| Platform | Status |
|---|---|
| Kalshi | v1 — primary |
| Interactive Brokers (event contracts) | v2 — planned |

## Development Phases

- **Phase 1:** Signal engine — score active markets, log to Postgres
- **Phase 2:** Paper trading — simulate trades, measure calibration
- **Phase 3:** Live trading — real orders with hard risk caps on AWS ECS

## Tech Stack

- **Backend:** Python (FastAPI)
- **Frontend:** React
- **Database:** PostgreSQL (RDS)
- **Cache:** Redis (ElastiCache)
- **Infrastructure:** AWS ECS + Fargate
- **LLM:** Claude (Anthropic)
- **News Retrieval:** Tavily, NewsAPI, GDELT

## License

MIT
