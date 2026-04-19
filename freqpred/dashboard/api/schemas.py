"""Pydantic response schemas for the dashboard API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class SignalOut(BaseModel):
    id: str
    market_id: str
    market_question: str | None
    estimated_probability: float
    confidence: float
    edge: float
    market_mid_at_signal: float
    direction: str
    reasoning: str
    sources: list[str]
    retrieval_hash: str
    model_used: str
    prompt_version: str
    trigger: str
    created_at: datetime
    social_sentiment_summary: str | None


class DocumentLinkOut(BaseModel):
    document_id: str
    source_url: str
    title: str
    relevance_score: float
    source_type: str
    source_name: str
    published_at: datetime | None
    fetched_at: datetime
    summary: str | None
    body_excerpt: str  # first 400 chars of body, for display when summary is absent


class SignalAssessmentOut(BaseModel):
    trust_score: float
    size_multiplier: float
    verdict: str
    reasoning: str
    key_factors: list[str]
    warnings: list[str]
    source_breakdown: list[dict[str, Any]]
    similar_market_summary: dict[str, Any]
    llm_query_id: int | None
    created_at: datetime


class SignalDetailOut(SignalOut):
    document_links: list[DocumentLinkOut]
    assessment: SignalAssessmentOut | None


class SignalListResponse(BaseModel):
    items: list[SignalOut]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


class PositionOut(BaseModel):
    id: str
    market_id: str
    signal_id: str
    strategy_name: str
    strategy_version: str
    signal_confidence: float
    signal_edge: float
    signal_estimated_prob: float
    direction: str
    contracts: int
    entry_price: float
    entry_time: datetime
    mode: str
    status: str
    exit_price: float | None
    exit_time: datetime | None
    exit_reason: str | None
    resolution: int | None
    pnl: float | None
    pnl_pct: float | None
    unrealized_pnl: float | None
    unrealized_pnl_pct: float | None
    created_at: datetime


class PositionListResponse(BaseModel):
    items: list[PositionOut]
    total: int


class PositionDetailOut(PositionOut):
    market_question: str | None
    current_mid: float | None
    entry_signal: SignalDetailOut
    market_signals: list[SignalOut]


# ---------------------------------------------------------------------------
# Strategy decisions (exited-position post-mortem)
# ---------------------------------------------------------------------------


class StrategyDecisionOut(PositionOut):
    """A closed position enriched with counterfactual and entry-efficiency metrics."""

    market_question: str | None
    market_result: str | None                   # "yes" | "no" | None (unresolved)

    # Exit decision counterfactual (None if market unresolved).
    # counterfactual_pnl  = (our_side_win_value - entry_price)   [per contract]
    # exit_delta_vs_hold  = (exit_price - our_side_win_value)    [per contract]
    counterfactual_pnl_per_contract: float | None
    counterfactual_pnl_usd: float | None
    exit_delta_per_contract: float | None
    exit_delta_usd: float | None

    # Entry efficiency vs best prior signal with edge > 0 for same (market, direction).
    # best_prior_ask          = min(signals.market_ask_at_signal) prior to entry_time
    # entry_efficiency_delta  = (best_prior_ask - entry_price)    [per contract]
    # Negative = we paid more than the earlier opportunity.
    best_prior_ask: float | None
    entry_efficiency_per_contract: float | None
    entry_efficiency_usd: float | None


class StrategyDecisionListResponse(BaseModel):
    items: list[StrategyDecisionOut]
    total: int
    limit: int
    offset: int
    # Populated from the full closed-position set so filter dropdowns are stable
    # regardless of the current filter selection.
    distinct_strategies: list[str]
    distinct_exit_reasons: list[str]


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class LedgerResponse(BaseModel):
    open_count: int
    total_exposure_usd: float
    daily_pnl_usd: float
    all_time_pnl_usd: float


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class CalibrationBucketOut(BaseModel):
    lower: float
    upper: float
    count: int
    mean_estimated_prob: float
    actual_resolution_rate: float


class CalibrationResponse(BaseModel):
    brier_score: float
    market_brier_score: float
    n_samples: int
    buckets: list[CalibrationBucketOut]
    market_buckets: list[CalibrationBucketOut]
    available_categories: list[str]


class SourceQualityScoreOut(BaseModel):
    source_name: str
    market_category: str | None
    weighted_brier: float
    overall_brier: float
    n_signals: int
    total_doc_uses: int
    computed_at: datetime


class SourceQualityListResponse(BaseModel):
    items: list[SourceQualityScoreOut]


# ---------------------------------------------------------------------------
# LLM cost
# ---------------------------------------------------------------------------


class LLMCostResponse(BaseModel):
    today_usd: float
    weekly_usd: float
    daily_cap_usd: float
    pct_used: float
    by_query_type: dict[str, float]


# ---------------------------------------------------------------------------
# LLM query list / detail
# ---------------------------------------------------------------------------


class LLMQueryOut(BaseModel):
    id: int
    timestamp: datetime
    query_type: str
    market_id: str | None
    model_used: str
    tokens_total: int
    cost_usd: float
    latency_ms: int
    success: bool


class LLMQueryListResponse(BaseModel):
    items: list[LLMQueryOut]
    total: int
    limit: int
    offset: int


class LLMQueryDetailOut(LLMQueryOut):
    prompt: str
    response: str
    error_message: str | None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str                          # "ok" | "degraded"
    db: str                              # "connected" | "error"
    open_positions: int
    llm_daily_budget_remaining_usd: float


# ---------------------------------------------------------------------------
# Strategy config
# ---------------------------------------------------------------------------


class StrategyConfigOut(BaseModel):
    name: str
    min_edge: float
    min_confidence: float
    kelly_fraction: float
    max_exposure_per_market: float
    categories: list[str]
    min_volume_24h: float
    max_days_to_close: float
    min_days_to_close: float
    stoploss: float
    trailing_stop: bool
    trailing_stop_positive: float | None
    trailing_stop_positive_offset: float
    min_mid_price: float | None
    max_mid_price: float | None
    max_spread: float | None
    block_reentry_after_stoploss: bool
    stoploss_cooldown_hours: float
    assessment_scale_min: float
    assessment_scale_max: float
    similar_market_min_signals: int
    similar_market_min_trades: int


class StrategyConfigUpdateRequest(BaseModel):
    """Mutable fields accepted by PUT /api/strategy/config.

    ``name`` and ``categories`` are intentionally included so the endpoint can
    detect them and return a 422 explaining they require a restart.
    All other fields are truly mutable at runtime.
    """

    model_config = ConfigDict(extra="forbid")

    # Immutable — included only to return a clear 422 if a client sends them.
    name: str | None = None
    categories: list[str] | None = None

    # Mutable
    min_edge: float | None = None
    min_confidence: float | None = None
    kelly_fraction: float | None = None
    max_exposure_per_market: float | None = None
    min_volume_24h: float | None = None
    max_days_to_close: float | None = None
    min_days_to_close: float | None = None
    stoploss: float | None = None
    trailing_stop: bool | None = None
    trailing_stop_positive: float | None = None
    trailing_stop_positive_offset: float | None = None
    min_mid_price: float | None = None
    max_mid_price: float | None = None
    max_spread: float | None = None
    block_reentry_after_stoploss: bool | None = None
    stoploss_cooldown_hours: float | None = None
    assessment_scale_min: float | None = None
    assessment_scale_max: float | None = None
    similar_market_min_signals: int | None = None
    similar_market_min_trades: int | None = None


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------


class MarketOut(BaseModel):
    id: str
    question: str
    status: str
    yes_bid: float
    yes_ask: float
    mid_price: float
    volume_24h: float
    close_time: datetime
    last_fetched_at: datetime
    current_signal_id: str | None


class MarketDetailOut(MarketOut):
    current_signal: SignalOut | None


class MarketListResponse(BaseModel):
    items: list[MarketOut]
    total: int
    limit: int
    offset: int


class AnalyzeResponse(BaseModel):
    signal: SignalOut
    cached: bool


# ---------------------------------------------------------------------------
# System health
# ---------------------------------------------------------------------------


class CircuitBreakerStateOut(BaseModel):
    trading_halted: bool
    reason: str | None
    daily_loss_pct: float
    daily_loss_limit_pct: float
    daily_loss_window_start: datetime  # lower bound of the loss window the risk engine enforces
    daily_loss_ack_at: datetime | None  # last /start acknowledgement; null if never acked
    llm_budget_used_usd: float
    llm_budget_cap_usd: float


class WebSocketStateOut(BaseModel):
    connected: bool | None          # null = not applicable (paper/standalone)
    subscribed_markets: int | None
    last_message_at: datetime | None


class ApiErrorStateOut(BaseModel):
    kalshi_errors_last_hour: int
    llm_errors_last_hour: int
    consecutive_llm_errors: int | None  # null = not available in standalone mode


class SystemHealthResponse(BaseModel):
    run_state: str                  # "running" | "paused" | "stopped"
    mode: str                       # "paper" | "live"
    circuit_breakers: CircuitBreakerStateOut
    websocket: WebSocketStateOut
    api_errors: ApiErrorStateOut
    pending_orders: int
    open_positions: int
    db_ok: bool
    uptime_seconds: int
