"""Pydantic response schemas for the dashboard API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class SignalOut(BaseModel):
    id: str
    market_id: str
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


class SignalDetailOut(SignalOut):
    document_links: list[DocumentLinkOut]


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
    resolution: int | None
    pnl: float | None
    pnl_pct: float | None
    created_at: datetime


class PositionListResponse(BaseModel):
    items: list[PositionOut]
    total: int


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
    naive_brier_score: float
    n_samples: int
    buckets: list[CalibrationBucketOut]


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
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str                          # "ok" | "degraded"
    db: str                              # "connected" | "error"
    redis: str                           # "connected" | "not_configured" | "error"
    open_positions: int
    llm_daily_budget_remaining_usd: float
