"""LLMQuery audit log dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class LLMQuery:
    # auto-increment PK (int, not UUID)
    id: int | None

    # When & why
    timestamp: datetime
    strategy: str
    query_type: str  # "market_analysis" | "social_summarization" | "movement_prediction" | "daily_digest"
    model_used: str
    prompt_version: str

    # Full request/response
    prompt: str
    response: str

    # Cost
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost_usd: float

    # Extracted outputs
    latency_ms: int
    success: bool

    market_id: str | None = None
    signal_id: str | None = None
    confidence_extracted: float | None = None
    decision_extracted: str | None = None
    error_message: str | None = None
