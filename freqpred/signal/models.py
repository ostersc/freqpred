"""Signal dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    id: str
    market_id: str

    # Estimate
    estimated_probability: float
    confidence: float
    edge: float
    market_mid_at_signal: float
    direction: str   # "YES" | "NO" | "SKIP"

    # Context
    reasoning: str
    sources: list[str]
    retrieval_hash: str

    # Provenance
    model_used: str
    prompt_version: str
    trigger: str     # "scheduled" | "price_moved" | "new_evidence" | "manual"
    created_at: datetime
    raw_context: str

    social_sentiment_summary: str | None = None
