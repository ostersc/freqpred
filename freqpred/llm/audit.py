"""LLMQuery logging, cost tracking, and budget circuit breaker.

IMPORTANT: Every LLM call must log a row here before returning,
even failed calls (success=False). This is non-negotiable.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.llm.models import LLMQueryRow


class LLMBudgetExceededError(Exception):
    """Raised when the configured daily LLM spend cap has been reached."""


async def get_daily_spend_usd(session: AsyncSession) -> float:
    """Return the total LLM spend in USD for today (UTC).

    Args:
        session: Open async session.

    Returns:
        Sum of cost_usd for all llm_queries rows timestamped today, or 0.0.
    """
    # Half-open UTC day range instead of date(timestamp) == today: a range
    # predicate can use ix_llm_queries_timestamp; date() over the column cannot.
    now = datetime.now(UTC)
    day_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
    result = await session.execute(
        select(func.coalesce(func.sum(LLMQueryRow.cost_usd), 0.0)).where(
            LLMQueryRow.timestamp >= day_start,
            LLMQueryRow.timestamp < day_start + timedelta(days=1),
        )
    )
    return float(result.scalar_one())

log = structlog.get_logger(__name__)


async def log_llm_query(
    session: AsyncSession,
    *,
    strategy: str,
    query_type: str,
    model_used: str,
    prompt_version: str,
    prompt: str,
    response: str,
    tokens_input: int,
    tokens_output: int,
    cost_usd: float,
    latency_ms: int,
    success: bool,
    market_id: str | None = None,
    signal_id: str | None = None,
    confidence_extracted: float | None = None,
    decision_extracted: str | None = None,
    error_message: str | None = None,
) -> int:
    """Insert an LLMQuery audit row and return its auto-increment ID.

    Must be called for every LLM API call — including failed ones.
    Caller is responsible for committing the session.

    Args:
        session:              Open async session.
        strategy:             Strategy name or "system" for non-strategy calls.
        query_type:           One of: market_analysis, catalyst_generation,
                              social_summarization, movement_prediction, daily_digest.
        model_used:           Model identifier, e.g. "claude-haiku-4-5-20251001".
        prompt_version:       Versioned prompt template ID, e.g. "catalyst-v1".
        prompt:               Full prompt sent to the LLM.
        response:             Full raw LLM response (or error text if failed).
        tokens_input:         Input token count.
        tokens_output:        Output token count.
        cost_usd:             Dollar cost of this query.
        latency_ms:           Wall-clock response time in milliseconds.
        success:              False if the LLM call failed or response unparseable.
        market_id:            Market being analyzed (optional).
        signal_id:            Signal produced by this call (optional).
        confidence_extracted: Confidence score parsed from response (optional).
        decision_extracted:   Decision parsed from response (optional).
        error_message:        Populated when success=False (optional).

    Returns:
        The auto-increment integer ID of the inserted row.
    """
    row = LLMQueryRow(
        timestamp=datetime.now(UTC),
        strategy=strategy,
        query_type=query_type,
        market_id=market_id,
        signal_id=signal_id,
        model_used=model_used,
        prompt_version=prompt_version,
        prompt=prompt,
        response=response,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_total=tokens_input + tokens_output,
        cost_usd=cost_usd,
        confidence_extracted=confidence_extracted,
        decision_extracted=decision_extracted,
        latency_ms=latency_ms,
        success=success,
        error_message=error_message,
    )
    session.add(row)
    await session.flush()  # populates auto-increment id without committing

    log.debug(
        "llm_query_logged",
        query_type=query_type,
        model=model_used,
        market_id=market_id,
        tokens_total=tokens_input + tokens_output,
        cost_usd=round(cost_usd, 6),
        success=success,
        llm_query_id=row.id,
    )

    return row.id  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Cost calculation helpers
# ---------------------------------------------------------------------------

# Pricing constants — update when Anthropic changes rates.
# Sources: https://www.anthropic.com/pricing
_COST_PER_TOKEN: dict[str, tuple[float, float]] = {
    # (input $/token, output $/token)
    "claude-haiku-4-5-20251001": (1.00 / 1_000_000, 5.00 / 1_000_000),
    "claude-sonnet-4-6":         (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-opus-4-6":           (5.00 / 1_000_000, 25.00 / 1_000_000),
    "claude-opus-4-7":           (5.00 / 1_000_000, 25.00 / 1_000_000),
    # Introductory pricing through 2026-08-31; standard rate is $3.00/$15.00 per MTok.
    # TODO: update to (3.00/1e6, 15.00/1e6) after the introductory window ends.
    "claude-sonnet-5":           (2.00 / 1_000_000, 10.00 / 1_000_000),
}
# Only hit for a model string with no pricing entry above — logged loudly
# (see calculate_cost) because a silent fallback here previously mispriced
# every claude-opus-4-7 call as claude-sonnet-4-6 for days before it was caught.
_DEFAULT_COST_PER_TOKEN = (3.00 / 1_000_000, 15.00 / 1_000_000)

# Cache tokens are billed at a different rate than regular input tokens.
# Writes use the 1h-TTL rate (2x base input) since llm/client.py's cache_system
# flag always requests ttl="1h"; reads are discounted to 0.1x base input.
_CACHE_WRITE_MULTIPLIER = 2.0
_CACHE_READ_MULTIPLIER = 0.1


def calculate_cost(
    model: str,
    tokens_input: int,
    tokens_output: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Return the USD cost for a given model and token counts.

    ``cache_creation_tokens``/``cache_read_tokens`` are the
    ``cache_creation_input_tokens``/``cache_read_input_tokens`` fields from the
    Anthropic usage object — billed at different rates than plain input tokens
    (see module-level multipliers above). Omitting them (the pre-cache-aware
    behavior) undercounts cost for any call made with ``cache_system=True``.
    """
    if model not in _COST_PER_TOKEN:
        log.warning("llm.unknown_model_pricing", model=model, using_default=_DEFAULT_COST_PER_TOKEN)
    input_rate, output_rate = _COST_PER_TOKEN.get(model, _DEFAULT_COST_PER_TOKEN)
    return (
        tokens_input * input_rate
        + tokens_output * output_rate
        + cache_creation_tokens * input_rate * _CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * input_rate * _CACHE_READ_MULTIPLIER
    )
