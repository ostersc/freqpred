"""Cheap LLM pre-summarizer for raw social posts (Reddit/Twitter)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from uuid import uuid4

import anthropic
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.ingestion.store import RawDocument
from freqpred.llm.models import LLMQueryRow

log = structlog.get_logger()

_PROMPT_VERSION = "v1"
_MAX_INPUT_CHARS = 8000

# Haiku pricing (per million tokens)
_COST_PER_M_INPUT = 0.25
_COST_PER_M_OUTPUT = 1.25

_SYSTEM_PROMPT = """\
You are a financial signal analyst. Given a collection of Reddit posts about a topic,
produce a concise structured summary as valid JSON with exactly these keys:
- sentiment: overall sentiment ("bullish", "bearish", "neutral", or "mixed")
- key_claims: list of the most important factual claims or opinions (max 5 strings)
- notable_threads: list of thread titles or summaries worth highlighting (max 3 strings)
- overall_signal: one-sentence summary of the signal for a prediction market trader

Return only the JSON object, no markdown fences or extra text."""

_USER_TEMPLATE = """\
Topic: {topic}

Reddit posts:
{posts}

Return a JSON summary."""


def _build_prompt(docs: list[RawDocument], topic: str) -> str:
    parts: list[str] = []
    total = 0
    for doc in docs:
        entry = f"[{doc.source_name}] {doc.title}\n{doc.body}"
        if total + len(entry) > _MAX_INPUT_CHARS:
            break
        parts.append(entry)
        total += len(entry)
    posts_text = "\n\n---\n\n".join(parts)
    return _USER_TEMPLATE.format(topic=topic, posts=posts_text)


def _compute_cost(tokens_input: int, tokens_output: int) -> float:
    return (tokens_input / 1_000_000) * _COST_PER_M_INPUT + (
        tokens_output / 1_000_000
    ) * _COST_PER_M_OUTPUT


async def summarize(
    docs: list[RawDocument],
    topic: str,
    session: AsyncSession,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> RawDocument:
    """Summarize a list of Reddit RawDocuments into a single structured RawDocument.

    Calls Claude Haiku to produce a JSON summary with sentiment, key_claims,
    notable_threads, and overall_signal. Always logs the LLM call to llm_queries,
    even on failure.

    Args:
        docs:    Raw Reddit documents to summarize.
        topic:   Topic label (used in prompt and synthetic source_url).
        session: SQLAlchemy async session for audit logging.
        api_key: Anthropic API key.
        model:   Model to use (defaults to claude-haiku-4-5-20251001).

    Returns:
        A single RawDocument with source_type="reddit_summary" and body as JSON.
    """
    now = datetime.now(timezone.utc)
    prompt_text = _build_prompt(docs, topic)
    synthetic_url = f"reddit_summary://{topic}/{uuid4()}"

    client = anthropic.AsyncAnthropic(api_key=api_key)

    t_start = time.monotonic()
    response_text = ""
    tokens_input = 0
    tokens_output = 0
    success = False
    error_message: str | None = None

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_text}],
        )
        response_text = response.content[0].text
        tokens_input = response.usage.input_tokens
        tokens_output = response.usage.output_tokens
        success = True
    except Exception as exc:
        error_message = str(exc)
        log.warning("social_summarizer.llm_error", topic=topic, exc_info=True)

    latency_ms = int((time.monotonic() - t_start) * 1000)
    tokens_total = tokens_input + tokens_output
    cost_usd = _compute_cost(tokens_input, tokens_output)

    audit_row = LLMQueryRow(
        timestamp=now,
        strategy="social_summarizer",
        query_type="social_summarization",
        market_id=None,
        signal_id=None,
        model_used=model,
        prompt_version=_PROMPT_VERSION,
        prompt=prompt_text,
        response=response_text,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_total=tokens_total,
        cost_usd=cost_usd,
        confidence_extracted=None,
        decision_extracted=None,
        latency_ms=latency_ms,
        success=success,
        error_message=error_message,
    )
    session.add(audit_row)
    await session.flush()

    # Parse JSON or fall back to raw text body
    body: str
    if success:
        try:
            parsed = json.loads(response_text)
            body = json.dumps(parsed)
        except json.JSONDecodeError:
            log.warning(
                "social_summarizer.parse_error",
                topic=topic,
                response_snippet=response_text[:200],
            )
            body = response_text
    else:
        body = json.dumps(
            {
                "sentiment": "unknown",
                "key_claims": [],
                "notable_threads": [],
                "overall_signal": f"Summarization failed: {error_message}",
            }
        )

    return RawDocument(
        source_url=synthetic_url,
        title=f"Reddit summary: {topic}",
        body=body,
        source_type="reddit_summary",
        source_name="social_summarizer",
        category="",
        tags=[],
        published_at=now,
        fetched_at=now,
    )
