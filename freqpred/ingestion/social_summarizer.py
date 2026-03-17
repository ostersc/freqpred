"""Cheap LLM pre-summarizer for raw social posts (Reddit/Twitter)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import structlog

from freqpred.ingestion.store import RawDocument
from freqpred.llm.client import LLMClient, LLMError

log = structlog.get_logger()

_PROMPT_VERSION = "social-summarizer-v1"
_MAX_INPUT_CHARS = 8000

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


async def summarize(
    docs: list[RawDocument],
    topic: str,
    llm_client: LLMClient,
    model: str = "claude-haiku-4-5-20251001",
) -> RawDocument:
    """Summarize a list of Reddit RawDocuments into a single structured RawDocument.

    Calls Claude Haiku to produce a JSON summary with sentiment, key_claims,
    notable_threads, and overall_signal. Always logs the LLM call to llm_queries,
    even on failure (handled by LLMClient).

    Args:
        docs:       Raw Reddit documents to summarize.
        topic:      Topic label (used in prompt and synthetic source_url).
        llm_client: Authenticated LLMClient (handles audit logging internally).
        model:      Model to use (defaults to claude-haiku-4-5-20251001).

    Returns:
        A single RawDocument with source_type="reddit_summary" and body as JSON.
    """
    now = datetime.now(timezone.utc)
    prompt_text = _build_prompt(docs, topic)
    synthetic_url = f"reddit_summary://{topic}/{uuid4()}"

    response_text = ""
    success = False
    error_message: str | None = None

    try:
        llm_resp = await llm_client.complete(
            prompt_text,
            model,
            "social_summarization",
            system=_SYSTEM_PROMPT,
            strategy="social_summarizer",
        )
        response_text = llm_resp.content
        success = True
    except LLMError as exc:
        error_message = str(exc)
        log.warning("social_summarizer.llm_error", topic=topic, error=error_message)

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
