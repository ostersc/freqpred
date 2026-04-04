"""LLM summarizer for long document bodies.

When a document body exceeds _SUMMARY_THRESHOLD chars, a cheap LLM call
produces a focused summary (≤ _MAX_SUMMARY_CHARS) framed around the market
question. The summary is used for both embedding and signal evidence.

The gating logic (length check + BM25 pre-check) lives in store.py where an
open session is available to run ts_rank. This module only handles prompt
building and the LLM call.
"""
from __future__ import annotations

import structlog

from freqpred.ingestion.store import RawDocument
from freqpred.llm.client import LLMClient, LLMError

log = structlog.get_logger(__name__)

_MAX_SUMMARY_CHARS = 500      # matches the evidence excerpt limit in signal/llm.py
_PROMPT_VERSION = "body-summarizer-v2"
_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are a research assistant helping a prediction market analyst. \
Summarize the article below in {max_chars} characters or fewer. \
Write only what the article says — key facts, dates, events, and outcomes. \
If the article happens to contain information directly relevant to the market question, include those facts. \
If it does not, simply summarize the article content without referencing the market question. \
Be concise and factual — no preamble, no opinions, no commentary on relevance."""


def _build_user_prompt(
    raw_doc: RawDocument,
    query_text: str,
    market_question: str,
) -> str:
    return (
        f"Market question: {market_question}\n"
        f"Search query used to find this article: {query_text}\n\n"
        f"Article title: {raw_doc.title}\n\n"
        f"Article body:\n{raw_doc.body}"
    )


async def summarize_body(
    raw_doc: RawDocument,
    query_text: str,
    market_question: str,
    llm_client: LLMClient,
    model: str = _MODEL,
) -> str | None:
    """Summarize a long document body focused on the market question.

    Assumes the caller has already verified that the body is long enough and
    passes the BM25 relevance gate. This function only builds the prompt, calls
    the LLM, and returns the summary text.

    Args:
        raw_doc:         The raw document whose body needs summarizing.
        query_text:      The catalyst query that retrieved this document (used as context in the prompt, not for relevance gating).
        market_question: The full market question (used for prompt context).
        llm_client:      Authenticated LLMClient (handles audit logging internally).
        model:           Model to use (defaults to claude-haiku-4-5-20251001).

    Returns:
        Summary string (≤ _MAX_SUMMARY_CHARS chars) on success, or None on failure.
        Never raises — LLMError is caught and logged.
    """
    system = _SYSTEM_PROMPT.format(max_chars=_MAX_SUMMARY_CHARS)
    user_prompt = _build_user_prompt(raw_doc, query_text, market_question)

    try:
        resp = await llm_client.complete(
            user_prompt,
            model,
            "body_summarization",
            system=system,
            strategy="body_summarizer",
            prompt_version=_PROMPT_VERSION,
            max_tokens=256,
        )
        summary = resp.content.strip()[:_MAX_SUMMARY_CHARS]
        log.debug(
            "body_summarizer.summarized",
            source_url=raw_doc.source_url,
            original_len=len(raw_doc.body),
            summary_len=len(summary),
        )
        return summary
    except LLMError as exc:
        log.warning(
            "body_summarizer.llm_error",
            source_url=raw_doc.source_url,
            error=str(exc),
        )
        return None
