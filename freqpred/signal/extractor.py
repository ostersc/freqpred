"""Question-focused evidence extraction at retrieval time (T101).

``build_prompt`` historically rendered each retrieved document as
``(summary or body)[:500]`` — a raw prefix cut. Because articles open with
navigation chrome, cookie notices and live-blog sign-offs, 40.3% of evidence
reached the signal LLM as boilerplate rather than content, and the worst band
(unsummarised bodies averaging 22k chars) was shown at 6.5%.

This module replaces the prefix cut with an extraction pass keyed on the market
actually being analysed. Two properties matter:

* The question is always the *current* market, not whichever question happened
  to trigger the document's ingestion — that gate mismatch is the root cause.
* ``relevance="none"`` lets a genuinely unrelated document be dropped from the
  prompt entirely instead of occupying an evidence slot with chrome.

Failure is always open: any LLM or parse error yields a
``DocumentExtract`` carrying the legacy ``[:500]`` cut, so a signal never
disappears because extraction had a bad day.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from freqpred.llm.client import LLMError
from freqpred.signal.models import DocumentExtractRow

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from freqpred.llm.client import LLMClient
    from freqpred.markets.models import Market
    from freqpred.rag.models import Document

log = structlog.get_logger(__name__)

EXTRACT_PROMPT_VERSION = "extract-v1"

QUERY_TYPE = "evidence_extraction"

#: Bodies at or below this length are rendered whole — there is nothing for an
#: extractor to do, and 40.5% of signal-linked evidence is in this band.
MIN_EXTRACT_BODY_CHARS = 500

#: Ceiling on what is sent to the extractor. A manual probe on real articles
#: showed raising a summariser's input cap from 4k to 16k changed almost
#: nothing (news front-loads its facts), so this is a cost guard, not a
#: quality knob.
MAX_EXTRACT_INPUT_CHARS = 16_000

#: Defensive cap on what comes back. The instruction asks for ~100 words; a
#: model that ignores it must not be able to blow up the signal prompt.
MAX_EXTRACT_CHARS = 800

RELEVANCE_VALUES = ("direct", "contextual", "none")

EXTRACT_TOOL: dict = {
    "name": "record_extract",
    "description": (
        "Record the question-relevant content of one retrieved document, "
        "together with how directly it bears on the market question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relevance": {
                "type": "string",
                "enum": list(RELEVANCE_VALUES),
                "description": (
                    "How this document bears on the market question. "
                    "'none' is reserved for genuinely unconnected documents."
                ),
            },
            "extract": {
                "type": "string",
                "description": (
                    "The question-relevant content, ~100 words or fewer. "
                    "Empty string when relevance is 'none'."
                ),
            },
        },
        "required": ["relevance", "extract"],
    },
}


@dataclass(frozen=True)
class DocumentExtract:
    """One document's extraction result for one market.

    ``fallback`` marks a result produced by the legacy ``[:500]`` path rather
    than by the model — either because the body was short enough to render
    whole, or because extraction failed and fell open. Callers rendering the
    prompt do not care, but the probe and the audit trail do.
    """

    document_id: str
    relevance: str
    extract: str
    model_used: str
    prompt_version: str
    fallback: bool = False


def legacy_excerpt(doc: Document) -> str:
    """The pre-T101 rendering: prefer summary, cut at 500, flatten newlines.

    Kept as the fail-open path so a failed extraction degrades to exactly the
    behaviour that shipped before this module existed.
    """
    excerpt = (doc.summary or doc.body or "")[:MIN_EXTRACT_BODY_CHARS]
    return excerpt.replace("\n", " ").strip()


def _fallback_extract(doc: Document, *, model: str, short_body: bool) -> DocumentExtract:
    return DocumentExtract(
        document_id=str(doc.id),
        # Never "none": a fail-open result must not remove the document from
        # the prompt. Dropping evidence is only ever a model decision.
        relevance="contextual",
        extract=legacy_excerpt(doc),
        model_used="" if short_body else model,
        prompt_version=EXTRACT_PROMPT_VERSION,
        fallback=True,
    )


def needs_extraction(doc: Document) -> bool:
    """Whether this document is worth an LLM call.

    Keyed on ``body`` alone, never ``summary or body``. Gating on the summary
    looks right — it is what ``build_prompt`` renders — but it hands the
    decision back to the ingestion-time gate this whole change exists to
    escape: a 486-char summary written against whichever question triggered
    the fetch would mark a 43,000-char article as "nothing to extract". A
    probe run on 2026-08-11 skipped documents of 48,114, 45,691 and 43,337
    chars for exactly that reason. The summary is retired by T102 regardless.
    """
    return len(doc.body or "") > MIN_EXTRACT_BODY_CHARS


def build_extract_prompt(market: Market, doc: Document) -> str:
    """Render the per-document extraction prompt.

    The framing deliberately mirrors ``catalyst_generator._build_prompt``'s
    "background signals, behavioral patterns, contextual factors" language.
    An earlier draft that asked only for facts bearing on *resolution*
    produced a false ``none`` on a Davos scheduling article — precisely the
    circumstantial evidence the catalyst had gone looking for.
    """
    body = (doc.body or "")[:MAX_EXTRACT_INPUT_CHARS]
    published = doc.published_at.isoformat() if doc.published_at else "unknown"

    return "\n".join(
        [
            "You are filtering retrieved evidence for a prediction market analyst.",
            "",
            "=== MARKET QUESTION ===",
            market.question,
            "",
            f"Market closes: {market.close_time.isoformat()}",
            "",
            "=== DOCUMENT ===",
            f"Title: {doc.title}",
            f"Source: {doc.source_name} ({doc.source_type})",
            f"Published: {published}",
            "",
            body,
            "",
            "=== YOUR TASK ===",
            "Pull out everything in this document that helps estimate the probability",
            "that the market resolves YES before it closes. Facts that would settle the",
            "question are not the only thing that counts: background signals, behavioral",
            "patterns, contextual factors, scheduling, stated intentions, and recent",
            "developments all shift the probability. This document was retrieved because",
            "a search for predictive evidence surfaced it.",
            "",
            "Label how directly it bears on the question:",
            "  direct     — speaks to the market's subject and its outcome",
            "  contextual — circumstantial or background evidence that moves the",
            "               probability without settling it (the subject's schedule,",
            "               related events, past behavior in similar situations)",
            "  none       — genuinely unconnected: a different topic, a different",
            "               person, a different event. If you can articulate any way",
            "               this document bears on the question, it is contextual,",
            "               not none.",
            "",
            "A document that covers the question's subject matter at all is contextual",
            "at minimum, even when it never mentions the person or event the question",
            "turns on. Coverage of a subject is itself evidence about how salient that",
            "subject currently is.",
            "",
            "Long documents are often bundles — a daily briefing, a live blog, a news",
            "roundup. Read to the end: the relevant passage is frequently nowhere near",
            "the top, and the headline may be about something else entirely.",
            "",
            "Write the extract in about 100 words or fewer. Preserve specific facts,",
            "dates, numbers and named actors; quote or paraphrase, do not analyse and",
            "do not state a probability. Do not reason about whether the market",
            "resolves, and do not compare dates against the resolution window — that",
            "is the analyst's job, not yours; spending the extract on it costs the",
            "analyst the evidence it replaced. Exclude navigation menus, app-download",
            "prompts, tracking pixels, cookie notices, newsletter sign-ups and other",
            "site chrome. Return an empty extract when relevance is none.",
            "",
            "Call record_extract with your result.",
        ]
    )


def parse_extract_response(content: str, doc: Document) -> tuple[str, str] | None:
    """Parse the forced tool call into ``(relevance, extract)``.

    Returns ``None`` on anything malformed so the caller can fall open rather
    than raise. A plain-text sentinel was tried first and proved unreliable —
    the model appended prose after the sentinel and leaked
    ``**Relevance: contextual**`` preambles into the extract — which is why
    this path is tool-forced JSON.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.warning(
            "signal.extractor.parse_error",
            document_id=str(doc.id),
            error=str(exc),
            content_preview=content[:200],
        )
        return None

    if not isinstance(data, dict):
        log.warning("signal.extractor.not_a_dict", document_id=str(doc.id))
        return None

    relevance = data.get("relevance")
    extract = data.get("extract")
    if relevance not in RELEVANCE_VALUES or not isinstance(extract, str):
        log.warning(
            "signal.extractor.invalid_fields",
            document_id=str(doc.id),
            relevance=relevance,
            extract_type=type(extract).__name__,
        )
        return None

    extract = extract.replace("\n", " ").strip()[:MAX_EXTRACT_CHARS]
    if relevance != "none" and not extract:
        # A non-none label with nothing behind it would render an empty
        # evidence slot — worse than the raw cut it replaced.
        log.warning("signal.extractor.empty_extract", document_id=str(doc.id))
        return None
    return relevance, extract


async def extract_document(
    llm_client: LLMClient,
    market: Market,
    doc: Document,
    *,
    model: str,
    strategy: str = "system",
) -> DocumentExtract:
    """Extract one document against one market question. Never raises.

    Short bodies skip the call entirely; every other outcome — success, API
    failure, malformed tool input — returns a usable ``DocumentExtract``.
    """
    if not needs_extraction(doc):
        return _fallback_extract(doc, model=model, short_body=True)

    prompt = build_extract_prompt(market, doc)
    try:
        response = await llm_client.complete(
            prompt,
            model,
            QUERY_TYPE,
            market_id=market.id,
            strategy=strategy,
            prompt_version=EXTRACT_PROMPT_VERSION,
            json_tool=EXTRACT_TOOL,
        )
    except LLMError as exc:
        log.warning(
            "signal.extractor.llm_error",
            document_id=str(doc.id),
            market_id=market.id,
            error=str(exc),
        )
        return _fallback_extract(doc, model=model, short_body=False)

    parsed = parse_extract_response(response.content, doc)
    if parsed is None:
        return _fallback_extract(doc, model=response.model, short_body=False)

    relevance, extract = parsed
    return DocumentExtract(
        document_id=str(doc.id),
        relevance=relevance,
        extract=extract,
        model_used=response.model,
        prompt_version=EXTRACT_PROMPT_VERSION,
        fallback=False,
    )


async def _load_cached(
    session: AsyncSession, market_id: str, doc_ids: list[str]
) -> dict[str, DocumentExtract]:
    if not doc_ids:
        return {}
    rows = (
        await session.execute(
            select(DocumentExtractRow).where(
                DocumentExtractRow.market_id == market_id,
                DocumentExtractRow.prompt_version == EXTRACT_PROMPT_VERSION,
                DocumentExtractRow.document_id.in_([uuid.UUID(d) for d in doc_ids]),
            )
        )
    ).scalars()
    return {
        str(row.document_id): DocumentExtract(
            document_id=str(row.document_id),
            relevance=row.relevance,
            extract=row.extract,
            model_used=row.model_used,
            prompt_version=row.prompt_version,
        )
        for row in rows
    }


async def _persist(
    session: AsyncSession, market_id: str, results: list[DocumentExtract]
) -> None:
    """Write fresh extracts, ignoring rows a concurrent analysis already wrote.

    Fallbacks are deliberately not persisted: caching a failure would make one
    bad API minute stick to a (document, market) pair until the prompt version
    changes.
    """
    fresh = [r for r in results if not r.fallback]
    if not fresh:
        return
    await session.execute(
        pg_insert(DocumentExtractRow)
        .values(
            [
                {
                    "id": uuid.uuid4(),
                    "document_id": uuid.UUID(r.document_id),
                    "market_id": market_id,
                    "relevance": r.relevance,
                    "extract": r.extract,
                    "model_used": r.model_used,
                    "prompt_version": r.prompt_version,
                }
                for r in fresh
            ]
        )
        .on_conflict_do_nothing(constraint="uq_document_extracts_doc_market_version")
    )
    # Committed here rather than left to the caller's transaction. Extraction
    # runs before the pipeline writes anything, so there is no partial work to
    # commit prematurely — and if the analysis call downstream fails, extracts
    # already paid for must survive rather than roll back and be re-bought on
    # the next pass.
    await session.commit()


async def extract_for_documents(
    session: AsyncSession,
    llm_client: LLMClient,
    market: Market,
    docs: list[Document],
    *,
    model: str,
    strategy: str = "system",
    max_concurrency: int = 5,
) -> dict[str, DocumentExtract]:
    """Extract every retrieved document for *market*, reading through the cache.

    Returns a ``{document_id: DocumentExtract}`` map covering every document in
    *docs* — callers can index it unconditionally.

    Only the uncached, long-bodied documents cost an API call. Measured over
    227 consecutive signal pairs (2026-08-11), just 15.6% of a market's
    retrieved documents are new since its previous signal, so the cache carries
    the overwhelming majority of a steady-state run.
    """
    if not docs:
        return {}

    cached = await _load_cached(session, market.id, [str(d.id) for d in docs])
    todo = [d for d in docs if str(d.id) not in cached]
    if not todo:
        return cached

    limiter = asyncio.Semaphore(max_concurrency)

    async def _one(doc: Document) -> DocumentExtract:
        async with limiter:
            return await extract_document(
                llm_client, market, doc, model=model, strategy=strategy
            )

    fresh = await asyncio.gather(*(_one(d) for d in todo))
    await _persist(session, market.id, list(fresh))

    log.info(
        "signal.extractor.batch",
        market_id=market.id,
        docs=len(docs),
        cache_hits=len(cached),
        extracted=len(fresh),
        dropped=sum(1 for r in fresh if r.relevance == "none"),
        fallbacks=sum(1 for r in fresh if r.fallback),
    )

    return {**cached, **{r.document_id: r for r in fresh}}
