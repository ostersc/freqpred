"""Prompt building and response parsing for signal analysis."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

from freqpred.markets.models import Market
from freqpred.rag.models import Document

log = structlog.get_logger(__name__)

PROMPT_VERSION = "signal-v2"

SYSTEM_PROMPT = """\
You are a prediction market probability analyst. Your task is to estimate the \
probability that a given market question resolves YES, based on the provided evidence.

Your response must be a single valid JSON object and nothing else — no prose, no \
reasoning, no markdown. Start your response with { and end with }."""


def build_prompt(market: Market, docs: list[Document]) -> str:
    """Build the user prompt for signal analysis.

    Includes the market question, current price context, and document excerpts
    as evidence.
    """
    now = datetime.now(tz=timezone.utc)
    days_to_close = (market.close_time - now).total_seconds() / 86400

    lines: list[str] = [
        f"Market Question: {market.question}",
        f"Category: {market.category}",
        f"Current Date (UTC): {now.strftime('%Y-%m-%d %H:%M')}",
        f"Market Closes: {market.close_time.isoformat()} ({days_to_close:.1f} days from now)",
        "",
        "=== EVIDENCE ===",
    ]

    if docs:
        for i, doc in enumerate(docs, start=1):
            # Prefer summary when available; fall back to body excerpt
            excerpt = doc.summary or doc.body[:500]
            excerpt = excerpt.replace("\n", " ").strip()
            lines += [
                f"[{i}] {doc.title}",
                f"    Source: {doc.source_name} ({doc.source_type})",
                f"    Published: {doc.published_at.isoformat()}",
                f"    ID: {doc.id}",
                f"    {excerpt}",
                "",
            ]
    else:
        lines.append("No evidence available.")
        lines.append("")

    lines += [
        "=== TASK ===",
        "Estimate the probability this market resolves YES based solely on the evidence above.",
        "Set direction to YES if you believe the event is more likely than not,",
        "NO if less likely than not, or SKIP if the evidence is insufficient to form a view.",
        "",
        "Respond with ONLY this JSON object (no markdown fences):",
        "{",
        '  "probability": <float 0.0-1.0>,',
        '  "confidence": <float 0.0-1.0>,',
        '  "direction": "<YES|NO|SKIP>",',
        '  "reasoning": "<concise explanation>",',
        '  "evidence_used": ["<doc_id_1>", ...]',
        "}",
    ]

    return "\n".join(lines)


def parse_signal_response(content: str) -> dict | None:
    """Parse the LLM's JSON response into a validated signal dict.

    Returns a dict with keys: probability, confidence, direction, reasoning,
    evidence_used.  Returns None on any parse or validation error so the
    caller can log and skip rather than crash.
    """
    text = content.strip()

    # Strip markdown code fences if present
    if "```" in text:
        lines = text.splitlines()
        inner = [
            line for line in lines
            if not line.strip().startswith("```")
        ]
        text = "\n".join(inner).strip()

    # Extract JSON object from prose preamble/postamble (model may reason before answering)
    if not text.startswith("{") and "{" in text and "}" in text:
        text = text[text.index("{") : text.rindex("}") + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning(
            "signal.llm.parse_error",
            error=str(exc),
            content_preview=content[:200],
        )
        return None

    if not isinstance(data, dict):
        log.warning("signal.llm.not_a_dict", type=type(data).__name__)
        return None

    required = {"probability", "confidence", "direction", "reasoning", "evidence_used"}
    missing = required - data.keys()
    if missing:
        log.warning("signal.llm.missing_fields", missing=sorted(missing))
        return None

    try:
        probability = float(data["probability"])
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        log.warning("signal.llm.invalid_numbers", error=str(exc))
        return None

    # Clamp to [0, 1]
    probability = max(0.0, min(1.0, probability))
    confidence = max(0.0, min(1.0, confidence))

    direction = str(data["direction"]).upper()
    if direction not in {"YES", "NO", "SKIP"}:
        log.warning("signal.llm.invalid_direction", direction=direction)
        return None

    reasoning = str(data.get("reasoning", ""))
    evidence_used: list[str] = [str(e) for e in data.get("evidence_used", [])]

    return {
        "probability": probability,
        "confidence": confidence,
        "direction": direction,
        "reasoning": reasoning,
        "evidence_used": evidence_used,
    }
