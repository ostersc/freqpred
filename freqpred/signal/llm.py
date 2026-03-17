"""Prompt building and response parsing for signal analysis."""
from __future__ import annotations

import json

import structlog

from freqpred.markets.models import Market
from freqpred.rag.models import Document

log = structlog.get_logger(__name__)

PROMPT_VERSION = "signal-v1"

SYSTEM_PROMPT = """\
You are a prediction market probability analyst. Your task is to estimate the \
probability that a given market question resolves YES, based on the provided evidence.

You must respond ONLY with a valid JSON object. No markdown, no explanation outside \
the JSON."""


def build_prompt(market: Market, docs: list[Document]) -> str:
    """Build the user prompt for signal analysis.

    Includes the market question, current price context, and document excerpts
    as evidence.
    """
    lines: list[str] = [
        f"Market Question: {market.question}",
        f"Category: {market.category}",
        f"Market Mid Price (implied probability): {market.mid_price:.4f}",
        f"Market Closes: {market.close_time.isoformat()}",
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
        "Estimate the probability this market resolves YES.",
        "Set direction to YES if your estimate is meaningfully above the market mid,",
        "NO if meaningfully below, or SKIP if you lack sufficient evidence.",
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
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line; drop closing fence if present
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(inner)

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
