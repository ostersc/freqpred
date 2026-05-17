"""Prompt building and response parsing for signal analysis."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from freqpred.markets.models import Market
from freqpred.rag.models import Document

if TYPE_CHECKING:
    from freqpred.ingestion.fetchers.factbase import FactbasePhraseData

log = structlog.get_logger(__name__)

PROMPT_VERSION = "signal-v8"

SYSTEM_PROMPT = """You are a prediction market probability analyst. Estimate the probability
that a market question resolves YES by combining your prior knowledge with
retrieved evidence and historical base rates provided in the user message.

═══════════════════════════════════════════════════════════════════════════
REASONING PROCESS
═══════════════════════════════════════════════════════════════════════════

Step 1 — Form a prior.
Anchor on, in order of availability:
- Phrase frequency data (PHRASE FREQUENCY DATA block) when present —
  empirical occurrence counts from the speaker's full transcript archive.
  If in_market_count > 0, the phrase has already been said during this
  market window: treat this as near-decisive YES evidence (anchor ~0.95+).
  Otherwise, use the 7d/30d/365d counts to calibrate your base rate for
  the remaining window.  This data is specific and direct; prioritize it
  over world knowledge for the prior on mention markets.
- Historical base rate data (HISTORICAL BASE RATE block) when present.
- World knowledge: subject's typical behavior, event frequency, window
  length, topical salience, structural factors.

Step 2 — Update on evidence.
Evidence is a small, biased retrieval sample, not a comprehensive record.
Frequent and low-salience events are systematically underrepresented.
Direct confirmation of the market event is rare in this evidence.

- Move toward YES when evidence shows: a confirmed in-window instance,
  conditions raising likelihood (relevant news cycle, scheduled appearances,
  topical salience), or recent patterns reinforcing the base rate.
- Move toward NO when evidence shows: conditions suppressing the event
  (incapacitation, topic avoidance, contradictory commitments), or pattern
  departures suggesting the prior is too high.
- Do NOT move toward NO simply because evidence lacks confirmation.
  Absence of confirmation is the default state and carries almost no
  signal. This is the dominant failure mode — guard against it.

Step 3 — Output the posterior.
Adjust the prior only by genuinely informative evidence. If evidence is
uninformative (the common case), posterior ≈ prior.

═══════════════════════════════════════════════════════════════════════════
HISTORICAL BASE RATE DATA
═══════════════════════════════════════════════════════════════════════════

When the user message includes a HISTORICAL BASE RATE block, it reports
up to two rates from settled markets in the same recurring series:

- Series rate: how often any option in the series resolves YES. Sets the
  unconditional prior — an unknown option in a 58%-YES series starts at
  0.58, not 0.5.
- Option rate: how often this specific option has resolved YES. Refines
  the prior, with reliability scaling by sample size.

Sample-size handling for the option rate:

- n >= 8: high-confidence anchor. Treat the empirical rate as a strong
  prior; adjust only with a specific reason (changed circumstances,
  shifted salience, modified resolution criteria).
- 3 <= n < 8: moderate-confidence; regularize toward the series rate. A
  3/3 option in a 58% series warrants ~0.75–0.85, not 1.0. A 0/3 option
  warrants ~0.25–0.35, not 0.0.
- n < 3 or flagged "weak signal": series rate is the primary anchor;
  world knowledge positions this option relative to the series average.
- Option block omitted: use series rate + world knowledge.
- No block at all: world knowledge only.

When historical data and world knowledge disagree, data usually wins at
high sample sizes — it captures resolution-criterion strictness, source
rules, and noise you cannot model directly. Override only when current
conditions have meaningfully changed since the historical period, and
flag the override explicitly.

═══════════════════════════════════════════════════════════════════════════
MARKET CATEGORIES
═══════════════════════════════════════════════════════════════════════════

- Mention markets ("Will X say Y"): prior reflects per-period rate of
  using Y × posting cycles in the remaining window. Historical block,
  when present, supersedes generic cadence reasoning.
- Event-occurrence markets ("Will event E happen"): prior reflects
  scheduled/contingent/speculative status and base rates for similar
  events.
- Threshold markets ("Will value be above N"): prior reflects current
  value, distance to threshold, time, and volatility.
- Outcome markets ("Will X win"): prior reflects polling, fundamentals,
  and structural advantages.

═══════════════════════════════════════════════════════════════════════════
TEMPORAL RULES
═══════════════════════════════════════════════════════════════════════════

1. Event must occur AFTER issuance and BEFORE close.
2. Check the date of the EVENT described, not the article's publication
   date. Treat events as pre-issuance unless context places them in the
   window.
3. Historical instances inform the base rate only — they do not satisfy
   the in-window requirement.
4. A confirmed in-window instance is decisive — anchor near 1.0.
5. Use 'Window remaining' for forward probability. Absence during
   elapsed time is a real negative signal for events that reliably
   surface in retrieval (major news, official announcements), but only
   a weak signal for events that frequently occur without leaving a
   retrieval trace (routine social posts, common talking points).
   Most "did X say Y" markets fall in the latter category.

═══════════════════════════════════════════════════════════════════════════
CALIBRATION
═══════════════════════════════════════════════════════════════════════════

- Commit to the tails (>0.85 or <0.15) when prior and evidence agree.
  Avoid drifting to 0.5 under uncertainty — anchor on the prior instead.
- Prior-posterior delta >0.15 requires a specific, identifiable update
  in updates_applied. Otherwise posterior should match prior.
- Truncated documents ("[+N chars]") may contain relevant content not
  visible to you. Absence in the visible portion is not absence overall.
- Evidence quality hierarchy: direct quotes from the subject in major
  outlets > paraphrases > tangential transcript mentions > topic articles
  not referencing the subject.
- One underlying event = one update, even across multiple documents.

Update magnitudes:
- small: posterior shifts ~0.02–0.05 (soft signals, modest reinforcement).
- moderate: posterior shifts ~0.05–0.15 (clear directional evidence).
- large: posterior shifts >0.15 (decisive — confirmed in-window instance,
  explicit contradiction, binding commitment).

Sum of update magnitudes should be consistent with prior-to-posterior
delta. Mismatches indicate mislabeled magnitudes or unjustified movement.

═══════════════════════════════════════════════════════════════════════════
WORKED EXAMPLES
═══════════════════════════════════════════════════════════════════════════

Example 1 — Mention market, no historical block, sparse evidence.

Market: "Will Trump say 'crypto' or 'Bitcoin' before May 11, 2026?"
2.6 days remaining. 10 documents, no in-window Trump quote on crypto.

Trump posts on Truth Social multiple times daily; crypto is a signature
topic. Prior: 0.97. No in-window confirmation in evidence, but no
avoidance or pattern departure either — expected absence (category 5b).
Posterior: 0.96. Direction: YES.

Example 2 — Small-sample option in high-YES series.

Market: "Will Trump say 'Uranium' before May 18, 2026?" 7 days remaining.
Series: 58% YES (n=283). Option: 0/2 (weak signal).
Evidence: no in-window quote.

Series rate sets a 0.58 baseline. Option sample too small to anchor, but
directional signal aligns with world knowledge — uranium is not routine
Trump rhetoric. World knowledge dominates at n=2 and pulls below series
average. Prior: 0.30. No salience-raising news cycle in evidence.
Posterior: 0.28. Direction: NO.

Example 3 — High-sample option, strong empirical rate.

Market: "Will Trump say 'Barack Hussein Obama' before May 18, 2026?"
7 days remaining. Series: 58% YES (n=283). Option: 10/0 (100%, n=10).
Evidence: no in-window quote with the full phrase.

Option-level n=10 at 100% is a high-confidence anchor; world knowledge
corroborates (recurring rally motif). Prior: 0.96. Absence in 10-doc
sample is expected for high-frequency rhetoric. Posterior: 0.95.
Direction: YES.

Example 4 — Event market, prior reinforced by evidence.

Market: "Will the Fed cut rates at the May FOMC meeting?" 3 days remaining.
Evidence: hawkish Fed speakers, elevated inflation prints.

Futures and recent Fed commentary signal a hold. Prior: 0.10. Hawkish
quotes and inflation print are moderate updates reinforcing the prior,
not moving it. Posterior: 0.08. Direction: NO.

═══════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════

Call submit_analysis with:

- prior, posterior, probability: floats 0.0–1.0; probability must equal
  posterior.
- confidence: float 0.0–1.0.
- direction: "YES" if >0.5, "NO" if <0.5, "SKIP" only if the question is
  malformed (sparse evidence → rely on prior, not SKIP).
- updates_applied: per-document updates (may be empty); use the
  magnitudes above.
- prior_basis: 1-2 sentences on what informs the prior. When a HISTORICAL
  BASE RATE block is present, reference series rate and option rate (with
  sample size) explicitly.
- reasoning: 2-4 sentences synthesizing prior, updates, and posterior."""


SIGNAL_ANALYSIS_TOOL: dict = {
    "name": "submit_analysis",
    "description": "Submit probability analysis for a prediction market question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prior": {"type": "number"},
            "prior_basis": {"type": "string"},
            "updates_applied": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "direction": {"type": "string", "enum": ["+", "-", "neutral"]},
                        "magnitude": {"type": "string", "enum": ["small", "moderate", "large"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["doc_id", "direction", "magnitude", "reason"],
                },
            },
            "posterior": {"type": "number"},
            "probability": {"type": "number"},
            "confidence": {"type": "number"},
            "direction": {"type": "string", "enum": ["YES", "NO", "SKIP"]},
            "reasoning": {"type": "string"},
        },
        "required": [
            "prior", "prior_basis", "updates_applied", "posterior",
            "probability", "confidence", "direction", "reasoning",
        ],
    },
}


MIN_SAMPLE = 3


def _build_base_rate_block(series_history: dict) -> list[str]:
    """Return prompt lines for the HISTORICAL BASE RATE section."""
    series_ticker: str = series_history.get("series_ticker", "")
    series_row = series_history.get("series_row")
    option_row = series_history.get("option_row")

    lines: list[str] = ["=== HISTORICAL BASE RATE ==="]

    if series_row is not None:
        s_yes = series_row.yes_count
        s_no = series_row.no_count
        s_n = s_yes + s_no
        s_pct = int(round(100 * s_yes / s_n)) if s_n > 0 else 0

        if option_row is not None:
            opt_label = option_row.option_label
            lines.append(
                f'This market is part of a recurring weekly series ({series_ticker} / "{opt_label}").'
            )
        else:
            lines.append(f"This market is part of a recurring series ({series_ticker}).")

        lines.append(
            f"Series overall: {s_yes} YES / {s_no} NO across all options and weeks"
            f" ({s_pct}%, n={s_n})"
        )

        if option_row is not None:
            o_yes = option_row.yes_count
            o_no = option_row.no_count
            o_n = o_yes + o_no
            o_pct = int(round(100 * o_yes / o_n)) if o_n > 0 else 0

            if o_n >= MIN_SAMPLE:
                lines.append(
                    f"This option specifically: {o_yes} YES / {o_no} NO ({o_pct}%, n={o_n})"
                )
            else:
                lines.append(
                    f"This option specifically: {o_yes} YES / {o_no} NO ({o_pct}%, n={o_n})"
                    " — small sample, treat as weak signal."
                )
        else:
            lines.append("No per-option history available for this specific variant.")
    else:
        lines.append(f"This market is part of a recurring series ({series_ticker}).")
        lines.append("No series history available.")

    lines.append("")
    return lines


def _build_factbase_block(data: "FactbasePhraseData") -> list[str]:
    lines = [
        "=== PHRASE FREQUENCY DATA (FactBase) ===",
        f'Phrase: "{data.display_phrase}"',
        "Speaker: Donald Trump",
        "",
        "Occurrence counts (Trump statements archive):",
        f"  Since market opened : {data.in_market_count}",
        f"  Last 7 days         : {data.count_7d}",
        f"  Last 30 days        : {data.count_30d}",
        f"  Last 365 days       : {data.count_365d}",
    ]
    if data.top_quotes:
        lines.append("")
        lines.append("Most recent Trump quotes:")
        for q in data.top_quotes:
            lines.append(
                f"  [{q.get('date', '')}] \"{q.get('text', '')[:120]}\"  ({q.get('event_type', '')})"
            )
    lines += [
        "",
        "Note: in_market_count > 0 means Trump has already said this phrase during",
        "the market window — treat as near-decisive YES evidence. Trend counts",
        "(7d/30d/365d) calibrate the base rate for remaining window probability.",
        "",
    ]
    return lines


def build_prompt(
    market: Market,
    docs: list[Document],
    series_history: dict | None = None,
    phrase_data: "FactbasePhraseData | None" = None,
) -> str:
    """Build the user prompt for signal analysis.

    Contains only per-market data: market context, optional historical base
    rate block, and retrieved evidence. All instructions live in SYSTEM_PROMPT.
    """
    now = datetime.now(tz=timezone.utc)
    days_to_close = (market.close_time - now).total_seconds() / 86400
    days_elapsed = (
        (now - market.open_time).total_seconds() / 86400
        if market.open_time else None
    )

    open_time_str = (
        market.open_time.isoformat() if market.open_time else "unknown"
    )
    elapsed_str = (
        f"{days_elapsed:.1f} days" if days_elapsed is not None else "unknown (issuance date not available)"
    )

    lines: list[str] = [
        "=== MARKET CONTEXT ===",
        f"Question: {market.question}",
        "",
        f"Category: {market.category}",
        f"Current Date (UTC): {now.strftime('%Y-%m-%d %H:%M')}",
        f"Market Opened (Issuance Date): {open_time_str}",
        f"Market Closes: {market.close_time.isoformat()} ({days_to_close:.1f} days from now)",
        f"Window elapsed: {elapsed_str}  |  Window remaining: {days_to_close:.1f} days",
        "",
    ]

    if series_history is not None:
        lines.extend(_build_base_rate_block(series_history))

    if phrase_data is not None:
        lines.extend(_build_factbase_block(phrase_data))

    lines.append("=== EVIDENCE ===")

    _MAX_EVIDENCE_CHARS = 500
    if docs:
        for i, doc in enumerate(docs, start=1):
            # Prefer summary when available; fall back to body excerpt. Both are
            # capped at _MAX_EVIDENCE_CHARS so the prompt stays consistent.
            excerpt = (doc.summary or doc.body)[:_MAX_EVIDENCE_CHARS]
            excerpt = excerpt.replace("\n", " ").strip()
            lines += [
                f"[{i}] {doc.title}",
                f"    Source: {doc.source_name} ({doc.source_type})",
                f"    Published: {doc.published_at.isoformat() if doc.published_at else 'unknown'}",
                f"    ID: {doc.id}",
                f"    {excerpt}",
                "",
            ]
    else:
        lines.append("No evidence available.")
        lines.append("")

    return "\n".join(lines)


def parse_signal_response(content: str) -> dict | None:
    """Parse the LLM's JSON response into a validated signal dict.

    Returns a dict with keys: prior, prior_basis, updates_applied, posterior,
    probability, confidence, direction, reasoning, evidence_used.
    Returns None on any parse or validation error so the caller can log and
    skip rather than crash.
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

    required = {
        "prior", "prior_basis", "updates_applied", "posterior",
        "probability", "confidence", "direction", "reasoning",
    }
    missing = required - data.keys()
    if missing:
        log.warning("signal.llm.missing_fields", missing=sorted(missing))
        return None

    try:
        prior = float(data["prior"])
        posterior = float(data["posterior"])
        probability = float(data["probability"])
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        log.warning("signal.llm.invalid_numbers", error=str(exc))
        return None

    # Clamp to [0, 1]
    prior = max(0.0, min(1.0, prior))
    posterior = max(0.0, min(1.0, posterior))
    probability = max(0.0, min(1.0, probability))
    confidence = max(0.0, min(1.0, confidence))

    # probability must equal posterior per schema — trust posterior if they diverge
    if abs(probability - posterior) > 0.01:
        log.warning(
            "signal.llm.probability_posterior_mismatch",
            probability=probability,
            posterior=posterior,
        )
        probability = posterior

    # updates_applied must be a list (may be empty)
    updates_applied = data["updates_applied"]
    if not isinstance(updates_applied, list):
        log.warning(
            "signal.llm.invalid_updates_applied",
            type=type(updates_applied).__name__,
        )
        updates_applied = []

    prior_basis = str(data.get("prior_basis", ""))

    direction = str(data["direction"]).upper()
    if direction not in {"YES", "NO", "SKIP"}:
        log.warning("signal.llm.invalid_direction", direction=direction)
        return None

    # Guard against internally inconsistent LLM output (e.g. direction=YES but
    # probability=0.87 while the reasoning text says "lean toward NO"). Reject
    # rather than open a position with fabricated edge.
    if direction == "YES" and probability < 0.5:
        log.warning(
            "signal.llm.direction_probability_mismatch",
            direction=direction,
            probability=probability,
        )
        return None
    if direction == "NO" and probability > 0.5:
        log.warning(
            "signal.llm.direction_probability_mismatch",
            direction=direction,
            probability=probability,
        )
        return None

    reasoning = str(data.get("reasoning", ""))
    evidence_used: list[str] = [
        str(u["doc_id"])
        for u in updates_applied
        if isinstance(u, dict) and u.get("doc_id")
    ]

    return {
        "prior": prior,
        "prior_basis": prior_basis,
        "updates_applied": updates_applied,
        "posterior": posterior,
        "probability": probability,
        "confidence": confidence,
        "direction": direction,
        "reasoning": reasoning,
        "evidence_used": evidence_used,
    }
