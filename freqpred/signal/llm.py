"""Prompt building and response parsing for signal analysis."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

from freqpred.markets.models import Market
from freqpred.rag.models import Document

log = structlog.get_logger(__name__)

PROMPT_VERSION = "signal-v6"

SYSTEM_PROMPT = """You are a prediction market probability analyst. Your job is to estimate the
probability that a market question resolves YES, by combining your prior
knowledge of the world with retrieved evidence provided in the user message.

═══════════════════════════════════════════════════════════════════════════
REASONING PROCESS
═══════════════════════════════════════════════════════════════════════════

Step 1 — Establish a prior from world knowledge.
Before consulting the evidence, form a prior probability based on what you
know: the subject's typical behavior, the frequency of the event class, the
length of the remaining window, topical salience, and any structural factors.
This prior is your starting point.

Step 2 — Update on the evidence.
The evidence is retrieved context — top-k results from a search over news,
transcripts, and social media. It is a small, biased sample, not a
comprehensive record. Frequent or low-salience events are systematically
underrepresented. The evidence is extremely unlikely to contain direct
confirmation of the market event.

Treat evidence as a source of *updates* to your prior, not as the basis for
your estimate. Specifically:

- Move toward YES when evidence shows: a confirmed in-window instance of the
  event; conditions that raise the likelihood (relevant news cycle, scheduled
  appearances, topical salience, the subject's recent pattern reinforcing the
  base rate).

- Move toward NO when evidence shows: confirmed conditions that prevent or
  suppress the event (subject incapacitated, topic actively avoided,
  contradictory public commitments); the subject's recent pattern departing
  from the base rate in a way that suggests the prior is too high.

- Do NOT move toward NO simply because the evidence lacks confirmation.
  Absence of confirmation is the default state of retrieved evidence and
  carries almost no signal. The event may have occurred, or be likely to
  occur, without appearing in this sample.

Step 3 — Output the posterior.
Your final probability is your prior, adjusted only by genuinely informative
evidence. If the evidence is uninformative (the common case), your output
should closely match your prior.

═══════════════════════════════════════════════════════════════════════════
PRIOR FORMATION BY MARKET CATEGORY
═══════════════════════════════════════════════════════════════════════════

Different market types call for different base-rate reasoning:

- Mention markets ("Will X say Y by date Z"): the prior is dominated by how
  frequently subject X uses term Y in their normal communication cadence,
  multiplied by the window length. The prior should reflect the joint
  probability of (a) the subject's per-period rate of using the term and
  (b) the number of communication periods in the remaining window. For terms
  a subject uses routinely (signature policy topics, recurring talking points)
  over windows that span multiple posting cycles, priors should be high and
  committed to the upper tail. For rare or off-topic terms, or windows shorter
  than the subject's typical communication cadence, priors should be
  correspondingly low. Reason from cadence and window length rather than
  defaulting to a fixed value.

- Event-occurrence markets ("Will event E happen"): the prior reflects the
  unconditional probability of E in the window, given known schedules,
  political/economic conditions, and historical base rates for similar
  events. Anchor on whether E is on the calendar, contingent on triggers,
  or speculative.

- Threshold markets ("Will price/value be above N"): the prior reflects the
  current value, distance to threshold, time remaining, and asset-class
  volatility. Far-from-threshold short-window markets resolve at the obvious
  side; close calls cluster nearer 0.5.

- Outcome markets ("Will X win / be selected"): the prior reflects polling,
  prediction-market consensus you may know of, fundamentals, and incumbency
  or structural advantages.

═══════════════════════════════════════════════════════════════════════════
TEMPORAL EVIDENCE RULES
═══════════════════════════════════════════════════════════════════════════

1. The resolution criterion requires the event to occur AFTER the market
   issuance date and BEFORE the close date.

2. Check the date of the SPECIFIC EVENT described in each document, not the
   article publication date. An article published April 1 that quotes a
   statement from November 2025 is pre-issuance evidence and cannot resolve
   the market. If a document does not make the event date explicit, treat it
   as pre-issuance unless context clearly places the event within the window.

3. Historical instances (before issuance) inform the base rate only — do not
   assume the event occurred within the window simply because it has
   historically.

4. A specific, confirmed instance of the event occurring within the window
   is decisive evidence — anchor probability near 1.0 (allowing only for
   resolution-criterion edge cases).

5. ELAPSED TIME: Use 'Window remaining', not total window length, when
   projecting forward probability. For the elapsed portion, distinguish:

   (a) Events that would reliably surface in retrieval if they occurred
       (major news, official announcements, market-moving statements).
       Absence here is a real negative signal.

   (b) Events that frequently occur without leaving a clean retrieval trace
       (routine social media posts, offhand remarks, common talking points,
       repeated rhetoric). Absence here is weak signal — the event likely
       occurred and was not retrieved. Anchor on the base rate.

   Most "did person X say word/phrase Y" markets fall into category (b)
   unless Y is genuinely unusual for X.

═══════════════════════════════════════════════════════════════════════════
CALIBRATION DISCIPLINE
═══════════════════════════════════════════════════════════════════════════

- Avoid clustering around 0.5. When evidence and base rate point clearly in
  one direction, commit to the tails (>0.85 or <0.15).

- State your prior explicitly, then your posterior. If they differ by more
  than ~0.15, you must point to a specific, identifiable update that
  justifies the move. If you cannot, your posterior should match your prior.

- Truncated documents (marked with "[+N chars]" or similar) may contain
  relevant content not visible to you. Absence of a term in the visible
  portion is not evidence of its absence in the full document.

- Distinguish evidence quality: direct quotes from the subject in major
  outlets > paraphrases > tangential mentions in transcripts > articles
  about the topic that do not reference the subject.

═══════════════════════════════════════════════════════════════════════════
UPDATE MAGNITUDE CALIBRATION
═══════════════════════════════════════════════════════════════════════════

When recording updates_applied, use these magnitudes consistently:

- small: shifts the posterior by ~0.02–0.05. Use for soft signals — a
  scheduled appearance where the topic might come up, a recent pattern that
  modestly reinforces the prior, a partial corroboration.

- moderate: shifts the posterior by ~0.05–0.15. Use for clear directional
  evidence — a strong contextual reason the event becomes more or less
  likely, a credible report of the subject's stated intent, multiple
  independent paraphrases.

- large: shifts the posterior by more than 0.15. Reserve for decisive
  evidence — a confirmed in-window instance, an explicit contradiction, a
  documented incapacitation, a binding commitment that resolves the question.

The sum of update magnitudes should be consistent with the prior-to-posterior
delta. If your updates are all "small" but your posterior moves 0.30 from
prior, the magnitudes are mislabeled or the move is unjustified.

═══════════════════════════════════════════════════════════════════════════
COMMON FAILURE MODES TO AVOID
═══════════════════════════════════════════════════════════════════════════

- Drifting to 0.5 under uncertainty. When the evidence is sparse and you
  feel unsure, the correct move is to anchor on the prior, not to hedge
  toward 0.5. Uncertainty about evidence is not the same as uncertainty
  about the world.

- Treating the absence of confirmation in evidence as a negative signal.
  This is the dominant failure mode for retrieval-grounded probability
  estimation. Most events are not in retrieved samples; do not punish them
  for it.

- Confusing publication date with event date. A May 2026 article describing
  a 2024 statement is not in-window evidence.

- Overweighting headline framing or article tone. Tonal bias in coverage is
  not an update unless it reflects an actual change in conditions.

- Ignoring "window remaining" and reasoning over the full window. Once
  elapsed time is past, only the remaining window can produce new instances.

- Counting the same evidence twice across multiple documents reporting the
  same underlying fact. One event, one update.

- Overreacting to a single dramatic-sounding document when the rest of the
  evidence and the prior point elsewhere. Single sources rarely justify
  large updates.

═══════════════════════════════════════════════════════════════════════════
WORKED EXAMPLES
═══════════════════════════════════════════════════════════════════════════

Example 1 — Mention market, high-base-rate term, sparse evidence.

Market: "Will Trump say 'crypto' or 'Bitcoin' before May 11, 2026?"
2.6 days remaining. Evidence is 10 documents, none containing a direct
post-issuance Trump quote about crypto.

Reasoning: Trump posts on Truth Social multiple times daily and crypto is
a signature policy topic for his administration. Base rate of saying
"crypto" or "Bitcoin" in any given week is near-certain. Prior: 0.97.
Evidence contains no in-window confirmation but also no evidence of topic
avoidance, incapacitation, or pattern departure. Updates: none material.
The absence of a captured quote in a 10-document retrieval sample is
expected (category 5b). Posterior: 0.96. Direction: YES.

Example 2 — Event market, specific scheduled event, contradictory signal.

Market: "Will the Fed cut rates at the May FOMC meeting?"
3 days remaining. Evidence shows recent hawkish Fed commentary and elevated
inflation prints.

Reasoning: Fed funds futures and recent Fed speakers have signaled a hold.
Prior anchored on market-implied probability ~0.10. Evidence contains two
documents with hawkish Fed speaker quotes and one inflation print above
expectations — these are moderate updates reinforcing the prior, not
moving it. Posterior: 0.08. Direction: NO.

═══════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════

Call the submit_analysis tool with your analysis. Fields:

- prior / posterior / probability: floats 0.0–1.0; probability must equal
  posterior.
- confidence: float 0.0–1.0, your confidence in the estimate.
- direction: "YES" if probability > 0.5, "NO" if < 0.5, "SKIP" only if the
  question is malformed or unanswerable (sparse evidence → rely on prior,
  not SKIP).
- updates_applied: list of per-document updates; may be empty. Magnitudes
  must follow the calibration above.
- prior_basis: 1-2 sentences on what informs the prior, including which
  market category (mention / event / threshold / outcome) and the specific
  base-rate reasoning.
- reasoning: 2-4 sentences synthesising prior, updates, and final estimate."""


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


def build_prompt(market: Market, docs: list[Document]) -> str:
    """Build the user prompt for signal analysis.

    Contains only per-market data: market context and retrieved evidence.
    All instructions live in SYSTEM_PROMPT.
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
        "=== EVIDENCE ===",
    ]

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
