"""Live-Opus audit: does adding point-in-time edge-calibration data to the
assessor's prompt improve its trust_score's correlation with actual outcomes?

Ad hoc /goal deliverable, not a tracked T-task. Reuses real production code
paths rather than reimplementing them:
  - freqpred.metrics.assessment._load_source_breakdown /
    _load_similar_market_summary / _build_prompt_payload / _SYSTEM_PROMPT /
    _ASSESSMENT_TOOL — the actual assessor prompt construction.
  - freqpred.replay.recorder._reconstruct_prices — recovers (yes_bid, yes_ask)
    at signal time from the signal's own stored fields, so the live call sees
    the market as it looked when the signal fired, not its (leaky) settled
    state today.
  - freqpred.llm.client.LLMClient with query_type="model_eval" — the same
    non-production query_type scripts/benchmark_signals.py uses for candidate
    calls, so this never writes to signal_assessments (no pollution of the
    real sizing-decision audit trail) but still logs real spend to
    llm_queries under a clearly-labeled bucket.

One live call per sampled signal per arm. Two arms, selectable via --arms:
  current    — the live production package: production _SYSTEM_PROMPT +
               production section loaders + production _PROMPT_VERSION, on
               the production judgment model (config.anthropic.
               judgment_model). Needs no maintenance; it is whatever shipped.
  challenger — the proposed package: defined per experiment via the
               CHALLENGER block near the top of this script (version string,
               system prompt, payload builder, optional judgment-model
               override for model-swap experiments). Undefined by default;
               the run fails loudly if requested without being defined.
Both arms share the same PIT base payload and the same PIT edge-band
calibration, so the paired contrast isolates exactly the challenger's change.
market.days_to_close is corrected to be relative to the signal's own
created_at in both arms (the real function uses wall-clock "now", which for
a since-resolved market leaks "this already closed" — a point-in-time bug
for this audit, not a feature under test).

POINT-IN-TIME REVISION (v2): the first run of this audit used the production
_load_similar_market_summary/_load_source_breakdown verbatim against current
DB state. That leaked outcomes: the sampled markets have since finalized, so
the "exact question" Brier history included the assessed market's OWN
resolution — control correlation jumped to 0.85 vs 0.08 for the real
production assessments, which is answer-leakage, not skill, and it saturated
both conditions. v2 replaces both loaders with point-in-time copies that
(a) exclude the assessed market entirely and (b) only admit markets closed /
positions exited / snapshots computed before the signal's created_at. In
production the equivalent leak cannot occur (the assessed market is never
finalized when the assessor runs), so the PIT copies approximate the
production information set, not a new behavior.

Remaining accepted scope limit: phrase_data is passed as None for both
conditions (production sometimes passes FactBase phrase-frequency context) —
its count_365d field would leak future occurrences for a point-in-time
signal, so it's dropped entirely rather than reconstructed.

CURRENT-VS-CHALLENGER REVISION (v4): earlier revisions pinned each
historical package (assessment-v4/v5/v6) as its own frozen arm to settle the
T94/T95 adoption decisions (results: README → "Auditing the sizing
assessor" reference runs; result CSVs remain in scripts/.audit_output/).
Going forward the harness only ever needs to answer one question — does the
proposed assessor beat the one in production? — so it now runs exactly the
two arms above. --reuse-csv imports per-signal columns for arms NOT being
re-run from a previous run over the same seed/sample, so a budget-
constrained day pays only for the new arm(s).

Note: the shipped edge_band_calibration loader reads the
edge_calibration_scores snapshot table, which has no as_of filter
(production always wants the latest snapshot — fine there, leaky here).
Both arms therefore use this script's PIT _edge_band_calibration
computation; it is byte-identical across the two arms, so it cancels out of
the current-vs-challenger contrast.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import uuid
from datetime import datetime

import anthropic
import pandas as pd
from sqlalchemy import select

import freqpred.ingestion.models  # noqa: F401 — registers mapper
import freqpred.rag.models  # noqa: F401
from freqpred.config import load_config
from freqpred.db import make_engine, make_session_factory
from freqpred.llm.client import LLMClient
from freqpred.markets.models import Market, MarketRow
from freqpred.metrics.assessment import (
    _ASSESSMENT_TOOL,
    _PROMPT_VERSION,
    _SYSTEM_PROMPT,
    _brier_from_rows,
    _build_prompt_payload,
    _clamp_multiplier,
    _load_market_reevaluation_history,
    _parse_assessment_response,
    _question_first_line,
    _trust_score_to_multiplier,
)
from freqpred.metrics.models import SignalAssessmentRow, SourceQualityScoreRow
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.replay.recorder import _reconstruct_prices
from freqpred.signal.models import Signal, SignalRow
from freqpred.strategy.loader import load_strategy

SEED = 42
SAMPLE_N = 30
MAX_PER_MARKET = 2
JUDGMENT_QUERY_TYPE = "model_eval"  # never "signal_assessment" — no DB pollution
EDGE_BINS = [-100, 0, 15, 40, 200]
EDGE_LABELS = ["<0", "0-15", "15-40", ">40"]
ARM_NAMES = ("current", "challenger")

# ---------------------------------------------------------------------------
# CHALLENGER definition — edit this block when screening a proposed change.
#
# The `current` arm always runs the live production package (production
# _SYSTEM_PROMPT + production section loaders) and needs no maintenance.
# The `challenger` arm is whatever you are proposing: set the three hooks
# below before running with challenger in --arms. Leaving them as None makes
# the run fail loudly instead of silently measuring current-vs-current.
#
#   CHALLENGER_VERSION       — prompt_version logged to llm_queries, e.g.
#                              "assessment-v7-audit-pit-challenger"
#   CHALLENGER_SYSTEM_PROMPT — the proposed system prompt, verbatim
#   CHALLENGER_MODEL         — judgment model override for the challenger arm
#                              only, e.g. "claude-opus-4-8"; None = same model
#                              as production (config.anthropic.judgment_model)
#   _challenger_payload      — builds the proposed payload from the same
#                              PIT base payload + PIT edge calibration the
#                              current arm gets, so the contrast isolates
#                              exactly what you changed
#
# One axis per experiment. For a model-only swap: set CHALLENGER_MODEL, set
# CHALLENGER_SYSTEM_PROMPT = _SYSTEM_PROMPT (production, unchanged), and make
# _challenger_payload mirror the current arm exactly:
#
#     base_payload["edge_band_calibration"] = edge_calib
#     base_payload["market_reevaluation_history"] = (
#         await _load_market_reevaluation_history(session, signal)
#     )
#     return base_payload
# ---------------------------------------------------------------------------
# ADOPTED 2026-07-25: assessment-v8 + claude-opus-5 shipped to production
# (freqpred/metrics/assessment.py). The block below is kept as the record of the
# adopted package; redefine it for the next experiment.
#
# Screened on the frozen, direction-balanced 76-signal eval set
# (scripts/freeze_assessor_eval_set.py + scripts/run_frozen_eval.py), NOT the old
# reshuffling _pick_sample draw -- the identical v6/opus-4-7 package had scored
# corr +0.529 and +0.246 on two consecutive draws, so sample composition was
# swamping the package effect.
#
# Result: capital tilt (mean multiplier on winners minus losers) +0.0751x vs
# +0.0170x, 95% CI on the paired diff (+0.0161, +0.0988) -- the only significant
# arm difference found across the entire screening effort. Ranking was a wash
# (AUC 0.700 vs 0.674; corr diff +0.033, CI -0.172..+0.236). v6 was effectively
# inert on this set (sd 0.027, six distinct scores, 76/76 size_down); v8 issued
# 19 size_up verdicts at a 78.9% hit rate against a 54.1% base.
#
# Shipped with three known limitations, recorded in SPEC.md:
#   * v8's advantage is almost entirely BETWEEN directions -- within-YES AUC is
#     0.294, below random -- so it has largely learned a direction base rate.
#   * Neither arm beat a free direction x band lookup (prior AUC 0.685, v8
#     increment +0.014, CI -0.077..+0.067).
#   * The audited package also filtered calibration history to the signal's own
#     prompt-version cohort; production cannot do this yet (edge_calibration_scores
#     has no prompt_version column), so it was omitted from the shipped prompt.
#     It applied to only 38 of 76 audited signals in any case.
CHALLENGER_VERSION: str | None = None
CHALLENGER_MODEL: str | None = None

# Deliberately a BUNDLED change (prompt v7 + opus-5), at the user's direction:
# staying on opus-4-7 indefinitely is not viable, so the two axes move together.
# Interpretive cost, recorded up front: a win does not attribute between the
# prompt and the model, and a loss does not say which half dragged. The
# reasoning-text diff is the only attribution available afterwards — if the v7
# reference-class framing shows up in the challenger's reasoning and its scores
# spread out, that is prompt; if the scores stay compressed in the 0.12-0.28
# band the way the v6/opus-5 screen did, that is the model.
#
# v7 targets the three failure modes found in the 2026-07-24 opus-5 screen
# (corr +0.355 vs +0.529, rejected):
#   1. Within-family discrimination collapsed to noise (r=+0.155 on the n=20
#      KXTRUMPSAY subset, vs +0.665 for opus-4-7) because the judge anchored on
#      family/band statistics that are IDENTICAL across every signal in a series
#      and therefore carry zero discriminating information. v7 makes the
#      reference-class-then-move-off-it structure explicit and demands the score
#      be non-transferable between siblings.
#   2. The v6 rule "any warning => trust_score MUST be <= 0.5" is a one-sided
#      ceiling. opus-5 emitted warnings on 30/30 responses, so it could never
#      score above neutral; across all 60 calls in both arms of that screen,
#      exactly one response exceeded 0.5 — the only one with zero warnings.
#      A first v7 attempt tried to fix this by raising the BAR for what counts as
#      a warning (material concerns only). That failed: opus-5 still found three
#      "material" concerns on 8/8 calls and still never exceeded 0.36. So v7 now
#      severs the coupling outright — warnings are a pure audit annotation with no
#      arithmetic tie to trust_score. NOTE this is a deliberate loosening of the
#      assessor's risk posture: it is the first package that can recommend sizing
#      ABOVE 1.0x while holding reservations. Nothing changes in production sizing
#      from this run (audits never write to signal_assessments), but adopting it
#      means the assessor gains genuine upside authority up to
#      assessment_scale_max. The audit is what tells us whether its above-0.5
#      judgments are actually predictive — the v6 ceiling made that unmeasurable,
#      since there was almost no above-0.5 data to evaluate.
#   3. Score compression (opus-5 sd 0.0824 over 7 distinct values, range
#      0.12-0.45; opus-4-7 sd 0.1006 over 10 values, range 0.10-0.62). v7 asks
#      for the full range explicitly.
# The liquidity-note fix that was the fourth finding is handled in the harness
# (_mask_unreconstructible_liquidity) rather than the prompt, because it was a
# point-in-time leak affecting BOTH arms, not a prompt defect.
# The adopted v8 prompt, kept verbatim as the record of what was screened.
# It is NOT wired to CHALLENGER_SYSTEM_PROMPT: v8 is production now, so
# arming it would measure current-vs-current and burn a paid run on a
# guaranteed null result. Set CHALLENGER_SYSTEM_PROMPT = your new prompt.
ADOPTED_V8_SYSTEM_PROMPT = """\
You are a risk and sizing judge for a prediction-market trading system.

The trade direction and base edge have already been decided upstream.
Do not re-predict the market outcome.
Do not change the trade direction.
Do not discuss exits or stop losses.
Your task is only to judge how much we should trust the base position sizing.

How to reach a score:
- Start from the reference class. Family-level statistics — family and \
exact-question Brier, and the strategy's win rate and mean PnL for this series — \
describe the population this signal belongs to. Within a single series they are \
frequently IDENTICAL for every signal, so on their own they cannot distinguish \
this signal from its siblings. Use them to set a starting point, and say what \
that starting point is. (Across DIFFERENT series they do discriminate, and there \
they are a primary signal — weight them fully when the comparison is between \
families.)
- Then move off that starting point using evidence specific to THIS signal: the \
trajectory in `market_reevaluation_history`, the source mix and \
weighted_delta_vs_overall in `source_quality_summary`, the exact-question subset \
where its sample is meaningful, days_to_close, and any genuine liquidity data. \
If nothing signal-specific distinguishes it, say so and stay at the starting point.
- A score that could be copied unchanged onto any other signal in the same series \
has not done its job. Two signals in the same family with different trajectories \
should not receive the same score.
- Use the full 0.0-1.0 range. A score above 0.5 is the correct output when the \
signal-specific evidence is favourable, even if the family baseline is weak. Do \
not compress every judgment into a narrow band.

Guidelines:
- Be conservative when sample sizes are small, mixed, or noisy.
- Prefer neutral output when the data is weak or conflicting.
- Exact-family history is more important than broad analogies.
- If source quality and similar-market history disagree, explain the conflict \
and stay closer to neutral unless one side clearly has stronger data.
- Treat this as a sizing-confidence judgment, not a market-prediction task.
- `edge_band_calibration` is the most important block, and \
`profit_edge_vs_price` is the figure that matters: hit_rate minus the average \
price actually paid. Positive means signals in this cell historically BEAT the \
price they traded at; negative means they lost money. Let that govern. Prefer \
`same_direction_only` wherever its sample is adequate — trade direction is a \
strong and persistent discriminator in this data, and it is one of the few \
inputs that genuinely varies between signals.
- `avg_model_implied_p` is the model's own claim about itself. The gap between \
it and hit_rate measures the model's SELF-KNOWLEDGE, which is a different \
question from whether the trade earns money, and the two frequently disagree: a \
cell can be badly overconfident and still profitable, or look perfectly \
calibrated and still lose. Do NOT size down on an overconfidence gap alone. \
Check profit_edge_vs_price first; where the two conflict, profit wins. Mention \
the gap only if it changes your conclusion.
- The raw size of this signal's own edge is not itself evidence of anything. A \
large edge in a cell with a positive profit_edge_vs_price is not suspect.
- When `market_reevaluation_history` is present, read `sampled_history` for \
the actual trajectory: is the edge widening or narrowing, are the model's \
and the market's probabilities converging or diverging, and is the traded \
direction stable (see direction_change_count)? Judge the observed \
trajectory on its own terms — no single pattern is inherently suspect.
- Any field marked 'unavailable_at_signal_time' is UNKNOWN, not zero. Draw no \
inference from it in either direction.

Call the submit_assessment tool with your judgment.

trust_score is the only output that affects position sizing; reasoning, \
key_factors, and warnings exist solely so the decision can be audited later. \
Spend your output budget accordingly — do not restate the payload, do not \
recite figures that are already in the input, and do not explain your method. \
Emit trust_score as the FIRST field in the tool call, before any prose.

- trust_score: 0.0–1.0; 0.5 = neutral, < 0.5 = lower confidence, > 0.5 = higher confidence
- reasoning: at most 2 sentences. Give the reference-class starting point as a \
number, then what moved you off it and by roughly how much. Nothing else.
- key_factors: 1-3 short strings — routine observations and background context \
belong here
- warnings: 0-3 short strings recording concerns a reviewer should see. This is \
an audit annotation and has NO arithmetic relationship to trust_score. Noting a \
concern does not oblige you to lower the score, and a trust_score above 0.5 \
alongside a genuine warning is a legitimate, expected combination when the \
signal-specific evidence supports it. Record what concerned you, then set \
trust_score on the merits.
"""


def _add_profit_edge(edge_calib: dict | None) -> dict | None:
    """Surface `profit_edge_vs_price` = hit_rate - avg_market_implied_p.

    Every figure needed for this was already in the payload; what was missing was
    the subtraction and the instruction to care about it. The shipped description
    directs the judge at hit_rate vs avg_model_implied_p, which measures the
    model's SELF-CONSISTENCY, not whether the trade makes money. Measured
    2026-07-24 across the 34 calibration cells with n>=50, the two correlate only
    +0.397 and 8 cells covering 2,880 signals point opposite ways — e.g.
    KXTRUMPSAY/NO/>40 earns +0.043 over the price while showing a -0.558
    overconfidence gap, so a judge following the shipped wording sizes DOWN a
    profitable band. Applied to the challenger arm only; the current arm keeps the
    production payload so the contrast stays clean.
    """
    if edge_calib is None:
        return None
    out = dict(edge_calib)
    for key in ("all_directions", "same_direction_only", "this_direction_all_bands"):
        cell = out.get(key)
        if not isinstance(cell, dict) or "hit_rate" not in cell:
            continue
        cell = dict(cell)
        cell["profit_edge_vs_price"] = round(
            cell["hit_rate"] - cell["avg_market_implied_p"], 4
        )
        out[key] = cell
    out["description"] = (
        "Empirical calibration computed ONLY from markets that had already closed "
        "before this signal fired (no lookahead), and ONLY from signals produced "
        "by the same signal prompt version as this one (see "
        "cohort_prompt_version) — historical performance is strongly "
        "version-dependent, so older eras describe a model this one no longer "
        "resembles. profit_edge_vs_price = hit_rate minus avg_market_implied_p is "
        "what signals in this cell actually EARNED relative to the price paid: "
        "positive means the cell historically beat its price, negative means it "
        "lost money. That is the figure that determines profitability. "
        "avg_model_implied_p is the model's own claim; the gap between it and "
        "hit_rate measures the model's self-knowledge, which is a DIFFERENT "
        "question and frequently points the other way. "
        "this_direction_all_bands aggregates every band for this signal's traded "
        "direction — direction is the single most persistent discriminator in "
        "this data, so consult it when the band-level cell is thin."
    )
    return out


CHALLENGER_SYSTEM_PROMPT: str | None = None


async def _challenger_payload(
    session, signal: Signal, base_payload: dict, edge_calib: dict | None
) -> dict:
    """Build the challenger arm's payload. Edit per experiment.

    Disarmed after the v8 adoption (2026-07-25): v8 is production, so wiring the
    old challenger back up would measure current-vs-current and spend real money
    on a guaranteed null result. For a model-only swap, mirror the current arm:

        base_payload["edge_band_calibration"] = edge_calib
        base_payload["market_reevaluation_history"] = (
            await _load_market_reevaluation_history(session, signal)
        )
        return base_payload

    The v8 shape is preserved above as ADOPTED_V8_SYSTEM_PROMPT and _add_profit_edge
    if you need to diff a new proposal against what was actually screened.
    """
    raise NotImplementedError(
        "define the challenger package (CHALLENGER_VERSION, "
        "CHALLENGER_SYSTEM_PROMPT, CHALLENGER_MODEL, _challenger_payload) "
        "before running with the challenger arm"
    )


def _edge_band(edge_pct: float) -> str:
    for lo, hi, label in zip(EDGE_BINS[:-1], EDGE_BINS[1:], EDGE_LABELS, strict=True):
        if lo <= edge_pct < hi:
            return label
    return EDGE_LABELS[-1]


async def _pick_sample(session) -> list[uuid.UUID]:
    result = await session.execute(
        select(SignalAssessmentRow.signal_id, SignalRow.market_id, SignalRow.created_at)
        .join(SignalRow, SignalRow.id == SignalAssessmentRow.signal_id)
        .where(SignalAssessmentRow.llm_query_id.is_not(None))
    )
    rows = result.all()
    rng = random.Random(SEED)
    rows = list(rows)
    rng.shuffle(rows)
    per_market: dict[str, int] = {}
    picked: list[uuid.UUID] = []
    for signal_id, market_id, _ in rows:
        if per_market.get(market_id, 0) >= MAX_PER_MARKET:
            continue
        per_market[market_id] = per_market.get(market_id, 0) + 1
        picked.append(signal_id)
        if len(picked) >= SAMPLE_N:
            break
    return picked


def _row_to_signal(row: SignalRow) -> Signal:
    return Signal(
        id=str(row.id),
        market_id=row.market_id,
        estimated_probability=row.estimated_probability,
        confidence=row.confidence,
        edge=row.edge,
        market_mid_at_signal=row.market_mid_at_signal,
        direction=row.direction,
        reasoning=row.reasoning,
        sources=list(row.sources or []),
        retrieval_hash=row.retrieval_hash,
        model_used=row.model_used,
        prompt_version=row.prompt_version,
        trigger=row.trigger,
        created_at=row.created_at,
        raw_context=row.raw_context,
        market_ask_at_signal=row.market_ask_at_signal,
        social_sentiment_summary=row.social_sentiment_summary,
    )


def _row_to_market_at_signal_time(row: MarketRow, signal: Signal) -> Market:
    """Historical Market snapshot: prices reconstructed at signal time, never
    the (settled, leaky) current DB state."""
    yes_bid, yes_ask = _reconstruct_prices(
        signal.direction,
        signal.market_mid_at_signal,
        signal.market_ask_at_signal,
        signal.estimated_probability,
        signal.edge,
    )
    return Market(
        id=row.id,
        platform=row.platform,
        question=row.question,
        category=row.category,
        close_time=row.close_time,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        mid_price=signal.market_mid_at_signal,
        volume_24h=row.volume_24h,
        open_interest=row.open_interest,
        last_fetched_at=row.last_fetched_at,
        price_updated_at=row.price_updated_at,
        metadata_fetched_at=row.metadata_fetched_at,
        current_signal_id=None,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
        open_time=row.open_time,
        status=row.status,
        result=None,  # never pass settlement result into the prompt
        settlement_value=None,
        last_price=row.last_price,
        yes_bid_size=row.yes_bid_size,
        yes_ask_size=row.yes_ask_size,
        series_ticker=row.series_ticker,
        volume_total=row.volume_total,
        settlement_sources=[],
    )


def _fix_days_to_close(payload: dict, market: Market, signal_created_at: datetime) -> dict:
    days = (market.close_time - signal_created_at).total_seconds() / 86400
    payload["market"]["days_to_close"] = round(days, 1)
    return payload


LIQUIDITY_UNAVAILABLE = "unavailable_at_signal_time"
BOOK_UNRECONSTRUCTIBLE = "could_not_be_reconstructed"


def _book_reconstruction_is_consistent(signal: Signal) -> bool:
    """Whether _reconstruct_prices yields a physically possible book.

    _reconstruct_prices recovers only the TRADED side directly from stored data
    (market_ask_at_signal is the real cost paid); the other side is *derived* by
    reflecting around the mid — yes_bid = 2*mid - yes_ask for YES, and the mirror
    for NO. That reflection is sound only if market_mid_at_signal really is the
    midpoint of the same book snapshot as market_ask_at_signal. On some signals it
    is not: the two fields were captured on different poll cycles, so the price
    moved between reads. Measured 2026-07-24 on the n=30 audit sample, 3 signals
    (10%) reflect to bid > ask, one at a -0.280 spread — e.g. KXTRUMPSAY-26JUL06-AUTO
    stores mid 0.35 against ask 0.33, and a YES mid cannot exceed its own ask.

    A crossed book is not a market state; it is a reconstruction failure. Feeding
    one to the judge invites it to discount a signal for degenerate pricing that
    never existed — observed live: "the inverted quoted spread (bid 0.37 > ask 0.33)
    ... keep[s] it from moving further". That is an artifact penalty, and it is not
    reliably symmetric across arms, since a judge that weights liquidity more
    heavily absorbs more of it.
    """
    try:
        yes_bid, yes_ask = _reconstruct_prices(
            signal.direction,
            signal.market_mid_at_signal,
            signal.market_ask_at_signal,
            signal.estimated_probability,
            signal.edge,
        )
    except Exception:  # noqa: BLE001 — unreconstructible is unreliable by definition
        return False
    return yes_bid <= yes_ask


def _mask_unreconstructible_liquidity(payload: dict, *, book_reliable: bool = True) -> dict:
    """Blank the order-book fields that cannot be recovered point-in-time.

    _reconstruct_prices recovers yes_bid/yes_ask (and therefore spread) from the
    signal's own stored fields, but book depth, 24h volume, open interest, and
    price_updated_at are read straight off the current MarketRow. Every market in
    this sample has since finalized, and those fields settle to zero — measured
    2026-07-24: all 6 sampled markets spot-checked had yes_bid_size = yes_ask_size
    = 0, against 154/74,634 live active markets. So production sees a populated
    book while the audit prompt saw "zero depth, zero volume" on nearly every
    signal, and the shipped note tells the judge a large edge in that context is
    "likely artificial". That is post-resolution state leaking in as a uniformly
    bearish input — the same class of bug already fixed here for days_to_close and
    phrase_data. It is not neutral across arms: it biases the contrast against
    whichever arm weights liquidity more heavily (in the 2026-07-24 opus-5 screen,
    that arm cited illiquidity in 29/30 responses vs 23/30 for opus-4-7).

    Prices and spread stay — they are genuinely reconstructed. The rest is marked
    unknown so the judge discounts the section rather than reading absence as
    thinness.
    """
    liq = payload.get("market_liquidity")
    if not isinstance(liq, dict):
        return payload
    for key in (
        "yes_bid_size_dollars",
        "yes_ask_size_dollars",
        "volume_24h",
        "open_interest",
        "price_updated_at",
    ):
        if key in liq:
            liq[key] = LIQUIDITY_UNAVAILABLE
    note = (
        "Book depth, 24h volume, open interest, and the price timestamp could not "
        f"be reconstructed for this historical signal and are marked "
        f"'{LIQUIDITY_UNAVAILABLE}'. Treat them as UNKNOWN, not as zero or thin — "
        "draw no liquidity inference from their absence in either direction. "
    )
    if book_reliable:
        note += (
            "yes_bid, yes_ask, and spread are genuine signal-time values and may be used."
        )
    else:
        # Only the traded side is stored directly; the other side is derived by
        # reflecting around the mid, and on this signal the stored mid and ask are
        # mutually inconsistent so the reflection crosses. Masking both sides is
        # the honest move — publishing one real price beside one impossible one
        # would still let the judge compute a bogus spread.
        for key in ("yes_bid", "yes_ask", "spread"):
            if key in liq:
                liq[key] = BOOK_UNRECONSTRUCTIBLE
        note += (
            "The bid/ask book could not be reconstructed consistently for this "
            f"signal and is marked '{BOOK_UNRECONSTRUCTIBLE}'. This is a limitation "
            "of historical reconstruction, NOT evidence of a crossed or degenerate "
            "market — do not treat it as a sign of bad pricing. The traded-side "
            "cost and the market's own probability remain available in "
            "trade_context and market_reevaluation_history; use those instead."
        )
    liq["note"] = note
    return payload


async def _load_source_breakdown_pit(session, signal: Signal, market: Market, as_of: datetime) -> list[dict]:
    """Point-in-time copy of assessment._load_source_breakdown: only source-
    quality snapshots computed before *as_of* are eligible."""
    from sqlalchemy import case, func, or_  # noqa: PLC0415

    counts_result = await session.execute(
        select(DocumentRow.source_name, func.count(DocumentRow.id).label("doc_count"))
        .join(DocumentMarketLinkRow, DocumentMarketLinkRow.document_id == DocumentRow.id)
        .where(DocumentMarketLinkRow.signal_id == uuid.UUID(signal.id))
        .group_by(DocumentRow.source_name)
        .order_by(func.count(DocumentRow.id).desc(), DocumentRow.source_name.asc())
    )
    counts = [(row.source_name, int(row.doc_count)) for row in counts_result.all()]
    total_docs = sum(count for _, count in counts)
    if total_docs == 0:
        return []
    breakdown: list[dict] = []
    for source_name, doc_count in counts:
        snapshot_result = await session.execute(
            select(SourceQualityScoreRow)
            .where(
                SourceQualityScoreRow.source_name == source_name,
                SourceQualityScoreRow.computed_at < as_of,  # PIT
                or_(
                    SourceQualityScoreRow.market_category == market.category,
                    SourceQualityScoreRow.market_category.is_(None),
                ),
            )
            .order_by(
                case((SourceQualityScoreRow.market_category == market.category, 0), else_=1),
                SourceQualityScoreRow.computed_at.desc(),
            )
            .limit(1)
        )
        snapshot = snapshot_result.scalar_one_or_none()
        if snapshot is None:
            continue
        breakdown.append(
            {
                "source_name": source_name,
                "document_share": round(doc_count / total_docs, 6),
                "doc_count": doc_count,
                "weighted_brier": float(snapshot.weighted_brier),
                "overall_brier": float(snapshot.overall_brier),
                "delta_vs_overall": float(snapshot.weighted_brier - snapshot.overall_brier),
                "n_signals": int(snapshot.n_signals),
                "total_doc_uses": int(snapshot.total_doc_uses),
                "market_category_used": snapshot.market_category,
                "lookback_days": int(snapshot.lookback_days),
                "computed_at": snapshot.computed_at.isoformat(),
            }
        )
    return breakdown


async def _load_similar_market_summary_pit(
    session, market: Market, strategy_name: str, as_of: datetime, *, min_signals: int, min_trades: int
) -> dict:
    """Point-in-time copy of assessment._load_similar_market_summary.

    Two filters added everywhere: the assessed market itself is excluded, and
    only markets closed (or positions exited) before *as_of* are admitted —
    matching what the assessor could actually have known at signal time.
    """
    from sqlalchemy import case  # noqa: PLC0415

    from freqpred.markets.models import PositionRow  # noqa: PLC0415

    if not market.series_ticker:
        return {"available": False, "reason": "missing_series_ticker"}

    signal_rows_result = await session.execute(
        select(
            SignalRow.estimated_probability,
            case((MarketRow.result == "yes", 1), else_=0).label("resolution"),
            MarketRow.question,
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(
            MarketRow.series_ticker == market.series_ticker,
            MarketRow.status == "finalized",
            MarketRow.result.is_not(None),
            MarketRow.id != market.id,          # PIT: never the assessed market
            MarketRow.close_time < as_of,       # PIT: resolved before signal
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
    )
    signal_rows = [
        (float(r.estimated_probability), int(r.resolution), r.question)
        for r in signal_rows_result.all()
    ]
    family_pairs = [(p, y) for p, y, _ in signal_rows]
    family_brier = _brier_from_rows(family_pairs)

    overall_result = await session.execute(
        select(
            SignalRow.estimated_probability,
            case((MarketRow.result == "yes", 1), else_=0).label("resolution"),
        )
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(
            MarketRow.status == "finalized",
            MarketRow.result.is_not(None),
            MarketRow.id != market.id,          # PIT
            MarketRow.close_time < as_of,       # PIT
            SignalRow.model_used != "demo_harness",
            SignalRow.prompt_version != "demo",
        )
    )
    overall_pairs = [(float(r.estimated_probability), int(r.resolution)) for r in overall_result.all()]
    overall_brier = _brier_from_rows(overall_pairs)

    first_line = _question_first_line(market.question)
    exact_pairs = [(p, y) for p, y, q in signal_rows if _question_first_line(q) == first_line]
    exact_brier = _brier_from_rows(exact_pairs)

    strategy_rows_result = await session.execute(
        select(PositionRow.pnl_pct, PositionRow.pnl)
        .join(MarketRow, MarketRow.id == PositionRow.market_id)
        .where(
            PositionRow.strategy_name == strategy_name,
            PositionRow.status == "closed",
            PositionRow.exit_time < as_of,      # PIT
            MarketRow.id != market.id,          # PIT
            MarketRow.series_ticker == market.series_ticker,
        )
    )
    strategy_rows = [(float(r.pnl_pct), float(r.pnl)) for r in strategy_rows_result.all()]

    baseline_result = await session.execute(
        select(PositionRow.pnl_pct, PositionRow.pnl).where(
            PositionRow.strategy_name == strategy_name,
            PositionRow.status == "closed",
            PositionRow.exit_time < as_of,      # PIT
            PositionRow.market_id != market.id,  # PIT
        )
    )
    baseline_rows = [(float(r.pnl_pct), float(r.pnl)) for r in baseline_result.all()]

    family_signal_count = len(family_pairs)
    family_trade_count = len(strategy_rows)
    strategy_mean = (
        sum(p for p, _ in strategy_rows) / family_trade_count if family_trade_count > 0 else None
    )
    baseline_mean = (
        sum(p for p, _ in baseline_rows) / len(baseline_rows) if baseline_rows else None
    )
    win_rate = (
        sum(1 for _, pnl in strategy_rows if pnl > 0.0) / family_trade_count
        if family_trade_count > 0
        else None
    )
    available = family_signal_count >= min_signals or family_trade_count >= min_trades
    summary: dict = {
        "available": available,
        "match_rule": "series_ticker",
        "series_ticker": market.series_ticker,
        "family_match": {
            "resolved_signals": family_signal_count,
            "family_signal_brier": family_brier,
            "family_signal_delta_vs_overall": (
                family_brier - overall_brier
                if family_brier is not None and overall_brier is not None
                else None
            ),
        },
        "exact_question_subset": {
            "resolved_signals": len(exact_pairs),
            "signal_brier": exact_brier,
            "signal_delta_vs_overall": (
                exact_brier - overall_brier
                if exact_brier is not None and overall_brier is not None
                else None
            ),
            "small_sample": 0 < len(exact_pairs) < min_signals,
        },
        "strategy_trade_history": {
            "strategy_name": strategy_name,
            "closed_trades": family_trade_count,
            "win_rate": win_rate,
            "mean_pnl_pct": strategy_mean,
            "delta_vs_strategy_overall_mean_pnl_pct": (
                strategy_mean - baseline_mean
                if strategy_mean is not None and baseline_mean is not None
                else None
            ),
        },
        "minimums": {"min_signals": min_signals, "min_trades": min_trades},
    }
    if not available:
        summary["reason"] = "insufficient_matched_history"
    return summary


MIN_PRIOR_CELL = 30


def _pit_baseline_prior(calib_df: pd.DataFrame, signal: Signal, as_of: datetime) -> float:
    """Free, LLM-free baseline score for this signal: the historical PROFIT edge
    of its (direction, edge band) cell, computed point-in-time.

    This is the bar any LLM package has to clear to be worth its cost. Measured
    2026-07-24, a direction-only lookup scored AUC 0.604 on the audit sample
    against 0.624 for the live v6/opus-4-7 package — i.e. the entire measured
    contribution of the LLM was +0.020 AUC.

    The cell statistic is `hit_rate - avg_market_implied_p`, NOT the
    `hit_rate - avg_model_implied_p` gap the prompt has directed the judge at
    since v4. Those diverge: across the 34 calibration cells with n>=50 they
    correlate only +0.397, and 8 cells covering 2,880 signals actively conflict.
    KXTRUMPSAY/NO/>40 is the clearest — it earns +0.043 over the market price
    (profitable) while showing a -0.558 overconfidence gap, so a judge told to
    treat that gap as the warning sign sizes DOWN a profitable band. Profit over
    the price is what pays; model self-consistency is not.

    Falls back cell -> direction -> global pool when a cell is too thin, so the
    baseline is always defined.

    Returns the cell's HIT RATE, not its profit edge, because the audit's AUC is
    measured against `hit` and the baseline must be matched to the target it is
    being scored on. (Profit edge is the economically meaningful statistic and is
    what the PROMPT should surface to the judge — but as an AUC-vs-hit baseline it
    scored 0.434, below random, on the 2026-07-24 sample.) Both are recorded so
    the economic and ranking views can be read separately.
    """
    prior = calib_df[calib_df["close_time"] < as_of]
    if prior.empty:
        return 0.5
    cell = prior[
        (prior["direction"] == signal.direction)
        & (prior["edge_band"] == _edge_band(signal.edge * 100.0))
    ]
    if len(cell) < MIN_PRIOR_CELL:
        cell = prior[prior["direction"] == signal.direction]
    if len(cell) < MIN_PRIOR_CELL:
        cell = prior
    return float(cell["hit"].mean())


def _pit_baseline_profit_edge(calib_df: pd.DataFrame, signal: Signal, as_of: datetime) -> float:
    """Same cell, but the PROFIT edge (hit_rate - avg market price paid).

    Recorded alongside the hit-rate prior for the economic view; see the note in
    _pit_baseline_prior on why the two are not interchangeable.
    """
    prior = calib_df[calib_df["close_time"] < as_of]
    if prior.empty:
        return 0.0
    cell = prior[
        (prior["direction"] == signal.direction)
        & (prior["edge_band"] == _edge_band(signal.edge * 100.0))
    ]
    if len(cell) < MIN_PRIOR_CELL:
        cell = prior[prior["direction"] == signal.direction]
    if len(cell) < MIN_PRIOR_CELL:
        cell = prior
    return float(cell["hit"].mean() - cell["market_p_side"].mean())


async def build_calibration_pool(session) -> pd.DataFrame:
    """Resolved-signal pool backing every point-in-time calibration lookup.

    Public (no underscore) because scripts/freeze_assessor_eval_set.py renders the
    frozen fixtures with the identical pool — a frozen payload built from a
    different pool than the audit uses would defeat the purpose.
    """
    rows = (
        await session.execute(
            select(
                SignalRow.market_id,
                SignalRow.direction,
                SignalRow.edge,
                SignalRow.estimated_probability,
                SignalRow.market_ask_at_signal,
                SignalRow.created_at,
                SignalRow.prompt_version,
                MarketRow.close_time,
                MarketRow.result,
                MarketRow.status,
            )
            .join(MarketRow, MarketRow.id == SignalRow.market_id)
            .where(SignalRow.trigger == "scheduled", SignalRow.confidence >= 0.60)
        )
    ).all()
    records = []
    for r in rows:
        resolved = r.status == "finalized" and r.result in ("yes", "no")
        records.append(
            {
                "market_id": r.market_id,
                "direction": r.direction,
                "edge_band": _edge_band(r.edge * 100.0),
                "prompt_version": r.prompt_version,
                "close_time": r.close_time,
                "resolved": resolved,
                "hit": (
                    (r.direction == "YES" and r.result == "yes")
                    or (r.direction == "NO" and r.result == "no")
                )
                if resolved
                else None,
                "p_side": (
                    r.estimated_probability
                    if r.direction == "YES"
                    else 1.0 - r.estimated_probability
                ),
                "market_p_side": r.market_ask_at_signal,
            }
        )
    df = pd.DataFrame(records)
    return df[df["resolved"]].copy()


MIN_VERSION_COHORT = 200


def _version_cohort(prior: pd.DataFrame, signal: Signal) -> tuple[pd.DataFrame, str]:
    """Restrict history to signals produced by the SAME signal prompt version.

    Measured 2026-07-24 on KXTRUMPSAY, the NO-side profit edge is strongly
    version-dependent: -0.240 (signal-v7), -0.067 (v4), +0.120 (v9), +0.133 (v11,
    current). Pooling all versions yields +0.073 — roughly half the current
    regime's, and contaminated by eras the production model no longer resembles.
    YES, by contrast, is negative under every version (-0.023 to -0.364), so the
    direction asymmetry itself transfers even though its magnitude does not.

    Falls back to the full pool when the cohort is too thin, so a freshly-shipped
    prompt version does not start from no history at all.
    """
    cohort = prior[prior["prompt_version"] == signal.prompt_version]
    if len(cohort) >= MIN_VERSION_COHORT:
        return cohort, signal.prompt_version
    return prior, "all_versions_fallback"


def _edge_band_calibration(calib_df: pd.DataFrame, signal: Signal, as_of: datetime) -> dict:
    """Point-in-time mirror of production's `_load_edge_band_calibration`.

    This MUST track production's shape exactly. The `current` arm is defined as
    "whatever shipped", so any divergence means the baseline is being measured on
    a payload production never sends, and every adoption decision inherits that
    error. It exists as a separate implementation only because production reads
    the `edge_calibration_scores` snapshot table, which has no as_of filter —
    fine live (it always wants the latest), leaky for a retrospective audit.

    Mirrors, as of the v8 adoption (2026-07-25):
      - `profit_edge_vs_price` on every cell, the figure that determines whether
        a cell earned money rather than whether the model knew itself;
      - `this_direction_all_bands`, because the band cell is often thin while
        direction is the most persistent discriminator in the data;
      - prompt-version cohort filtering, with `cohort_prompt_version` naming the
        cohort actually used.
    """
    band = _edge_band(signal.edge * 100.0)
    prior = calib_df[calib_df["close_time"] < as_of]
    cohort, cohort_label = _version_cohort(prior, signal)
    same_band = cohort[cohort["edge_band"] == band]

    def _summarize(df: pd.DataFrame) -> dict:
        if len(df) == 0:
            return {"n_signals": 0, "n_markets": 0}
        return {
            "n_signals": int(len(df)),
            "n_markets": int(df["market_id"].nunique()),
            "hit_rate": round(float(df["hit"].mean()), 3),
            "avg_market_implied_p": round(float(df["market_p_side"].mean()), 3),
            "avg_model_implied_p": round(float(df["p_side"].mean()), 3),
        }

    return _add_profit_edge(
        {
            "this_signal_edge_band": band,
            "cohort_prompt_version": cohort_label,
            "all_directions": _summarize(same_band),
            "same_direction_only": _summarize(
                same_band[same_band["direction"] == signal.direction]
            ),
            "this_direction_all_bands": _summarize(
                cohort[cohort["direction"] == signal.direction]
            ),
        }
    )


async def main(
    arms: set[str], reuse_csv: str | None, out_path: str, dry_run: bool = False
) -> None:
    unknown = arms - set(ARM_NAMES)
    if unknown:
        raise SystemExit(f"ERROR: unknown arm(s) {sorted(unknown)}; valid: {ARM_NAMES}")
    if "challenger" in arms and (CHALLENGER_VERSION is None or CHALLENGER_SYSTEM_PROMPT is None):
        raise SystemExit(
            "ERROR: challenger arm requested but no challenger package is defined — "
            "set CHALLENGER_VERSION, CHALLENGER_SYSTEM_PROMPT, and _challenger_payload "
            "(see the CHALLENGER block at the top of this script)."
        )
    config = load_config()
    if not config.database.url:
        raise SystemExit("ERROR: DATABASE_URL not configured.")
    if not config.anthropic.api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not configured.")

    judgment_model = config.anthropic.judgment_model
    strategy = load_strategy("PoliticsEdgeStrategy")

    print("Loading point-in-time calibration reference set from DB...")
    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    async with session_factory() as session:
        calib_df = await build_calibration_pool(session)
    print(f"  point-in-time calibration pool: {len(calib_df)} resolved signals")

    reused: dict[str, dict] | None = None
    if reuse_csv:
        # The prior run's signals ARE the sample — pairing requires identical
        # signals, and re-drawing with the seed is not reproducible once the
        # assessment population has grown since that run.
        prev_df = pd.read_csv(reuse_csv)
        sample_ids = [uuid.UUID(str(s)) for s in prev_df["signal_id"]]
        reused = {
            str(r["signal_id"]): {k: v for k, v in r.items() if k != "signal_id"}
            for _, r in prev_df.iterrows()
        }
        print(f"Sample = the {len(sample_ids)} signals from {reuse_csv} (paired reuse)")
    else:
        async with session_factory() as session:
            sample_ids = await _pick_sample(session)
        print(f"Sampled {len(sample_ids)} signals (seed={SEED}, max {MAX_PER_MARKET}/market)")
    print(f"Live arms this run: {sorted(arms)}")

    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        session_factory,
        default_strategy="model_eval",
        prompt_version="assessor-audit-pit-v3",
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
        max_consecutive_errors=config.risk.max_consecutive_llm_errors,
    )

    out_rows = []
    async with session_factory() as session:
        for i, signal_id in enumerate(sample_ids, start=1):
            sig_row = (
                await session.execute(select(SignalRow).where(SignalRow.id == signal_id))
            ).scalar_one()
            market_row = (
                await session.execute(select(MarketRow).where(MarketRow.id == sig_row.market_id))
            ).scalar_one()
            existing = (
                await session.execute(
                    select(SignalAssessmentRow).where(SignalAssessmentRow.signal_id == signal_id)
                )
            ).scalar_one()

            signal = _row_to_signal(sig_row)
            market = _row_to_market_at_signal_time(market_row, signal)

            outcome_result = market_row.result
            hit = (signal.direction == "YES" and outcome_result == "yes") or (
                signal.direction == "NO" and outcome_result == "no"
            )

            source_breakdown = await _load_source_breakdown_pit(
                session, signal, market, signal.created_at
            )
            similar_market_summary = await _load_similar_market_summary_pit(
                session,
                market,
                strategy.config.name,
                signal.created_at,
                min_signals=strategy.config.similar_market_min_signals,
                min_trades=strategy.config.similar_market_min_trades,
            )
            base_payload = _build_prompt_payload(
                signal,
                market,
                strategy.config.name,
                source_breakdown,
                similar_market_summary,
                scale_min=strategy.config.assessment_scale_min,
                scale_max=strategy.config.assessment_scale_max,
                phrase_data=None,
            )
            base_payload = _fix_days_to_close(base_payload, market, signal.created_at)
            base_payload = _mask_unreconstructible_liquidity(
                base_payload,
                book_reliable=_book_reconstruction_is_consistent(signal),
            )

            async def _call(
                payload: dict,
                label: str,
                *,
                system: str,
                version: str,
                model: str | None = None,
                _market=market,
                _signal=signal,
                _i=i,
            ) -> dict | None:
                prompt = json.dumps(payload, indent=2, sort_keys=True)
                if dry_run:  # payload built + serialized; skip the paid call
                    return None
                try:
                    resp = await llm_client.complete(
                        prompt=prompt,
                        model=model or judgment_model,
                        query_type=JUDGMENT_QUERY_TYPE,
                        system=system,
                        market_id=_market.id,
                        signal_id=_signal.id,
                        strategy=f"audit_{label}",
                        prompt_version=version,
                        # 768 was tuned for the v6 package and is too tight for
                        # any wordier one: in the first v7 attempt (2026-07-24)
                        # the challenger averaged 721 output tokens and hit the
                        # cap on 2/9 calls, truncating mid-sentence, while the
                        # current arm never exceeded 547. That clips only the
                        # wordier arm, so a challenger loss would be
                        # uninterpretable. 1024 restores headroom symmetrically;
                        # the prompt itself is what keeps output short (see
                        # CHALLENGER_SYSTEM_PROMPT: trust_score first, <=2
                        # sentences, <=3 key_factors).
                        max_tokens=1024,
                        json_tool=_ASSESSMENT_TOOL,
                    )
                    parsed = _parse_assessment_response(resp.content)
                    mult = _clamp_multiplier(
                        _trust_score_to_multiplier(
                            parsed["trust_score"],
                            scale_min=strategy.config.assessment_scale_min,
                            scale_max=strategy.config.assessment_scale_max,
                        ),
                        scale_min=strategy.config.assessment_scale_min,
                        scale_max=strategy.config.assessment_scale_max,
                    )
                    return {
                        "trust_score": parsed["trust_score"],
                        "multiplier": mult,
                        "verdict": parsed["verdict"],
                    }
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{_i}/{len(sample_ids)}] {label} call failed: {exc}")
                    return None

            arm_results: dict[str, dict | None] = {}
            edge_calib = _edge_band_calibration(calib_df, signal, signal.created_at)
            if "current" in arms:
                current_payload = json.loads(json.dumps(base_payload))  # deep copy
                current_payload["edge_band_calibration"] = edge_calib
                current_payload["market_reevaluation_history"] = (
                    await _load_market_reevaluation_history(session, signal)
                )
                arm_results["current"] = await _call(
                    current_payload,
                    "current",
                    system=_SYSTEM_PROMPT,
                    version=f"{_PROMPT_VERSION}-audit-pit-current",
                )
            if "challenger" in arms:
                challenger_payload = await _challenger_payload(
                    session,
                    signal,
                    json.loads(json.dumps(base_payload)),  # deep copy
                    # v8 uses its own cohort-filtered calibration block; the
                    # current arm keeps the production one so the contrast is
                    # exactly "production package vs proposed package".
                    _edge_band_calibration(calib_df, signal, signal.created_at),
                )
                arm_results["challenger"] = await _call(
                    challenger_payload,
                    "challenger",
                    system=CHALLENGER_SYSTEM_PROMPT,
                    version=CHALLENGER_VERSION,
                    model=CHALLENGER_MODEL,
                )

            row_out: dict = {
                "signal_id": str(signal_id),
                "market_id": signal.market_id,
                "direction": signal.direction,
                "edge_pct": signal.edge * 100.0,
                "confidence": signal.confidence,
                "hit": hit,
                "existing_trust_score": existing.trust_score,
                "existing_multiplier": existing.size_multiplier,
                # LLM-free baseline; adoption is judged INCREMENTAL to this.
                "baseline_prior": _pit_baseline_prior(calib_df, signal, signal.created_at),
                "baseline_profit_edge": _pit_baseline_profit_edge(
                    calib_df, signal, signal.created_at
                ),
            }
            if reused is not None:
                fresh_prefixes = tuple(f"{a}_" for a in arms)
                for col, val in reused[str(signal_id)].items():
                    if col in row_out or col.startswith(fresh_prefixes):
                        continue
                    row_out[col] = val
            for arm, res in arm_results.items():
                row_out[f"{arm}_trust_score"] = res["trust_score"] if res else None
                row_out[f"{arm}_multiplier"] = res["multiplier"] if res else None
                row_out[f"{arm}_verdict"] = res["verdict"] if res else None
            out_rows.append(row_out)

            live_bits = " ".join(
                f"{a}={r['trust_score']:.2f}" if r else f"{a}=FAIL"
                for a, r in arm_results.items()
            )
            print(
                f"  [{i}/{len(sample_ids)}] {signal.market_id[:30]:30s} "
                f"hit={hit} existing={existing.trust_score:.2f} {live_bits}"
            )

    out_df = pd.DataFrame(out_rows)
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paired live-Opus assessor audit (PIT). See module docstring."
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARM_NAMES),
        help="Comma-separated arms to run live this invocation (default: all).",
    )
    parser.add_argument(
        "--reuse-csv",
        default=None,
        help="Prior run's CSV (same seed/sample); its columns for arms not in "
        "--arms are carried into the output instead of re-bought.",
    )
    parser.add_argument(
        "--out",
        default="scripts/.audit_output/assessor_audit_pit.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and serialize every arm payload but make no LLM calls.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            {a.strip() for a in args.arms.split(",") if a.strip()},
            args.reuse_csv,
            args.out,
            dry_run=args.dry_run,
        )
    )
