"""Scenario bank for model/prompt benchmarking (T93).

A Scenario freezes one historical signal-analysis decision on a market whose
outcome is now known: the exact prompt to send a candidate, the incumbent
model's parsed output (recovered from the ``llm_queries`` audit trail), the
prices at signal time, and the actual resolution. Candidates are scored
against the outcome — never against training-data-era history.

Two sources:

- **Model mode** (``build_db_scenarios``): every LLM-backed signal on
  finalized binary markets. The stored ``raw_context`` is replayed to the
  candidate **verbatim**, so this mode is exact but cannot vary the prompt
  template.
- **Prompt mode** (``build_fixture_scenarios``): replay fixtures, whose
  structured inputs are **re-rendered** through the current ``build_prompt`` +
  ``SYSTEM_PROMPT``. Only snapshotted fixture inputs are trustworthy for
  re-rendering — reconstructing series-history/FactBase state as-of signal
  time from live tables would introduce lookahead.

Both modes return *all* decision points per market; which markets and which
of each market's signals to actually benchmark is decided at run time by
``sample_markets`` (seeded random market selection) and ``sample_per_market``
(last / all / spread:K across the signal timeline). Signals within a market
share one outcome — score them with market-clustered statistics.

Contamination guard: scenarios whose market closed on or before the
candidate's training-data cutoff are excluded by ``filter_contaminated``
(the outcome could be in the training data); ``include_contaminated=True``
keeps them but flags each one.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import MarketRow
from freqpred.replay.engine import render_prompt_from_inputs
from freqpred.replay.fixtures import ReplayFixture, load_fixture
from freqpred.replay.recorder import _reconstruct_prices
from freqpred.signal.extractor import DocumentExtract, extract_for_documents
from freqpred.signal.llm import parse_signal_response
from freqpred.signal.models import SignalRow


@dataclass
class ModelOutput:
    """One model's parsed signal output plus call economics (where known)."""

    model: str
    prior: float
    posterior: float
    confidence: float
    direction: str
    updates_count: int
    reasoning: str = ""
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None


@dataclass
class Scenario:
    """One frozen decision on a resolved market, ready to benchmark against."""

    id: str                     # signal id (db mode) or fixture name (prompt mode)
    source: str                 # "db" | "fixture"
    market_id: str
    market_question: str
    close_time: datetime
    outcome: float              # 1.0 = resolved YES, 0.0 = resolved NO
    prompt: str                 # exactly what the candidate model receives
    incumbent: ModelOutput
    yes_bid: float              # prices at signal time (frozen)
    yes_ask: float
    mid_price: float
    signal_time: datetime | None = None  # when the signal fired (frozen clock)
    contaminated: bool = False  # close_time <= candidate training cutoff
    notes: list[str] = field(default_factory=list)


def _outcome_from_result(result: str | None) -> float | None:
    if result == "yes":
        return 1.0
    if result == "no":
        return 0.0
    return None  # scalar / unknown — never silently miscategorized as NO


def scenario_from_signal(
    signal: SignalRow,
    market: MarketRow,
    llm_row: LLMQueryRow,
    *,
    prompt: str | None = None,
    source: str = "db",
    scenario_id: str | None = None,
) -> Scenario | None:
    """Assemble a Scenario from audit-trail rows. Returns None when the
    market has no binary outcome or the stored response no longer parses.

    *prompt* defaults to the signal's stored ``raw_context`` — verbatim
    replay, byte-identical to what the incumbent saw.
    """
    outcome = _outcome_from_result(market.result)
    if outcome is None:
        return None

    parsed = parse_signal_response(llm_row.response or "")
    if parsed is None:
        return None

    yes_bid, yes_ask = _reconstruct_prices(
        signal.direction,
        signal.market_mid_at_signal,
        signal.market_ask_at_signal,
        signal.estimated_probability,
        signal.edge,
    )

    return Scenario(
        id=scenario_id or str(signal.id),
        source=source,
        market_id=market.id,
        market_question=market.question,
        close_time=market.close_time,
        outcome=outcome,
        prompt=prompt if prompt is not None else signal.raw_context,
        incumbent=ModelOutput(
            model=llm_row.model_used,
            prior=parsed["prior"],
            posterior=parsed["posterior"],
            confidence=parsed["confidence"],
            direction=parsed["direction"],
            updates_count=len(parsed["updates_applied"]),
            reasoning=parsed["reasoning"],
            tokens_input=llm_row.tokens_input,
            tokens_output=llm_row.tokens_output,
            cost_usd=llm_row.cost_usd,
            latency_ms=llm_row.latency_ms,
        ),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        mid_price=signal.market_mid_at_signal,
        signal_time=signal.created_at,
    )


def sample_markets(
    scenarios: list[Scenario],
    limit: int | None,
    *,
    seed: int = 42,
) -> tuple[list[Scenario], int]:
    """Randomly sample *limit* markets (seeded, reproducible) and keep all of
    their scenarios. Returns (kept, n_markets_kept).

    Sampling whole markets — not scenarios — keeps per-market signal series
    intact for spread sampling and market-clustered statistics, and avoids the
    recency bias of taking the first/last N.
    """
    market_ids = sorted({s.market_id for s in scenarios})
    if limit is None or len(market_ids) <= limit:
        return scenarios, len(market_ids)
    chosen = set(random.Random(seed).sample(market_ids, limit))
    return [s for s in scenarios if s.market_id in chosen], len(chosen)


def sample_per_market(
    scenarios: list[Scenario],
    per_market: str,
) -> list[Scenario]:
    """Select which of each market's signals to benchmark.

    - ``"first"``: the earliest LLM signal only — the purest early-entry
      decision point, before the market converges.
    - ``"last"``: the final pre-resolution signal only (the market has usually
      converged by then — favorites-heavy, least tradeable).
    - ``"all"``: every recorded signal (use market-clustered stats downstream).
    - ``"spread:K"``: K signals per market evenly spaced across its signal
      timeline (first and last always included; markets with <= K signals keep
      all) — covers early/mid/late decision points at bounded cost.

    Scenarios lacking ``signal_time`` sort by close_time as a fallback.
    """
    if per_market == "all":
        return scenarios

    by_market: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        by_market.setdefault(scenario.market_id, []).append(scenario)

    kept: list[Scenario] = []
    for series in by_market.values():
        series.sort(key=lambda s: s.signal_time or s.close_time)
        if per_market == "first":
            kept.append(series[0])
            continue
        if per_market == "last":
            kept.append(series[-1])
            continue
        if not per_market.startswith("spread:"):
            raise ValueError(f"unknown --per-market mode: {per_market!r}")
        k = int(per_market.split(":", 1)[1])
        if k < 1:
            raise ValueError("--per-market spread:K requires K >= 1")
        if len(series) <= k:
            kept.extend(series)
        elif k == 1:
            kept.append(series[-1])
        else:
            indices = sorted({round(i * (len(series) - 1) / (k - 1)) for i in range(k)})
            kept.extend(series[i] for i in indices)
    kept.sort(key=lambda s: (s.market_id, s.signal_time or s.close_time))
    return kept


def filter_contaminated(
    scenarios: list[Scenario],
    training_cutoff: datetime,
    *,
    include_contaminated: bool = False,
) -> tuple[list[Scenario], list[Scenario]]:
    """Split scenarios into (kept, excluded) around the training-data cutoff.

    A market that closed on or before the cutoff resolved inside the
    candidate's training window — its outcome may be memorized, so scoring
    against it is not evidence of forecasting skill. With
    ``include_contaminated=True`` such scenarios are kept but flagged
    (``contaminated=True``) so every downstream report can mark them.
    """
    kept: list[Scenario] = []
    excluded: list[Scenario] = []
    for scenario in scenarios:
        if scenario.close_time > training_cutoff:
            kept.append(scenario)
        elif include_contaminated:
            scenario.contaminated = True
            kept.append(scenario)
        else:
            excluded.append(scenario)
    return kept, excluded


async def build_db_scenarios(
    session,
    *,
    market_id: str | None = None,
) -> list[Scenario]:
    """Model-mode bank: every LLM-backed signal on finalized binary markets.

    Returns all decision points — market selection (``sample_markets``) and
    per-market signal sampling (``sample_per_market``) happen at benchmark
    time, so the caller controls cost and coverage. Scans from signals: only
    a small fraction of markets ever get LLM analysis, so scanning markets
    first mostly produces skips.
    """
    stmt = (
        select(SignalRow)
        .join(MarketRow, MarketRow.id == SignalRow.market_id)
        .where(
            SignalRow.llm_query_id.isnot(None),
            MarketRow.status == "finalized",
            # in_(("yes","no")) — not isnot(None) — so scalar markets are never
            # silently miscategorized as a NO resolution.
            MarketRow.result.in_(("yes", "no")),
        )
        .order_by(SignalRow.market_id, SignalRow.created_at)
    )
    if market_id:
        stmt = stmt.where(SignalRow.market_id == market_id)

    signals = (await session.execute(stmt)).scalars().all()
    if not signals:
        return []

    markets = {
        m.id: m
        for m in (
            await session.execute(
                select(MarketRow).where(MarketRow.id.in_([s.market_id for s in signals]))
            )
        ).scalars()
    }
    llm_rows = {
        row.id: row
        for row in (
            await session.execute(
                select(LLMQueryRow).where(
                    LLMQueryRow.id.in_([s.llm_query_id for s in signals])
                )
            )
        ).scalars()
    }

    scenarios: list[Scenario] = []
    for signal in signals:
        market = markets.get(signal.market_id)
        llm_row = llm_rows.get(signal.llm_query_id)
        if market is None or llm_row is None:
            continue
        scenario = scenario_from_signal(signal, market, llm_row)
        if scenario is not None:
            scenarios.append(scenario)
    return scenarios


def scenario_from_fixture(
    fixture: ReplayFixture,
    *,
    outcome: float,
    close_time: datetime,
    incumbent_model: str = "",
    extracts: dict[str, DocumentExtract] | None = None,
) -> Scenario | None:
    """Prompt-mode scenario: fixture inputs re-rendered through the *current*
    prompt template; the fixture's stored LLM response is the incumbent.

    Returns None when the stored response no longer parses.
    """
    inputs = fixture.inputs
    parsed = parse_signal_response(inputs.llm_response)
    if parsed is None:
        return None

    scenario = Scenario(
        id=fixture.name,
        source="fixture",
        market_id=inputs.market.id,
        market_question=inputs.market.question,
        close_time=close_time,
        outcome=outcome,
        prompt=render_prompt_from_inputs(inputs, extracts),
        incumbent=ModelOutput(
            model=incumbent_model or "unknown",
            prior=parsed["prior"],
            posterior=parsed["posterior"],
            confidence=parsed["confidence"],
            direction=parsed["direction"],
            updates_count=len(parsed["updates_applied"]),
            reasoning=parsed["reasoning"],
        ),
        yes_bid=inputs.market.yes_bid,
        yes_ask=inputs.market.yes_ask,
        mid_price=inputs.market.mid_price,
        signal_time=inputs.now,
    )
    if scenario.prompt != fixture.expectations.rendered_prompt:
        scenario.notes.append(
            "prompt template differs from the one the fixture was recorded "
            f"under ({fixture.expectations.prompt_version})"
        )
    return scenario


async def build_fixture_scenarios(
    session,
    fixture_dir: Path,
) -> tuple[list[Scenario], list[str], dict[str, ReplayFixture]]:
    """Prompt-mode bank from T66 fixtures.

    Returns ``(scenarios, skip_reasons, fixtures_by_scenario_id)``. The third
    element exists so the caller can extract evidence (T101) *after* sampling —
    see ``apply_extraction``. Building every scenario is free; extracting is
    not, and the bank is ~6x larger than a typical ``--limit``.

    Outcomes come from the live markets table: only fixtures whose market has
    since finalized with a binary result are usable; the rest are reported in
    *skip_reasons* rather than silently dropped.
    """
    paths = sorted(Path(fixture_dir).glob("*.json"))
    fixtures = [load_fixture(p) for p in paths]
    if not fixtures:
        return [], [f"no fixtures found under {fixture_dir}"]

    # Every fixture becomes a scenario — multiple signals on one market are
    # correlated snapshots of the same outcome, which downstream handles via
    # per-market sampling (sample_per_market) and market-clustered statistics,
    # not by discarding decision points here. Same-signal duplicates (identical
    # frozen clock) are collapsed.
    skipped: list[str] = []
    by_key: dict[tuple[str, datetime | None], ReplayFixture] = {}
    for fixture in fixtures:
        key = (fixture.inputs.market.id, fixture.inputs.now)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = fixture
        else:
            skipped.append(
                f"{fixture.name}: duplicate of {existing.name} "
                f"(same market and frozen clock)"
            )
    fixtures = list(by_key.values())

    market_rows = {
        m.id: m
        for m in (
            await session.execute(
                select(MarketRow).where(
                    MarketRow.id.in_([f.inputs.market.id for f in fixtures])
                )
            )
        ).scalars()
    }

    # Recover the incumbent model name from the recorded signal where possible.
    signal_ids = [
        f.recorded_from_signal_id for f in fixtures if f.recorded_from_signal_id
    ]
    model_by_signal: dict[str, str] = {}
    if signal_ids:
        rows = (
            await session.execute(
                select(SignalRow.id, SignalRow.model_used).where(
                    SignalRow.id.in_(signal_ids)
                )
            )
        ).all()
        model_by_signal = {str(sid): model for sid, model in rows}

    scenarios: list[Scenario] = []
    fixtures_by_id: dict[str, ReplayFixture] = {}
    for fixture in fixtures:
        market = market_rows.get(fixture.inputs.market.id)
        if market is None:
            skipped.append(f"{fixture.name}: market {fixture.inputs.market.id} not in DB")
            continue
        outcome = _outcome_from_result(market.result)
        if market.status != "finalized" or outcome is None:
            skipped.append(
                f"{fixture.name}: market {market.id} not finalized with a binary "
                f"result yet (status={market.status}, result={market.result})"
            )
            continue
        scenario = scenario_from_fixture(
            fixture,
            outcome=outcome,
            close_time=market.close_time,
            incumbent_model=model_by_signal.get(fixture.recorded_from_signal_id or "", ""),
        )
        if scenario is None:
            skipped.append(f"{fixture.name}: stored llm_response failed to parse")
            continue
        scenarios.append(scenario)
        fixtures_by_id[scenario.id] = fixture
    return scenarios, skipped, fixtures_by_id


async def apply_extraction(
    session,
    llm_client,  # noqa: ANN001 — LLMClient; typed loosely to avoid an import cycle
    scenarios: list[Scenario],
    fixtures_by_id: dict[str, ReplayFixture],
    *,
    model: str,
) -> int:
    """Re-render *scenarios* with T101 question-focused extracts, in place.

    **Call this after sampling, never before.** Extraction costs an API call
    per uncached (document, market) pair, and the bank holds several times more
    scenarios than any ``--limit`` keeps — extracting during the scenario build
    bills the whole bank to answer a question about 50 markets. That mistake
    cost $3.82 before it was caught on 2026-08-11.

    Extraction reads ``full_body``, not the frozen 500-char excerpt: a bank
    whose bodies are capped at exactly the length T101 replaces would report a
    null change however good the extractor is. Fixtures recorded before the
    ``full_body`` backfill, and documents that drifted since their signal,
    carry ``None`` there and fall back to the excerpt — contributing no
    extraction signal rather than a wrong one.

    Returns the number of scenarios whose prompt actually changed.
    """
    changed = 0
    for scenario in scenarios:
        fixture = fixtures_by_id.get(scenario.id)
        if fixture is None:
            continue
        extracts = await extract_for_documents(
            session,
            llm_client,
            fixture.inputs.market.to_market(fixture.inputs.now),
            [fd.to_document(full=True) for fd in fixture.inputs.documents],
            model=model,
            strategy="benchmark",
        )
        rendered = render_prompt_from_inputs(fixture.inputs, extracts)
        if rendered != scenario.prompt:
            changed += 1
        scenario.prompt = rendered
    return changed
