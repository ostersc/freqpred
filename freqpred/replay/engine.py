"""Deterministic replay engine (T66).

Recomputes every stage of one frozen signal-analysis decision from a
ReplayFixture's inputs — retrieval hash, rendered prompt, parsed LLM output,
edge, scheduled skip/cooldown decisions, and the entry-decision chain through
the hard risk caps — and compares the results against the fixture's stored
expectations.

Determinism boundaries:
- No live network calls and no real LLM calls: the LLM is "mocked" by parsing
  the verbatim response stored in the fixture.
- No DB access and no writes: retrieval is bypassed (the fixture carries the
  retrieved document set); risk caps run against the fixture's frozen
  PortfolioSnapshot via RiskEngine.evaluate_position_caps().
- The clock is pinned: build_prompt and the cache decision helpers take the
  fixture's ``now`` explicitly; the strategy layer (should_trade /
  is_market_interesting, whose signatures don't accept a clock) runs under
  freezegun.

The same computation doubles as the recorder's expectation generator and the
``FREQPRED_UPDATE_FIXTURES=1`` regeneration path: ``compute_expectations()``
produces a fresh FixtureExpectations from inputs alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from freezegun import freeze_time

from freqpred.config import RiskConfig
from freqpred.markets.models import Market
from freqpred.rag.retriever import compute_retrieval_hash
from freqpred.replay.fixtures import (
    FixtureEntryDecision,
    FixtureExpectations,
    FixtureInputs,
    FixtureParsed,
    FixtureSkipDecisions,
    ReplayFixture,
)
from freqpred.signal.cache import cooldown_decision, scheduled_skip_decision
from freqpred.signal.llm import PROMPT_VERSION, build_prompt, parse_signal_response
from freqpred.signal.models import Signal
from freqpred.signal.pipeline import compute_signal_edge
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.loader import load_strategy
from freqpred.trading.risk import RiskEngine

_FLOAT_TOL = 1e-9


class ReplayError(Exception):
    """Raised when a fixture cannot be replayed at all (e.g. unparseable LLM response)."""


@dataclass
class ReplayCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ReplayResult:
    fixture_name: str
    checks: list[ReplayCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[ReplayCheck]:
        return [c for c in self.checks if not c.passed]


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL)


def replay_entry_decision(
    signal: Signal,
    market: Market,
    strategy: IPredictionStrategy,
    *,
    bankroll: float,
    existing_market_exposure: float,
    portfolio_snapshot,
    risk_engine: RiskEngine,
) -> FixtureEntryDecision:
    """Mirror order_manager.submit()'s entry-decision chain against frozen state.

    Gates run in production order: SKIP direction (checked by the run loop
    before submit), spread, strategy.should_trade, position_size, hard risk
    caps, contracts >= 1. Circuit breakers and the opposite-side /
    price-improvement guards are portfolio-history checks outside a single
    frozen decision and are not replayed.

    The strategy layer runs under freezegun pinned to signal.created_at so
    days-to-close filters are evaluated at the frozen decision time, not the
    wall clock.
    """
    decision = FixtureEntryDecision(would_trade=False)

    if signal.direction == "SKIP":
        decision.decline_reason = "skip_direction"
        return decision

    spread = round(market.yes_ask - market.yes_bid, 4)
    effective_max_spread = (
        strategy.config.max_spread
        if strategy.config.max_spread is not None
        else strategy.config.min_edge / 2
    )
    if spread > effective_max_spread:
        decision.decline_reason = "spread_too_wide"
        return decision

    with freeze_time(signal.created_at):
        if not strategy.should_trade(signal, market):
            decision.decline_reason = "strategy_declined"
            return decision

        decision.position_size_raw = strategy.position_size(
            signal, bankroll, existing_market_exposure
        )

    risk = risk_engine.evaluate_position_caps(
        signal_edge=signal.edge,
        requested_size=decision.position_size_raw,
        bankroll=bankroll,
        market_id=market.id,
        max_market_exposure=strategy.config.max_exposure_per_market * bankroll,
        snapshot=portfolio_snapshot,
        block_reentry_after_stoploss=strategy.config.block_reentry_after_stoploss,
        stoploss_cooldown_hours=strategy.config.stoploss_cooldown_hours,
    )
    decision.risk_allowed = risk.allowed
    decision.risk_capped_size = risk.capped_size
    decision.risk_reason = risk.reason
    if not risk.allowed:
        decision.decline_reason = "risk_blocked"
        return decision

    # Entry pricing mirrors submit(): limit entries post at
    # estimated_probability - min_edge (or the strategy's custom price);
    # market entries cross the spread at the side's ask. The replay stops at
    # "order would be placed" — resting-order fill lifecycle is out of scope
    # for a single frozen decision.
    if strategy.config.order_types.entry == "limit":
        with freeze_time(signal.created_at):
            custom_price = strategy.custom_entry_price(signal, market)
        if custom_price is not None:
            decision.entry_price = custom_price
        elif signal.direction == "YES":
            decision.entry_price = round(
                signal.estimated_probability - strategy.config.min_edge, 4
            )
        else:
            decision.entry_price = round(
                (1.0 - signal.estimated_probability) - strategy.config.min_edge, 4
            )
    else:
        decision.entry_price = (
            market.yes_ask if signal.direction == "YES" else round(1.0 - market.yes_bid, 4)
        )
    decision.contracts = (
        math.floor(risk.capped_size / decision.entry_price)
        if decision.entry_price > 0
        else 0
    )
    if decision.contracts < 1:
        decision.decline_reason = "contracts_below_minimum"
        return decision

    decision.would_trade = True
    return decision


def compute_expectations(
    inputs: FixtureInputs,
    strategy: IPredictionStrategy | None = None,
    *,
    fixture_name: str = "",
) -> FixtureExpectations:
    """Recompute every expected output from the fixture's inputs alone.

    Raises ReplayError when the stored LLM response cannot be parsed — a
    fixture whose response no longer parses cannot express expectations.
    """
    docs = [fd.to_document() for fd in inputs.documents]
    market = inputs.market.to_market(inputs.now)

    retrieval_hash = compute_retrieval_hash([d.id for d in docs])

    rendered_prompt = build_prompt(
        market,
        docs,
        series_history=(
            inputs.series_history.to_series_history() if inputs.series_history else None
        ),
        phrase_data=inputs.phrase_data.to_phrase_data() if inputs.phrase_data else None,
        _now=inputs.now,
    )

    parsed = parse_signal_response(inputs.llm_response)
    if parsed is None:
        raise ReplayError(
            f"fixture {fixture_name or inputs.market.id}: stored llm_response "
            "failed parse_signal_response()"
        )

    edge, market_ask_at_signal = compute_signal_edge(
        parsed["direction"], parsed["probability"], market.yes_bid, market.yes_ask
    )

    ctx = inputs.decision_context
    signal = Signal(
        id="replay",
        market_id=market.id,
        estimated_probability=parsed["probability"],
        confidence=parsed["confidence"],
        edge=edge,
        market_mid_at_signal=market.mid_price,
        direction=parsed["direction"],
        reasoning=parsed["reasoning"],
        sources=[d.id for d in docs],
        retrieval_hash=retrieval_hash,
        model_used="replay",
        prompt_version=PROMPT_VERSION,
        trigger=inputs.trigger,
        created_at=inputs.now,
        raw_context=rendered_prompt,
        market_ask_at_signal=market_ask_at_signal,
    )

    strategy = strategy if strategy is not None else load_strategy(ctx.strategy)
    risk_engine = RiskEngine(RiskConfig(**ctx.risk_config))
    entry = replay_entry_decision(
        signal,
        market,
        strategy,
        bankroll=ctx.bankroll,
        existing_market_exposure=ctx.existing_market_exposure,
        portfolio_snapshot=ctx.portfolio.to_snapshot(),
        risk_engine=risk_engine,
    )

    skip_decisions = None
    if inputs.prior_scheduled_signal is not None:
        prior = inputs.prior_scheduled_signal
        skip_decisions = FixtureSkipDecisions(
            scheduled_skip=scheduled_skip_decision(
                prior.retrieval_hash,
                prior.created_at,
                retrieval_hash,
                inputs.max_scheduled_interval_hours,
                prior.factbase_refreshed_at,
                inputs.now,
            ),
            cooldown_hours_remaining=cooldown_decision(
                prior.confidence, prior.created_at, inputs.now
            ),
        )

    return FixtureExpectations(
        prompt_version=PROMPT_VERSION,
        retrieval_hash=retrieval_hash,
        rendered_prompt=rendered_prompt,
        parsed=FixtureParsed(
            prior=parsed["prior"],
            posterior=parsed["posterior"],
            probability=parsed["probability"],
            confidence=parsed["confidence"],
            direction=parsed["direction"],
            updates_count=len(parsed["updates_applied"]),
        ),
        edge=edge,
        market_ask_at_signal=market_ask_at_signal,
        entry=entry,
        skip_decisions=skip_decisions,
    )


def _first_diff(a: str, b: str, context: int = 40) -> str:
    """Return a short description of the first divergence between two strings."""
    for i, (ca, cb) in enumerate(zip(a, b, strict=False)):
        if ca != cb:
            lo = max(0, i - context)
            return (
                f"first diff at char {i}: "
                f"expected …{a[lo:i + context]!r} got …{b[lo:i + context]!r}"
            )
    return f"length differs: expected {len(a)} chars, got {len(b)}"


def replay_fixture(
    fixture: ReplayFixture,
    strategy: IPredictionStrategy | None = None,
) -> ReplayResult:
    """Replay one fixture and compare every stage against its stored expectations."""
    result = ReplayResult(fixture_name=fixture.name)
    expected = fixture.expectations

    try:
        actual = compute_expectations(
            fixture.inputs, strategy=strategy, fixture_name=fixture.name
        )
    except ReplayError as exc:
        result.checks.append(ReplayCheck("llm_response_parse", False, str(exc)))
        return result

    def check(name: str, passed: bool, detail: str = "") -> None:
        result.checks.append(ReplayCheck(name, passed, "" if passed else detail))

    check(
        "retrieval_hash",
        actual.retrieval_hash == expected.retrieval_hash,
        f"expected {expected.retrieval_hash} got {actual.retrieval_hash} — the "
        "selected document set changed",
    )

    # Prompt snapshot + PROMPT_VERSION guard.
    prompt_matches = actual.rendered_prompt == expected.rendered_prompt
    version_matches = PROMPT_VERSION == expected.prompt_version
    if prompt_matches and version_matches:
        check("rendered_prompt", True)
    elif not prompt_matches and version_matches:
        check(
            "rendered_prompt",
            False,
            "prompt output changed but PROMPT_VERSION is still "
            f"{PROMPT_VERSION!r} — bump PROMPT_VERSION for an intentional prompt "
            "change, or regenerate fixtures (FREQPRED_UPDATE_FIXTURES=1 pytest / "
            "freqpred fixtures record) if inputs changed. "
            + _first_diff(expected.rendered_prompt, actual.rendered_prompt),
        )
    else:
        # Version bumped (with or without a prompt diff): the fixture predates
        # the current prompt version and must be regenerated so it snapshots
        # the new template.
        check(
            "prompt_version",
            False,
            f"fixture recorded under prompt_version {expected.prompt_version!r} "
            f"but code is {PROMPT_VERSION!r} — regenerate the fixture "
            "(FREQPRED_UPDATE_FIXTURES=1 pytest, or re-record)",
        )

    p_exp, p_act = expected.parsed, actual.parsed
    check(
        "parsed_signal",
        (
            _close(p_act.prior, p_exp.prior)
            and _close(p_act.posterior, p_exp.posterior)
            and _close(p_act.probability, p_exp.probability)
            and _close(p_act.confidence, p_exp.confidence)
            and p_act.direction == p_exp.direction
            and p_act.updates_count == p_exp.updates_count
        ),
        f"expected {p_exp} got {p_act} — response parsing changed under a frozen "
        "LLM response",
    )

    check(
        "edge",
        _close(actual.edge, expected.edge)
        and _close(actual.market_ask_at_signal, expected.market_ask_at_signal),
        f"expected edge={expected.edge} ask={expected.market_ask_at_signal} "
        f"got edge={actual.edge} ask={actual.market_ask_at_signal}",
    )

    e_exp, e_act = expected.entry, actual.entry
    check(
        "entry_decision",
        (
            e_act.would_trade == e_exp.would_trade
            and e_act.decline_reason == e_exp.decline_reason
            and _close(e_act.position_size_raw, e_exp.position_size_raw)
            and e_act.risk_allowed == e_exp.risk_allowed
            and _close(e_act.risk_capped_size, e_exp.risk_capped_size)
            and e_act.risk_reason == e_exp.risk_reason
            and _close(e_act.entry_price, e_exp.entry_price)
            and e_act.contracts == e_exp.contracts
        ),
        f"expected {e_exp} got {e_act} — the trade/no-trade decision chain "
        "changed under frozen inputs",
    )

    if expected.skip_decisions is not None or actual.skip_decisions is not None:
        s_exp, s_act = expected.skip_decisions, actual.skip_decisions
        check(
            "skip_decisions",
            s_exp is not None
            and s_act is not None
            and s_act.scheduled_skip == s_exp.scheduled_skip
            and _close(s_act.cooldown_hours_remaining, s_exp.cooldown_hours_remaining),
            f"expected {s_exp} got {s_act}",
        )

    return result
