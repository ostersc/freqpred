"""Replay fixture schema (T66).

A ReplayFixture freezes everything one signal-analysis decision consumed —
market snapshot, retrieved documents with scores, catalyst queries, optional
series-history/FactBase context, the clock, and the verbatim stored LLM
response — plus the expected outputs at every stage (retrieval hash, rendered
prompt, parsed signal, edge, and the downstream entry decision through the
risk caps).

Fixtures store *structured inputs*, not just the rendered prompt, so the same
scenario can be re-rendered under a modified prompt template (the scenario-bank
benchmark harness, T93, depends on this). Fixtures are recorded from real
production signals by ``freqpred fixtures record`` and checked in under
``tests/fixtures/replay/``. They must never contain secrets.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, Field

from freqpred.ingestion.fetchers.factbase import FactbasePhraseData
from freqpred.markets.models import Market
from freqpred.rag.models import Document
from freqpred.trading.risk import PortfolioSnapshot

SCHEMA_VERSION = 1

DEFAULT_FIXTURE_DIR = Path("tests/fixtures/replay")


class FixtureDocument(BaseModel):
    """A retrieved document frozen at record time.

    Embedding vectors are deliberately NOT stored: the fixture freezes the
    *outcome* of retrieval (which documents, in what order, with what scores).
    Retrieval-code correctness (vector scoring, column selection, age cutoff)
    is covered by targeted tests in tests/integration/test_retriever_integration.py
    with small synthetic vectors — not by this harness.
    """

    id: str
    source_url: str
    content_hash: str = ""
    title: str
    body: str
    summary: str | None = None
    #: The document's full text, carried alongside the frozen excerpt so
    #: retrieval-time extraction (T101) has something to extract from.
    #:
    #: ``body`` in a frozen-context fixture is the 500-char excerpt the model
    #: actually saw, which is what makes the byte-exact re-render check work —
    #: but it is also, by construction, the exact thing T101 replaces, so a
    #: bank recorded with only ``body`` measures a null change. Populated by
    #: the recorder only when the live row still reproduces that excerpt
    #: exactly; ``None`` means the source drifted (or was summarised away) and
    #: extraction must fall back to ``body``.
    full_body: str | None = None
    source_type: str
    source_name: str
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    fetched_at: datetime
    similarity_score: float = 0.0

    def to_document(self, *, full: bool = False) -> Document:
        """Rebuild the domain ``Document``.

        ``full=True`` substitutes ``full_body`` for ``body`` when it is
        available — for the extraction pass only. Prompt rendering and the
        expectations round-trip must keep the default, since they are verified
        against the excerpt the model was actually shown.
        """
        body = self.full_body if (full and self.full_body) else self.body
        return Document(
            id=self.id,
            source_url=self.source_url,
            content_hash=self.content_hash,
            title=self.title,
            body=body,
            source_type=self.source_type,
            source_name=self.source_name,
            category=self.category,
            tags=list(self.tags),
            published_at=self.published_at,
            fetched_at=self.fetched_at,
            embedding=[],
            embedding_model="",
            summary=self.summary,
        )


class FixtureMarket(BaseModel):
    """Market snapshot as of the frozen decision time.

    ``yes_bid``/``yes_ask`` are the prices at signal time (reconstructed by the
    recorder from the signal's stored mid and side-specific ask), not current
    prices.
    """

    id: str
    platform: str = "kalshi"
    question: str
    category: str
    close_time: datetime
    open_time: datetime | None = None
    yes_bid: float
    yes_ask: float
    mid_price: float
    volume_24h: float = 0.0
    open_interest: float = 0.0
    status: str = "active"
    series_ticker: str | None = None

    def to_market(self, now: datetime) -> Market:
        return Market(
            id=self.id,
            platform=self.platform,
            question=self.question,
            category=self.category,
            close_time=self.close_time,
            yes_bid=self.yes_bid,
            yes_ask=self.yes_ask,
            mid_price=self.mid_price,
            volume_24h=self.volume_24h,
            open_interest=self.open_interest,
            last_fetched_at=now,
            price_updated_at=now,
            metadata_fetched_at=now,
            open_time=self.open_time,
            status=self.status,
            series_ticker=self.series_ticker,
        )


class FixtureSeriesCounts(BaseModel):
    """One row of series/option settlement history (mirrors SeriesOptionHistoryRow)."""

    option_label: str = ""
    yes_count: int = 0
    no_count: int = 0


class FixtureSeriesHistory(BaseModel):
    """HISTORICAL BASE RATE inputs consumed by build_prompt."""

    series_ticker: str
    option_code: str = ""
    series_row: FixtureSeriesCounts | None = None
    option_row: FixtureSeriesCounts | None = None

    def to_series_history(self) -> dict:
        def _ns(row: FixtureSeriesCounts | None) -> SimpleNamespace | None:
            if row is None:
                return None
            return SimpleNamespace(
                option_label=row.option_label,
                yes_count=row.yes_count,
                no_count=row.no_count,
            )

        return {
            "series_ticker": self.series_ticker,
            "option_code": self.option_code,
            "series_row": _ns(self.series_row),
            "option_row": _ns(self.option_row),
        }


class FixturePhraseData(BaseModel):
    """PHRASE FREQUENCY DATA inputs (mirrors FactbasePhraseData)."""

    display_phrase: str
    api_query: str = ""
    speaker_slug: str = ""
    in_market_count: int = 0
    count_7d: int = 0
    count_30d: int = 0
    count_365d: int = 0
    top_quotes: list[dict] = Field(default_factory=list)
    fetched_at: datetime

    def to_phrase_data(self) -> FactbasePhraseData:
        return FactbasePhraseData(
            display_phrase=self.display_phrase,
            api_query=self.api_query,
            speaker_slug=self.speaker_slug,
            in_market_count=self.in_market_count,
            count_7d=self.count_7d,
            count_30d=self.count_30d,
            count_365d=self.count_365d,
            top_quotes=list(self.top_quotes),
            fetched_at=self.fetched_at,
        )


class FixturePriorScheduledSignal(BaseModel):
    """State of the last scheduled signal, for replaying skip/cooldown decisions."""

    retrieval_hash: str
    created_at: datetime
    confidence: float
    factbase_refreshed_at: datetime | None = None


class FixturePortfolio(BaseModel):
    """Frozen portfolio state for the risk-cap evaluation (defaults: empty book)."""

    loss_exit_count: int = 0
    market_open_exposure: float = 0.0
    market_pending_exposure: float = 0.0
    open_count: int = 0
    pending_count: int = 0
    open_exposure: float = 0.0
    pending_exposure: float = 0.0
    daily_pnl: float = 0.0

    def to_snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            loss_exit_count=self.loss_exit_count,
            market_open_exposure=self.market_open_exposure,
            market_pending_exposure=self.market_pending_exposure,
            open_count=self.open_count,
            pending_count=self.pending_count,
            open_exposure=self.open_exposure,
            pending_exposure=self.pending_exposure,
            daily_pnl=self.daily_pnl,
        )


class FixtureDecisionContext(BaseModel):
    """Strategy + bankroll context for the entry-decision replay.

    ``risk_config`` holds RiskConfig field overrides (e.g. ``min_edge_floor``);
    unset fields use RiskConfig defaults so fixtures stay small.
    """

    strategy: str = "ConservativeDefault"
    bankroll: float = 1000.0
    existing_market_exposure: float = 0.0
    portfolio: FixturePortfolio = Field(default_factory=FixturePortfolio)
    risk_config: dict[str, float | int] = Field(default_factory=dict)


class FixtureInputs(BaseModel):
    now: datetime
    trigger: str = "scheduled"
    market: FixtureMarket
    documents: list[FixtureDocument]
    catalyst_queries: list[str] = Field(default_factory=list)
    series_history: FixtureSeriesHistory | None = None
    phrase_data: FixturePhraseData | None = None
    # Verbatim tool-call JSON from the llm_queries audit row — the mocked LLM.
    llm_response: str
    prior_scheduled_signal: FixturePriorScheduledSignal | None = None
    max_scheduled_interval_hours: float = 24.0
    decision_context: FixtureDecisionContext = Field(default_factory=FixtureDecisionContext)


class FixtureParsed(BaseModel):
    prior: float
    posterior: float
    probability: float
    confidence: float
    direction: str
    updates_count: int


class FixtureSkipDecisions(BaseModel):
    """Expected scheduled skip/cooldown outcomes (requires prior_scheduled_signal)."""

    scheduled_skip: bool
    cooldown_hours_remaining: float


class FixtureEntryDecision(BaseModel):
    """Expected outcome of the entry-decision chain (mirrors order_manager.submit)."""

    would_trade: bool
    # Which gate declined, "" when the trade goes through. One of:
    # skip_direction | spread_too_wide | strategy_declined | risk_blocked |
    # contracts_below_minimum
    decline_reason: str = ""
    position_size_raw: float = 0.0
    risk_allowed: bool = False
    risk_capped_size: float = 0.0
    risk_reason: str = ""
    entry_price: float | None = None
    contracts: int = 0


class FixtureExpectations(BaseModel):
    prompt_version: str
    retrieval_hash: str
    rendered_prompt: str
    # SHA-256 of SYSTEM_PROMPT at record time. The rendered_prompt snapshot only
    # covers build_prompt's user-prompt output — without this, SYSTEM_PROMPT
    # could change without a PROMPT_VERSION bump and every fixture stays green.
    system_prompt_sha256: str = ""
    parsed: FixtureParsed
    edge: float
    market_ask_at_signal: float | None = None
    entry: FixtureEntryDecision
    skip_decisions: FixtureSkipDecisions | None = None


class ReplayFixture(BaseModel):
    schema_version: int = SCHEMA_VERSION
    name: str
    description: str = ""
    recorded_from_signal_id: str | None = None
    recorded_at: datetime | None = None
    inputs: FixtureInputs
    expectations: FixtureExpectations


def load_fixture(path: Path | str) -> ReplayFixture:
    fixture = ReplayFixture.model_validate_json(Path(path).read_text())
    if fixture.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"fixture {path} has schema_version={fixture.schema_version}, "
            f"this code supports {SCHEMA_VERSION} — re-record the fixture"
        )
    return fixture


def save_fixture(fixture: ReplayFixture, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fixture.model_dump_json(indent=2) + "\n")
