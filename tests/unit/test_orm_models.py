"""Unit tests for SQLAlchemy ORM models and freqpred.db helpers.

These tests are pure Python — no database connection required.
They verify:
- All ORM models can be imported and instantiated without errors
- Table names and column names match the spec
- The Base metadata contains the expected tables
- db.py helpers return the expected types
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from freqpred.alerts.models import RunStateRow  # registers the table
from freqpred.db import Base, make_engine, make_session_factory
from freqpred.ingestion.models import ApiDailyCounterRow  # registers the table
from freqpred.llm.models import LLMQueryRow
from freqpred.markets.models import MarketRow, PositionRow
from freqpred.metrics.models import SeriesOptionHistoryRow, SignalAssessmentRow, SourceQualityScoreRow
from freqpred.rag.models import DocumentMarketLinkRow, DocumentRow
from freqpred.runtime.models import RuntimeEventRow, ServiceHeartbeatRow  # registers the tables
from freqpred.signal.models import SignalRow
from freqpred.strategy.models import RuntimeConfigOverrideRow  # registers the table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
MARKET_ID = "MARKET-001"
SIGNAL_ID = uuid.uuid4()
DOCUMENT_ID = uuid.uuid4()
POSITION_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# Base / metadata tests
# ---------------------------------------------------------------------------


def test_all_tables_registered():
    """All expected tables must appear in Base.metadata."""
    expected = {
        "markets",
        "signals",
        "positions",
        "documents",
        "document_market_links",
        "llm_queries",
        "catalyst_runs",
        "catalyst_queries",
        "api_daily_counters",
        "fetcher_cursors",
        "fetcher_rate_limits",
        "factbase_phrase_frequency",
        "run_state",
        "runtime_config_overrides",
        "source_quality_scores",
        "signal_assessments",
        "service_heartbeats",
        "runtime_events",
        "series_option_history",
    }
    registered = set(Base.metadata.tables.keys())
    assert expected == registered


# ---------------------------------------------------------------------------
# MarketRow
# ---------------------------------------------------------------------------


def test_market_row_table_name():
    assert MarketRow.__tablename__ == "markets"


def test_market_row_instantiation():
    row = MarketRow(
        id=MARKET_ID,
        platform="kalshi",
        question="Will X happen?",
        category="politics",
        close_time=NOW,
        yes_bid=0.45,
        yes_ask=0.50,
        mid_price=0.475,
        volume_24h=10_000.0,
        open_interest=5_000.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
        current_signal_id=None,
        metadata_={"raw": "data"},
    )
    assert row.id == MARKET_ID
    assert row.current_signal_id is None


def test_market_row_columns():
    cols = {c.name for c in MarketRow.__table__.columns}
    required = {
        "id", "platform", "question", "category", "close_time",
        "yes_bid", "yes_ask", "mid_price", "volume_24h", "open_interest",
        "last_fetched_at", "price_updated_at", "metadata_fetched_at",
        "current_signal_id", "metadata", "created_at",
    }
    assert required.issubset(cols)


# ---------------------------------------------------------------------------
# SignalRow
# ---------------------------------------------------------------------------


def test_signal_row_table_name():
    assert SignalRow.__tablename__ == "signals"


def test_signal_row_instantiation():
    row = SignalRow(
        id=SIGNAL_ID,
        market_id=MARKET_ID,
        estimated_probability=0.71,
        confidence=0.78,
        edge=0.235,
        market_mid_at_signal=0.475,
        direction="YES",
        reasoning="Strong news coverage suggests YES.",
        sources=["https://example.com/article"],
        retrieval_hash="abc123",
        model_used="claude-3-5-sonnet-20241022",
        prompt_version="politics-v2",
        trigger="scheduled",
        created_at=NOW,
        raw_context="<context>...</context>",
    )
    assert row.id == SIGNAL_ID
    assert row.direction == "YES"
    assert row.social_sentiment_summary is None


def test_signal_row_columns():
    cols = {c.name for c in SignalRow.__table__.columns}
    required = {
        "id", "market_id", "estimated_probability", "confidence", "edge",
        "market_mid_at_signal", "direction", "reasoning", "sources",
        "social_sentiment_summary", "retrieval_hash", "model_used",
        "prompt_version", "trigger", "created_at", "raw_context",
    }
    assert required.issubset(cols)


# ---------------------------------------------------------------------------
# PositionRow
# ---------------------------------------------------------------------------


def test_position_row_table_name():
    assert PositionRow.__tablename__ == "positions"


def test_position_row_instantiation():
    row = PositionRow(
        id=POSITION_ID,
        market_id=MARKET_ID,
        signal_id=SIGNAL_ID,
        strategy_name="PoliticsEdgeStrategy",
        strategy_version="1.0.0",
        signal_confidence=0.78,
        signal_edge=0.235,
        signal_estimated_prob=0.71,
        direction="YES",
        contracts=10,
        entry_price=0.475,
        entry_time=NOW,
        mode="paper",
        status="open",
    )
    assert row.mode == "paper"
    assert row.exit_price is None
    assert row.pnl is None


def test_position_row_columns():
    cols = {c.name for c in PositionRow.__table__.columns}
    required = {
        "id", "market_id", "signal_id", "strategy_name", "strategy_version",
        "signal_confidence", "signal_edge", "signal_estimated_prob",
        "direction", "contracts", "entry_price", "entry_time", "mode", "status",
        "exit_price", "exit_time", "resolution", "pnl", "pnl_pct", "created_at",
    }
    assert required.issubset(cols)


# ---------------------------------------------------------------------------
# DocumentRow
# ---------------------------------------------------------------------------


def test_document_row_table_name():
    assert DocumentRow.__tablename__ == "documents"


def test_document_row_instantiation():
    row = DocumentRow(
        id=DOCUMENT_ID,
        source_url="https://example.com/article",
        content_hash="deadbeef",
        title="Example Article",
        body="Article body text.",
        source_type="news",
        source_name="Reuters",
        category="politics",
        tags=["election", "senate"],
        published_at=NOW,
        fetched_at=NOW,
        embedding=[0.1] * 384,
        embedding_model="all-MiniLM-L6-v2",
    )
    assert row.source_type == "news"
    assert row.summary is None
    assert len(row.embedding) == 384


def test_source_quality_score_row_table_name():
    assert SourceQualityScoreRow.__tablename__ == "source_quality_scores"


def test_source_quality_score_row_instantiation():
    row = SourceQualityScoreRow(
        source_name="Reuters",
        market_category="politics",
        lookback_days=90,
        weighted_brier=0.143,
        overall_brier=0.171,
        n_signals=25,
        total_doc_uses=81,
    )
    assert row.source_name == "Reuters"
    assert row.market_category == "politics"


def test_signal_assessment_row_table_name():
    assert SignalAssessmentRow.__tablename__ == "signal_assessments"


def test_signal_assessment_row_instantiation():
    row = SignalAssessmentRow(
        signal_id=SIGNAL_ID,
        trust_score=0.62,
        size_multiplier=1.048,
        verdict="size_up",
        reasoning="Family history is stronger than baseline.",
        key_factors=["Strong family match"],
        warnings=[],
        source_breakdown=[],
        similar_market_summary={"available": True},
        llm_query_id=None,
    )
    assert row.signal_id == SIGNAL_ID
    assert row.verdict == "size_up"


def test_document_row_source_url_unique():
    """source_url column must have a unique constraint."""
    col = DocumentRow.__table__.c["source_url"]
    assert col.unique or any(
        set(uc.columns.keys()) == {"source_url"}
        for uc in DocumentRow.__table__.constraints
        if hasattr(uc, "columns")
    )


def test_document_row_columns():
    cols = {c.name for c in DocumentRow.__table__.columns}
    required = {
        "id", "source_url", "content_hash", "title", "body", "summary",
        "source_type", "source_name", "category", "tags",
        "published_at", "fetched_at", "embedding", "embedding_model", "created_at",
    }
    assert required.issubset(cols)


# ---------------------------------------------------------------------------
# DocumentMarketLinkRow
# ---------------------------------------------------------------------------


def test_document_market_link_row_table_name():
    assert DocumentMarketLinkRow.__tablename__ == "document_market_links"


def test_document_market_link_row_instantiation():
    row = DocumentMarketLinkRow(
        id=uuid.uuid4(),
        document_id=DOCUMENT_ID,
        market_id=MARKET_ID,
        signal_id=SIGNAL_ID,
        relevance_score=0.92,
        linked_at=NOW,
    )
    assert row.relevance_score == 0.92


# ---------------------------------------------------------------------------
# LLMQueryRow
# ---------------------------------------------------------------------------


def test_llm_query_row_table_name():
    assert LLMQueryRow.__tablename__ == "llm_queries"


def test_llm_query_row_instantiation():
    row = LLMQueryRow(
        timestamp=NOW,
        strategy="PoliticsEdgeStrategy",
        query_type="market_analysis",
        model_used="claude-3-5-sonnet-20241022",
        prompt_version="politics-v2",
        prompt="Analyze this market...",
        response='{"probability": 0.71}',
        tokens_input=500,
        tokens_output=200,
        tokens_total=700,
        cost_usd=0.0042,
        latency_ms=1200,
        success=True,
    )
    assert row.id is None  # auto-increment, not set until DB insert
    assert row.success is True
    assert row.error_message is None


def test_llm_query_row_columns():
    cols = {c.name for c in LLMQueryRow.__table__.columns}
    required = {
        "id", "timestamp", "strategy", "query_type", "market_id", "signal_id",
        "model_used", "prompt_version", "prompt", "response",
        "tokens_input", "tokens_output", "tokens_total", "cost_usd",
        "confidence_extracted", "decision_extracted", "latency_ms",
        "success", "error_message", "created_at",
    }
    assert required.issubset(cols)


# ---------------------------------------------------------------------------
# db.py helpers
# ---------------------------------------------------------------------------


def test_make_engine_returns_async_engine():
    engine = make_engine("postgresql+asyncpg://user:pass@localhost/freqpred_test")
    assert isinstance(engine, AsyncEngine)


def test_make_session_factory_returns_async_sessionmaker():
    engine = make_engine("postgresql+asyncpg://user:pass@localhost/freqpred_test")
    factory = make_session_factory(engine)
    assert isinstance(factory, async_sessionmaker)
