"""Unit tests for freqpred/ingestion/realtime_scheduler.py.

All external dependencies (fetchers, store, DB) are mocked.
No real API calls or DB connections are made.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.ingestion.realtime_scheduler import run_realtime_cycle
from freqpred.ingestion.store import DocumentSkipped, RawDocument, UpsertStatus
from freqpred.rag.models import Document

NOW = datetime(2026, 3, 23, 12, 0, 0, tzinfo=UTC)
CLOSE_TIME = datetime(2026, 3, 30, 0, 0, 0, tzinfo=UTC)
FAKE_EMBEDDING = [0.1] * 384


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_backoff(monkeypatch):
    monkeypatch.setattr(
        "freqpred.ingestion.realtime_scheduler.tick_and_load",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.realtime_scheduler.record_rate_limit",
        AsyncMock(return_value=1),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.realtime_scheduler.record_success",
        AsyncMock(return_value=None),
    )


@pytest.fixture(autouse=True)
def mock_cursors(monkeypatch):
    monkeypatch.setattr(
        "freqpred.ingestion.realtime_scheduler.get_cursor",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "freqpred.ingestion.realtime_scheduler.set_cursor",
        AsyncMock(return_value=None),
    )


def _make_raw_doc(url: str = "https://example.com/chyron") -> RawDocument:
    return RawDocument(
        source_url=url,
        title="FOXNEWSW: Fox News Tonight",
        body="TRUMP: IRAN DEAL",
        source_type="tv_chyron",
        source_name="TVThirdEye",
        category="",
        tags=["FOXNEWSW"],
        published_at=NOW,
        fetched_at=NOW,
    )


def _make_document(url: str = "https://example.com/chyron") -> Document:
    return Document(
        id=str(uuid.uuid4()),
        source_url=url,
        content_hash="abc123",
        title="FOXNEWSW: Fox News Tonight",
        body="TRUMP: IRAN DEAL",
        summary=None,
        source_type="tv_chyron",
        source_name="TVThirdEye",
        category="",
        tags=["FOXNEWSW"],
        published_at=NOW,
        fetched_at=NOW,
        embedding=FAKE_EMBEDDING,
        embedding_model="all-MiniLM-L6-v2",
    )


def _make_session() -> AsyncMock:
    session = AsyncMock()
    nested_ctx = MagicMock()
    nested_ctx.__aenter__ = AsyncMock(return_value=nested_ctx)
    nested_ctx.__aexit__ = AsyncMock(return_value=None)
    session.begin_nested = MagicMock(return_value=nested_ctx)
    return session


def _make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_text = AsyncMock(return_value=FAKE_EMBEDDING)
    return embedder


# ---------------------------------------------------------------------------
# run_realtime_cycle — both sources disabled
# ---------------------------------------------------------------------------


class TestBothDisabled:
    @pytest.mark.asyncio
    async def test_returns_zero_stats(self) -> None:
        session = AsyncMock()
        embedder = _make_embedder()

        with patch(
            "freqpred.ingestion.realtime_scheduler._load_active_market_queries",
            new_callable=AsyncMock,
            return_value=[],
        ):
            stats = await run_realtime_cycle(
                session, embedder, tv_chyron_enabled=False, truthsocial_enabled=False
            )

        assert stats == {"docs_fetched": 0, "docs_stored": 0, "docs_error": 0}


# ---------------------------------------------------------------------------
# run_realtime_cycle — TV chyron phase
# ---------------------------------------------------------------------------


class TestChyronPhase:
    @pytest.mark.asyncio
    async def test_chyron_disabled_skips_fetch(self) -> None:
        session = _make_session()
        embedder = _make_embedder()

        with (
            patch(
                "freqpred.ingestion.realtime_scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.fetch_all",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            await run_realtime_cycle(
                session, embedder, tv_chyron_enabled=False, truthsocial_enabled=False
            )

        mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_chyron_enabled_calls_fetch_with_1_hour(self) -> None:
        session = _make_session()
        embedder = _make_embedder()

        with (
            patch(
                "freqpred.ingestion.realtime_scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.fetch_all",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_fetch,
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.parse_and_groups",
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.filter_chyrons",
                return_value=[],
            ),
        ):
            await run_realtime_cycle(
                session, embedder, tv_chyron_enabled=True, truthsocial_enabled=False
            )

        mock_fetch.assert_called_once_with(lookback_hours=1)

    @pytest.mark.asyncio
    async def test_chyron_matched_docs_upserted_and_linked(self) -> None:
        session = _make_session()
        embedder = _make_embedder()
        raw_doc = _make_raw_doc()
        doc = _make_document()
        market_queries = [
            ("MKT-1", "politics", "Will this market resolve Yes?", CLOSE_TIME, [("fed rate", 'trump AND "iran"')])
        ]

        with (
            patch(
                "freqpred.ingestion.realtime_scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.fetch_all",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.parse_and_groups",
                return_value=[["trump"], ["iran"]],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.filter_chyrons",
                return_value=[raw_doc],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.upsert_document",
                new_callable=AsyncMock,
                return_value=(doc, UpsertStatus.INSERTED),
            ) as mock_upsert,
            patch(
                "freqpred.ingestion.realtime_scheduler.link_document_to_market",
                new_callable=AsyncMock,
            ) as mock_link,
        ):
            stats = await run_realtime_cycle(
                session, embedder, tv_chyron_enabled=True, truthsocial_enabled=False
            )

        assert stats["docs_fetched"] == 1
        assert stats["docs_stored"] == 1
        assert stats["docs_error"] == 0
        mock_upsert.assert_called_once()
        mock_link.assert_called_once_with(session, doc.id, "MKT-1")

    @pytest.mark.asyncio
    async def test_chyron_cursor_updated_after_cycle(self, monkeypatch) -> None:
        set_cursor_mock = AsyncMock()
        monkeypatch.setattr(
            "freqpred.ingestion.realtime_scheduler.set_cursor",
            set_cursor_mock,
        )
        session = _make_session()
        embedder = _make_embedder()

        with (
            patch(
                "freqpred.ingestion.realtime_scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.fetch_all",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            await run_realtime_cycle(
                session, embedder, tv_chyron_enabled=True, truthsocial_enabled=False
            )

        calls = [c.args for c in set_cursor_mock.call_args_list]
        assert any(args[1] == "tv_chyron" and args[2] == "global" for args in calls)

    @pytest.mark.asyncio
    async def test_chyron_upsert_error_counted(self) -> None:
        session = _make_session()
        embedder = _make_embedder()
        raw_doc = _make_raw_doc()
        market_queries = [("MKT-1", "politics", "Will this market resolve Yes?", CLOSE_TIME, [("fed rate", 'trump')])]

        with (
            patch(
                "freqpred.ingestion.realtime_scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.fetch_all",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.parse_and_groups",
                return_value=[["trump"]],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.filter_chyrons",
                return_value=[raw_doc],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.upsert_document",
                new_callable=AsyncMock,
                side_effect=Exception("DB error"),
            ),
        ):
            stats = await run_realtime_cycle(
                session, embedder, tv_chyron_enabled=True, truthsocial_enabled=False
            )

        assert stats["docs_error"] == 1
        assert stats["docs_stored"] == 0

    @pytest.mark.asyncio
    async def test_chyron_skipped_doc_not_counted(self) -> None:
        session = _make_session()
        embedder = _make_embedder()
        raw_doc = _make_raw_doc()
        market_queries = [("MKT-1", "politics", "Will this market resolve Yes?", CLOSE_TIME, [("fed rate", 'trump')])]

        with (
            patch(
                "freqpred.ingestion.realtime_scheduler._load_active_market_queries",
                new_callable=AsyncMock,
                return_value=market_queries,
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.fetch_all",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.parse_and_groups",
                return_value=[["trump"]],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.tv_chyron_fetcher.filter_chyrons",
                return_value=[raw_doc],
            ),
            patch(
                "freqpred.ingestion.realtime_scheduler.upsert_document",
                new_callable=AsyncMock,
                side_effect=DocumentSkipped(),
            ),
        ):
            stats = await run_realtime_cycle(
                session, embedder, tv_chyron_enabled=True, truthsocial_enabled=False
            )

        assert stats["docs_stored"] == 0
        assert stats["docs_error"] == 0
