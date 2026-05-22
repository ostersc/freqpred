"""Unit tests for the Kalshi changelog RSS monitor."""
from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from freqpred.ingestion.kalshi_changelog import (
    ChangelogEntry,
    fetch_changelog_entries,
    run_changelog_monitor,
)


# ---------------------------------------------------------------------------
# Sample RSS XML
# ---------------------------------------------------------------------------

_RSS_BREAKING = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Kalshi API Changelog</title>
    <item>
      <title>Legacy fields removed</title>
      <pubDate>Wed, 21 May 2026 00:00:00 GMT</pubDate>
      <category>Breaking Change</category>
      <category>Upcoming</category>
      <link>https://docs.kalshi.com/changelog#2026-05-21</link>
    </item>
    <item>
      <title>New candlestick endpoint</title>
      <pubDate>Mon, 18 May 2026 00:00:00 GMT</pubDate>
      <category>New Feature</category>
      <link>https://docs.kalshi.com/changelog#2026-05-18</link>
    </item>
  </channel>
</rss>"""

_RSS_NO_BREAKING = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Kalshi API Changelog</title>
    <item>
      <title>Bug fix in search</title>
      <pubDate>Thu, 15 May 2026 00:00:00 GMT</pubDate>
      <category>Bug Fix</category>
      <link>https://docs.kalshi.com/changelog#2026-05-15</link>
    </item>
  </channel>
</rss>"""

_RSS_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Kalshi API Changelog</title>
  </channel>
</rss>"""


# ---------------------------------------------------------------------------
# fetch_changelog_entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_changelog_entries_parses_rss():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = _RSS_BREAKING

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)

    entries = await fetch_changelog_entries(mock_client)

    assert len(entries) == 2
    assert entries[0].title == "Legacy fields removed"
    assert entries[0].pub_date == date(2026, 5, 21)
    assert entries[0].is_breaking_change is True
    assert "Breaking Change" in entries[0].categories

    assert entries[1].title == "New candlestick endpoint"
    assert entries[1].pub_date == date(2026, 5, 18)
    assert entries[1].is_breaking_change is False


@pytest.mark.asyncio
async def test_fetch_changelog_entries_empty_channel():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = _RSS_EMPTY

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)

    entries = await fetch_changelog_entries(mock_client)
    assert entries == []


# ---------------------------------------------------------------------------
# ChangelogEntry.is_breaking_change
# ---------------------------------------------------------------------------


def test_is_breaking_change_case_insensitive():
    entry = ChangelogEntry(
        title="X", pub_date=date(2026, 1, 1), categories=["breaking change", "Upcoming"]
    )
    assert entry.is_breaking_change is True


def test_is_breaking_change_false_for_non_breaking():
    entry = ChangelogEntry(
        title="X", pub_date=date(2026, 1, 1), categories=["New Feature", "Released"]
    )
    assert entry.is_breaking_change is False


def test_is_breaking_change_empty_categories():
    entry = ChangelogEntry(title="X", pub_date=date(2026, 1, 1), categories=[])
    assert entry.is_breaking_change is False


# ---------------------------------------------------------------------------
# run_changelog_monitor
# ---------------------------------------------------------------------------


def _make_state(last_reviewed_at: date, last_checked_at: datetime | None = None):
    state = MagicMock()
    state.last_reviewed_at = last_reviewed_at
    state.last_checked_at = last_checked_at
    return state


def _make_session_factory(state):
    """Return an async context-manager session factory that yields a mock session."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=state))
    )
    mock_session.commit = AsyncMock()

    async def _ctx():
        return mock_session

    class _CM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    def factory():
        return _CM()

    return factory


@pytest.mark.asyncio
async def test_run_changelog_monitor_sends_critical_alert():
    """Breaking-change entries → changelog_critical_alert called."""
    last_reviewed = date(2026, 5, 10)
    state = _make_state(last_reviewed, last_checked_at=None)
    session_factory = _make_session_factory(state)

    dispatcher = AsyncMock()
    dispatcher.changelog_critical_alert = AsyncMock()
    dispatcher.changelog_warning_alert = AsyncMock()

    breaking_entry = ChangelogEntry(
        title="Legacy fields removed",
        pub_date=date(2026, 5, 21),
        categories=["Breaking Change"],
    )
    non_breaking_entry = ChangelogEntry(
        title="New endpoint",
        pub_date=date(2026, 5, 18),
        categories=["New Feature"],
    )

    with patch(
        "freqpred.ingestion.kalshi_changelog.fetch_changelog_entries",
        AsyncMock(return_value=[breaking_entry, non_breaking_entry]),
    ), patch("freqpred.ingestion.kalshi_changelog.httpx.AsyncClient") as mock_hc_cls:
        mock_hc_cls.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_hc_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        # Patch asyncio.sleep to stop after one cycle
        with patch("freqpred.ingestion.kalshi_changelog.asyncio.sleep", side_effect=asyncio.CancelledError):
            try:
                await run_changelog_monitor(
                    session_factory=session_factory,
                    dispatcher=dispatcher,
                    telemetry=None,
                )
            except asyncio.CancelledError:
                pass

    dispatcher.changelog_critical_alert.assert_awaited_once()
    critical_entries = dispatcher.changelog_critical_alert.call_args[0][0]
    assert any(e.title == "Legacy fields removed" for e in critical_entries)

    dispatcher.changelog_warning_alert.assert_awaited_once()
    warning_entries = dispatcher.changelog_warning_alert.call_args[0][0]
    assert any(e.title == "New endpoint" for e in warning_entries)


@pytest.mark.asyncio
async def test_run_changelog_monitor_sends_warning_only():
    """Non-breaking entries → changelog_warning_alert called, not critical."""
    last_reviewed = date(2026, 5, 10)
    state = _make_state(last_reviewed, last_checked_at=None)
    session_factory = _make_session_factory(state)

    dispatcher = AsyncMock()
    dispatcher.changelog_critical_alert = AsyncMock()
    dispatcher.changelog_warning_alert = AsyncMock()

    entry = ChangelogEntry(
        title="New endpoint", pub_date=date(2026, 5, 18), categories=["New Feature"]
    )

    with patch(
        "freqpred.ingestion.kalshi_changelog.fetch_changelog_entries",
        AsyncMock(return_value=[entry]),
    ), patch("freqpred.ingestion.kalshi_changelog.httpx.AsyncClient") as mock_hc_cls:
        mock_hc_cls.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_hc_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("freqpred.ingestion.kalshi_changelog.asyncio.sleep", side_effect=asyncio.CancelledError):
            try:
                await run_changelog_monitor(
                    session_factory=session_factory,
                    dispatcher=dispatcher,
                    telemetry=None,
                )
            except asyncio.CancelledError:
                pass

    dispatcher.changelog_warning_alert.assert_awaited_once()
    dispatcher.changelog_critical_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_changelog_monitor_no_alert_when_up_to_date():
    """All entries on or before last_reviewed_at → no alert sent."""
    last_reviewed = date(2026, 5, 21)
    state = _make_state(last_reviewed, last_checked_at=None)
    session_factory = _make_session_factory(state)

    dispatcher = AsyncMock()
    dispatcher.changelog_critical_alert = AsyncMock()
    dispatcher.changelog_warning_alert = AsyncMock()

    old_entry = ChangelogEntry(
        title="Old fix", pub_date=date(2026, 5, 10), categories=["Bug Fix"]
    )
    same_day_entry = ChangelogEntry(
        title="Same day", pub_date=date(2026, 5, 21), categories=["New Feature"]
    )

    with patch(
        "freqpred.ingestion.kalshi_changelog.fetch_changelog_entries",
        AsyncMock(return_value=[old_entry, same_day_entry]),
    ), patch("freqpred.ingestion.kalshi_changelog.httpx.AsyncClient") as mock_hc_cls:
        mock_hc_cls.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_hc_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("freqpred.ingestion.kalshi_changelog.asyncio.sleep", side_effect=asyncio.CancelledError):
            try:
                await run_changelog_monitor(
                    session_factory=session_factory,
                    dispatcher=dispatcher,
                    telemetry=None,
                )
            except asyncio.CancelledError:
                pass

    dispatcher.changelog_warning_alert.assert_not_awaited()
    dispatcher.changelog_critical_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_changelog_monitor_skips_initial_run_when_recently_checked():
    """If last_checked_at is recent, skip the immediate check and wait for sleep."""
    recently = datetime.now(UTC).replace(microsecond=0)
    state = _make_state(date(2026, 5, 10), last_checked_at=recently)
    session_factory = _make_session_factory(state)

    dispatcher = AsyncMock()
    dispatcher.changelog_warning_alert = AsyncMock()
    dispatcher.changelog_critical_alert = AsyncMock()

    fetch_mock = AsyncMock(return_value=[])

    with patch("freqpred.ingestion.kalshi_changelog.fetch_changelog_entries", fetch_mock), \
         patch("freqpred.ingestion.kalshi_changelog.httpx.AsyncClient") as mock_hc_cls:
        mock_hc_cls.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_hc_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("freqpred.ingestion.kalshi_changelog.asyncio.sleep", side_effect=asyncio.CancelledError):
            try:
                await run_changelog_monitor(
                    session_factory=session_factory,
                    dispatcher=dispatcher,
                    telemetry=None,
                    interval_seconds=86400,
                )
            except asyncio.CancelledError:
                pass

    # fetch should NOT have been called since last_checked_at is recent
    fetch_mock.assert_not_awaited()


# Need asyncio in scope for the CancelledError references
import asyncio  # noqa: E402
