"""Kalshi API changelog monitor.

Fetches https://docs.kalshi.com/changelog/rss.xml once per day, compares
publication dates against ``kalshi_changelog_state.last_reviewed_at``, and
fires alerts when unreviewed entries exist.  Breaking-change entries trigger
a critical alert; other new entries trigger a warning alert.

To mark entries as reviewed, run an Alembic migration that updates
``last_reviewed_at`` to the current date:

    UPDATE kalshi_changelog_state SET last_reviewed_at = CURRENT_DATE WHERE id = 1;
"""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, AsyncGenerator

import httpx
import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from freqpred.runtime.models import KalshiChangelogStateRow
from freqpred.runtime.telemetry import SERVICE_KALSHI_CHANGELOG

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator as _AG

    from freqpred.alerts.dispatcher import AlertDispatcher
    from freqpred.runtime.telemetry import RuntimeTelemetry

log = structlog.get_logger(__name__)

RSS_URL = "https://docs.kalshi.com/changelog/rss.xml"
_SINGLETON_ID = 1


@dataclass
class ChangelogEntry:
    title: str
    pub_date: date
    categories: list[str] = field(default_factory=list)
    link: str = ""

    @property
    def is_breaking_change(self) -> bool:
        return any("breaking" in c.lower() for c in self.categories)


async def fetch_changelog_entries(http_client: httpx.AsyncClient) -> list[ChangelogEntry]:
    """Fetch and parse the Kalshi changelog RSS feed."""
    resp = await http_client.get(RSS_URL, timeout=15.0)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    channel = root.find("channel")
    if channel is None:
        return []

    entries: list[ChangelogEntry] = []
    for item in channel.findall("item"):
        title_el = item.find("title")
        pubdate_el = item.find("pubDate")
        link_el = item.find("link")
        if title_el is None or pubdate_el is None:
            continue
        try:
            pub_dt = parsedate_to_datetime(pubdate_el.text or "")
            pub_d = pub_dt.date()
        except Exception:
            continue
        categories = [el.text for el in item.findall("category") if el.text]
        entries.append(
            ChangelogEntry(
                title=title_el.text or "",
                pub_date=pub_d,
                categories=categories,
                link=link_el.text or "" if link_el is not None else "",
            )
        )
    return entries


async def _get_state(session: AsyncSession) -> KalshiChangelogStateRow | None:
    result = await session.execute(
        select(KalshiChangelogStateRow).where(KalshiChangelogStateRow.id == _SINGLETON_ID)
    )
    return result.scalar_one_or_none()


async def _update_state(
    session: AsyncSession,
    *,
    unreviewed_count: int,
    has_unreviewed_breaking_change: bool,
    last_checked_at: datetime,
) -> None:
    await session.execute(
        update(KalshiChangelogStateRow)
        .where(KalshiChangelogStateRow.id == _SINGLETON_ID)
        .values(
            unreviewed_count=unreviewed_count,
            has_unreviewed_breaking_change=has_unreviewed_breaking_change,
            last_checked_at=last_checked_at,
        )
    )
    await session.commit()


async def run_changelog_monitor(
    session_factory: AsyncGenerator,
    dispatcher: AlertDispatcher | None,
    telemetry: RuntimeTelemetry | None,
    interval_seconds: int = 86400,
) -> None:
    """Daily loop: fetch Kalshi changelog RSS, alert on new entries, update DB state.

    Runs immediately on startup if ``last_checked_at`` is None or older than
    ``interval_seconds``.  Subsequent runs sleep for ``interval_seconds``.
    """
    log.info("kalshi_changelog_monitor.started", interval_seconds=interval_seconds)

    async def _check_now() -> None:
        now = datetime.now(UTC)
        try:
            async with httpx.AsyncClient() as client:
                entries = await fetch_changelog_entries(client)

            async with session_factory() as session:
                state = await _get_state(session)
                if state is None:
                    log.error("kalshi_changelog_monitor.no_state_row")
                    if telemetry is not None:
                        await telemetry.mark_error(
                            SERVICE_KALSHI_CHANGELOG,
                            "kalshi_changelog_state row missing — run migration 0036",
                            now=now,
                        )
                    return

                cutoff: date = state.last_reviewed_at
                new_entries = [e for e in entries if e.pub_date > cutoff]
                breaking = [e for e in new_entries if e.is_breaking_change]
                non_breaking = [e for e in new_entries if not e.is_breaking_change]

                await _update_state(
                    session,
                    unreviewed_count=len(new_entries),
                    has_unreviewed_breaking_change=bool(breaking),
                    last_checked_at=now,
                )

            log.info(
                "kalshi_changelog_monitor.checked",
                total_entries=len(entries),
                unreviewed=len(new_entries),
                breaking=len(breaking),
                last_reviewed_at=str(cutoff),
            )

            if dispatcher is not None:
                if breaking:
                    await dispatcher.changelog_critical_alert(breaking)
                if non_breaking:
                    await dispatcher.changelog_warning_alert(non_breaking)

            if telemetry is not None:
                await telemetry.mark_success(SERVICE_KALSHI_CHANGELOG, now=now)

        except Exception as exc:
            log.exception("kalshi_changelog_monitor.error")
            if telemetry is not None:
                await telemetry.mark_error(
                    SERVICE_KALSHI_CHANGELOG, str(exc), now=datetime.now(UTC)
                )

    # On startup: run immediately if never checked or overdue.
    async with session_factory() as session:
        state = await _get_state(session)

    needs_immediate_run = state is None or state.last_checked_at is None or (
        (datetime.now(UTC) - state.last_checked_at).total_seconds() >= interval_seconds
    )

    if needs_immediate_run:
        await _check_now()

    while True:
        await asyncio.sleep(interval_seconds)
        await _check_now()
