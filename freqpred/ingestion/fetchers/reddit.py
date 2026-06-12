"""Reddit RSS fetcher (no credentials required).

Reddit shut down unauthenticated access to its public JSON API in June 2026
(the "Responsible Builder Policy" requires pre-approval for all API access).
The Atom/RSS feeds remain publicly served, and ``search.rss`` carries the full
selftext for self posts — which is everything the old JSON fetcher extracted
(it never fetched comments).

Differences vs the retired JSON fetcher:
  - RSS entries carry no upvote score, so the ``score >= 10`` quality filter is
    gone. Per-source Brier weighting (``source_quality_scores``) is the quality
    control instead.
  - All subreddits are searched in a single multireddit request
    (``/r/a+b+c/search.rss``) — unauthenticated tolerance is ~10 requests/min
    per IP, so request count matters more than anything. Each entry's
    ``<category term>`` identifies its actual subreddit for per-source Brier
    attribution.
  - A failed search raises :class:`RedditBlockedError` so the scheduler can
    back off and surface the outage, instead of the source dying silently
    (which is how the JSON shutdown went unnoticed for 12 days).
"""
from __future__ import annotations

import asyncio
import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx
import structlog

from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()

_MAX_AGE_DAYS = 7
_BASE_URL = "https://www.reddit.com"
_SEARCH_PATH = "/r/{multireddit}/search.rss"
_ATOM = "{http://www.w3.org/2005/Atom}"

# Global spacing between Reddit requests, across all fetch() calls in the
# process. Unauthenticated access tolerates roughly 10 requests/min per IP
# (2.5s spacing still drew sustained 429s); 6.5s ≈ 9/min leaves headroom.
_REQUEST_SPACING_SECONDS = 6.5
_throttle_lock = asyncio.Lock()
_last_request_at = 0.0  # time.monotonic()


async def _throttle() -> None:
    """Enforce minimum spacing between Reddit requests process-wide."""
    global _last_request_at
    if _REQUEST_SPACING_SECONDS <= 0:
        return
    async with _throttle_lock:
        wait = _last_request_at + _REQUEST_SPACING_SECONDS - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_at = time.monotonic()

_TAG_RE = re.compile(r"<[^>]+>")
# Reddit appends "submitted by /u/<user> [link] [comments]" to every entry body.
_FOOTER_RE = re.compile(r"submitted by\s+/u/\S+.*$", re.DOTALL)
_WS_RE = re.compile(r"\s+")


class RedditBlockedError(Exception):
    """The search request failed — Reddit is blocking or unreachable."""


def _strip_html(content: str) -> str:
    """Render Atom entry HTML content down to plain text."""
    text = html.unescape(content)
    text = _TAG_RE.sub(" ", text)
    text = _FOOTER_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def _parse_published(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _entry_subreddit(entry: ET.Element, fallback: str) -> str:
    """Resolve the entry's subreddit from its <category term> element."""
    category = entry.find(f"{_ATOM}category")
    if category is not None:
        term = (category.get("term") or "").strip()
        if term:
            return term
    return fallback


async def fetch(
    subreddits: list[str],
    query: str,
    user_agent: str = "freqpred/0.1",
    limit: int = 25,
) -> list[RawDocument]:
    """Fetch Reddit posts matching a query across the given subreddits.

    Uses Reddit's public Atom search feeds — no credentials required.
    All subreddits are combined into one multireddit search request.
    Filters posts by recency (last 7 days).

    Args:
        subreddits: List of subreddit names to search (without 'r/' prefix).
        query:      Search query string.
        user_agent: User-Agent header sent with requests.
        limit:      Maximum submissions to request (shared across subreddits).

    Returns:
        List of RawDocument objects with source_type="reddit". Each document's
        source_name reflects the entry's actual subreddit (from the Atom
        category term), not the requested list.

    Raises:
        RedditBlockedError: the search request failed (HTTP 403/429, transport
            error, or unparseable response). A 404 (no such subreddits) is a
            silent skip, not a hard failure.
    """
    if not subreddits:
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_MAX_AGE_DAYS)
    headers = {"User-Agent": user_agent}
    docs: list[RawDocument] = []
    multireddit = "+".join(subreddits)

    async with httpx.AsyncClient(base_url=_BASE_URL, headers=headers, timeout=15.0) as client:
        try:
            await _throttle()
            response = await client.get(
                _SEARCH_PATH.format(multireddit=multireddit),
                params={
                    "q": query,
                    "sort": "new",
                    "limit": limit,
                    "restrict_sr": 1,
                },
            )
            if response.status_code == 404:
                # No such subreddit(s) — a config issue, not blocking.
                log.debug("reddit.fetch.skip", multireddit=multireddit, status=404)
                return []
            if response.status_code in (403, 429):
                log.warning(
                    "reddit.fetch.blocked",
                    multireddit=multireddit,
                    status=response.status_code,
                )
                raise RedditBlockedError(
                    f"r/{multireddit}: HTTP {response.status_code}"
                )
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except RedditBlockedError:
            raise
        except Exception as exc:
            log.warning(
                "reddit.fetch.error",
                multireddit=multireddit,
                query=query,
                exc_info=True,
            )
            raise RedditBlockedError(f"r/{multireddit}: {exc}") from exc

        for entry in root.findall(f"{_ATOM}entry"):
            title = (entry.findtext(f"{_ATOM}title") or "").strip()
            link_el = entry.find(f"{_ATOM}link")
            source_url = (link_el.get("href") or "") if link_el is not None else ""
            published_at = _parse_published(
                entry.findtext(f"{_ATOM}published") or entry.findtext(f"{_ATOM}updated")
            )

            if not source_url:
                log.debug("reddit.fetch.skip", reason="no_link", title=title[:80])
                continue

            if published_at is not None and published_at < cutoff:
                log.debug(
                    "reddit.fetch.skip",
                    reason="too_old",
                    published_at=published_at.isoformat(),
                    source_url=source_url,
                )
                continue

            body = _strip_html(entry.findtext(f"{_ATOM}content") or "") or title
            if not body:
                log.warning(
                    "reddit.fetch.skip", reason="empty_body", source_url=source_url
                )
                continue

            docs.append(
                RawDocument(
                    source_url=source_url,
                    title=title,
                    body=body,
                    source_type="reddit",
                    source_name=f"r/{_entry_subreddit(entry, subreddits[0])}",
                    category="",
                    tags=[],
                    published_at=published_at or now,
                    fetched_at=now,
                )
            )

    return docs
