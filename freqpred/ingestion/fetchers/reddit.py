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
  - A blanket failure across all subreddits raises :class:`RedditBlockedError`
    so the scheduler can back off and surface the outage, instead of the source
    dying silently (which is how the JSON shutdown went unnoticed for 12 days).
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx
import structlog

from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()

_MAX_AGE_DAYS = 7
_BASE_URL = "https://www.reddit.com"
_SEARCH_PATH = "/r/{subreddit}/search.rss"
_ATOM = "{http://www.w3.org/2005/Atom}"

_TAG_RE = re.compile(r"<[^>]+>")
# Reddit appends "submitted by /u/<user> [link] [comments]" to every entry body.
_FOOTER_RE = re.compile(r"submitted by\s+/u/\S+.*$", re.DOTALL)
_WS_RE = re.compile(r"\s+")


class RedditBlockedError(Exception):
    """Every subreddit request failed — Reddit is blocking or unreachable."""


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


async def fetch(
    subreddits: list[str],
    query: str,
    user_agent: str = "freqpred/0.1",
    limit: int = 25,
) -> list[RawDocument]:
    """Fetch Reddit posts matching a query across the given subreddits.

    Uses Reddit's public Atom search feeds — no credentials required.
    Filters posts by recency (last 7 days).

    Args:
        subreddits: List of subreddit names to search (without 'r/' prefix).
        query:      Search query string.
        user_agent: User-Agent header sent with requests.
        limit:      Maximum submissions to request per subreddit.

    Returns:
        List of RawDocument objects with source_type="reddit".

    Raises:
        RedditBlockedError: every subreddit request hard-failed (HTTP 403/429,
            transport error, or unparseable response). A 404 (subreddit does
            not exist) is a per-subreddit skip, not a hard failure.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_MAX_AGE_DAYS)
    headers = {"User-Agent": user_agent}
    docs: list[RawDocument] = []
    skipped_404 = 0
    hard_failures = 0
    last_failure = ""

    async with httpx.AsyncClient(base_url=_BASE_URL, headers=headers, timeout=15.0) as client:
        for subreddit_name in subreddits:
            try:
                response = await client.get(
                    _SEARCH_PATH.format(subreddit=subreddit_name),
                    params={
                        "q": query,
                        "sort": "new",
                        "limit": limit,
                        "restrict_sr": 1,
                    },
                )
                if response.status_code == 404:
                    # Subreddit doesn't exist — a config issue, not blocking.
                    # Excluded from the blanket-failure count below.
                    skipped_404 += 1
                    log.debug(
                        "reddit.fetch.skip",
                        subreddit=subreddit_name,
                        status=404,
                    )
                    continue
                if response.status_code in (403, 429):
                    hard_failures += 1
                    last_failure = f"r/{subreddit_name}: HTTP {response.status_code}"
                    log.warning(
                        "reddit.fetch.blocked",
                        subreddit=subreddit_name,
                        status=response.status_code,
                    )
                    continue
                response.raise_for_status()
                root = ET.fromstring(response.text)
            except Exception as exc:
                hard_failures += 1
                last_failure = f"r/{subreddit_name}: {exc}"
                log.warning(
                    "reddit.fetch.error",
                    subreddit=subreddit_name,
                    query=query,
                    exc_info=True,
                )
                continue

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
                        source_name=f"r/{subreddit_name}",
                        category="",
                        tags=[],
                        published_at=published_at or now,
                        fetched_at=now,
                    )
                )

    attempted = len(subreddits) - skipped_404
    if attempted > 0 and hard_failures == attempted:
        raise RedditBlockedError(
            f"all {hard_failures} subreddit request(s) failed; last: {last_failure}"
        )

    return docs
