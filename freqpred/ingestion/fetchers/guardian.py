"""The Guardian API fetcher.

Uses the Guardian Content API (https://open-platform.theguardian.com/).
Free developer tier: 500 requests/day, 1 req/sec.

Key behaviours:
- Accepts a Solr/Lucene query string (the ``tv_query`` from catalyst generation
  is ideal — the Guardian ``q`` parameter supports full boolean syntax).
- Requests ``show-fields=body`` to get the full article HTML, then strips tags.
- Passes ``from-date`` for recency filtering.
- Returns ``published_at`` from ``webPublicationDate``.
"""
from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime, timezone

import httpx
import structlog

from freqpred.ingestion.fetchers import is_domain_excluded
from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()

_BASE_URL = "https://content.guardianapis.com/search"
_RATE_LIMIT_SLEEP = 1.0  # Guardian free tier: ≤1 req/sec
_DEFAULT_EXCLUDED_DOMAINS: frozenset[str] = frozenset({"kalshi.com"})
_REQUEST_TIMEOUT = 15.0


class GuardianRateLimitError(Exception):
    """Raised when the Guardian API returns HTTP 429."""


def _strip_html(raw: str) -> str:
    """Strip HTML tags and unescape entities from Guardian article body."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _sanitize_query(query: str) -> str:
    """Remove Google Search syntax unsupported by the Guardian Solr/Lucene API.

    The catalyst generator produces ``query_text`` for general web search
    (Tavily, Google) which may include ``site:`` filters.  These are not valid
    Lucene syntax and cause the Guardian API to return HTTP 400.
    """
    # Strip bare site: tokens (e.g. "site:truthsocial.com").
    sanitized = re.sub(r"\bsite:\S+", "", query, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", sanitized).strip()


async def fetch(
    api_key: str,
    query: str,
    from_date: datetime | None = None,
    max_results: int = 20,
    excluded_domains: frozenset[str] = _DEFAULT_EXCLUDED_DOMAINS,
) -> list[RawDocument]:
    """Fetch articles from The Guardian Content API.

    Enforces max 1 req/sec via a fixed sleep before the call.
    The Guardian ``q`` parameter accepts Solr/Lucene boolean syntax, so passing
    the catalyst ``tv_query`` (when available) gives tighter results than a
    plain keyword string.

    Args:
        api_key:          Guardian API key (free registration at open-platform.theguardian.com).
        query:            Search query — plain text or Solr boolean syntax.
        from_date:        Earliest article date. Passed as ``from-date=YYYY-MM-DD``.
        max_results:      Number of results to request (Guardian max page-size: 200).
        excluded_domains: Domains to skip. Matched against ``webUrl`` hostname.

    Returns:
        List of ``RawDocument`` objects with ``source_type="news"``.

    Raises:
        GuardianRateLimitError: if the API returns HTTP 429.
    """
    now = datetime.now(timezone.utc)
    query = _sanitize_query(query)
    if not query:
        log.debug("guardian.fetch.skip", reason="empty_query_after_sanitize")
        return []

    params: dict[str, str | int] = {
        "q": query,
        "show-fields": "body,headline",
        "order-by": "relevance",
        "page-size": min(max_results, 200),
        "api-key": api_key,
    }
    if from_date is not None:
        params["from-date"] = from_date.strftime("%Y-%m-%d")

    await asyncio.sleep(_RATE_LIMIT_SLEEP)

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.get(_BASE_URL, params=params)
    except Exception:
        log.warning("guardian.fetch.error", query=query, exc_info=True)
        return []

    if response.status_code == 429:
        raise GuardianRateLimitError(f"Guardian API rate limit hit (query={query!r})")

    if response.status_code != 200:
        log.warning(
            "guardian.fetch.http_error",
            query=query,
            status=response.status_code,
        )
        return []

    try:
        data = response.json()
    except Exception:
        log.warning("guardian.fetch.json_error", query=query, exc_info=True)
        return []

    results = data.get("response", {}).get("results", [])
    docs: list[RawDocument] = []

    for item in results:
        url = (item.get("webUrl") or "").strip()
        fields = item.get("fields") or {}
        raw_body = (fields.get("body") or "").strip()
        title = (fields.get("headline") or item.get("webTitle") or "").strip()
        published_str = item.get("webPublicationDate")

        if not url or not raw_body:
            log.debug("guardian.fetch.skip", reason="missing_url_or_body", url=url)
            continue

        if is_domain_excluded(url, excluded_domains):
            log.debug("guardian.fetch.skip", reason="excluded_domain", url=url)
            continue

        body = _strip_html(raw_body)
        if not body:
            log.debug("guardian.fetch.skip", reason="empty_body_after_strip", url=url)
            continue

        published_at: datetime | None = None
        if published_str:
            try:
                published_at = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        docs.append(
            RawDocument(
                source_url=url,
                title=title,
                body=body,
                source_type="news",
                source_name="The Guardian",
                category="",
                tags=[],
                published_at=published_at,
                fetched_at=now,
            )
        )

    return docs
