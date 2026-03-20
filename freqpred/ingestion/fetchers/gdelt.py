"""GDELT Doc API fetcher — no API key required."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import structlog

from freqpred.ingestion.fetchers import is_domain_excluded
from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()

_GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTRateLimitError(Exception):
    """Raised when GDELT returns a 429 response."""
_GDELT_MIN_KEYWORD_LEN = 4  # GDELT rejects tokens shorter than this


def _sanitize_query(query: str) -> str | None:
    """Strip tokens shorter than _GDELT_MIN_KEYWORD_LEN from *query*.

    Returns None if no tokens remain after filtering (caller should skip).
    """
    tokens = []
    for t in query.split():
        if len(t) < _GDELT_MIN_KEYWORD_LEN:
            continue
        # GDELT rejects bare hyphens in keywords; wrap hyphenated tokens in quotes.
        if "-" in t and not (t.startswith('"') and t.endswith('"')):
            t = f'"{t}"'
        tokens.append(t)
    return " ".join(tokens) if tokens else None


_DEFAULT_EXCLUDED_DOMAINS: frozenset[str] = frozenset({"kalshi.com"})
_ARTICLE_FETCH_TIMEOUT = 10.0  # seconds per URL
_API_TIMEOUT = 30.0
_RATE_LIMIT_SLEEP = 5.0  # GDELT enforces 1 req/5s; no Retry-After header is returned


async def _fetch_article_body(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch article body from *url* with a timeout. Returns None on any failure."""
    try:
        response = await client.get(url, timeout=_ARTICLE_FETCH_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except Exception:
        log.debug("gdelt.fetch_article.skip", url=url)
        return None


async def fetch(
    query: str,
    timespan: str = "1d",
    max_results: int = 20,
    excluded_domains: frozenset[str] = _DEFAULT_EXCLUDED_DOMAINS,
) -> list[RawDocument]:
    """Fetch news articles from the GDELT Doc API.

    Queries GDELT for article URLs matching *query*, then fetches article bodies
    in parallel with a per-URL timeout of 10 s. Failed or paywalled URLs are
    skipped silently. No API key is required.

    Args:
        query:            Search query string (catalyst query text).
        timespan:         GDELT timespan parameter (e.g. "1d", "1h", "15min").
        max_results:      Maximum number of results to return.
        excluded_domains: Set of domain strings to skip.

    Returns:
        List of RawDocument objects with source_type="news", source_name="GDELT".
    """
    now = datetime.now(timezone.utc)

    sanitized = _sanitize_query(query)
    if sanitized is None:
        log.warning("gdelt.fetch.query_empty_after_sanitize", query=query)
        return []
    if sanitized != query:
        log.debug("gdelt.fetch.query_sanitized", original=query, sanitized=sanitized)

    params: dict[str, str | int] = {
        "query": sanitized,
        "mode": "artlist",
        "format": "json",
        "timespan": timespan,
        "maxrecords": max_results,
    }

    await asyncio.sleep(_RATE_LIMIT_SLEEP)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(_GDELT_API_URL, params=params, timeout=_API_TIMEOUT)
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            log.warning("gdelt.fetch.error", query=query, error=type(exc).__name__)
            return []
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise GDELTRateLimitError(f"GDELT rate limited (429)") from exc
            log.warning("gdelt.fetch.error", query=query, status_code=exc.response.status_code)
            return []
        except Exception:
            log.warning("gdelt.fetch.error", query=query, exc_info=True)
            return []

        try:
            data = response.json()
        except Exception:
            # GDELT returns plain-text error messages (e.g. "keyword too short") with HTTP 200.
            log.warning("gdelt.fetch.non_json_response", query=query, body=response.text.strip())
            return []

        articles = data.get("articles") or []
        if not articles:
            return []

        # Filter excluded domains before fetching bodies.
        candidates = []
        for article in articles[:max_results]:
            url = (article.get("url") or "").strip()
            if not url:
                continue
            if is_domain_excluded(url, excluded_domains):
                log.debug("gdelt.fetch.skip", reason="excluded_domain", url=url)
                continue
            candidates.append(article)

        if not candidates:
            return []

        # Fetch article bodies in parallel.
        bodies: list[str | None] = list(
            await asyncio.gather(*[_fetch_article_body(client, a["url"]) for a in candidates])
        )

    docs: list[RawDocument] = []
    for article, body in zip(candidates, bodies):
        if not body or not body.strip():
            log.debug("gdelt.fetch.skip", reason="empty_body", url=article.get("url"))
            continue

        url = article["url"]
        title = (article.get("title") or "").strip()
        seen_date_str = article.get("seendate") or ""

        try:
            # GDELT seendate format: "20260101T120000Z"
            published_at = datetime.strptime(seen_date_str, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            published_at = now

        docs.append(
            RawDocument(
                source_url=url,
                title=title,
                body=body,
                source_type="news",
                source_name="GDELT",
                category="",
                tags=[],
                published_at=published_at,
                fetched_at=now,
            )
        )

    return docs
