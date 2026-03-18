"""NewsAPI fetcher with 1 req/sec rate limit."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from newsapi import NewsApiClient

from freqpred.ingestion.fetchers import is_domain_excluded
from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()

_RATE_LIMIT_SLEEP = 1.0  # max 1 req/sec
_DEFAULT_EXCLUDED_DOMAINS: frozenset[str] = frozenset({"kalshi.com"})


async def fetch(
    api_key: str,
    query: str,
    from_date: datetime | None = None,
    max_results: int = 20,
    excluded_domains: frozenset[str] = _DEFAULT_EXCLUDED_DOMAINS,
) -> list[RawDocument]:
    """Fetch news articles from NewsAPI.

    Enforces max 1 req/sec via a fixed sleep before the call.
    Skips articles missing url or content, and any URL whose hostname
    contains a domain in *excluded_domains*.

    Note: the NewsAPI free tier restricts ``/everything`` to the past 24 hours
    regardless of ``from_date``.  Omitting ``from_date`` (or passing None) lets
    the API return whatever it can within its allowed window.  On paid plans,
    passing a ``from_date`` extends coverage back up to 30 days.

    Args:
        api_key:          NewsAPI API key.
        query:            Search query string.
        from_date:        Earliest article date (paid plans only; None = API default).
        max_results:      Maximum number of results (capped at 100 by NewsAPI).
        excluded_domains: Set of domain strings to skip (e.g. ``{"kalshi.com"}``).
                          Matched as a substring of the URL so subdomains are also excluded.

    Returns:
        List of RawDocument objects with source_type="news".
    """
    client = NewsApiClient(api_key=api_key)
    now = datetime.now(timezone.utc)

    kwargs: dict = dict(
        q=query,
        page_size=min(max_results, 100),
        sort_by="relevancy",
        language="en",
    )
    if from_date is not None:
        kwargs["from_param"] = from_date.strftime("%Y-%m-%dT%H:%M:%S")

    await asyncio.sleep(_RATE_LIMIT_SLEEP)

    try:
        response = await asyncio.to_thread(
            client.get_everything,
            **kwargs,
        )
    except Exception:
        log.warning("newsapi.fetch.error", query=query, exc_info=True)
        return []

    articles = response.get("articles", [])
    docs: list[RawDocument] = []

    for article in articles:
        url = (article.get("url") or "").strip()
        body = (article.get("content") or article.get("description") or "").strip()
        title = (article.get("title") or "").strip()
        source_name = (article.get("source", {}).get("name") or "NewsAPI").strip()
        published_str = article.get("publishedAt")

        if not url or not body:
            log.warning("newsapi.fetch.skip", reason="missing_url_or_body", url=url)
            continue

        if is_domain_excluded(url, excluded_domains):
            log.debug("newsapi.fetch.skip", reason="excluded_domain", url=url)
            continue

        try:
            published_at = (
                datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                if published_str
                else now
            )
        except ValueError:
            published_at = now

        docs.append(
            RawDocument(
                source_url=url,
                title=title,
                body=body,
                source_type="news",
                source_name=source_name,
                category="",
                tags=[],
                published_at=published_at,
                fetched_at=now,
            )
        )

    return docs
