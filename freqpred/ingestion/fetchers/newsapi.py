"""NewsAPI fetcher with 1 req/sec rate limit."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from newsapi import NewsApiClient

from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()

_RATE_LIMIT_SLEEP = 1.0  # max 1 req/sec


async def fetch(
    api_key: str,
    query: str,
    from_date: datetime,
    max_results: int = 20,
) -> list[RawDocument]:
    """Fetch news articles from NewsAPI.

    Enforces max 1 req/sec via a fixed sleep before the call.
    Skips articles missing url or content.

    Args:
        api_key:     NewsAPI API key.
        query:       Search query string.
        from_date:   Earliest article date to include.
        max_results: Maximum number of results (capped at 100 by NewsAPI).

    Returns:
        List of RawDocument objects with source_type="news".
    """
    client = NewsApiClient(api_key=api_key)
    now = datetime.now(timezone.utc)
    from_str = from_date.strftime("%Y-%m-%dT%H:%M:%S")

    await asyncio.sleep(_RATE_LIMIT_SLEEP)

    try:
        response = await asyncio.to_thread(
            client.get_everything,
            q=query,
            from_param=from_str,
            page_size=min(max_results, 100),
            sort_by="publishedAt",
            language="en",
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
