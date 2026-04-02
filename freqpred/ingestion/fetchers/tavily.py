"""Tavily Search API fetcher."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog
from tavily import TavilyClient
from tavily.errors import ForbiddenError, UsageLimitExceededError

from freqpred.ingestion.fetchers import is_domain_excluded
from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()


_DEFAULT_EXCLUDED_DOMAINS: frozenset[str] = frozenset({"kalshi.com"})


async def fetch(
    api_key: str,
    query: str,
    max_results: int = 20,
    excluded_domains: frozenset[str] = _DEFAULT_EXCLUDED_DOMAINS,
) -> list[RawDocument]:
    """Fetch news articles from Tavily Search API.

    Runs the synchronous Tavily client in a thread to avoid blocking.
    Skips results missing url or body content, and any URL whose hostname
    contains a domain in *excluded_domains*.

    Args:
        api_key:          Tavily API key.
        query:            Search query string.
        max_results:      Maximum number of results to request (default 20).
        excluded_domains: Set of domain strings to skip (e.g. ``{"kalshi.com"}``).
                          Matched as a substring of the URL so subdomains are also excluded.

    Returns:
        List of RawDocument objects with source_type="news".
    """
    client = TavilyClient(api_key=api_key)
    now = datetime.now(timezone.utc)

    try:
        response: dict[str, Any] = await asyncio.to_thread(
            client.search,
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
        )
    except (ForbiddenError, UsageLimitExceededError):
        # Plan/usage limit — re-raise so the caller can short-circuit remaining queries.
        raise
    except Exception:
        log.warning("tavily.fetch.error", query=query, exc_info=True)
        return []

    results = response.get("results", [])
    docs: list[RawDocument] = []

    for item in results:
        url = item.get("url", "").strip()
        body = (item.get("raw_content") or item.get("content") or "").strip()
        title = (item.get("title") or "").strip()
        published_str = item.get("published_date")

        if not url or not body:
            log.warning("tavily.fetch.skip", reason="missing_url_or_body", url=url)
            continue

        if is_domain_excluded(url, excluded_domains):
            log.debug("tavily.fetch.skip", reason="excluded_domain", url=url)
            continue

        published_at = None
        if published_str:
            try:
                published_at = datetime.fromisoformat(published_str)
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        docs.append(
            RawDocument(
                source_url=url,
                title=title,
                body=body,
                source_type="news",
                source_name="Tavily",
                category="",
                tags=[],
                published_at=published_at,
                fetched_at=now,
            )
        )

    return docs
