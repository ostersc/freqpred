"""Truth Social fetcher via truthbrush library.

Two modes:
- Search mode: catalyst-driven keyword search (fetch_search).
- Account feed mode: standing per-account feeds (fetch_account).

Both use asyncio.to_thread because truthbrush is synchronous.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser

import structlog

from freqpred.ingestion.fetchers import is_domain_excluded
from freqpred.ingestion.store import RawDocument

log = structlog.get_logger(__name__)

_DEFAULT_LOOKBACK = timedelta(hours=48)

try:
    from truthbrush.api import Api, LoginErrorException
except ImportError:  # pragma: no cover
    Api = None  # type: ignore[assignment,misc]

    class LoginErrorException(Exception):  # type: ignore[no-redef]
        """Raised when truthbrush cannot authenticate."""


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._text: list[str] = []

    def handle_data(self, data: str) -> None:
        self._text.append(data)

    def get_text(self) -> str:
        return " ".join("".join(self._text).split())


def _strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


def _statuses_to_docs(
    statuses: list[dict],
    created_after: datetime,
    excluded_domains: frozenset[str],
    now: datetime,
) -> list[RawDocument]:
    docs: list[RawDocument] = []
    for status in statuses:
        # Skip reblogs (reposts).
        if status.get("reblog"):
            continue

        url = (status.get("url") or "").strip()
        if not url:
            continue

        if is_domain_excluded(url, excluded_domains):
            log.debug("truthsocial.skip.excluded_domain", url=url)
            continue

        created_str = status.get("created_at") or ""
        try:
            published_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published_at = now

        # Client-side recency filter.
        if published_at <= created_after:
            log.debug("truthsocial.skip.too_old", url=url, created_at=created_str)
            continue

        content_html = status.get("content") or ""
        body = _strip_html(content_html).strip()
        if not body:
            log.debug("truthsocial.skip.empty_body", url=url)
            continue

        account = status.get("account") or {}
        author = account.get("username") or account.get("display_name") or "unknown"

        docs.append(
            RawDocument(
                source_url=url,
                title=f"@{author}: {body[:120]}",
                body=body,
                source_type="social",
                source_name="TruthSocial",
                category="",
                tags=[],
                published_at=published_at,
                fetched_at=now,
            )
        )

    return docs


async def fetch_search(
    api: object,
    query: str,
    created_after: datetime | None = None,
    max_results: int = 40,
    excluded_domains: frozenset[str] = frozenset(),
) -> list[RawDocument]:
    """Fetch Truth Social posts matching a keyword search.

    Raises LoginErrorException if credentials are invalid — callers should
    treat this as a cycle-level circuit-breaker.

    Args:
        api:              truthbrush.Api instance (synchronous).
        query:            Search query string.
        created_after:    Earliest post datetime (default: now - 48h).
        max_results:      Max number of results to return.
        excluded_domains: URL domains to exclude.

    Returns:
        List of RawDocument objects with source_type="social".
    """
    now = datetime.now(UTC)
    if created_after is None:
        created_after = now - _DEFAULT_LOOKBACK

    def _sync_search() -> list[dict]:
        results: list[dict] = []
        for page in api.search(searchtype="statuses", query=query, limit=max_results):  # type: ignore[union-attr]
            statuses = (page or {}).get("statuses", [])
            results.extend(statuses)
            if len(results) >= max_results:
                break
        return results[:max_results]

    try:
        raw_statuses = await asyncio.to_thread(_sync_search)
    except LoginErrorException:
        raise
    except Exception:
        log.warning("truthsocial.fetch_search.error", query=query, exc_info=True)
        return []
    return _statuses_to_docs(raw_statuses, created_after, excluded_domains, now)


async def fetch_account(
    api: object,
    username: str,
    created_after: datetime,
    excluded_domains: frozenset[str] = frozenset(),
) -> list[RawDocument]:
    """Fetch Truth Social posts from a specific account.

    Raises LoginErrorException if credentials are invalid — callers should
    treat this as a cycle-level circuit-breaker.

    Args:
        api:              truthbrush.Api instance (synchronous).
        username:         Truth Social username (without @).
        created_after:    Only fetch posts after this datetime.
        excluded_domains: URL domains to exclude.

    Returns:
        List of RawDocument objects with source_type="social".
    """
    now = datetime.now(UTC)

    def _sync_pull() -> list[dict]:
        return list(api.pull_statuses(username, created_after=created_after))  # type: ignore[union-attr]

    try:
        raw_statuses = await asyncio.to_thread(_sync_pull)
    except LoginErrorException:
        raise
    except Exception:
        log.warning("truthsocial.fetch_account.error", username=username, exc_info=True)
        return []
    return _statuses_to_docs(raw_statuses, created_after, excluded_domains, now)
