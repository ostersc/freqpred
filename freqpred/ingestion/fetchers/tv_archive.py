"""Internet Archive TV News Archive fetcher.

Searches closed-caption transcripts from 163+ U.S. TV stations (CNN, MSNBC,
Fox News, CSPAN, BBC, etc.) via the archive.org internal beta search endpoint.
Data runs from July 2009 to present day — no auth required.

NOTE: This uses an unofficial/internal API reverse-engineered from the
archive.org website. The endpoint could change without notice.

Queries should be Solr/Lucene boolean syntax produced by the catalyst
generator's tv_query field, e.g.:
    trump AND ("communist" OR "communism")
    "federal reserve" AND ("rate cut" OR "interest rates")

Document body is taken from highlight.text (the query-relevant transcript
excerpt with {{{ }}} markers stripped) rather than cc_excerpt, which is
raw caption from the block start and not query-relevant.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
import structlog

from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()

_TV_SEARCH_URL = "https://archive.org/services/search/beta/page_production/"
_API_TIMEOUT = 45.0  # cache misses can be slow (~30s); add margin

_HIGHLIGHT_MARKER_RE = re.compile(r"\{\{\{|\}\}\}")


def _strip_highlight_markers(text: str) -> str:
    """Remove {{{ }}} emphasis markers from highlight.text excerpts."""
    return _HIGHLIGHT_MARKER_RE.sub("", text)


def _build_filter_map(close_time: datetime | None) -> dict[str, Any]:
    """Build the filter_map for the archive.org TV search API.

    Scopes the search to the 60 days leading up to *close_time* (or today
    if close_time is None/past), English-language news programs only.

    Date format must be YYYY-MM (month-level) for range operators to work.
    """
    now = datetime.now(timezone.utc)
    end = min(close_time, now) if close_time and close_time > now else now
    start = end - timedelta(days=60)

    start_str = start.strftime("%Y-%m")
    end_str = end.strftime("%Y-%m")

    return {
        "date": {start_str: "gte", end_str: "lte"},
        "language": {"English": "inc"},
        "program": {"News": "inc"},
    }


async def fetch(
    query: str,
    close_time: datetime | None = None,
    max_results: int = 20,
) -> list[RawDocument]:
    """Fetch TV transcript clips from the Internet Archive TV News Archive.

    Args:
        query:       Solr/Lucene boolean query (tv_query from catalyst generator).
        close_time:  Market close time used to scope the date range.
                     Defaults to a 60-day lookback from now.
        max_results: Maximum number of results to return.

    Returns:
        List of RawDocument objects with source_type="tv_transcript",
        source_name="TVArchive".
    """
    now = datetime.now(timezone.utc)
    filter_map = _build_filter_map(close_time)

    params: dict[str, Any] = {
        "service_backend": "tvs",
        "user_query": query,
        "hits_per_page": max_results,
        "page": 1,
        "aggregations": "false",
        "filter_map": json.dumps(filter_map),
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(_TV_SEARCH_URL, params=params, timeout=_API_TIMEOUT)
            response.raise_for_status()
        except httpx.TimeoutException:
            log.warning("tv_archive.fetch.timeout", query=query)
            return []
        except httpx.HTTPStatusError as exc:
            log.warning(
                "tv_archive.fetch.http_error",
                query=query,
                status_code=exc.response.status_code,
            )
            return []
        except Exception:
            log.warning("tv_archive.fetch.error", query=query, exc_info=True)
            return []

        try:
            data = response.json()
        except Exception:
            log.warning("tv_archive.fetch.non_json_response", query=query, body=response.text[:200])
            return []

    hits: list[dict[str, Any]] = (
        data.get("response", {}).get("body", {}).get("hits", {}).get("hits", [])
    )
    if not hits:
        return []

    docs: list[RawDocument] = []
    for hit in hits[:max_results]:
        fields = hit.get("fields", {})
        highlight = hit.get("highlight", {})

        # Use highlight.text as document body — it's the query-relevant excerpt.
        highlight_texts = highlight.get("text", [])
        if not highlight_texts:
            log.debug("tv_archive.fetch.skip", reason="no_highlight", identifier=fields.get("identifier"))
            continue

        body = _strip_highlight_markers(highlight_texts[0]).strip()
        if not body:
            continue

        # Build a unique source_url deep-linking to the exact timestamp.
        href = fields.get("__href__", "")
        source_url = f"https://archive.org{href}" if href else ""
        if not source_url:
            log.debug("tv_archive.fetch.skip", reason="no_href", identifier=fields.get("identifier"))
            continue

        title = (fields.get("title") or "").strip()

        # Parse broadcast date from fields.date ("2026-03-20T00:00:00Z").
        date_str = fields.get("date") or ""
        try:
            published_at = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            published_at = now

        # Named entities from auto-NER — stored as tags.
        subjects: list[str] = fields.get("subject") or []
        tags = [s for s in subjects if isinstance(s, str)]

        docs.append(
            RawDocument(
                source_url=source_url,
                title=title,
                body=body,
                source_type="tv_transcript",
                source_name="TVArchive",
                category="",
                tags=tags,
                published_at=published_at,
                fetched_at=now,
            )
        )

    log.info(
        "tv_archive.fetch.complete",
        query=query,
        hits_returned=len(hits),
        docs_extracted=len(docs),
    )
    return docs
