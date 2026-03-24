"""Internet Archive Third Eye API fetcher — TV chyron (lower-third) text.

Third Eye provides OCR-extracted chyron text from live US TV broadcasts
(CNN, Fox News, MSNBC, BBC) in near-real-time, no authentication required.

Unlike the per-market tv_archive fetcher, Third Eye uses a bulk-pull +
local-filter architecture: pull all chyrons once per cycle, then distribute
matching rows to each market using its tv_query AND-groups.

API endpoint: https://archive.org/services/third-eye.php?last=N
Response: tab-separated, 5 columns, header row, no authentication required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import structlog

from freqpred.ingestion.store import RawDocument

log = structlog.get_logger(__name__)

_THIRD_EYE_URL = "https://archive.org/services/third-eye.php"
_API_TIMEOUT = 30.0
_ARCHIVE_DETAILS_BASE = "https://archive.org/details/"

_AND_SPLIT_RE = re.compile(r"\s+AND\s+", re.IGNORECASE)
_OR_SPLIT_RE = re.compile(r"\s+OR\s+", re.IGNORECASE)


@dataclass
class ChyronRow:
    dt: datetime          # date_time_(UTC) column
    channel: str
    duration_s: int
    identifier_path: str  # col 4 — e.g. "FOXNEWSW_.../start/60"
    text: str             # raw chyron text (may contain \n between lines)


def _parse_show_name(identifier_path: str) -> str:
    """Extract show name from archive identifier like 'FOXNEWSW_20260323_230100_Fox_News/start/60'."""
    base = identifier_path.split("/")[0]
    parts = base.split("_")
    if len(parts) > 3:
        return " ".join(parts[3:])
    return base


async def fetch_all(lookback_hours: int = 1) -> list[ChyronRow]:
    """Pull all chyrons from the Third Eye API for the last lookback_hours.

    Args:
        lookback_hours: How far back to query (passed as ?last=N to the API).
                        The Third Eye API only accepts whole-number values.

    Returns:
        ChyronRow objects sorted by dt ascending. Returns [] on any error.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                _THIRD_EYE_URL,
                params={"last": lookback_hours},
                timeout=_API_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            log.warning("tv_chyron.fetch_all.timeout")
            return []
        except httpx.HTTPStatusError as exc:
            log.warning(
                "tv_chyron.fetch_all.http_error",
                status_code=exc.response.status_code,
            )
            return []
        except Exception:
            log.warning("tv_chyron.fetch_all.error", exc_info=True)
            return []

    lines = response.text.splitlines()
    if len(lines) < 2:
        return []

    rows: list[ChyronRow] = []
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        cols = line.split("\t", 4)
        if len(cols) < 5:
            log.debug("tv_chyron.fetch_all.malformed_row", cols=len(cols))
            continue
        try:
            dt = datetime.strptime(cols[0].strip(), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            channel = cols[1].strip()
            duration_s = int(cols[2].strip())
            identifier_path = cols[3].strip()
            text = cols[4]
        except (ValueError, IndexError):
            log.debug("tv_chyron.fetch_all.parse_error", line=line[:80])
            continue

        rows.append(
            ChyronRow(
                dt=dt,
                channel=channel,
                duration_s=duration_s,
                identifier_path=identifier_path,
                text=text,
            )
        )

    rows.sort(key=lambda r: r.dt)
    log.info("tv_chyron.fetch_all.complete", chyrons_fetched=len(rows))
    return rows


def parse_and_groups(tv_query: str | None) -> list[list[str]]:
    """Parse a Solr/Lucene boolean query into AND-groups of OR'd terms.

    Examples:
        'trump' → [['trump']]
        'trump AND ("communist" OR "communism")' → [['trump'], ['communist', 'communism']]
        '"federal reserve" AND ("rate cut" OR "interest rates")' →
            [['federal reserve'], ['rate cut', 'interest rates']]

    A chyron matches only if EVERY group has at least one term present
    (case-insensitive substring match).
    """
    if not tv_query or not tv_query.strip():
        return []

    and_parts = _AND_SPLIT_RE.split(tv_query)
    result: list[list[str]] = []

    for part in and_parts:
        part = part.strip()
        # Strip surrounding parentheses
        if part.startswith("(") and part.endswith(")"):
            part = part[1:-1].strip()

        or_terms = _OR_SPLIT_RE.split(part)
        group: list[str] = []
        for term in or_terms:
            term = term.strip().strip('"')
            if term:
                group.append(term)

        if group:
            result.append(group)

    return result


def filter_chyrons(
    chyrons: list[ChyronRow],
    and_groups: list[list[str]],
    since: datetime | None = None,
) -> list[RawDocument]:
    """Return RawDocument objects for chyrons that match all AND-groups.

    Args:
        chyrons:    Rows from fetch_all().
        and_groups: Parsed query groups from parse_and_groups().
        since:      Skip rows with dt <= since (deduplication via cursor).

    Returns:
        Matching RawDocument objects with source_type='tv_chyron'.
    """
    if not and_groups:
        return []

    now = datetime.now(timezone.utc)
    docs: list[RawDocument] = []

    for row in chyrons:
        if since is not None and row.dt <= since:
            continue

        text_lower = row.text.lower()
        if not all(
            any(term.lower() in text_lower for term in group)
            for group in and_groups
        ):
            continue

        show_name = _parse_show_name(row.identifier_path)
        docs.append(
            RawDocument(
                source_url=_ARCHIVE_DETAILS_BASE + row.identifier_path,
                title=f"{row.channel}: {show_name}",
                body=row.text.replace("\n", " | "),
                source_type="tv_chyron",
                source_name="TVThirdEye",
                category="",
                tags=[row.channel],
                published_at=row.dt,
                fetched_at=now,
            )
        )

    return docs
