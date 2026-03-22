"""Reddit JSON API fetcher (no credentials required)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import structlog

from freqpred.ingestion.store import RawDocument

log = structlog.get_logger()

_MIN_SCORE = 10
_MAX_AGE_DAYS = 7
_BASE_URL = "https://www.reddit.com"
_SEARCH_PATH = "/r/{subreddit}/search.json"
_DEFAULT_HEADERS = {"User-Agent": "freqpred/0.1"}


async def fetch(
    subreddits: list[str],
    query: str,
    user_agent: str = "freqpred/0.1",
    limit: int = 50,
) -> list[RawDocument]:
    """Fetch Reddit posts matching a query across the given subreddits.

    Uses Reddit's public JSON API — no credentials required.
    Filters posts by minimum upvote score (>=10) and recency (last 7 days).

    Args:
        subreddits: List of subreddit names to search (without 'r/' prefix).
        query:      Search query string.
        user_agent: User-Agent header sent with requests.
        limit:      Maximum submissions to fetch per subreddit.

    Returns:
        List of RawDocument objects with source_type="reddit".
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_MAX_AGE_DAYS)
    headers = {"User-Agent": user_agent}
    docs: list[RawDocument] = []

    async with httpx.AsyncClient(base_url=_BASE_URL, headers=headers, timeout=15.0) as client:
        for subreddit_name in subreddits:
            try:
                # Use streaming so we can inspect the status code before reading the body.
                # Reddit sends 429 and then closes the connection, causing a ReadError if
                # we try to read the body — streaming lets us bail out cleanly first.
                async with client.stream(
                    "GET",
                    _SEARCH_PATH.format(subreddit=subreddit_name),
                    params={
                        "q": query,
                        "sort": "new",
                        "limit": limit,
                        "restrict_sr": 1,
                    },
                ) as response:
                    status = response.status_code
                    if status in (403, 404, 429):
                        # 403: subreddit restricted/private
                        # 404: subreddit doesn't exist
                        # 429: rate limited
                        log.debug(
                            "reddit.fetch.skip",
                            subreddit=subreddit_name,
                            status=status,
                        )
                        continue
                    response.raise_for_status()
                    await response.aread()
                    data = response.json()

                    posts = data.get("data", {}).get("children", [])
                    for post in posts:
                        p = post.get("data", {})

                        score: int = p.get("score", 0)
                        created_utc: float = p.get("created_utc", 0.0)
                        permalink: str = p.get("permalink", "")
                        title: str = (p.get("title") or "").strip()

                        published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc)

                        if score < _MIN_SCORE:
                            log.debug(
                                "reddit.fetch.skip",
                                reason="low_score",
                                score=score,
                                permalink=permalink,
                            )
                            continue

                        if published_at < cutoff:
                            log.debug(
                                "reddit.fetch.skip",
                                reason="too_old",
                                published_at=published_at.isoformat(),
                                permalink=permalink,
                            )
                            continue

                        body = (p.get("selftext") or title).strip()
                        if not body:
                            log.warning(
                                "reddit.fetch.skip", reason="empty_body", permalink=permalink
                            )
                            continue

                        display_name: str = p.get("subreddit", subreddit_name)

                        docs.append(
                            RawDocument(
                                source_url=f"https://reddit.com{permalink}",
                                title=title,
                                body=body,
                                source_type="reddit",
                                source_name=f"r/{display_name}",
                                category="",
                                tags=[],
                                published_at=published_at,
                                fetched_at=now,
                            )
                        )
            except httpx.HTTPStatusError as exc:
                log.warning(
                    "reddit.fetch.error",
                    subreddit=subreddit_name,
                    query=query,
                    status=exc.response.status_code,
                )
            except Exception:
                log.warning(
                    "reddit.fetch.error",
                    subreddit=subreddit_name,
                    query=query,
                    exc_info=True,
                )

    return docs
