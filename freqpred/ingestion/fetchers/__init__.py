"""Fetcher utilities shared across all fetcher modules."""
from __future__ import annotations

from urllib.parse import urlparse


def is_domain_excluded(url: str, excluded_domains: frozenset[str]) -> bool:
    """Return True if *url*'s hostname matches any entry in *excluded_domains*.

    A match occurs when the hostname equals the excluded domain or ends with
    ``.<domain>``, so subdomains are included but unrelated domains that happen
    to contain the string (e.g. ``notkalshi.com``) are not.

    Any path, query string, or fragment in *url* is ignored — only the netloc
    is inspected.
    """
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    for domain in excluded_domains:
        if host == domain or host.endswith("." + domain):
            return True
    return False
