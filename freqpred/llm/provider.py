"""Transport selection for LLM calls: Anthropic direct, or OpenRouter.

OpenRouter serves an Anthropic-compatible Messages API, so the same
``anthropic.AsyncAnthropic`` client speaks to both — only ``base_url`` and the
key differ. Every feature the signal path relies on survives the hop: system
prompts, ``cache_control``, forced tool use via ``tool_choice``, ``thinking``,
and ``output_tokens_details.thinking_tokens``.

Two details worth knowing before changing anything here:

* The SDK appends ``/v1/messages`` to ``base_url`` itself, so the base must
  stop at ``/api``. Setting it to ``https://openrouter.ai/api/v1`` produces
  ``/api/v1/v1/messages``, which returns a 404 *HTML* page — surfacing as an
  ``anthropic.NotFoundError`` whose message is a whole web page rather than
  anything resembling a routing error.
* OpenRouter reports the real dollar cost of each call on ``usage.cost``, so
  cost for these models is measured rather than derived from a local price
  table. See ``freqpred.llm.audit.calculate_cost``, which has no rates for
  non-Anthropic slugs and would silently apply a default.
"""
from __future__ import annotations

import anthropic
import httpx
import structlog

log = structlog.get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def is_openrouter_model(model: str) -> bool:
    """Return True when ``model`` is an OpenRouter slug rather than an Anthropic id.

    OpenRouter slugs are always ``vendor/model`` (``anthropic/claude-sonnet-4.5``,
    ``openai/gpt-5``); Anthropic's own model ids never contain a slash
    (``claude-sonnet-4-6``). That makes the separator a reliable discriminator,
    and it means a config value alone decides the transport — no extra flag has
    to be kept in sync with the model name.
    """
    return "/" in model


def make_openrouter_client(api_key: str) -> anthropic.AsyncAnthropic:
    """Build an Anthropic-SDK client pointed at OpenRouter's Messages API."""
    return anthropic.AsyncAnthropic(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def maybe_openrouter_client(api_key: str | None) -> anthropic.AsyncAnthropic | None:
    """Build an OpenRouter client when a key is configured, else ``None``.

    Call sites pass the result straight to ``LLMClient(openrouter_client=...)``,
    so an unset key leaves behaviour exactly as it was before OpenRouter
    support existed.
    """
    return make_openrouter_client(api_key) if api_key else None


def fetch_openrouter_pricing(model: str, *, timeout: float = 30.0) -> tuple[float, float] | None:
    """Return ``(input, output)`` dollars per million tokens for an OpenRouter slug.

    Reads the public catalogue, which needs no credential. Returns ``None`` when
    the slug is absent or the catalogue cannot be reached — callers should treat
    that as "no pricing known" rather than substituting a guess, since the whole
    point is to avoid a wrong number that looks authoritative.
    """
    try:
        response = httpx.get(OPENROUTER_MODELS_URL, timeout=timeout)
        response.raise_for_status()
        entries = response.json()["data"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("openrouter.pricing_lookup_failed", model=model, error=str(exc))
        return None

    for entry in entries:
        if entry.get("id") != model:
            continue
        pricing = entry.get("pricing") or {}
        try:
            return (
                float(pricing["prompt"]) * 1_000_000,
                float(pricing["completion"]) * 1_000_000,
            )
        except (KeyError, TypeError, ValueError):
            log.warning("openrouter.pricing_unparseable", model=model, pricing=pricing)
            return None

    log.warning("openrouter.model_not_in_catalogue", model=model)
    return None


def openrouter_call_cost(usage: object) -> float | None:
    """Return the dollar cost OpenRouter reported for a call, if it reported one.

    Returns ``None`` when the field is absent, which is what an Anthropic-direct
    response looks like. Callers fall back to the local price table then, rather
    than recording a zero that would read as a free call.

    The numeric check is deliberate rather than a bare ``is not None``: a
    ``MagicMock`` usage object auto-creates any attribute asked of it, so a
    truthiness test would treat every mocked Anthropic response as if it
    carried a cost.
    """
    cost = getattr(usage, "cost", None)
    return float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
