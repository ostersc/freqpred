"""TypeSafe System One transport (Jev).

Jev is not reachable through the OpenRouter path in ``provider.py``. That path
routes on a ``/`` in the model id to an Anthropic-Messages-compatible endpoint;
System One is a different protocol entirely — one ``state`` blob plus a map of
typed ``questions``, with no messages, no tools and no ``max_tokens``. So it
gets its own transport.

Two properties make it interesting here and both are load-bearing:

- **Questions are batched against one state.** The context is billed once no
  matter how many judgments are asked of it, which is why a signal estimate, a
  sizing score and an exit call can ride in a single request.
- **Output tokens are free and there is no string generation**, so the
  truncation failure mode that the assessor's ``max_tokens`` rule exists to
  prevent (a cut-off response failing open to neutral sizing) cannot occur.

The corresponding limitation: Jev cannot produce free text, so it cannot
satisfy any contract that asks for a ``reasoning`` field.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from freqpred.llm.audit import register_model_pricing

log = structlog.get_logger(__name__)

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
SYSTEM_ONE_PATH = "/v1/systemone"

# Pin the version rather than the ``jev-latest`` alias: a benchmark that silently
# changes model underneath itself is not a benchmark.
JEV_MODEL = "jev-1.13.0"

# Dollars per million tokens. Output is free ("too cheap to meter").
JEV_INPUT_PER_MTOK = 0.042
JEV_OUTPUT_PER_MTOK = 0.0

# 64k per request total, of which 32k is state plus the longest single question.
JEV_STATE_TOKEN_BUDGET = 32_000

_RETRY_STATUSES = frozenset({429, 529})
_MAX_ATTEMPTS = 4
_BACKOFF_BASE_SECONDS = 0.5


class TypeSafeError(Exception):
    """A System One call failed and its answer cannot be used."""


def register_jev_pricing() -> None:
    """Teach ``calculate_cost`` Jev's rates.

    Without this the audit layer falls back to Sonnet's $3/$15 per Mtok for any
    model it has no entry for — the same silent fallback that mispriced every
    ``claude-opus-4-7`` call for days. Here it would overstate Jev by ~70x and
    invert the one comparison the model is being evaluated on.
    """
    register_model_pricing(JEV_MODEL, JEV_INPUT_PER_MTOK, JEV_OUTPUT_PER_MTOK)


# --- Question builders -----------------------------------------------------


def noul(instructions: str, *, when_true: str, when_false: str) -> dict[str, Any]:
    """A yes/no judgment. The answer is a probability in [0, 1], 1 = yes."""
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": when_true, "false": when_false},
    }


def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
    """An ordered scale. ``criteria`` names each level, lowest first."""
    if len(criteria) < 2:
        raise ValueError("a score question needs at least two levels")
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def choice(instructions: str, criteria: dict[str, str | None]) -> dict[str, Any]:
    """A pick from named options, each mapped to a rubric (or None)."""
    if len(criteria) < 2:
        raise ValueError("a choice question needs at least two options")
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


# --- Response --------------------------------------------------------------


@dataclass(frozen=True)
class SystemOneResponse:
    """One System One call's answers plus what it cost."""

    model: str
    answers: dict[str, dict[str, Any]]
    tokens_input: int
    tokens_output: int
    latency_ms: int
    raw: dict[str, Any]

    def noul(self, key: str) -> float:
        """The probability for a noul question.

        Validated rather than trusted: a provider that types this as a "number"
        while meaning percent would otherwise sail through as a certainty. That
        has happened before on this codebase (deepseek returning 45 for 45%).
        """
        answer = self._answer(key, "noul")
        value = answer["noul"]
        if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
            raise TypeSafeError(
                f"noul {key!r} returned {value!r}, which is not a probability in [0,1]"
            )
        return float(value)

    def score(self, key: str) -> float:
        return float(self._answer(key, "score")["score"])

    def choice(self, key: str) -> str:
        return str(self._answer(key, "choice")["choice"])

    def confidence(self, key: str) -> float | None:
        """Jev reports confidence on choice and score answers, never on noul."""
        value = self.answers.get(key, {}).get("confidence")
        return float(value) if isinstance(value, int | float) else None

    def probabilities(self, key: str) -> dict[str, float] | None:
        value = self.answers.get(key, {}).get("probabilities")
        return {k: float(v) for k, v in value.items()} if isinstance(value, dict) else None

    def _answer(self, key: str, expected_type: str) -> dict[str, Any]:
        answer = self.answers.get(key)
        if answer is None:
            raise TypeSafeError(
                f"no answer for question {key!r}; got {sorted(self.answers)}"
            )
        if answer.get("type") != expected_type:
            raise TypeSafeError(
                f"question {key!r} answered as {answer.get('type')!r}, expected {expected_type!r}"
            )
        return answer


class TypeSafeTransport:
    """Minimal async client for ``POST /v1/systemone``.

    Retries 429 and 529 with exponential backoff, as the API reference asks.
    Retries are transport-level and consume no tokens, so the caller still
    audits exactly one row per logical call.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = TYPESAFE_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http = http_client
        self._owns_http = http_client is None

    async def __aenter__(self) -> TypeSafeTransport:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def system_one(
        self,
        *,
        state: dict[str, Any] | list[str] | str,
        questions: dict[str, dict[str, Any]],
        model: str = JEV_MODEL,
    ) -> SystemOneResponse:
        if not questions:
            raise ValueError("system_one needs at least one question")
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)

        body = {"model": model, "state": state, "questions": questions}
        start = time.monotonic()
        last_error: str | None = None

        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await self._http.post(
                    f"{self._base_url}{SYSTEM_ONE_PATH}",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue

            if response.status_code in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS - 1:
                log.warning(
                    "typesafe.retrying",
                    status=response.status_code,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
                continue

            if response.status_code != 200:
                # The body carries the validation detail on a 422, which is the
                # one failure a caller can actually fix.
                raise TypeSafeError(
                    f"System One returned HTTP {response.status_code}: {response.text[:500]}"
                )

            payload = response.json()
            usage = payload.get("usage") or {}
            return SystemOneResponse(
                model=payload.get("model", model),
                answers=payload.get("answers") or {},
                tokens_input=int(usage.get("input_tokens", 0)),
                tokens_output=int(usage.get("output_tokens", 0)),
                latency_ms=int((time.monotonic() - start) * 1000),
                raw=payload,
            )

        raise TypeSafeError(
            f"System One unreachable after {_MAX_ATTEMPTS} attempts: {last_error}"
        )


def audit_prompt(state: object, questions: dict[str, dict[str, Any]]) -> str:
    """Serialize a request for the audit log.

    The full point-in-time request is what made the postmortem's Brier and cost
    analysis possible at all, so it is stored verbatim rather than summarized.
    """
    return json.dumps({"state": state, "questions": questions}, sort_keys=True, default=str)
