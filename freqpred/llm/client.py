"""LLM API wrapper (Claude via Anthropic SDK).

Every call to complete() logs an audit row via audit.log_llm_query —
including failed calls (success=False). This is non-negotiable.
"""
from __future__ import annotations

import json
import time

import anthropic
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.llm.audit import (
    LLMBudgetExceededError,
    calculate_cost,
    get_daily_spend_usd,
    log_llm_query,
)
from freqpred.llm.models import LLMResponse
from freqpred.llm.provider import is_openrouter_model, openrouter_call_cost

log = structlog.get_logger(__name__)


class LLMError(Exception):
    """Raised when the Anthropic API call fails."""


class LLMConsecutiveErrorsError(Exception):
    """Raised when the LLM API has failed N consecutive times without recovery."""


class LLMClient:
    """Thin wrapper around the Anthropic async client that ensures every
    call is audited in the ``llm_queries`` table.

    Args:
        anthropic_client:  An ``anthropic.AsyncAnthropic`` instance.
        session_factory:   SQLAlchemy async session factory used for
                           fire-and-forget audit writes.
        default_strategy:  Strategy name written to audit rows when the
                           caller does not supply one (default: "system").
        prompt_version:    Versioned prompt template ID (default: "v1").
        openrouter_client: Optional second ``anthropic.AsyncAnthropic`` pointed
                           at OpenRouter. When supplied, any model whose id is
                           an OpenRouter slug (``vendor/model``) is routed
                           through it; everything else keeps going direct to
                           Anthropic. Both transports speak the same Messages
                           API, so the call path below is shared.
    """

    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        default_strategy: str = "system",
        prompt_version: str = "v1",
        daily_spend_cap_usd: float | None = None,
        max_consecutive_errors: int = 3,
        openrouter_client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._client = anthropic_client
        self._openrouter_client = openrouter_client
        self._session_factory = session_factory
        self._default_strategy = default_strategy
        self._prompt_version = prompt_version
        self._daily_spend_cap_usd = daily_spend_cap_usd
        self._max_consecutive_errors = max_consecutive_errors
        self._consecutive_errors = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        query_type: str,
        *,
        system: str | None = None,
        cache_system: bool = False,
        market_id: str | None = None,
        signal_id: str | None = None,
        strategy: str | None = None,
        prompt_version: str | None = None,
        max_tokens: int = 1024,
        json_tool: dict | None = None,
        thinking: dict | None = None,
    ) -> LLMResponse:
        """Call the Anthropic messages API and return an ``LLMResponse``.

        Logs exactly one ``llm_queries`` row regardless of success or failure.
        Raises ``LLMError`` on API failure after logging.

        Args:
            prompt:     Full prompt to send as a single user message.
            model:      Anthropic model ID, e.g. "claude-haiku-4-5-20251001".
            query_type: Audit category (e.g. "market_analysis").
            system:     Optional system prompt.
            market_id:  Optional market being analyzed.
            signal_id:  Optional signal produced by this call.
            strategy:   Strategy name; falls back to ``default_strategy``.
            max_tokens: Maximum tokens in the response (default: 1024).
            json_tool:  If provided, forces a tool call with this definition
                        (must have "name", "description", "input_schema" keys).
                        The tool input dict is serialised to JSON and returned
                        as ``LLMResponse.content``, guaranteeing valid JSON
                        output without any prose preamble.
            thinking:   Optional ``thinking`` config passed straight through to
                        the Anthropic API (e.g. ``{"type": "adaptive", "display":
                        "summarized"}``). Omitted by default so existing callers
                        keep relying on each model's own default thinking
                        behavior unchanged. When thinking blocks come back with
                        visible text, they're concatenated into
                        ``LLMResponse.thinking``; the actual thinking token
                        count (including 0 when adaptive thinking chose not to
                        think at all) is reported separately on
                        ``LLMResponse.thinking_tokens`` so "no visible text" and
                        "no thinking happened" aren't conflated.

        Returns:
            ``LLMResponse`` on success.

        Raises:
            ``LLMError`` on Anthropic API failure.
        """
        strategy_name = strategy or self._default_strategy
        effective_prompt_version = prompt_version or self._prompt_version

        if self._daily_spend_cap_usd is not None:
            async with self._session_factory() as session:
                daily_spend = await get_daily_spend_usd(session)
            if daily_spend >= self._daily_spend_cap_usd:
                log.warning(
                    "llm_budget_exceeded",
                    daily_spend_usd=round(daily_spend, 4),
                    cap_usd=self._daily_spend_cap_usd,
                )
                raise LLMBudgetExceededError(
                    f"Daily LLM spend cap of ${self._daily_spend_cap_usd:.2f} reached "
                    f"(spent ${daily_spend:.4f} today)"
                )

        start = time.monotonic()

        create_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            if cache_system:
                create_kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
                ]
            else:
                create_kwargs["system"] = system
        if json_tool:
            tool_entry = dict(json_tool)
            if cache_system:
                tool_entry["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
            create_kwargs["tools"] = [tool_entry]
            create_kwargs["tool_choice"] = {"type": "tool", "name": json_tool["name"]}
        if thinking:
            create_kwargs["thinking"] = thinking

        try:
            message = await self._transport_for(model).messages.create(**create_kwargs)
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            await self._write_audit(
                prompt_version=effective_prompt_version,
                strategy=strategy_name,
                query_type=query_type,
                model_used=model,
                prompt=prompt,
                response="",
                tokens_input=0,
                tokens_output=0,
                cost_usd=0.0,
                latency_ms=latency_ms,
                success=False,
                market_id=market_id,
                signal_id=signal_id,
                error_message=str(exc),
            )
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._max_consecutive_errors:
                raise LLMConsecutiveErrorsError(
                    f"LLM API failed {self._consecutive_errors} consecutive times"
                ) from exc
            raise LLMError(str(exc)) from exc

        self._consecutive_errors = 0
        latency_ms = int((time.monotonic() - start) * 1000)

        # A forced tool_choice is honored by Anthropic's own models but not by
        # every model reachable through OpenRouter: some answer in prose and
        # stop with end_turn, others spend the whole budget thinking and emit
        # no tool block at all. Falling back to the text block there hands the
        # caller prose (or "", when the only block is a thinking block) while
        # the audit row still says success, so a model that cannot satisfy the
        # contract reads as a working one producing unparseable signals.
        tool_contract_violated = False
        text_block_missing = False
        if json_tool:
            tool_block = next((b for b in message.content if b.type == "tool_use"), None)
            if tool_block is not None:
                content = json.dumps(tool_block.input)
            else:
                content = next((b.text for b in message.content if hasattr(b, "text")), "")
                tool_contract_violated = True
        else:
            # Not content[0]: a reasoning model puts a thinking block first, and
            # indexing blindly raised AttributeError here — before the audit row
            # was written, so the call cost money and left no trace at all.
            text_block = next(
                (b for b in message.content if getattr(b, "type", None) == "text"), None
            )
            if text_block is None:
                content = ""
                text_block_missing = True
            else:
                content = text_block.text

        thinking_text = "".join(
            b.thinking for b in message.content
            if getattr(b, "type", None) == "thinking" and getattr(b, "thinking", "")
        ) or None
        thinking_tokens = (getattr(message.usage, "output_tokens_details", None) or {}).get("thinking_tokens")

        tokens_in = message.usage.input_tokens
        tokens_out = message.usage.output_tokens
        cache_read = getattr(message.usage, "cache_read_input_tokens", 0) or 0
        cache_created = getattr(message.usage, "cache_creation_input_tokens", 0) or 0
        # OpenRouter reports the actual dollar cost of the call; calculate_cost
        # has no rates for those slugs and would quietly apply its default.
        reported_cost = openrouter_call_cost(message.usage)
        if reported_cost is None:
            cost = calculate_cost(model, tokens_in, tokens_out, cache_created, cache_read)
        else:
            cost = reported_cost

        if tool_contract_violated or text_block_missing:
            blocks = ", ".join(sorted({str(getattr(b, "type", "?")) for b in message.content})) or "none"
            if tool_contract_violated:
                detail = (
                    f"Model {model!r} ignored the forced tool_choice for "
                    f"{json_tool['name']!r}: stop_reason={message.stop_reason}, "
                    f"content blocks=[{blocks}]"
                )
            else:
                # Reached when a model spends its whole budget thinking and stops
                # before emitting any text — an empty answer, not a usable one.
                detail = (
                    f"Model {model!r} returned no text block: "
                    f"stop_reason={message.stop_reason}, content blocks=[{blocks}]"
                )
            # The call happened and cost money, so it is still audited — as a
            # failure, since its output cannot be used.
            await self._write_audit(
                prompt_version=effective_prompt_version,
                strategy=strategy_name,
                query_type=query_type,
                model_used=model,
                prompt=prompt,
                response=content,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                cost_usd=cost,
                latency_ms=latency_ms,
                success=False,
                market_id=market_id,
                signal_id=signal_id,
                error_message=detail,
            )
            if tool_contract_violated:
                log.warning(
                    "llm.tool_contract_violated",
                    model=model,
                    tool=json_tool["name"],
                    stop_reason=message.stop_reason,
                    content_blocks=blocks,
                )
            else:
                log.warning(
                    "llm.no_text_block",
                    model=model,
                    stop_reason=message.stop_reason,
                    content_blocks=blocks,
                )
            raise LLMError(detail)

        llm_query_id = await self._write_audit(
            prompt_version=effective_prompt_version,
            strategy=strategy_name,
            query_type=query_type,
            model_used=model,
            prompt=prompt,
            response=content,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            success=True,
            market_id=market_id,
            signal_id=signal_id,
        )

        total_input = tokens_in + cache_read + cache_created
        log.debug(
            "llm_complete",
            model=model,
            query_type=query_type,
            tokens_input_uncached=tokens_in,
            tokens_output=tokens_out,
            tokens_total_input=total_input,
            cache_read_tokens=cache_read,
            cache_created_tokens=cache_created,
            cache_hit=cache_read > 0,
            cost_usd=round(cost, 6),
            latency_ms=latency_ms,
        )

        return LLMResponse(
            content=content,
            model=model,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            llm_query_id=llm_query_id,
            thinking=thinking_text,
            thinking_tokens=thinking_tokens,
        )

    def _transport_for(self, model: str) -> anthropic.AsyncAnthropic:
        """Pick the client that serves ``model``.

        An OpenRouter slug with no OpenRouter client configured is a
        misconfiguration, not something to paper over: sent to the Anthropic
        endpoint it would fail as an unknown model, which reads as a model
        problem rather than a missing key.
        """
        if not is_openrouter_model(model):
            return self._client
        if self._openrouter_client is None:
            raise LLMError(
                f"Model {model!r} is an OpenRouter slug but no OpenRouter client "
                "is configured (set OPENROUTER_API_KEY)"
            )
        return self._openrouter_client

    async def _write_audit(self, prompt_version: str, **kwargs) -> int:
        """Open a short-lived session and write one llm_queries row."""
        async with self._session_factory() as session:
            llm_query_id = await log_llm_query(
                session,
                prompt_version=prompt_version,
                **kwargs,
            )
            await session.commit()
        return llm_query_id
