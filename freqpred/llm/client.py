"""LLM API wrapper (Claude via Anthropic SDK).

Every call to complete() logs an audit row via audit.log_llm_query —
including failed calls (success=False). This is non-negotiable.
"""
from __future__ import annotations

import time

import anthropic
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.llm.audit import LLMBudgetExceededError, calculate_cost, get_daily_spend_usd, log_llm_query
from freqpred.llm.models import LLMResponse

log = structlog.get_logger(__name__)


class LLMError(Exception):
    """Raised when the Anthropic API call fails."""


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
    """

    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        default_strategy: str = "system",
        prompt_version: str = "v1",
        daily_spend_cap_usd: float | None = None,
    ) -> None:
        self._client = anthropic_client
        self._session_factory = session_factory
        self._default_strategy = default_strategy
        self._prompt_version = prompt_version
        self._daily_spend_cap_usd = daily_spend_cap_usd

    async def complete(
        self,
        prompt: str,
        model: str,
        query_type: str,
        *,
        system: str | None = None,
        market_id: str | None = None,
        signal_id: str | None = None,
        strategy: str | None = None,
        prompt_version: str | None = None,
        max_tokens: int = 1024,
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

        create_kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            create_kwargs["system"] = system

        try:
            message = await self._client.messages.create(**create_kwargs)
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
            raise LLMError(str(exc)) from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        content = message.content[0].text
        tokens_in = message.usage.input_tokens
        tokens_out = message.usage.output_tokens
        cost = calculate_cost(model, tokens_in, tokens_out)

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

        log.debug(
            "llm_complete",
            model=model,
            query_type=query_type,
            tokens_total=tokens_in + tokens_out,
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
        )

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
