"""LLMQuery logging, cost tracking, and budget circuit breaker.

IMPORTANT: Every LLM call must log a row here before returning,
even failed calls (success=False). This is non-negotiable.
"""
from __future__ import annotations

# TODO: implement audit logging to llm_queries table
# TODO: implement daily spend cap circuit breaker
