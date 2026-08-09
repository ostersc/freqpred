"""Unit tests for freqpred/llm/provider.py.

No network: the OpenRouter client is only constructed, never called.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from freqpred.llm.audit import calculate_cost, register_model_pricing
from freqpred.llm.provider import (
    OPENROUTER_BASE_URL,
    fetch_openrouter_pricing,
    is_openrouter_model,
    make_openrouter_client,
    maybe_openrouter_client,
    openrouter_call_cost,
)


class TestIsOpenRouterModel:
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.6-terra",
            "deepseek/deepseek-v3",
            "z-ai/glm-4.6",
        ],
    )
    def test_slugs_route_to_openrouter(self, model: str) -> None:
        assert is_openrouter_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-opus-5",
        ],
    )
    def test_anthropic_ids_stay_direct(self, model: str) -> None:
        assert is_openrouter_model(model) is False


class TestClientConstruction:
    def test_base_url_stops_at_api(self) -> None:
        """The SDK appends /v1/messages, so a /v1 suffix here would 404."""
        assert OPENROUTER_BASE_URL == "https://openrouter.ai/api"
        assert not OPENROUTER_BASE_URL.endswith("/v1")

    def test_make_openrouter_client_points_at_openrouter(self) -> None:
        client = make_openrouter_client("sk-test")
        assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL

    def test_maybe_returns_none_without_key(self) -> None:
        assert maybe_openrouter_client("") is None
        assert maybe_openrouter_client(None) is None

    def test_maybe_returns_client_with_key(self) -> None:
        assert maybe_openrouter_client("sk-test") is not None


class TestOpenRouterCallCost:
    def test_reads_reported_cost(self) -> None:
        usage = MagicMock()
        usage.cost = 3.65e-06
        assert openrouter_call_cost(usage) == pytest.approx(3.65e-06)

    def test_zero_cost_is_reported_not_discarded(self) -> None:
        """A genuinely free call must read as $0.00, not as 'no cost reported'."""
        usage = MagicMock()
        usage.cost = 0.0
        assert openrouter_call_cost(usage) == 0.0

    def test_absent_cost_returns_none(self) -> None:
        class Usage:
            input_tokens = 10
            output_tokens = 5

        assert openrouter_call_cost(Usage()) is None

    def test_magicmock_usage_does_not_fake_a_cost(self) -> None:
        """A bare MagicMock auto-creates .cost; that must not read as a real cost.

        Without this guard every mocked Anthropic response in the suite would be
        treated as an OpenRouter response carrying a cost.
        """
        assert openrouter_call_cost(MagicMock()) is None

    def test_bool_is_not_a_cost(self) -> None:
        usage = MagicMock()
        usage.cost = True
        assert openrouter_call_cost(usage) is None


class TestFetchOpenRouterPricing:
    """No network: httpx.get is mocked in every case."""

    def _catalogue(self, entries: list[dict]) -> MagicMock:
        response = MagicMock()
        response.json.return_value = {"data": entries}
        response.raise_for_status.return_value = None
        return response

    def test_returns_rates_per_million_tokens(self) -> None:
        entry = {"id": "deepseek/deepseek-v3.2", "pricing": {"prompt": "0.000000269", "completion": "0.0000004"}}
        with patch("freqpred.llm.provider.httpx.get", return_value=self._catalogue([entry])):
            rates = fetch_openrouter_pricing("deepseek/deepseek-v3.2")

        assert rates is not None
        assert rates[0] == pytest.approx(0.269, rel=1e-3)
        assert rates[1] == pytest.approx(0.400, rel=1e-3)

    def test_unknown_slug_returns_none_not_a_guess(self) -> None:
        """A wrong number that looks authoritative is worse than no number."""
        entry = {"id": "other/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
        with patch("freqpred.llm.provider.httpx.get", return_value=self._catalogue([entry])):
            assert fetch_openrouter_pricing("deepseek/deepseek-v3.2") is None

    def test_network_failure_returns_none(self) -> None:
        with patch("freqpred.llm.provider.httpx.get", side_effect=httpx.ConnectError("down")):
            assert fetch_openrouter_pricing("deepseek/deepseek-v3.2") is None

    def test_unparseable_pricing_returns_none(self) -> None:
        entry = {"id": "weird/model", "pricing": {"prompt": "free"}}
        with patch("freqpred.llm.provider.httpx.get", return_value=self._catalogue([entry])):
            assert fetch_openrouter_pricing("weird/model") is None


class TestRegisterModelPricing:
    def test_registered_rates_replace_the_default_fallback(self) -> None:
        """The bug this exists to prevent: a cheap slug priced at Sonnet's rates."""
        slug = "test-vendor/test-model-pricing"
        default_cost = calculate_cost(slug, 1_000_000, 0)
        assert default_cost == pytest.approx(3.00)  # the Sonnet-rate fallback

        register_model_pricing(slug, 0.269, 0.400)
        assert calculate_cost(slug, 1_000_000, 0) == pytest.approx(0.269)
        assert calculate_cost(slug, 0, 1_000_000) == pytest.approx(0.400)
