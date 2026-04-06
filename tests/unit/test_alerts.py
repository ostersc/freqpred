"""Unit tests for freqpred/alerts/*.

All HTTP calls are mocked — no real network requests made.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.alerts.discord import DiscordSender
from freqpred.alerts.dispatcher import AlertDispatcher
from freqpred.alerts.telegram import TelegramSender
from freqpred.markets.models import Market, Position
from freqpred.signal.models import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market(**overrides: object) -> Market:
    defaults = dict(
        id="MKTTEST-1",
        platform="kalshi",
        question="Will the Fed cut rates in May 2026?",
        category="economics",
        close_time=datetime(2026, 5, 31, tzinfo=timezone.utc),
        yes_bid=0.42,
        yes_ask=0.44,
        mid_price=0.43,
        volume_24h=1000.0,
        open_interest=500.0,
        last_fetched_at=datetime(2026, 3, 18, tzinfo=timezone.utc),
        price_updated_at=datetime(2026, 3, 18, tzinfo=timezone.utc),
        metadata_fetched_at=None,
        current_signal_id=None,
        metadata={},
    )
    defaults.update(overrides)
    return Market(**defaults)


def _make_signal(**overrides: object) -> Signal:
    defaults = dict(
        id="sig-abc",
        market_id="MKTTEST-1",
        estimated_probability=0.60,
        confidence=0.75,
        edge=0.17,
        market_mid_at_signal=0.43,
        direction="YES",
        reasoning="Strong signals point to a rate cut.",
        sources=[],
        retrieval_hash="abc123",
        model_used="claude-sonnet-4-6",
        prompt_version="signal-v1",
        trigger="scheduled",
        created_at=datetime(2026, 3, 18, tzinfo=timezone.utc),
        raw_context="",
    )
    defaults.update(overrides)
    return Signal(**defaults)


def _make_position(**overrides: object) -> Position:
    defaults = dict(
        id="pos-xyz",
        market_id="MKTTEST-1",
        signal_id="sig-abc",
        strategy_name="ConservativeDefault",
        strategy_version="0.1",
        signal_confidence=0.75,
        signal_edge=0.17,
        signal_estimated_prob=0.60,
        direction="YES",
        contracts=5,
        entry_price=0.43,
        entry_time=datetime(2026, 3, 18, tzinfo=timezone.utc),
        mode="paper",
        status="open",
    )
    defaults.update(overrides)
    return Position(**defaults)


# ---------------------------------------------------------------------------
# test_telegram_send_posts_correct_payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telegram_send_posts_correct_payload() -> None:
    """TelegramSender posts to the correct URL with token and message."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    sender = TelegramSender(bot_token="TOKEN123", chat_id="CHAT456")

    with patch("freqpred.alerts.telegram.httpx.AsyncClient", return_value=mock_client):
        await sender.send("hello telegram")

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    url = call_args.args[0]
    payload = call_args.kwargs["json"]

    assert "TOKEN123" in url
    assert "sendMessage" in url
    assert payload["chat_id"] == "CHAT456"
    assert payload["text"] == "hello telegram"


# ---------------------------------------------------------------------------
# test_discord_send_posts_correct_payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discord_send_posts_correct_payload() -> None:
    """DiscordSender posts to the webhook URL with content key."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    webhook = "https://discord.com/api/webhooks/123/abc"
    sender = DiscordSender(webhook_url=webhook)

    with patch("freqpred.alerts.discord.httpx.AsyncClient", return_value=mock_client):
        await sender.send("hello discord")

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args.args[0] == webhook
    assert call_args.kwargs["json"] == {"content": "hello discord"}


# ---------------------------------------------------------------------------
# test_dispatcher_fans_out_to_all_senders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_fans_out_to_all_senders() -> None:
    """AlertDispatcher calls send() on every configured sender."""
    s1 = AsyncMock()
    s2 = AsyncMock()
    dispatcher = AlertDispatcher([s1, s2])

    await dispatcher.send("broadcast")

    s1.send.assert_called_once_with("broadcast")
    s2.send.assert_called_once_with("broadcast")


# ---------------------------------------------------------------------------
# test_dispatcher_continues_if_one_sender_fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_continues_if_one_sender_fails() -> None:
    """If one sender raises, the other is still called and no exception propagates."""
    s1 = AsyncMock()
    s1.send = AsyncMock(side_effect=RuntimeError("network timeout"))
    s2 = AsyncMock()

    dispatcher = AlertDispatcher([s1, s2])

    # Must not raise
    await dispatcher.send("important message")

    s2.send.assert_called_once_with("important message")


# ---------------------------------------------------------------------------
# test_signal_alert_format_contains_key_fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_alert_format_contains_key_fields() -> None:
    """signal_alert message contains probability, market mid, and edge."""
    captured: list[str] = []

    class CaptureSender:
        async def send(self, message: str) -> None:
            captured.append(message)

    dispatcher = AlertDispatcher([CaptureSender()])
    market = _make_market(mid_price=0.43)
    signal = _make_signal(estimated_probability=0.60, edge=0.17, market_mid_at_signal=0.43)

    await dispatcher.signal_alert(signal, market)

    assert len(captured) == 1
    msg = captured[0]
    assert "60.0" in msg      # prob pct
    assert "43.0" in msg      # market mid pct
    assert "+17.0" in msg     # edge pct


# ---------------------------------------------------------------------------
# test_resolution_alert_win_vs_loss_prefix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolution_alert_win_vs_loss_prefix() -> None:
    """WIN prefix when pnl > 0, LOSS prefix when pnl < 0."""
    captured: list[str] = []

    class CaptureSender:
        async def send(self, message: str) -> None:
            captured.append(message)

    dispatcher = AlertDispatcher([CaptureSender()])
    market = _make_market()

    win_pos = _make_position(
        pnl=1.25,
        exit_price=1.0,
        exit_time=datetime(2026, 3, 18, tzinfo=timezone.utc),
        resolution=1,
        status="closed",
    )
    await dispatcher.resolution_alert(win_pos, market)

    loss_pos = _make_position(
        pnl=-0.50,
        exit_price=0.0,
        exit_time=datetime(2026, 3, 18, tzinfo=timezone.utc),
        resolution=0,
        status="closed",
    )
    await dispatcher.resolution_alert(loss_pos, market)

    assert captured[0].startswith("WIN")
    assert captured[1].startswith("LOSS")


# ---------------------------------------------------------------------------
# Circuit breaker alert format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_alert_format() -> None:
    """circuit_breaker_alert() produces the standard structured format."""
    captured: list[str] = []

    class CaptureSender:
        async def send(self, message: str) -> None:
            captured.append(message)

    dispatcher = AlertDispatcher([CaptureSender()])
    await dispatcher.circuit_breaker_alert("daily_loss", "Daily loss 16.2% exceeded 15% limit")

    assert len(captured) == 1
    msg = captured[0]
    assert "🚨" in msg
    assert "CIRCUIT BREAKER TRIPPED" in msg
    assert "Type: daily_loss" in msg
    assert "Reason: Daily loss 16.2% exceeded 15% limit" in msg
    assert "Action required:" in msg
    assert "Resume:" in msg
