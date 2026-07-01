"""Unit tests for TelegramCommandHandler.

All HTTP calls are mocked — no real Telegram API requests made.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.alerts.telegram_commands import TelegramCommandHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(
    update_id: int,
    text: str,
    user_id: int = 111,
    username: str = "alice",
    chat_id: int = 999,
) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": user_id, "username": username},
            "chat": {"id": chat_id},
            "text": text,
        },
    }


def _make_handler(reply: str = "ok") -> AsyncMock:
    handler = AsyncMock(return_value=reply)
    return handler


# ---------------------------------------------------------------------------
# test_disabled_when_no_token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_when_no_token() -> None:
    """run() returns immediately without making any API calls when token is empty."""
    handler = TelegramCommandHandler(bot_token="", authorized_users=["alice"])

    with patch("freqpred.alerts.telegram_commands.httpx.AsyncClient") as mock_cls:
        await handler.run()

    mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# test_auth_authorized_by_username
# ---------------------------------------------------------------------------


def test_auth_authorized_by_username() -> None:
    """_is_authorized returns True when username matches authorized_users."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice", "bob"])
    assert h._is_authorized({"id": 111, "username": "alice"}) is True


def test_auth_authorized_by_user_id() -> None:
    """_is_authorized returns True when numeric user ID (as string) matches."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["123456"])
    assert h._is_authorized({"id": 123456, "username": "stranger"}) is True


def test_auth_unauthorized_user() -> None:
    """_is_authorized returns False for unknown username and ID."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    assert h._is_authorized({"id": 999, "username": "eve"}) is False


def test_auth_empty_authorized_users() -> None:
    """_is_authorized returns False when authorized_users is empty."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=[])
    assert h._is_authorized({"id": 111, "username": "alice"}) is False


# ---------------------------------------------------------------------------
# test_command_parsing
# ---------------------------------------------------------------------------


def test_parse_command_no_args() -> None:
    cmd, args = TelegramCommandHandler._parse_command("/status")
    assert cmd == "status"
    assert args == []


def test_parse_command_with_args() -> None:
    cmd, args = TelegramCommandHandler._parse_command("/analyze MKTABC-1 verbose")
    assert cmd == "analyze"
    assert args == ["MKTABC-1", "verbose"]


def test_parse_command_strips_bot_mention() -> None:
    cmd, args = TelegramCommandHandler._parse_command("/help@MyBot")
    assert cmd == "help"
    assert args == []


# ---------------------------------------------------------------------------
# test_polling_offset_advancement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_polling_offset_advancement() -> None:
    """_get_updates advances _offset to last_update_id + 1."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    assert h._offset == 0

    fake_updates = [{"update_id": 10}, {"update_id": 11}, {"update_id": 12}]
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={"result": fake_updates})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=fake_response)

    with patch("freqpred.alerts.telegram_commands.httpx.AsyncClient", return_value=mock_client):
        result = await h._get_updates()

    assert result == fake_updates
    assert h._offset == 13  # 12 + 1


@pytest.mark.asyncio
async def test_polling_offset_unchanged_on_empty_result() -> None:
    """_offset stays at 0 when getUpdates returns no updates."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={"result": []})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=fake_response)

    with patch("freqpred.alerts.telegram_commands.httpx.AsyncClient", return_value=mock_client):
        result = await h._get_updates()

    assert result == []
    assert h._offset == 0


@pytest.mark.asyncio
async def test_polling_offset_sent_after_first_batch() -> None:
    """Second _get_updates call passes the advanced offset as a param."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={"result": [{"update_id": 5}]})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=fake_response)

    with patch("freqpred.alerts.telegram_commands.httpx.AsyncClient", return_value=mock_client):
        await h._get_updates()
        await h._get_updates()

    assert mock_client.get.call_count == 2
    second_call_params = mock_client.get.call_args_list[1].kwargs["params"]
    assert second_call_params["offset"] == 6


# ---------------------------------------------------------------------------
# test_authorized_command_dispatched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorized_command_dispatched() -> None:
    """An authorized user's command is dispatched and reply is sent."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    cmd_handler = _make_handler("all good")
    h.register("status", cmd_handler)

    update = _make_update(update_id=1, text="/status", username="alice", chat_id=42)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("freqpred.alerts.telegram_commands.httpx.AsyncClient", return_value=mock_client):
        await h._handle_update(update)

    cmd_handler.assert_awaited_once_with(42, [])
    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["chat_id"] == 42
    assert payload["text"] == "all good"


# ---------------------------------------------------------------------------
# test_unauthorized_command_silently_dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthorized_command_silently_dropped() -> None:
    """An unauthorized user's command is silently dropped — no reply sent."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    cmd_handler = _make_handler("secret data")
    h.register("secret", cmd_handler)

    update = _make_update(update_id=1, text="/secret", username="eve", user_id=999)

    with patch("freqpred.alerts.telegram_commands.httpx.AsyncClient") as mock_cls:
        await h._handle_update(update)

    cmd_handler.assert_not_awaited()
    mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# test_help_command_lists_registered_commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_help_command_lists_registered_commands() -> None:
    """/help replies with all registered commands."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    h.register("status", _make_handler())
    h.register("pause", _make_handler())

    update = _make_update(update_id=1, text="/help", username="alice", chat_id=77)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("freqpred.alerts.telegram_commands.httpx.AsyncClient", return_value=mock_client):
        await h._handle_update(update)

    payload = mock_client.post.call_args.kwargs["json"]
    reply_text = payload["text"]
    assert "/help" in reply_text
    assert "/status" in reply_text
    assert "/pause" in reply_text


# ---------------------------------------------------------------------------
# test_unknown_command_returns_error_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_command_returns_error_message() -> None:
    """An unknown command sends a helpful error reply."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    update = _make_update(update_id=1, text="/unknown", username="alice", chat_id=55)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("freqpred.alerts.telegram_commands.httpx.AsyncClient", return_value=mock_client):
        await h._handle_update(update)

    payload = mock_client.post.call_args.kwargs["json"]
    assert "Unknown command" in payload["text"]
    assert "/help" in payload["text"]


# ---------------------------------------------------------------------------
# test_run_loop_cancels_cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_loop_cancels_cleanly() -> None:
    """run() can be cancelled without raising anything other than CancelledError."""
    h = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    call_count = 0

    async def fake_get_updates() -> list:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise asyncio.CancelledError()
        return []

    h._get_updates = fake_get_updates  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await h.run()
