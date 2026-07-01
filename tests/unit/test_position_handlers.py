"""Unit tests for T30 Telegram position management command handlers.

All DB sessions and external calls are mocked — no real DB or API requests.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.alerts.position_handlers import register_position_commands
from freqpred.alerts.telegram_commands import TelegramCommandHandler
from freqpred.markets.models import Position
from freqpred.trading.order_manager import PositionNotFoundError, PositionNotOpenError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(mode: str = "paper") -> tuple[TelegramCommandHandler, MagicMock]:
    """Return (cmd_handler, session_factory_mock) with position commands registered."""
    config = MagicMock()
    session_factory = MagicMock()

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_position_commands(
        cmd_handler=cmd_handler,
        session_factory=session_factory,
        config=config,
        mode=mode,
    )
    return cmd_handler, session_factory


def _async_session_ctx():
    """Build a mock async session factory that returns a mock session."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session_factory = MagicMock(return_value=ctx)
    return session_factory, session


def _make_position(pos_id: str | None = None, status: str = "open") -> Position:
    return Position(
        id=pos_id or str(uuid.uuid4()),
        market_id="MKT-1",
        signal_id=str(uuid.uuid4()),
        strategy_name="Test",
        strategy_version="1.0",
        signal_confidence=0.7,
        signal_edge=0.05,
        signal_estimated_prob=0.6,
        direction="YES",
        contracts=10,
        entry_price=0.50,
        entry_time=datetime.now(tz=UTC),
        mode="paper",
        status=status,
        pnl=0.10,
        pnl_pct=0.02,
        exit_price=0.51 if status == "closed" else None,
    )


# ---------------------------------------------------------------------------
# /forceexit — paper mode, single position (happy path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forceexit_paper_single_position():
    """Paper mode: /forceexit <id> delegates to order_manager.force_exit() immediately."""
    pos_id = str(uuid.uuid4())
    closed = _make_position(pos_id, status="closed")
    closed.exit_price = 0.55
    closed.exit_reason = "force_exit:manual"

    mock_om = AsyncMock()
    mock_om.force_exit = AsyncMock(return_value=closed)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_position_commands(cmd_handler, MagicMock(), MagicMock(), "paper", order_manager=mock_om)

    handler = cmd_handler._handlers["forceexit"]
    reply = await handler(chat_id=1, args=[pos_id])

    assert reply is not None
    assert "closed" in reply.lower()
    assert pos_id in reply
    mock_om.force_exit.assert_awaited_once_with(pos_id, exit_reason="force_exit:manual")


@pytest.mark.asyncio
async def test_forceexit_paper_invalid_id():
    """Non-UUID string treated as market ID; if no open position found, returns informative message."""
    session_factory, session = _async_session_ctx()
    # DB query for market_id "not-a-uuid" returns no open position
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)

    mock_om = AsyncMock()
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_position_commands(cmd_handler, session_factory, MagicMock(), "paper", order_manager=mock_om)

    handler = cmd_handler._handlers["forceexit"]
    reply = await handler(chat_id=1, args=["not-a-uuid"])

    assert reply is not None
    assert "no open position" in reply.lower()
    mock_om.force_exit.assert_not_awaited()


@pytest.mark.asyncio
async def test_forceexit_paper_position_not_found():
    """PositionNotFoundError from order_manager returns an informative message."""
    pos_id = str(uuid.uuid4())

    mock_om = AsyncMock()
    mock_om.force_exit = AsyncMock(
        side_effect=PositionNotFoundError(f"Position {pos_id!r} not found for mode='paper'")
    )

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_position_commands(cmd_handler, MagicMock(), MagicMock(), "paper", order_manager=mock_om)

    handler = cmd_handler._handlers["forceexit"]
    reply = await handler(chat_id=1, args=[pos_id])

    assert reply is not None
    assert "not found" in reply.lower()


@pytest.mark.asyncio
async def test_forceexit_paper_already_closed():
    """PositionNotOpenError from order_manager returns an informative message, no further action."""
    pos_id = str(uuid.uuid4())

    mock_om = AsyncMock()
    mock_om.force_exit = AsyncMock(
        side_effect=PositionNotOpenError(f"Position {pos_id!r} is not open (status='closed')")
    )

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_position_commands(cmd_handler, MagicMock(), MagicMock(), "paper", order_manager=mock_om)

    handler = cmd_handler._handlers["forceexit"]
    reply = await handler(chat_id=1, args=[pos_id])

    assert reply is not None
    assert "closed" in reply.lower() or "not open" in reply.lower()


# ---------------------------------------------------------------------------
# /forceexit — missing args
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forceexit_no_args():
    """No arguments returns usage hint."""
    cmd_handler, _ = _make_handler(mode="paper")
    handler = cmd_handler._handlers["forceexit"]
    reply = await handler(chat_id=1, args=[])

    assert reply is not None
    assert "usage" in reply.lower() or "Usage" in reply


# ---------------------------------------------------------------------------
# /forceexit all — sends confirmation prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forceexit_all_sends_confirmation():
    """/forceexit all sends a confirmation inline keyboard and returns None."""
    cmd_handler, session_factory = _make_handler(mode="paper")

    with patch.object(cmd_handler, "send_inline_keyboard", new_callable=AsyncMock) as mock_kbd:
        mock_kbd.return_value = 42

        handler = cmd_handler._handlers["forceexit"]
        # Use asyncio.wait_for to avoid blocking on the timeout background task.
        reply = await asyncio.wait_for(handler(chat_id=1, args=["all"]), timeout=5)

    assert reply is None
    mock_kbd.assert_awaited_once()
    # Verify the keyboard has Confirm and Cancel buttons.
    call_args = mock_kbd.call_args
    buttons = call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs.get("buttons", [])
    flat_buttons = [btn for row in buttons for btn in row]
    callback_data_values = [b["callback_data"] for b in flat_buttons]
    assert any("confirm" in d for d in callback_data_values)
    assert any("cancel" in d for d in callback_data_values)


# ---------------------------------------------------------------------------
# /forceexit — live mode sends confirmation prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forceexit_live_sends_confirmation():
    """Live mode: /forceexit <id> sends confirmation prompt and returns None."""
    pos_id = str(uuid.uuid4())
    cmd_handler, session_factory = _make_handler(mode="live")

    with patch.object(cmd_handler, "send_inline_keyboard", new_callable=AsyncMock) as mock_kbd:
        mock_kbd.return_value = 42
        handler = cmd_handler._handlers["forceexit"]
        reply = await asyncio.wait_for(handler(chat_id=1, args=[pos_id]), timeout=5)

    assert reply is None
    mock_kbd.assert_awaited_once()


# ---------------------------------------------------------------------------
# /fx — alias for /forceexit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fx_alias():
    """/fx is registered and behaves identically to /forceexit."""
    cmd_handler, session_factory = _make_handler(mode="paper")

    # Both handlers should be registered.
    assert "fx" in cmd_handler._handlers
    assert "forceexit" in cmd_handler._handlers

    # Both should return usage hint when called with no args.
    fx_reply = await cmd_handler._handlers["fx"](chat_id=1, args=[])
    fe_reply = await cmd_handler._handlers["forceexit"](chat_id=1, args=[])
    assert fx_reply == fe_reply


# ---------------------------------------------------------------------------
# /delete — live mode rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_rejected_in_live_mode():
    """/delete in live mode returns an error immediately without any DB access."""
    pos_id = str(uuid.uuid4())
    cmd_handler, session_factory = _make_handler(mode="live")

    handler = cmd_handler._handlers["delete"]
    reply = await handler(chat_id=1, args=[pos_id])

    assert reply is not None
    assert "not available" in reply.lower() or "live mode" in reply.lower()
    session_factory.assert_not_called()


# ---------------------------------------------------------------------------
# /delete — paper mode sends confirmation prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_paper_sends_confirmation():
    """/delete in paper mode sends a confirmation inline keyboard and returns None."""
    pos_id = str(uuid.uuid4())
    cmd_handler, session_factory = _make_handler(mode="paper")

    with patch.object(cmd_handler, "send_inline_keyboard", new_callable=AsyncMock) as mock_kbd:
        mock_kbd.return_value = 42
        handler = cmd_handler._handlers["delete"]
        reply = await asyncio.wait_for(handler(chat_id=1, args=[pos_id]), timeout=5)

    assert reply is None
    mock_kbd.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_invalid_id():
    """Invalid UUID returns an error message immediately without sending confirmation."""
    cmd_handler, session_factory = _make_handler(mode="paper")

    with patch.object(cmd_handler, "send_inline_keyboard", new_callable=AsyncMock) as mock_kbd:
        handler = cmd_handler._handlers["delete"]
        reply = await handler(chat_id=1, args=["bad-uuid"])

    assert reply is not None
    assert "invalid" in reply.lower() or "Invalid" in reply
    mock_kbd.assert_not_awaited()


# ---------------------------------------------------------------------------
# Confirmation timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmation_timeout_sends_notice():
    """After 30 s with no button press the bot sends a timeout notice."""
    import freqpred.alerts.position_handlers as ph

    original_timeout = ph._CONFIRM_TIMEOUT_SECS
    ph._CONFIRM_TIMEOUT_SECS = 0  # Speed up test by setting timeout to 0 s.
    try:
        pos_id = str(uuid.uuid4())
        cmd_handler, session_factory = _make_handler(mode="live")

        sent_replies: list[str] = []

        async def fake_send_reply(chat_id: int, text: str) -> None:
            sent_replies.append(text)

        async def fake_send_inline_keyboard(chat_id, text, buttons):
            return 1

        cmd_handler._send_reply = fake_send_reply  # type: ignore[method-assign]
        cmd_handler.send_inline_keyboard = fake_send_inline_keyboard  # type: ignore[method-assign]

        handler = cmd_handler._handlers["forceexit"]
        await handler(chat_id=1, args=[pos_id])

        # Allow the timeout background task to fire.
        await asyncio.sleep(0.05)

        timeout_notices = [r for r in sent_replies if "timed out" in r.lower() or "cancelled" in r.lower()]
        assert timeout_notices, f"Expected timeout notice, got: {sent_replies}"
    finally:
        ph._CONFIRM_TIMEOUT_SECS = original_timeout


# ---------------------------------------------------------------------------
# TelegramCommandHandler — callback query dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_registered_and_dispatched():
    """register_callback / _handle_callback_query dispatches to the correct handler."""
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    received: list[tuple] = []

    async def my_callback(chat_id: int, data: str, cb_id: str) -> str:
        received.append((chat_id, data, cb_id))
        return "done"

    cmd_handler.register_callback("confirm:abc", my_callback)

    callback_query = {
        "id": "cq-001",
        "from": {"id": 999, "username": "alice"},
        "data": "confirm:abc",
        "message": {"chat": {"id": 42}},
    }

    with patch.object(cmd_handler, "_answer_callback_query", new_callable=AsyncMock):
        with patch.object(cmd_handler, "_send_reply", new_callable=AsyncMock) as mock_reply:
            await cmd_handler._handle_callback_query(callback_query)

    assert received == [(42, "confirm:abc", "cq-001")]
    mock_reply.assert_awaited_once_with(42, "done")


@pytest.mark.asyncio
async def test_callback_unknown_data_answers_with_notice():
    """Unknown callback_data answers the query with an expired/unknown notice."""
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    callback_query = {
        "id": "cq-002",
        "from": {"id": 999, "username": "alice"},
        "data": "confirm:nonexistent",
        "message": {"chat": {"id": 1}},
    }

    with patch.object(cmd_handler, "_answer_callback_query", new_callable=AsyncMock) as mock_answer:
        with patch.object(cmd_handler, "_send_reply", new_callable=AsyncMock):
            await cmd_handler._handle_callback_query(callback_query)

    mock_answer.assert_awaited_once()
    answered_text = mock_answer.call_args.args[1] if len(mock_answer.call_args.args) > 1 else ""
    assert "expired" in answered_text.lower() or "unknown" in answered_text.lower()


@pytest.mark.asyncio
async def test_unregister_callback():
    """unregister_callback removes the handler so subsequent dispatches hit the unknown path."""
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    async def my_callback(chat_id: int, data: str, cb_id: str) -> str:
        return "ok"

    cmd_handler.register_callback("confirm:xyz", my_callback)
    cmd_handler.unregister_callback("confirm:xyz")

    assert "confirm:xyz" not in cmd_handler._callback_handlers
