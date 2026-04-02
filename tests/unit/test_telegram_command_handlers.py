"""Unit tests for T28 Telegram command handlers.

All DB sessions and external calls are mocked — no real DB or API requests.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.alerts.command_handlers import LogBuffer, _LogBufferHandler, register_system_commands
from freqpred.alerts.telegram_commands import TelegramCommandHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler_under_test(config=None, mode="paper", strategy_name="TestStrategy"):
    """Return a (TelegramCommandHandler, session_factory_mock, log_buffer) triple."""
    if config is None:
        config = MagicMock()
        config.risk.min_edge_floor = 0.10
        config.risk.max_position_pct = 0.05
        config.risk.max_daily_llm_spend_usd = 10.0
        config.risk.max_open_positions = 20

    session_factory = MagicMock()
    log_buffer = LogBuffer()

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(
        cmd_handler=cmd_handler,
        session_factory=session_factory,
        config=config,
        mode=mode,
        strategy_name=strategy_name,
        log_buffer=log_buffer,
    )
    return cmd_handler, session_factory, log_buffer


def _async_session_ctx(return_value=None):
    """Build a mock async session factory that returns a mock session."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    session_factory = MagicMock(return_value=ctx)
    return session_factory, session


# ---------------------------------------------------------------------------
# LogBuffer
# ---------------------------------------------------------------------------


def test_log_buffer_last_fewer_than_capacity():
    buf = LogBuffer(maxlen=100)
    for i in range(5):
        buf.append(f"line {i}")
    assert buf.last(10) == [f"line {i}" for i in range(5)]


def test_log_buffer_last_truncates_correctly():
    buf = LogBuffer(maxlen=100)
    for i in range(20):
        buf.append(f"line {i}")
    result = buf.last(5)
    assert result == [f"line {i}" for i in range(15, 20)]


def test_log_buffer_maxlen_evicts_oldest():
    buf = LogBuffer(maxlen=3)
    for i in range(5):
        buf.append(f"line {i}")
    assert buf.last(10) == ["line 2", "line 3", "line 4"]


def test_log_buffer_handler_writes_to_buffer():
    import logging

    buf = LogBuffer()
    handler = _LogBufferHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello world", args=(), exc_info=None,
    )
    handler.emit(record)
    assert buf.last(1) == ["hello world"]


# ---------------------------------------------------------------------------
# /start, /pause, /stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_sets_state_running():
    session_factory, session = _async_session_ctx()
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    with patch("freqpred.alerts.command_handlers.set_run_state", new_callable=AsyncMock) as mock_set:
        register_system_commands(cmd_handler, session_factory, MagicMock(
            risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                           max_daily_llm_spend_usd=10.0, max_open_positions=20)
        ), mode="paper", strategy_name="Test")
        reply = await cmd_handler._handlers["start"](42, [])

    assert "running" in reply
    mock_set.assert_awaited_once()
    assert mock_set.await_args.args[1] == "running"


@pytest.mark.asyncio
async def test_pause_sets_state_paused():
    session_factory, _ = _async_session_ctx()
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    with patch("freqpred.alerts.command_handlers.set_run_state", new_callable=AsyncMock) as mock_set:
        register_system_commands(cmd_handler, session_factory, MagicMock(
            risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                           max_daily_llm_spend_usd=10.0, max_open_positions=20)
        ), mode="paper", strategy_name="Test")
        reply = await cmd_handler._handlers["pause"](42, [])

    assert "paused" in reply
    assert mock_set.await_args.args[1] == "paused"


@pytest.mark.asyncio
async def test_stop_sets_state_stopped():
    session_factory, _ = _async_session_ctx()
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])

    with patch("freqpred.alerts.command_handlers.set_run_state", new_callable=AsyncMock) as mock_set:
        register_system_commands(cmd_handler, session_factory, MagicMock(
            risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                           max_daily_llm_spend_usd=10.0, max_open_positions=20)
        ), mode="paper", strategy_name="Test")
        reply = await cmd_handler._handlers["stop"](42, [])

    assert "stopped" in reply
    assert mock_set.await_args.args[1] == "stopped"


# ---------------------------------------------------------------------------
# /show_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_show_config_includes_key_fields():
    cmd_handler, session_factory, _ = _make_handler_under_test(
        mode="paper", strategy_name="ConservativeDefault"
    )
    reply = await cmd_handler._handlers["show_config"](42, [])
    assert "ConservativeDefault" in reply
    assert "paper" in reply
    assert "10.00%" in reply   # min_edge_floor 0.10
    assert "5.00%" in reply    # max_position_pct 0.05
    assert "$10.00" in reply   # llm budget


# ---------------------------------------------------------------------------
# /logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logs_returns_last_n_lines():
    cmd_handler, _, log_buffer = _make_handler_under_test()
    for i in range(30):
        log_buffer.append(f"log line {i}")

    reply = await cmd_handler._handlers["logs"](42, ["5"])
    assert "log line 29" in reply
    assert "log line 25" in reply
    # Should not contain lines outside the last 5
    assert "log line 24" not in reply


@pytest.mark.asyncio
async def test_logs_defaults_to_20():
    cmd_handler, _, log_buffer = _make_handler_under_test()
    for i in range(25):
        log_buffer.append(f"line {i}")

    reply = await cmd_handler._handlers["logs"](42, [])
    # Should contain the last 20
    assert "line 5" in reply  # 25-20=5, so line 5 is the earliest included
    assert "line 4" not in reply


@pytest.mark.asyncio
async def test_logs_invalid_n_returns_usage():
    cmd_handler, _, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["logs"](42, ["notanumber"])
    assert "Usage" in reply


@pytest.mark.asyncio
async def test_logs_no_buffer_returns_unavailable():
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    session_factory = MagicMock()
    register_system_commands(
        cmd_handler, session_factory,
        MagicMock(risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                                  max_daily_llm_spend_usd=10.0, max_open_positions=20)),
        mode="paper", strategy_name="Test",
        log_buffer=None,
    )
    reply = await cmd_handler._handlers["logs"](42, [])
    assert "not available" in reply


# ---------------------------------------------------------------------------
# /version
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_version_includes_version_string():
    cmd_handler, _, _ = _make_handler_under_test()
    with patch("freqpred.alerts.command_handlers.importlib.metadata.version", return_value="0.1.0"):
        with patch("freqpred.alerts.command_handlers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n")
            reply = await cmd_handler._handlers["version"](42, [])
    assert "0.1.0" in reply
    assert "abc1234" in reply


@pytest.mark.asyncio
async def test_version_handles_missing_package():
    import importlib.metadata as _meta
    cmd_handler, _, _ = _make_handler_under_test()
    with patch("freqpred.alerts.command_handlers.importlib.metadata.version",
               side_effect=_meta.PackageNotFoundError):
        with patch("freqpred.alerts.command_handlers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="abc\n")
            reply = await cmd_handler._handlers["version"](42, [])
    assert "unknown" in reply


# ---------------------------------------------------------------------------
# /count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_returns_open_and_max():
    from sqlalchemy import func
    from freqpred.markets.models import PositionRow

    session_factory, session = _async_session_ctx()

    # Mock the scalar result
    scalar_result = MagicMock()
    scalar_result.scalar_one = MagicMock(return_value=3)
    session.execute = AsyncMock(return_value=scalar_result)

    config = MagicMock()
    config.risk.max_open_positions = 10
    config.risk.min_edge_floor = 0.1
    config.risk.max_position_pct = 0.05
    config.risk.max_daily_llm_spend_usd = 10.0

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(cmd_handler, session_factory, config, mode="paper", strategy_name="Test")

    reply = await cmd_handler._handlers["count"](42, [])
    assert "3" in reply
    assert "10" in reply
    assert "Open" in reply


# ---------------------------------------------------------------------------
# /status — no positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_no_open_positions():
    session_factory, session = _async_session_ctx()

    # execute() calls in order:
    # 1. get_run_state → scalar_one_or_none() → None (defaults to "running")
    # 2. get_drawdown_window → scalar_one_or_none() → None (no reset)
    # 3. get_net_bankroll → scalar_one() → 0.0
    # 4. open positions query → all() → []
    run_state_mock = MagicMock()
    run_state_mock.scalar_one_or_none.return_value = None
    reset_window_mock = MagicMock()
    reset_window_mock.scalar_one_or_none.return_value = None
    net_bankroll_mock = MagicMock()
    net_bankroll_mock.scalar_one.return_value = 0.0
    positions_mock = MagicMock()
    positions_mock.all.return_value = []
    session.execute = AsyncMock(side_effect=[run_state_mock, reset_window_mock, net_bankroll_mock, positions_mock])

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(cmd_handler, session_factory, MagicMock(
        trading=MagicMock(bankroll_usd=1000.0),
        risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                       max_daily_llm_spend_usd=10.0, max_open_positions=20)
    ), mode="paper", strategy_name="Test")

    reply = await cmd_handler._handlers["status"](42, [])
    assert "No open positions" in reply
    assert "drawdown=" in reply


# ---------------------------------------------------------------------------
# /status <position_id> — not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_single_position_not_found():
    session_factory, session = _async_session_ctx()
    result_mock = MagicMock()
    result_mock.one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result_mock)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(cmd_handler, session_factory, MagicMock(
        risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                       max_daily_llm_spend_usd=10.0, max_open_positions=20)
    ), mode="paper", strategy_name="Test")

    pos_id = str(uuid.uuid4())
    reply = await cmd_handler._handlers["status"](42, [pos_id])
    assert "not found" in reply


@pytest.mark.asyncio
async def test_status_invalid_position_id():
    cmd_handler, _, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["status"](42, ["not-a-uuid"])
    assert "Invalid position ID" in reply


# ---------------------------------------------------------------------------
# /trades — no resolved positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trades_no_resolved_positions():
    session_factory, session = _async_session_ctx()
    result_mock = MagicMock()
    result_mock.all = MagicMock(return_value=[])
    session.execute = AsyncMock(return_value=result_mock)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(cmd_handler, session_factory, MagicMock(
        risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                       max_daily_llm_spend_usd=10.0, max_open_positions=20)
    ), mode="paper", strategy_name="Test")

    reply = await cmd_handler._handlers["trades"](42, [])
    assert "No resolved positions" in reply


@pytest.mark.asyncio
async def test_trades_invalid_n():
    cmd_handler, _, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["trades"](42, ["abc"])
    assert "Usage" in reply


# ---------------------------------------------------------------------------
# /signals — no signals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signals_no_signals():
    session_factory, session = _async_session_ctx()
    result_mock = MagicMock()
    result_mock.all = MagicMock(return_value=[])
    session.execute = AsyncMock(return_value=result_mock)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(cmd_handler, session_factory, MagicMock(
        risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                       max_daily_llm_spend_usd=10.0, max_open_positions=20)
    ), mode="paper", strategy_name="Test")

    reply = await cmd_handler._handlers["signals"](42, [])
    assert "No signals" in reply


@pytest.mark.asyncio
async def test_signals_invalid_n():
    cmd_handler, _, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["signals"](42, ["xyz"])
    assert "Usage" in reply


# ---------------------------------------------------------------------------
# All T28 commands are registered
# ---------------------------------------------------------------------------


def test_all_t28_commands_registered():
    cmd_handler, _, _ = _make_handler_under_test()
    expected = {"start", "pause", "stop", "show_config", "logs", "version",
                "status", "count", "trades", "signals"}
    assert expected.issubset(set(cmd_handler._handlers.keys()))
