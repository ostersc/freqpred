"""Unit tests for T28 Telegram command handlers.

All DB sessions and external calls are mocked — no real DB or API requests.
"""
from __future__ import annotations

import uuid
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
        buf.append("test.logger", f"line {i}")
    assert buf.last(10) == [f"line {i}" for i in range(5)]


def test_log_buffer_last_truncates_correctly():
    buf = LogBuffer(maxlen=100)
    for i in range(20):
        buf.append("test.logger", f"line {i}")
    result = buf.last(5)
    assert result == [f"line {i}" for i in range(15, 20)]


def test_log_buffer_maxlen_evicts_oldest():
    buf = LogBuffer(maxlen=3)
    for i in range(5):
        buf.append("test.logger", f"line {i}")
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
        log_buffer.append("freqpred.test", f"log line {i}")

    reply = await cmd_handler._handlers["logs"](42, ["5"])
    assert "log line 29" in reply
    assert "log line 25" in reply
    # Should not contain lines outside the last 5
    assert "log line 24" not in reply


@pytest.mark.asyncio
async def test_logs_defaults_to_20():
    cmd_handler, _, log_buffer = _make_handler_under_test()
    for i in range(25):
        log_buffer.append("freqpred.test", f"line {i}")

    reply = await cmd_handler._handlers["logs"](42, [])
    # Should contain the last 20
    assert "line 5" in reply  # 25-20=5, so line 5 is the earliest included
    assert "line 4" not in reply


@pytest.mark.asyncio
async def test_logs_filter_returns_matching_lines():
    cmd_handler, _, log_buffer = _make_handler_under_test()
    log_buffer.append("freqpred.ingestion.scheduler", "scheduler line A")
    log_buffer.append("freqpred.signal.pipeline", "pipeline line B")
    log_buffer.append("freqpred.ingestion.scheduler", "scheduler line C")
    reply = await cmd_handler._handlers["logs"](42, ["scheduler"])
    assert "scheduler line A" in reply
    assert "scheduler line C" in reply
    assert "pipeline line B" not in reply


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
# Formatting helpers
# ---------------------------------------------------------------------------


def test_truncate_cuts_at_word_boundary():
    from freqpred.alerts.command_handlers import _truncate

    text = "Will Trump say the word tariff before the end of April"
    result = _truncate(text, 30)
    assert result.endswith("…")
    assert len(result) <= 31
    # No mid-word cut: everything before the ellipsis is whole words
    assert result[:-1].rstrip() in text


def test_truncate_short_text_unchanged():
    from freqpred.alerts.command_handlers import _truncate

    assert _truncate("short", 30) == "short"


def test_fmt_price_whole_and_fractional_cents():
    from freqpred.alerts.command_handlers import _fmt_price

    assert _fmt_price(0.43) == "43¢"
    assert _fmt_price(0.435) == "43.5¢"
    assert _fmt_price(1.0) == "100¢"


def test_fmt_usd_signed():
    from freqpred.alerts.command_handlers import _fmt_usd

    assert _fmt_usd(1.2) == "+$1.20"
    assert _fmt_usd(-0.35) == "-$0.35"
    assert _fmt_usd(0.0) == "+$0.00"


def test_fmt_age_secs_ranges():
    from freqpred.alerts.command_handlers import _fmt_age_secs

    assert _fmt_age_secs(12) == "12s"
    assert _fmt_age_secs(5 * 60) == "5m"
    assert _fmt_age_secs(2 * 3600 + 15 * 60) == "2h 15m"
    assert _fmt_age_secs(3 * 86400 + 4 * 3600) == "3d 4h"


def test_ago_with_injected_clock():
    from datetime import UTC, datetime

    from freqpred.alerts.command_handlers import _ago

    now = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
    then = datetime(2026, 1, 1, 9, 45, 0, tzinfo=UTC)
    assert _ago(then, now=now) == "1d 2h"
    # Naive datetimes are treated as UTC
    assert _ago(then.replace(tzinfo=None), now=now) == "1d 2h"


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("YES", 10 * (0.60 - 0.40)),          # bought YES at 40¢, mid now 60¢
        ("NO", 10 * ((1.0 - 0.60) - 0.40)),   # bought NO at 40¢, NO now worth 40¢
    ],
)
def test_unrealized_pnl_both_directions(direction, expected):
    from freqpred.alerts.command_handlers import _unrealized_pnl

    assert _unrealized_pnl(direction, 10, 0.40, 0.60) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# /status — no positions
# ---------------------------------------------------------------------------


def _status_config():
    return MagicMock(
        trading=MagicMock(bankroll_usd=1000.0),
        risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                       max_daily_llm_spend_usd=10.0, max_open_positions=20),
    )


def _mock_status_list_session(session, rows):
    """Wire session.execute side effects for the /status list query chain."""
    # execute() calls in order:
    # 1. get_run_state → scalar_one_or_none() → None (defaults to "running")
    # 2. get_drawdown_window → scalar_one_or_none() → None (no reset)
    # 3. get_net_bankroll → scalar_one() → 0.0
    # 4. open positions query → all() → rows
    run_state_mock = MagicMock()
    run_state_mock.scalar_one_or_none.return_value = None
    reset_window_mock = MagicMock()
    reset_window_mock.scalar_one_or_none.return_value = None
    net_bankroll_mock = MagicMock()
    net_bankroll_mock.scalar_one.return_value = 0.0
    positions_mock = MagicMock()
    positions_mock.all.return_value = rows
    session.execute = AsyncMock(
        side_effect=[run_state_mock, reset_window_mock, net_bankroll_mock, positions_mock]
    )


@pytest.mark.asyncio
async def test_status_no_open_positions():
    session_factory, session = _async_session_ctx()
    _mock_status_list_session(session, [])

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(
        cmd_handler, session_factory, _status_config(), mode="paper", strategy_name="Test"
    )

    reply = await cmd_handler._handlers["status"](42, [])
    assert "no open positions" in reply
    assert "0/20" in reply
    assert "Drawdown" in reply
    assert "RUNNING" in reply


def _make_open_position(direction, contracts, entry_price, market_id="KXTEST-26-A"):
    pos = MagicMock()
    pos.direction = direction
    pos.contracts = contracts
    pos.entry_price = entry_price
    pos.market_id = market_id
    pos.entry_time = None
    pos.status = "open"
    pos.signal_estimated_prob = 0.7
    pos.signal_edge = 0.1
    pos.signal_confidence = 0.8
    pos.mae = None
    pos.mfe = None
    return pos


@pytest.mark.asyncio
async def test_status_list_shows_yes_and_no_unrealized_pnl():
    session_factory, session = _async_session_ctx()
    # YES bought at 40¢, mid 60¢ → +$2.00 on 10 contracts
    yes_pos = _make_open_position("YES", 10, 0.40, market_id="KXYES-26")
    # NO bought at 40¢, mid 60¢ → NO now worth 40¢ → $0.00 P&L... use mid 0.50 → +$1.00
    no_pos = _make_open_position("NO", 10, 0.40, market_id="KXNO-26")
    rows = [
        (yes_pos, "Yes market question?", 0.60),
        (no_pos, "No market question?", 0.50),
    ]
    _mock_status_list_session(session, rows)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(
        cmd_handler, session_factory, _status_config(), mode="paper", strategy_name="Test"
    )

    reply = await cmd_handler._handlers["status"](42, [])
    assert "KXYES-26" in reply
    assert "KXNO-26" in reply
    assert "+$2.00" in reply           # YES leg unrealized
    assert "+$1.00" in reply           # NO leg unrealized (inverted payout)
    assert "unrealized +$3.00" in reply
    assert "2/20" in reply
    # Prices rendered in cents, entry → current, NO side shows NO price
    assert "40¢ → 60¢" in reply        # YES leg
    assert "40¢ → 50¢" in reply        # NO leg: 1 - 0.50


# ---------------------------------------------------------------------------
# /status <position_id_or_ticker> — detail view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_single_position_not_found():
    session_factory, session = _async_session_ctx()
    result_mock = MagicMock()
    result_mock.first = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result_mock)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(cmd_handler, session_factory, MagicMock(
        risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                       max_daily_llm_spend_usd=10.0, max_open_positions=20)
    ), mode="paper", strategy_name="Test")

    pos_id = str(uuid.uuid4())
    reply = await cmd_handler._handlers["status"](42, [pos_id])
    assert "No position found" in reply


@pytest.mark.asyncio
async def test_status_detail_accepts_market_ticker():
    """A non-UUID arg is looked up as a market ticker, not rejected."""
    session_factory, session = _async_session_ctx()
    result_mock = MagicMock()
    result_mock.first = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result_mock)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(cmd_handler, session_factory, MagicMock(
        risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                       max_daily_llm_spend_usd=10.0, max_open_positions=20)
    ), mode="paper", strategy_name="Test")

    reply = await cmd_handler._handlers["status"](42, ["KXTEST-26-A"])
    assert "No position found" in reply
    assert "KXTEST-26-A" in reply


@pytest.mark.parametrize(
    ("direction", "mid", "expected_pnl"),
    [
        ("YES", 0.60, "+$2.00"),   # 10 × (0.60 - 0.40)
        ("NO", 0.50, "+$1.00"),    # 10 × ((1 - 0.50) - 0.40)
    ],
)
@pytest.mark.asyncio
async def test_status_detail_unrealized_pnl_both_directions(direction, mid, expected_pnl):
    session_factory, session = _async_session_ctx()
    pos = _make_open_position(direction, 10, 0.40)
    pos.id = uuid.uuid4()
    result_mock = MagicMock()
    result_mock.first = MagicMock(return_value=(pos, "A test question?", mid))
    session.execute = AsyncMock(return_value=result_mock)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(cmd_handler, session_factory, MagicMock(
        risk=MagicMock(min_edge_floor=0.1, max_position_pct=0.05,
                       max_daily_llm_spend_usd=10.0, max_open_positions=20)
    ), mode="paper", strategy_name="Test")

    reply = await cmd_handler._handlers["status"](42, [str(pos.id)])
    assert expected_pnl in reply
    assert "A test question?" in reply
    assert str(pos.id) in reply


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_without_telemetry():
    cmd_handler, _, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["health"](42, [])
    assert "not available" in reply


@pytest.mark.asyncio
async def test_health_lists_service_states():
    from freqpred.runtime.telemetry import ServiceFreshnessState

    session_factory, session = _async_session_ctx()
    run_state_mock = MagicMock()
    run_state_mock.scalar_one_or_none.return_value = None
    heartbeats_mock = MagicMock()
    heartbeats_mock.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(side_effect=[run_state_mock, heartbeats_mock])

    telemetry = MagicMock()
    telemetry.websocket_state.return_value = {
        "connected": True, "subscribed_markets": 4,
        "last_message_at": None, "last_reconcile_at": None,
    }
    telemetry.evaluate_service_states.return_value = [
        ServiceFreshnessState(
            service_name="signal_loop", label="Signal loop", status="ok",
            last_success_at=None, last_error_at=None, last_error_message=None,
            stale_after_seconds=600, age_seconds=30, alertable=True,
        ),
        ServiceFreshnessState(
            service_name="fetcher_reddit", label="Reddit fetcher", status="stale",
            last_success_at=None, last_error_at=None,
            last_error_message="RedditBlockedError: 403",
            stale_after_seconds=86400, age_seconds=90000, alertable=True,
        ),
    ]

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_system_commands(
        cmd_handler, session_factory, _status_config(), mode="paper",
        strategy_name="Test", telemetry=telemetry,
    )

    reply = await cmd_handler._handlers["health"](42, [])
    assert "1/2 ok" in reply
    assert "Signal loop" in reply
    assert "Reddit fetcher" in reply
    assert "RedditBlockedError" in reply
    assert "WebSocket connected" in reply
    # Stale services sort before healthy ones
    assert reply.index("Reddit fetcher") < reply.index("Signal loop")


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
                "status", "trades", "signals", "health"}
    assert expected.issubset(set(cmd_handler._handlers.keys()))
    # /count was removed — its info lives in the /status header now
    assert "count" not in cmd_handler._handlers
