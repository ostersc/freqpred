"""Unit tests for T29 Telegram metrics command handlers.

All DB sessions and external calls are mocked — no real DB or API requests.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freqpred.alerts.metrics_handlers import register_metrics_commands
from freqpred.alerts.telegram_commands import TelegramCommandHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(bankroll: float = 1000.0, llm_cap: float = 10.0) -> MagicMock:
    cfg = MagicMock()
    cfg.trading.bankroll_usd = bankroll
    cfg.risk.max_daily_llm_spend_usd = llm_cap
    return cfg


def _make_handler_under_test(config=None, mode="paper"):
    """Return a (TelegramCommandHandler, session_factory_mock) pair."""
    if config is None:
        config = _make_config()
    session_factory = MagicMock()
    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(
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


def _make_closed_position(
    pnl: float = 0.05,
    contracts: int = 10,
    entry_price: float = 0.50,
    exit_reason: str = "market_resolved",
    hold_hours: int = 4,
) -> MagicMock:
    pos = MagicMock()
    pos.status = "closed"
    pos.pnl = pnl
    pos.pnl_pct = pnl / (contracts * entry_price)
    pos.contracts = contracts
    pos.entry_price = entry_price
    pos.exit_reason = exit_reason
    now = datetime.now(UTC)
    pos.entry_time = now - timedelta(hours=hold_hours)
    pos.exit_time = now
    pos.signal_id = uuid.uuid4()
    pos.resolution = 1
    return pos


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


def test_all_t29_commands_registered():
    cmd_handler, _ = _make_handler_under_test()
    expected = {"profit", "daily", "weekly", "monthly", "stats", "balance", "budget", "calibration"}
    assert expected.issubset(set(cmd_handler._handlers.keys()))


# ---------------------------------------------------------------------------
# /profit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profit_no_closed_trades():
    session_factory, session = _async_session_ctx()

    pos_result = MagicMock()
    pos_result.scalars.return_value.all.return_value = []
    brier_result = MagicMock()
    brier_result.all.return_value = []
    session.execute = AsyncMock(side_effect=[pos_result, brier_result])

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["profit"](42, [])
    assert "No closed trades" in reply


@pytest.mark.asyncio
async def test_profit_with_data():
    session_factory, session = _async_session_ctx()

    pos1 = _make_closed_position(pnl=0.048, contracts=10, entry_price=0.50, exit_reason="roi")
    pos2 = _make_closed_position(pnl=-0.020, contracts=5, entry_price=0.80, exit_reason="stoploss")

    pos_result = MagicMock()
    pos_result.scalars.return_value.all.return_value = [pos1, pos2]
    brier_result = MagicMock()
    brier_result.all.return_value = [(0.7, 1), (0.6, 0)]
    session.execute = AsyncMock(side_effect=[pos_result, brier_result])

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["profit"](42, [])
    assert "Total P&L" in reply
    assert "win rate" in reply
    assert "Brier score" in reply
    assert "Best trade" in reply
    assert "Worst trade" in reply


@pytest.mark.asyncio
async def test_profit_invalid_n():
    cmd_handler, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["profit"](42, ["notanumber"])
    assert "Usage" in reply


@pytest.mark.asyncio
async def test_profit_with_days_filter():
    session_factory, session = _async_session_ctx()

    pos_result = MagicMock()
    pos_result.scalars.return_value.all.return_value = []
    brier_result = MagicMock()
    brier_result.all.return_value = []
    session.execute = AsyncMock(side_effect=[pos_result, brier_result])

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["profit"](42, ["7"])
    assert "last 7 day(s)" in reply


# ---------------------------------------------------------------------------
# /daily
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_no_trades():
    session_factory, session = _async_session_ctx()

    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["daily"](42, [])
    assert "No closed trades" in reply


@pytest.mark.asyncio
async def test_daily_with_data():
    session_factory, session = _async_session_ctx()

    now = datetime.now(UTC)
    pos = _make_closed_position(pnl=0.048, contracts=10, entry_price=0.50)
    pos.exit_time = now

    result = MagicMock()
    result.scalars.return_value.all.return_value = [pos]
    session.execute = AsyncMock(return_value=result)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["daily"](42, [])
    assert "Date" in reply
    assert "Trades" in reply
    assert "P&amp;L" in reply
    assert reply.startswith("<pre>") or "<pre>" in reply


@pytest.mark.asyncio
async def test_daily_invalid_n():
    cmd_handler, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["daily"](42, ["xyz"])
    assert "Usage" in reply


# ---------------------------------------------------------------------------
# /weekly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weekly_no_trades():
    session_factory, session = _async_session_ctx()

    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["weekly"](42, [])
    assert "No closed trades" in reply


@pytest.mark.asyncio
async def test_weekly_with_data():
    session_factory, session = _async_session_ctx()

    now = datetime.now(UTC)
    pos1 = _make_closed_position(pnl=0.048)
    pos1.exit_time = now
    pos2 = _make_closed_position(pnl=-0.010)
    pos2.exit_time = now - timedelta(days=3)

    result = MagicMock()
    result.scalars.return_value.all.return_value = [pos1, pos2]
    session.execute = AsyncMock(return_value=result)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["weekly"](42, [])
    assert "Week start" in reply
    assert "P&amp;L" in reply


@pytest.mark.asyncio
async def test_weekly_invalid_n():
    cmd_handler, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["weekly"](42, ["abc"])
    assert "Usage" in reply


# ---------------------------------------------------------------------------
# /monthly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monthly_no_trades():
    session_factory, session = _async_session_ctx()

    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["monthly"](42, [])
    assert "No closed trades" in reply


@pytest.mark.asyncio
async def test_monthly_with_data():
    session_factory, session = _async_session_ctx()

    now = datetime.now(UTC)
    pos = _make_closed_position(pnl=1.20, contracts=20, entry_price=0.60)
    pos.exit_time = now

    result = MagicMock()
    result.scalars.return_value.all.return_value = [pos]
    session.execute = AsyncMock(return_value=result)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["monthly"](42, [])
    assert "Month" in reply
    assert "P&amp;L" in reply


@pytest.mark.asyncio
async def test_monthly_invalid_n():
    cmd_handler, _ = _make_handler_under_test()
    reply = await cmd_handler._handlers["monthly"](42, ["!"])
    assert "Usage" in reply


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_no_closed_trades():
    session_factory, session = _async_session_ctx()

    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["stats"](42, [])
    assert "No closed trades" in reply


@pytest.mark.asyncio
async def test_stats_with_data():
    session_factory, session = _async_session_ctx()

    positions = [
        _make_closed_position(pnl=0.048, exit_reason="roi"),
        _make_closed_position(pnl=-0.020, exit_reason="stoploss"),
        _make_closed_position(pnl=0.030, exit_reason="market_resolved"),
    ]

    result = MagicMock()
    result.scalars.return_value.all.return_value = positions
    session.execute = AsyncMock(return_value=result)

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")

    reply = await cmd_handler._handlers["stats"](42, [])
    assert "Total trades" in reply
    assert "win rate" in reply
    assert "Best trade" in reply
    assert "Worst trade" in reply
    assert "By exit reason" in reply
    assert "roi" in reply
    assert "stoploss" in reply


# ---------------------------------------------------------------------------
# /balance
# ---------------------------------------------------------------------------




@pytest.mark.asyncio
async def test_balance_zero_state():
    session_factory, session = _async_session_ctx()

    with patch(
        "freqpred.alerts.metrics_handlers.get_portfolio_summary",
        new_callable=AsyncMock,
        return_value={
            "open_count": 0,
            "total_exposure_usd": 0.0,
            "net_exposure_usd": 0.0,
            "daily_pnl_usd": 0.0,
            "all_time_pnl_usd": 0.0,
            "unrealized_pnl_usd": 0.0,
            "portfolio_mae_usd": None,
            "portfolio_mfe_usd": None,
            "portfolio_mae_pct": None,
            "portfolio_mfe_pct": None,
        },
    ), patch(
        "freqpred.alerts.metrics_handlers.get_drawdown_window",
        new=AsyncMock(return_value=(None, None)),
    ):
        cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
        register_metrics_commands(
            cmd_handler, session_factory, _make_config(bankroll=1000.0), mode="paper"
        )
        reply = await cmd_handler._handlers["balance"](42, [])

    assert "1,000.00" in reply
    assert "Net value" in reply
    assert "paper" in reply
    assert "Drawdown" in reply


@pytest.mark.asyncio
async def test_balance_with_pnl():
    session_factory, session = _async_session_ctx()

    with patch(
        "freqpred.alerts.metrics_handlers.get_portfolio_summary",
        new_callable=AsyncMock,
        return_value={
            "open_count": 3,
            "total_exposure_usd": 48.60,
            "net_exposure_usd": 30.00,
            "daily_pnl_usd": 0.48,
            "all_time_pnl_usd": 3.21,
            "unrealized_pnl_usd": 1.05,
            "portfolio_mae_usd": -3.50,
            "portfolio_mfe_usd": 5.20,
            "portfolio_mae_pct": -0.072,
            "portfolio_mfe_pct": 0.107,
        },
    ), patch(
        "freqpred.alerts.metrics_handlers.get_drawdown_window",
        new=AsyncMock(return_value=(None, None)),
    ):
        cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
        register_metrics_commands(
            cmd_handler, session_factory, _make_config(bankroll=1000.0), mode="paper"
        )
        reply = await cmd_handler._handlers["balance"](42, [])

    assert "1,003.21" in reply  # net value
    assert "3.21" in reply       # all-time P&L
    assert "48.60" in reply      # exposure
    assert "3" in reply          # open positions
    assert "Drawdown" in reply


# ---------------------------------------------------------------------------
# /budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_zero_spend():
    session_factory, session = _async_session_ctx()

    # today by type: empty
    today_result = MagicMock()
    today_result.all.return_value = []
    week_result = MagicMock()
    week_result.scalar_one.return_value = 0.0
    month_result = MagicMock()
    month_result.scalar_one.return_value = 0.0
    alltime_result = MagicMock()
    alltime_result.scalar_one.return_value = 0.0

    session.execute = AsyncMock(
        side_effect=[today_result, week_result, month_result, alltime_result]
    )

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(
        cmd_handler, session_factory, _make_config(llm_cap=10.0), mode="paper"
    )
    reply = await cmd_handler._handlers["budget"](42, [])

    assert "LLM budget" in reply
    assert "$10.00 cap" in reply
    assert "(0%)" in reply


@pytest.mark.asyncio
async def test_budget_with_spend():
    session_factory, session = _async_session_ctx()

    today_result = MagicMock()
    today_result.all.return_value = [("signal", 0.0123), ("catalyst", 0.0050)]
    week_result = MagicMock()
    week_result.scalar_one.return_value = 0.0847
    month_result = MagicMock()
    month_result.scalar_one.return_value = 0.3201
    alltime_result = MagicMock()
    alltime_result.scalar_one.return_value = 2.45

    session.execute = AsyncMock(
        side_effect=[today_result, week_result, month_result, alltime_result]
    )

    cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
    register_metrics_commands(
        cmd_handler, session_factory, _make_config(llm_cap=10.0), mode="paper"
    )
    reply = await cmd_handler._handlers["budget"](42, [])

    assert "$0.02 / $10.00 cap" in reply  # today total, 2dp
    assert "signal" in reply
    assert "0.0123" in reply  # per-type breakdown keeps 4dp
    assert "This week: $0.08" in reply
    assert "All-time: $2.45" in reply


# ---------------------------------------------------------------------------
# /calibration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_no_samples():
    from freqpred.metrics.calibration import CalibrationReport

    session_factory, session = _async_session_ctx()

    empty_report = CalibrationReport(
        brier_score=0.0,
        market_brier_score=0.0,
        n_samples=0,
        buckets=[],
    )

    with patch(
        "freqpred.alerts.metrics_handlers.compute_calibration",
        new_callable=AsyncMock,
        return_value=empty_report,
    ):
        cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
        register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")
        reply = await cmd_handler._handlers["calibration"](42, [])

    assert "calibration unavailable" in reply


@pytest.mark.asyncio
async def test_calibration_with_data():
    from freqpred.metrics.calibration import CalibrationBucket, CalibrationReport

    session_factory, session = _async_session_ctx()

    report = CalibrationReport(
        brier_score=0.127,
        market_brier_score=0.224,
        n_samples=20,
        buckets=[
            CalibrationBucket(
                lower=0.6, upper=0.7, count=8,
                mean_estimated_prob=0.643, actual_resolution_rate=0.625,
            ),
            CalibrationBucket(
                lower=0.7, upper=0.8, count=12,
                mean_estimated_prob=0.742, actual_resolution_rate=0.750,
            ),
        ],
    )

    with patch(
        "freqpred.alerts.metrics_handlers.compute_calibration",
        new_callable=AsyncMock,
        return_value=report,
    ):
        cmd_handler = TelegramCommandHandler(bot_token="TOKEN", authorized_users=["alice"])
        register_metrics_commands(cmd_handler, session_factory, _make_config(), mode="paper")
        reply = await cmd_handler._handlers["calibration"](42, [])

    assert "0.127" in reply
    assert "0.224" in reply
    assert "20" in reply
    assert "better" in reply
    assert "0.643" in reply  # bucket row data
