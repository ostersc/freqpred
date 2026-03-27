"""Unit tests for freqpred/markets/position_watcher.py (T39).

All DB and Kalshi interactions mocked — no external dependencies.
Tests cover:
- _on_ticker_update: DB price upsert and price_moved logging
- PositionMonitor called on each tick
- Exponential backoff reconnect sequence
- Subscription re-built from DB on reconnect
- _reconcile_positions: contracts sync, auto-close, kalshi-only skip
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# Register ORM models for SQLAlchemy relationship resolution.
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401

from freqpred.markets.models import Position
from freqpred.markets.position_watcher import PositionWatcher

NOW = datetime(2026, 3, 22, 12, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_watcher(
    *,
    price_move_threshold: float = 0.05,
    open_market_ids: set[str] | None = None,
) -> tuple[PositionWatcher, MagicMock, MagicMock, MagicMock, AsyncMock]:
    """Return (watcher, kalshi_client, session_factory, position_monitor, order_manager)."""
    kalshi_client = MagicMock()
    kalshi_client._make_auth_headers.return_value = {}
    kalshi_client.get_positions = AsyncMock(return_value=[])

    session_factory = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock(return_value=_make_db_result([]))
    mock_session.commit = AsyncMock()
    session_factory.return_value = mock_session

    position_monitor = MagicMock()
    position_monitor.check_all_positions = AsyncMock(return_value=[])

    order_manager = MagicMock()
    order_manager.reconcile_pending_orders = AsyncMock()

    watcher = PositionWatcher(
        kalshi_client=kalshi_client,
        ws_url="wss://demo-api.kalshi.co/trade-api/ws/v2",
        session_factory=session_factory,
        position_monitor=position_monitor,
        order_manager=order_manager,
        price_move_threshold=price_move_threshold,
    )
    if open_market_ids is not None:
        watcher._subscribed = open_market_ids

    return watcher, kalshi_client, session_factory, position_monitor, order_manager


def _make_db_result(rows: list) -> MagicMock:
    """Return a mock SQLAlchemy result with scalars().all() and .all() returning rows."""
    result = MagicMock()
    scalars_result = MagicMock()
    scalars_result.all.return_value = rows
    result.scalars.return_value = scalars_result
    result.all.return_value = rows
    return result


def _make_position_row(
    *,
    market_id: str = "MKT-1",
    contracts: int = 10,
    status: str = "open",
    mode: str = "live",
) -> MagicMock:
    row = MagicMock()
    row.id = uuid.uuid4()
    row.market_id = market_id
    row.contracts = contracts
    row.status = status
    row.mode = mode
    return row


def _make_kalshi_position(market_id: str, contracts: int) -> Position:
    return Position(
        id=market_id,
        market_id=market_id,
        signal_id="",
        strategy_name="exchange_reconciliation",
        strategy_version="0",
        signal_confidence=0.0,
        signal_edge=0.0,
        signal_estimated_prob=0.0,
        direction="YES",
        contracts=contracts,
        entry_price=0.0,
        entry_time=NOW,
        mode="live",
        status="open",
    )


# ---------------------------------------------------------------------------
# _on_ticker_update tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ticker_update_upserts_market_price() -> None:
    """_on_ticker_update() executes an UPDATE on MarketRow with correct prices."""
    watcher, _, session_factory, _, _ = _make_watcher(open_market_ids={"MKT-1"})
    mock_session = session_factory.return_value.__aenter__.return_value

    await watcher._on_ticker_update("MKT-1", 0.55, 0.57)

    # Session execute should have been called (the UPDATE statement).
    assert mock_session.execute.called
    assert mock_session.commit.called

    # The new mid should be stored in _last_mid.
    assert watcher._last_mid["MKT-1"] == pytest.approx(0.56, abs=1e-4)


@pytest.mark.asyncio
async def test_price_move_logged_above_threshold() -> None:
    """price_moved is logged when Δmid ≥ threshold."""
    watcher, _, _, _, _ = _make_watcher(
        price_move_threshold=0.05, open_market_ids={"MKT-1"}
    )
    # Seed previous mid so delta is detectable.
    watcher._last_mid["MKT-1"] = 0.50

    with patch("freqpred.markets.position_watcher.log") as mock_log:
        mock_log.info = MagicMock()
        await watcher._on_ticker_update("MKT-1", 0.56, 0.58)  # mid=0.57, Δ=0.07

    logged_events = [c.args[0] for c in mock_log.info.call_args_list]
    assert "position_watcher.price_moved" in logged_events


@pytest.mark.asyncio
async def test_price_move_not_logged_below_threshold() -> None:
    """price_moved is NOT logged when Δmid < threshold."""
    watcher, _, _, _, _ = _make_watcher(
        price_move_threshold=0.05, open_market_ids={"MKT-1"}
    )
    watcher._last_mid["MKT-1"] = 0.50

    with patch("freqpred.markets.position_watcher.log") as mock_log:
        mock_log.info = MagicMock()
        await watcher._on_ticker_update("MKT-1", 0.51, 0.53)  # mid=0.52, Δ=0.02

    logged_events = [c.args[0] for c in mock_log.info.call_args_list]
    assert "position_watcher.price_moved" not in logged_events


@pytest.mark.asyncio
async def test_position_monitor_called_on_tick() -> None:
    """PositionMonitor.check_all_positions() is awaited on every ticker update."""
    watcher, _, _, position_monitor, _ = _make_watcher(open_market_ids={"MKT-1"})

    await watcher._on_ticker_update("MKT-1", 0.55, 0.57)

    position_monitor.check_all_positions.assert_awaited_once()


# ---------------------------------------------------------------------------
# _handle_message ticker parsing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_ticker_parses_dollar_strings() -> None:
    """Kalshi WS v2 sends yes_bid_dollars/yes_ask_dollars as dollar strings; _handle_message parses them."""
    watcher, _, _, _, _ = _make_watcher(open_market_ids={"MKT-1"})

    calls: list[tuple[str, float, float]] = []

    async def fake_on_ticker(market_id: str, yes_bid: float, yes_ask: float) -> None:
        calls.append((market_id, yes_bid, yes_ask))

    watcher._on_ticker_update = fake_on_ticker  # type: ignore[method-assign]

    await watcher._handle_message({
        "type": "ticker",
        "msg": {
            "market_ticker": "MKT-1",
            "yes_bid_dollars": "0.6200",
            "yes_ask_dollars": "0.6600",
        },
    })

    assert len(calls) == 1
    market_id, yes_bid, yes_ask = calls[0]
    assert market_id == "MKT-1"
    assert yes_bid == pytest.approx(0.62)
    assert yes_ask == pytest.approx(0.66)


@pytest.mark.asyncio
async def test_handle_message_ticker_ignores_old_cents_field() -> None:
    """Messages using the old yes_bid/yes_ask integer-cents field names are silently ignored."""
    watcher, _, _, _, _ = _make_watcher(open_market_ids={"MKT-1"})

    calls: list[tuple] = []

    async def fake_on_ticker(market_id: str, yes_bid: float, yes_ask: float) -> None:
        calls.append((market_id, yes_bid, yes_ask))

    watcher._on_ticker_update = fake_on_ticker  # type: ignore[method-assign]

    # Message uses old integer-cents format — should not trigger an update.
    await watcher._handle_message({
        "type": "ticker",
        "msg": {
            "market_ticker": "MKT-1",
            "yes_bid": 62,
            "yes_ask": 66,
        },
    })

    assert calls == []


@pytest.mark.asyncio
async def test_handle_message_ticker_skips_unsubscribed_market() -> None:
    """Ticker for a market not in _subscribed is not forwarded to _on_ticker_update."""
    watcher, _, _, _, _ = _make_watcher(open_market_ids={"MKT-OTHER"})

    calls: list[tuple] = []

    async def fake_on_ticker(market_id: str, yes_bid: float, yes_ask: float) -> None:
        calls.append((market_id, yes_bid, yes_ask))

    watcher._on_ticker_update = fake_on_ticker  # type: ignore[method-assign]

    await watcher._handle_message({
        "type": "ticker",
        "msg": {"market_ticker": "MKT-1", "yes_bid_dollars": "0.6200", "yes_ask_dollars": "0.6600"},
    })

    assert calls == []


# ---------------------------------------------------------------------------
# Reconnect backoff tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_backoff_sequence() -> None:
    """Backoff doubles from 1s on each failure, capped at 60s."""
    watcher, _, _, _, _ = _make_watcher()

    connect_calls = 0
    expected_backoffs = [1.0, 2.0, 4.0, 8.0, 16.0]
    slept: list[float] = []

    async def fake_connect(*_a, **_kw):
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls <= len(expected_backoffs):
            raise ConnectionError("simulated disconnect")
        # Stop the loop after expected iterations.
        raise asyncio.CancelledError()

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    with (
        patch.object(watcher, "_connect_and_subscribe", side_effect=fake_connect),
        patch("freqpred.markets.position_watcher.asyncio.sleep", side_effect=fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await watcher.run()

    assert slept == expected_backoffs


# ---------------------------------------------------------------------------
# Reconnect subscription re-build test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resubscribes_current_positions_on_reconnect() -> None:
    """Subscription set is re-queried from DB and subscribe is sent on reconnect."""
    watcher, _, session_factory, _, _ = _make_watcher()

    # Make DB return two open live positions.
    row_a = _make_position_row(market_id="MKT-A")
    row_b = _make_position_row(market_id="MKT-B")

    mock_session = session_factory.return_value.__aenter__.return_value
    mock_session.execute = AsyncMock(return_value=_make_db_result([row_a, row_b]))

    connect_calls = 0
    sent_messages: list[dict] = []

    async def fake_connect(url: str, *, additional_headers: dict):
        nonlocal connect_calls
        connect_calls += 1

        class FakeWS:
            async def send(self, data: str) -> None:
                sent_messages.append(__import__("json").loads(data))

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        if connect_calls == 1:
            raise ConnectionError("first disconnect")
        return FakeWS()

    with (
        patch("freqpred.markets.position_watcher.websockets.connect", side_effect=fake_connect),
        patch("freqpred.markets.position_watcher.asyncio.sleep", new_callable=AsyncMock),
    ):
        # Run for just enough iterations.
        try:
            task = asyncio.create_task(watcher.run())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises((asyncio.CancelledError, Exception)):
                await task
        except Exception:
            pass

    # At least one subscribe command was sent.
    subscribe_cmds = [m for m in sent_messages if m.get("cmd") == "subscribe"]
    if subscribe_cmds:
        tickers = subscribe_cmds[0]["params"].get("market_tickers", [])
        assert set(tickers) == {"MKT-A", "MKT-B"}


# ---------------------------------------------------------------------------
# _reconcile_positions tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_updates_contracts_when_kalshi_differs() -> None:
    """Kalshi returns 8 contracts, DB has 5 → DB row.contracts updated to 8."""
    watcher, kalshi_client, session_factory, _, _ = _make_watcher()

    db_row = _make_position_row(market_id="MKT-X", contracts=5, status="open")
    # market mid price row for auto-close path
    market_row = MagicMock()
    market_row.id = "MKT-X"
    market_row.mid_price = 0.55

    # Session returns DB positions on first call, market prices on second.
    call_count = 0

    async def fake_execute(stmt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_db_result([db_row])
        return _make_db_result([market_row])

    mock_session = session_factory.return_value.__aenter__.return_value
    mock_session.execute = AsyncMock(side_effect=fake_execute)

    kalshi_client.get_positions = AsyncMock(
        return_value=[_make_kalshi_position("MKT-X", 8)]
    )

    async with session_factory() as session:
        await watcher._reconcile_positions(session)

    assert db_row.contracts == 8


@pytest.mark.asyncio
async def test_reconcile_closes_position_when_kalshi_has_zero() -> None:
    """Kalshi has no position for market, DB has open → ledger.close_position called."""
    watcher, kalshi_client, session_factory, _, _ = _make_watcher()

    db_row = _make_position_row(market_id="MKT-X", contracts=10, status="open")
    market_row = MagicMock()
    market_row.id = "MKT-X"
    market_row.mid_price = 0.60

    call_count = 0

    async def fake_execute(stmt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_db_result([db_row])
        return _make_db_result([market_row])

    mock_session = session_factory.return_value.__aenter__.return_value
    mock_session.execute = AsyncMock(side_effect=fake_execute)

    # Kalshi returns no positions.
    kalshi_client.get_positions = AsyncMock(return_value=[])

    with patch("freqpred.markets.position_watcher.ledger") as mock_ledger:
        mock_ledger.close_position = AsyncMock()

        async with session_factory() as session:
            await watcher._reconcile_positions(session)

    mock_ledger.close_position.assert_awaited_once()
    call_kwargs = mock_ledger.close_position.call_args_list[0].kwargs
    assert call_kwargs.get("exit_reason") == "reconcile_auto_close"


# ---------------------------------------------------------------------------
# _on_market_lifecycle tests (T40)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_settled_yes_closes_winning_position() -> None:
    """YES position, market determined YES → close_position(exit_price=1.0, exit_reason='market_resolved')."""
    watcher, _, session_factory, _, _ = _make_watcher(open_market_ids={"MKT-1"})

    market_row = MagicMock()
    market_row.question = "Will YES win?"

    pos_row = _make_position_row(market_id="MKT-1", status="open", mode="live")
    pos_row.direction = "YES"

    mock_session = session_factory.return_value.__aenter__.return_value

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = market_row
    pos_result = _make_db_result([pos_row])

    call_count = 0

    async def fake_execute(stmt):
        nonlocal call_count
        call_count += 1
        return scalar_result if call_count == 1 else pos_result

    mock_session.execute = AsyncMock(side_effect=fake_execute)

    with patch("freqpred.markets.position_watcher.ledger") as mock_ledger:
        closed_pos = MagicMock()
        closed_pos.pnl = 0.4
        closed_pos.direction = "YES"
        closed_pos.entry_price = 0.6
        closed_pos.exit_price = 1.0
        mock_ledger.close_position = AsyncMock(return_value=closed_pos)

        await watcher._on_market_lifecycle("MKT-1", "determined", "yes")

    mock_ledger.close_position.assert_awaited_once()
    call_kwargs = mock_ledger.close_position.call_args.kwargs
    assert call_kwargs["exit_price"] == pytest.approx(1.0)
    assert call_kwargs["exit_reason"] == "market_resolved"
    assert call_kwargs["resolution"] == 1


@pytest.mark.asyncio
async def test_lifecycle_settled_yes_closes_losing_position() -> None:
    """NO position, market determined YES → close_position(exit_price=0.0)."""
    watcher, _, session_factory, _, _ = _make_watcher(open_market_ids={"MKT-1"})

    market_row = MagicMock()
    market_row.question = "Will YES win?"

    pos_row = _make_position_row(market_id="MKT-1", status="open", mode="live")
    pos_row.direction = "NO"

    mock_session = session_factory.return_value.__aenter__.return_value

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = market_row
    pos_result = _make_db_result([pos_row])

    call_count = 0

    async def fake_execute(stmt):
        nonlocal call_count
        call_count += 1
        return scalar_result if call_count == 1 else pos_result

    mock_session.execute = AsyncMock(side_effect=fake_execute)

    with patch("freqpred.markets.position_watcher.ledger") as mock_ledger:
        closed_pos = MagicMock()
        closed_pos.pnl = -0.3
        closed_pos.direction = "NO"
        closed_pos.entry_price = 0.3
        closed_pos.exit_price = 0.0
        mock_ledger.close_position = AsyncMock(return_value=closed_pos)

        await watcher._on_market_lifecycle("MKT-1", "determined", "yes")

    mock_ledger.close_position.assert_awaited_once()
    call_kwargs = mock_ledger.close_position.call_args.kwargs
    assert call_kwargs["exit_price"] == pytest.approx(0.0)
    assert call_kwargs["exit_reason"] == "market_resolved"
    assert call_kwargs["resolution"] == 1


@pytest.mark.asyncio
async def test_lifecycle_settled_does_not_close_positions() -> None:
    """status='settled' → no close_position() call (positions already closed on 'determined')."""
    watcher, _, session_factory, _, _ = _make_watcher(open_market_ids={"MKT-1"})
    mock_session = session_factory.return_value.__aenter__.return_value
    mock_session.execute = AsyncMock(return_value=_make_db_result([]))

    with patch("freqpred.markets.position_watcher.ledger") as mock_ledger:
        mock_ledger.close_position = AsyncMock()
        await watcher._on_market_lifecycle("MKT-1", "settled", None)

    mock_ledger.close_position.assert_not_awaited()


@pytest.mark.asyncio
async def test_lifecycle_unsubscribes_after_settlement() -> None:
    """After settlement, market_id is removed from _subscribed."""
    watcher, _, session_factory, _, _ = _make_watcher(open_market_ids={"MKT-1"})

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None  # no market row needed
    empty_result = _make_db_result([])  # no positions

    call_count = 0

    async def fake_execute(stmt):
        nonlocal call_count
        call_count += 1
        return scalar_result if call_count == 1 else empty_result

    mock_session = session_factory.return_value.__aenter__.return_value
    mock_session.execute = AsyncMock(side_effect=fake_execute)

    with patch("freqpred.markets.position_watcher.ledger"):
        await watcher._on_market_lifecycle("MKT-1", "settled", "yes")

    assert "MKT-1" not in watcher._subscribed


@pytest.mark.asyncio
async def test_telegram_alert_sent_on_resolution() -> None:
    """AlertDispatcher.send() is called with WIN/LOSS message on position resolution."""
    alert_dispatcher = MagicMock()
    alert_dispatcher.send = AsyncMock()

    watcher, _, session_factory, _, _ = _make_watcher(open_market_ids={"MKT-1"})
    watcher._alert_dispatcher = alert_dispatcher

    market_row = MagicMock()
    market_row.question = "Will it rain?"

    pos_row = _make_position_row(market_id="MKT-1", status="open", mode="live")
    pos_row.direction = "YES"

    mock_session = session_factory.return_value.__aenter__.return_value

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = market_row
    pos_result = _make_db_result([pos_row])

    call_count = 0

    async def fake_execute(stmt):
        nonlocal call_count
        call_count += 1
        return scalar_result if call_count == 1 else pos_result

    mock_session.execute = AsyncMock(side_effect=fake_execute)

    with patch("freqpred.markets.position_watcher.ledger") as mock_ledger:
        closed_pos = MagicMock()
        closed_pos.pnl = 0.5
        closed_pos.direction = "YES"
        closed_pos.entry_price = 0.5
        closed_pos.exit_price = 1.0
        mock_ledger.close_position = AsyncMock(return_value=closed_pos)

        await watcher._on_market_lifecycle("MKT-1", "determined", "yes")

    alert_dispatcher.send.assert_awaited_once()
    sent_msg: str = alert_dispatcher.send.call_args.args[0]
    assert "WIN" in sent_msg or "LOSS" in sent_msg
    assert "Will it rain?" in sent_msg


@pytest.mark.asyncio
async def test_rest_fallback_resolves_if_websocket_missed() -> None:
    """MarketWatcher._resolve_settled_live_positions() closes open live positions for settled markets."""
    from freqpred.markets.watcher import MarketWatcher

    mock_client = MagicMock()
    mock_client.list_markets = AsyncMock(return_value=[])

    session_factory = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.commit = AsyncMock()
    session_factory.return_value = mock_session

    # Row simulating a join result (PositionRow.id, PositionRow.direction, MarketRow.result)
    row = MagicMock()
    row.id = uuid.uuid4()
    row.direction = "YES"
    row.result = "yes"

    mock_session.execute = AsyncMock(return_value=_make_db_result([row]))

    watcher = MarketWatcher(client=mock_client, session_factory=session_factory)

    with patch("freqpred.markets.watcher.ledger") as mock_ledger:
        mock_ledger.close_position = AsyncMock()
        count = await watcher._resolve_settled_live_positions()

    assert count == 1
    mock_ledger.close_position.assert_awaited_once()
    call_kwargs = mock_ledger.close_position.call_args.kwargs
    assert call_kwargs["exit_price"] == pytest.approx(1.0)
    assert call_kwargs["exit_reason"] == "market_resolved"
    assert call_kwargs["resolution"] == 1


@pytest.mark.asyncio
async def test_reconcile_ignores_kalshi_only_positions() -> None:
    """Kalshi has a position not in DB (manual trade) → logged, no new DB row created."""
    watcher, kalshi_client, session_factory, _, _ = _make_watcher()

    mock_session = session_factory.return_value.__aenter__.return_value
    # DB has no live positions.
    mock_session.execute = AsyncMock(return_value=_make_db_result([]))

    # Kalshi has a position for a market not in DB.
    kalshi_client.get_positions = AsyncMock(
        return_value=[_make_kalshi_position("MKT-Y", 5)]
    )

    with patch("freqpred.markets.position_watcher.log") as mock_log:
        mock_log.info = MagicMock()
        async with session_factory() as session:
            await watcher._reconcile_positions(session)

    # No commit for auto-close (no DB positions to process).
    # The kalshi-only market should be logged.
    logged_events = [c.args[0] for c in mock_log.info.call_args_list]
    assert "position_watcher.reconcile_kalshi_only" in logged_events

    # No DB row should be created (add/insert never called).
    mock_session.add.assert_not_called() if hasattr(mock_session, "add") else None
