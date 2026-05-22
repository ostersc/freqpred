"""freqpred CLI entry point."""
from __future__ import annotations

import asyncio
import logging
import signal as _signal
from pathlib import Path

import click
import structlog

from freqpred.config import load_config


def _configure_logging(log_level: str, log_file: str = "", log_backup_days: int = 14, log_module_levels: dict[str, str] | None = None) -> None:
    """Set up structlog with stdlib integration at the given level.

    Uses ProcessorFormatter so that console output gets colors while the
    rolling file gets plain text.  Creates the log directory automatically.
    """
    import logging.handlers
    import sys
    from pathlib import Path

    level = getattr(logging, log_level)

    # Shared pre-processors applied before the final renderer.
    # These run for both structlog-native records and foreign stdlib records.
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Use DEBUG as the structlog-level floor so that per-module stdlib level overrides
    # work. Actual filtering is done by stdlib's logger hierarchy: the root logger is
    # set to `level` (INFO by default), so debug output from other modules is dropped
    # before reaching any handler. Only loggers explicitly set to DEBUG below will emit.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console handler — colours on, no locals in tracebacks (avoids leaking secrets/large objects)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processor=structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        )
    )
    root.addHandler(console_handler)

    # Rolling file handler — plain text, one file per day, keep N days
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_path,
            when="midnight",
            backupCount=log_backup_days,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processor=structlog.dev.ConsoleRenderer(
                    colors=False,
                    exception_formatter=structlog.dev.plain_traceback,
                ),
            )
        )
        root.addHandler(file_handler)

    # Suppress chatty libraries unless debugging
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Route truthbrush's loguru messages through structlog at DEBUG level so they
    # only appear with --log-level DEBUG and use the same format as everything else.
    try:
        from loguru import logger as _loguru
        try:
            _loguru.remove(0)  # remove default stderr sink (always id=0 on first import)
        except ValueError:
            pass
        _loguru.add(
            lambda msg: logging.getLogger("truthbrush").debug(msg.record["message"]),
            filter=lambda r: r["name"].startswith("truthbrush"),
            level=0,
            format="{message}",
        )
    except ImportError:
        pass

    # Per-module level overrides from config
    for module, module_level in (log_module_levels or {}).items():
        logging.getLogger(module).setLevel(getattr(logging, module_level))



# Module-level log buffer shared between _configure_logging and _run_main.
_log_buffer: "LogBuffer | None" = None


def _get_or_create_log_buffer() -> "LogBuffer":
    from freqpred.alerts.command_handlers import LogBuffer, install_log_buffer

    global _log_buffer
    if _log_buffer is None:
        _log_buffer = LogBuffer()
        install_log_buffer(_log_buffer)
    return _log_buffer


# Forward reference to avoid circular import at module load time.
if False:  # TYPE_CHECKING
    from freqpred.alerts.command_handlers import LogBuffer


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """freqpred — LLM-driven prediction market trading framework."""
    ctx.ensure_object(dict)
    config = load_config()
    _configure_logging(config.log_level, log_file=config.log_file, log_backup_days=config.log_backup_days, log_module_levels=config.log_module_levels)
    ctx.obj["config"] = config


@main.command()
@click.option("--strategy", required=True, help="Strategy class name to run.")
@click.option(
    "--mode",
    type=click.Choice(["paper", "live", "signal-only"]),
    default="paper",
    show_default=True,
    help="Trading mode.",
)
@click.pass_context
def run(ctx: click.Context, strategy: str, mode: str) -> None:
    """Start market watcher, ingestion scheduler, and signal pipeline."""
    config = ctx.obj["config"]
    if config.database.url:
        from freqpred.db import run_migrations
        click.echo("Applying pending migrations...")
        run_migrations(config.database.url)
    try:
        asyncio.run(_run_main(config, strategy, mode))
    except KeyboardInterrupt:
        click.echo("\nShutting down.")


async def _run_main(config: object, strategy_name: str, mode: str) -> None:
    from datetime import UTC, datetime as _datetime  # noqa: PLC0415

    _process_started_at = _datetime.now(UTC)

    import anthropic

    import freqpred.ingestion.models  # noqa: F401
    import freqpred.metrics.models  # noqa: F401
    import freqpred.signal.models  # noqa: F401
    import freqpred.rag.models  # noqa: F401

    from freqpred.db import make_engine, make_session_factory
    from freqpred.ingestion.scheduler import run_scheduler
    from freqpred.ingestion.realtime_scheduler import run_realtime_scheduler
    from freqpred.llm.audit import LLMBudgetExceededError
    from freqpred.llm.client import LLMClient, LLMConsecutiveErrorsError
    from freqpred.markets.kalshi import KalshiClient
    from freqpred.markets.models import Market, MarketRow
    from freqpred.markets.watcher import MarketWatcher
    from freqpred.rag.embedder import LocalEmbedder
    from freqpred.signal.llm import PROMPT_VERSION
    from freqpred.signal.pipeline import SignalPipeline
    from freqpred.strategy.loader import load_strategy

    from sqlalchemy import select

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return
    if not config.anthropic.api_key:
        click.echo("ERROR: ANTHROPIC_API_KEY not configured.", err=True)
        return

    from freqpred.ingestion.fetchers.factbase import FactbasePhraseCache, run_factbase_scheduler

    strategy = load_strategy(strategy_name)
    click.echo(f"Loaded strategy: {strategy.config.name}")
    click.echo(f"Starting freqpred | strategy={strategy_name} | mode={mode}")

    phrase_cache = FactbasePhraseCache()
    if hasattr(strategy, "_phrase_cache"):
        strategy._phrase_cache = phrase_cache

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    embedder = LocalEmbedder()
    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        session_factory,
        prompt_version=PROMPT_VERSION,
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
        max_consecutive_errors=config.risk.max_consecutive_llm_errors,
    )
    _factbase_allowlist = frozenset(getattr(strategy.config, "factbase_series_allowlist", []))
    pipeline = SignalPipeline(
        session_factory=session_factory,
        embedder=embedder,
        llm_client=llm_client,
        model=config.anthropic.primary_model,
        top_k=config.signal.top_k_documents,
        factbase_series_allowlist=_factbase_allowlist,
        max_scheduled_interval_hours=config.signal.max_scheduled_interval_hours,
    )

    from freqpred.alerts.telegram import TelegramSender
    from freqpred.alerts.telegram_commands import TelegramCommandHandler
    from freqpred.alerts.discord import DiscordSender
    from freqpred.alerts.dispatcher import AlertDispatcher

    senders = []
    telegram = TelegramSender(
        bot_token=config.alerts.telegram_bot_token,
        chat_id=config.alerts.telegram_chat_id,
    )
    senders.append(telegram)
    discord = DiscordSender(webhook_url=config.alerts.discord_webhook_url)
    senders.append(discord)
    alert_dispatcher = AlertDispatcher(senders)

    telegram_cmd_handler = TelegramCommandHandler(
        bot_token=config.alerts.telegram_bot_token,
        authorized_users=config.alerts.telegram_authorized_users,
    )

    import freqpred.alerts.models  # noqa: F401 — register RunStateRow
    from freqpred.alerts.command_handlers import register_system_commands
    from freqpred.alerts.metrics_handlers import register_metrics_commands
    from freqpred.alerts.position_handlers import register_position_commands
    from freqpred.alerts.run_state import get_run_state, set_cb_state, set_mode, set_strategy_name
    from freqpred.runtime.telemetry import (
        SERVICE_SIGNAL_LOOP,
        RuntimeTelemetry,
        build_freshness_specs,
        run_stale_service_watchdog,
    )

    log_buffer = _get_or_create_log_buffer()
    register_system_commands(
        cmd_handler=telegram_cmd_handler,
        session_factory=session_factory,
        config=config,
        mode=mode,
        strategy_name=strategy_name,
        log_buffer=log_buffer,
    )
    register_metrics_commands(
        cmd_handler=telegram_cmd_handler,
        session_factory=session_factory,
        config=config,
        mode=mode,
        llm_client=llm_client,
    )
    from freqpred.trading.risk import RiskEngine, TradingCircuitBreakerError
    from freqpred.trading.order_manager import OrderManager
    runtime_telemetry = RuntimeTelemetry(
        session_factory=session_factory,
        freshness_specs=build_freshness_specs(
            ingestion_interval_seconds=config.ingestion.schedule_interval_seconds,
            realtime_interval_seconds=config.ingestion.realtime_interval_seconds,
            signal_interval_seconds=config.signal.interval_seconds,
            market_watcher_interval_seconds=config.kalshi.polling_interval_seconds,
        ),
    )

    order_manager = None
    position_watcher = None
    if mode == "paper":
        risk_engine = RiskEngine(config.risk)
        order_manager = OrderManager(
            risk=risk_engine,
            session_factory=session_factory,
            bankroll=config.trading.bankroll_usd,
            mode="paper",
            llm_client=llm_client,
            judgment_model=config.anthropic.judgment_model,
            runtime_telemetry=runtime_telemetry,
        )

    from freqpred.strategy.loader import _BUILTIN_STRATEGIES
    from freqpred.trading.position_monitor import PositionMonitor

    # Load all built-in strategies so the monitor can evaluate exits for
    # positions entered under a previous strategy (e.g. after a strategy switch).
    all_strategies = {}
    for name in _BUILTIN_STRATEGIES:
        try:
            all_strategies[name] = load_strategy(name)
        except Exception:
            pass  # don't let a broken built-in prevent startup
    all_strategies[strategy_name] = strategy  # active strategy always wins

    position_monitor = PositionMonitor(
        session_factory=session_factory,
        strategies=all_strategies,
        alert_dispatcher=alert_dispatcher,
        mode=mode,
        kalshi_client=None,  # set to kalshi_client below once the client is open
        runtime_telemetry=runtime_telemetry,
    )

    async def signal_loop() -> None:
        import structlog
        log = structlog.get_logger("freqpred.cli.signal_loop")
        log.info("signal_loop.started")
        circuit_breaker_active = False
        while True:
            _signal_loop_error: str | None = None
            try:
                # Check run-loop state; apply any runtime config overrides; skip if paused/stopped.
                async with session_factory() as rs_session:
                    run_state = await get_run_state(rs_session)
                    from freqpred.strategy.config_store import load_overrides  # noqa: PLC0415
                    _overrides = await load_overrides(rs_session, strategy.config.name)
                    for _k, _v in _overrides.items():
                        setattr(strategy.config, _k, _v)

                if run_state == "stopped":
                    log.debug("signal_loop.skipped", reason="stopped")
                    await asyncio.sleep(10)  # poll frequently so /start is picked up quickly
                    continue

                async with session_factory() as session:
                    from datetime import UTC, datetime
                    now = datetime.now(UTC)
                    result = await session.execute(
                        select(MarketRow).where(MarketRow.close_time > now)
                    )
                    market_rows = result.scalars().all()

                markets: list[Market] = [
                    Market(
                        id=row.id,
                        platform=row.platform,
                        question=row.question,
                        category=row.category,
                        status=row.status,
                        result=row.result,
                        close_time=row.close_time,
                        yes_bid=row.yes_bid,
                        yes_ask=row.yes_ask,
                        mid_price=row.mid_price,
                        last_price=row.last_price,
                        volume_24h=row.volume_24h,
                        open_interest=row.open_interest,
                        yes_bid_size=row.yes_bid_size,
                        yes_ask_size=row.yes_ask_size,
                        last_fetched_at=row.last_fetched_at,
                        price_updated_at=row.price_updated_at,
                        metadata_fetched_at=row.metadata_fetched_at,
                        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
                        metadata=dict(row.metadata_),
                        open_time=row.open_time,
                        series_ticker=row.series_ticker,
                    )
                    for row in market_rows
                ]

                interesting = strategy.filter_markets(markets)

                # Always re-analyze markets with open positions even if they
                # no longer pass filter_markets (e.g. price drifted outside
                # the min/max mid_price window). Signal-driven exits must
                # still fire for existing positions.
                from freqpred.markets.models import PositionRow  # noqa: PLC0415
                async with session_factory() as pos_session:
                    open_pos_result = await pos_session.execute(
                        select(PositionRow.market_id).where(
                            PositionRow.status == "open",
                            PositionRow.mode == mode,
                        ).distinct()
                    )
                    open_market_ids = {row.market_id for row in open_pos_result.all()}

                interesting_ids = {m.id for m in interesting}
                market_by_id = {m.id: m for m in markets}
                for mid in open_market_ids:
                    if mid not in interesting_ids and mid in market_by_id:
                        interesting.append(market_by_id[mid])

                log.info(
                    "signal_loop.cycle",
                    total_markets=len(markets),
                    selected=len(interesting),
                    open_position_markets=len(open_market_ids - interesting_ids),
                )

                # Circuit breaker check at the top of each cycle
                if order_manager is not None:
                    try:
                        async with session_factory() as cb_session:
                            from freqpred.alerts.run_state import (  # noqa: PLC0415
                                get_daily_loss_ack_at,
                                get_drawdown_window,
                            )
                            from freqpred.trading import ledger as _ledger  # noqa: PLC0415
                            _net_bankroll = await _ledger.get_net_bankroll(
                                cb_session, order_manager._bankroll, mode=order_manager._mode
                            )
                            _, _reset_bankroll = await get_drawdown_window(cb_session)
                            _daily_loss_ack_at = await get_daily_loss_ack_at(cb_session)
                            await order_manager._risk.check_circuit_breakers(
                                cb_session, _net_bankroll, mode=order_manager._mode,
                                drawdown_reset_bankroll=_reset_bankroll,
                                daily_loss_ack_at=_daily_loss_ack_at,
                            )
                            # CB check passed — clear any previously persisted CB state.
                            if circuit_breaker_active:
                                log.info("signal_loop.circuit_breaker_cleared")
                            circuit_breaker_active = False
                            await set_cb_state(cb_session, active=False, reason=None)
                    except TradingCircuitBreakerError as exc:
                        log.warning("signal_loop.circuit_breaker_fired", reason=str(exc))
                        cb_type = "daily_loss" if "daily loss" in str(exc) else "drawdown"
                        if not circuit_breaker_active:
                            # Only alert on the initial trip, not every subsequent cycle.
                            await alert_dispatcher.circuit_breaker_alert(cb_type, str(exc))
                        circuit_breaker_active = True
                        async with session_factory() as _cb_persist_session:
                            await set_cb_state(_cb_persist_session, active=True, reason=str(exc))

                # Pre-signal risk gate: check global capacity once per cycle so we can
                # skip LLM calls for new-entry markets that risk would block anyway.
                _gate_enabled = (
                    order_manager is not None
                    and not circuit_breaker_active
                    and strategy.config.pre_signal_risk_gate
                )
                _entry_globally_blocked = False
                _entry_blocked_reason = ""
                _effective_max_spread: float | None = None
                if _gate_enabled:
                    sc = strategy.config
                    _effective_max_spread = (
                        sc.max_spread if sc.max_spread is not None else sc.min_edge / 2
                    )
                    from freqpred.trading import ledger as _ledger  # noqa: PLC0415
                    async with session_factory() as _cap_session:
                        _cap_bankroll = await _ledger.get_net_bankroll(
                            _cap_session, order_manager._bankroll, mode=order_manager._mode
                        )
                        _entry_globally_blocked, _entry_blocked_reason = (
                            await order_manager._risk.check_entry_capacity(
                                _cap_session, _cap_bankroll, mode=order_manager._mode
                            )
                        )
                    if _entry_globally_blocked:
                        log.info(
                            "signal_loop.entry_capacity_full",
                            reason=_entry_blocked_reason,
                        )

                for market in interesting:
                    # Pre-signal gate: skip LLM for new-entry markets risk would block
                    if _gate_enabled and market.id not in open_market_ids:
                        if _entry_globally_blocked:
                            log.info("signal.skipped.capacity_full", market_id=market.id)
                            continue
                        assert _effective_max_spread is not None
                        sc = strategy.config
                        async with session_factory() as _gate_session:
                            _gate_blocked, _gate_reason = (
                                await order_manager._risk.pre_signal_gate(
                                    _gate_session,
                                    market,
                                    mode=order_manager._mode,
                                    effective_max_spread=_effective_max_spread,
                                    block_reentry_after_stoploss=sc.block_reentry_after_stoploss,
                                    stoploss_cooldown_hours=sc.stoploss_cooldown_hours,
                                )
                            )
                        if _gate_blocked:
                            log.info(
                                "signal.skipped.risk_gate",
                                market_id=market.id,
                                reason=_gate_reason,
                            )
                            continue

                    signal = await pipeline.analyze(market, trigger="scheduled")
                    if signal is None:
                        async with session_factory() as synth_session:
                            signal = await strategy.synthesize_signal(synth_session, market)
                    if signal:
                        click.echo(
                            f"[SIGNAL] market={market.id} "
                            f"prob={signal.estimated_probability:.3f} "
                            f"edge={signal.edge:+.3f} "
                            f"confidence={signal.confidence:.2f} "
                            f"direction={signal.direction} "
                            f"signal_id={signal.id}"
                        )
                        if mode == "signal-only" and signal.edge >= strategy.config.min_edge and signal.direction != "SKIP":
                            await alert_dispatcher.signal_alert(signal, market)
                        if (
                            order_manager is not None
                            and not circuit_breaker_active
                            and signal.direction != "SKIP"
                            and run_state == "running"
                        ):
                            # If this market has an open position, evaluate exits
                            # with the fresh signal before attempting entry.  This
                            # ensures should_exit() fires (e.g. direction flip) and
                            # the position is closed in the DB before submit() runs
                            # its opposite-side guard — all within the same cycle.
                            if market.id in open_market_ids:
                                await position_monitor.check_all_positions(
                                    fresh_signals={market.id: signal}
                                )
                            position = await order_manager.submit(signal, market, strategy)
                            if position:
                                log.info(
                                    "signal_loop.order_submitted",
                                    position_id=position.id,
                                    market_id=market.id,
                                    direction=signal.direction,
                                )
                                await alert_dispatcher.trade_alert(position, market)
                                await strategy.on_position_opened(position, market, session_factory)
                                if position_watcher is not None:
                                    await position_watcher.subscribe(position.market_id)
                            else:
                                strategy.on_order_failed(market)
                await runtime_telemetry.mark_success(
                    SERVICE_SIGNAL_LOOP,
                    details={
                        "total_markets": len(markets),
                        "selected_markets": len(interesting),
                        "run_state": run_state,
                    },
                )
            except asyncio.CancelledError:
                raise
            except LLMBudgetExceededError as exc:
                log.warning("signal_loop.llm_budget_exceeded", reason=str(exc))
                await alert_dispatcher.circuit_breaker_alert("llm_budget", str(exc))
                _signal_loop_error = str(exc)
            except LLMConsecutiveErrorsError as exc:
                log.warning("signal_loop.llm_consecutive_errors", reason=str(exc))
                await alert_dispatcher.circuit_breaker_alert("llm_errors", str(exc))
                _signal_loop_error = str(exc)
            except Exception as exc:
                import structlog
                structlog.get_logger("freqpred.cli.signal_loop").exception("signal_loop.error")
                _signal_loop_error = str(exc)

            if _signal_loop_error is not None:
                await runtime_telemetry.mark_error(
                    SERVICE_SIGNAL_LOOP,
                    _signal_loop_error,
                )

            await asyncio.sleep(config.signal.interval_seconds)

    tasks: list[asyncio.Task] = []

    async with KalshiClient(
        api_key=config.kalshi.api_key,
        base_url=config.kalshi.base_url,
        private_key_path=config.kalshi.private_key_path,
    ) as kalshi_client:
        if mode == "live":
            import structlog as _sl
            _log = _sl.get_logger("freqpred.cli")
            balance = await kalshi_client.get_balance()
            if balance < config.trading.bankroll_usd:
                _log.error(
                    "startup.balance_below_bankroll",
                    balance_usd=balance,
                    bankroll_usd=config.trading.bankroll_usd,
                )
                click.echo(
                    f"ERROR: Kalshi balance ${balance:.2f} is below configured "
                    f"bankroll ${config.trading.bankroll_usd:.2f}. Aborting.",
                    err=True,
                )
                await alert_dispatcher.circuit_breaker_alert(
                    "startup_balance",
                    f"Startup aborted: Kalshi balance ${balance:.2f} < "
                    f"bankroll ${config.trading.bankroll_usd:.2f}",
                )
                return
            _log.info(
                "startup.balance_ok",
                balance_usd=balance,
                bankroll_usd=config.trading.bankroll_usd,
            )
            risk_engine = RiskEngine(config.risk)
            order_manager = OrderManager(
                risk=risk_engine,
                session_factory=session_factory,
                bankroll=config.trading.bankroll_usd,
                mode="live",
                kalshi_client=kalshi_client,
                llm_client=llm_client,
                judgment_model=config.anthropic.judgment_model,
                runtime_telemetry=runtime_telemetry,
            )

        position_monitor._kalshi_client = kalshi_client

        # PositionWatcher runs in ALL modes — paper positions need the same
        # sub-second WS tick feed as live positions so TA/algo exits work.
        # Reconciliation inside PositionWatcher is already live-only.
        from freqpred.markets.position_watcher import PositionWatcher
        _ws_url = (
            config.kalshi.ws_demo_url
            if "demo" in config.kalshi.base_url.lower()
            else config.kalshi.ws_url
        )
        position_watcher = PositionWatcher(
            kalshi_client=kalshi_client,
            ws_url=_ws_url,
            session_factory=session_factory,
            position_monitor=position_monitor,
            order_manager=order_manager,
            runtime_telemetry=runtime_telemetry,
        )

        watcher = MarketWatcher(
            client=kalshi_client,
            session_factory=session_factory,
            polling_interval=config.kalshi.polling_interval_seconds,
            alert_dispatcher=alert_dispatcher,
            runtime_telemetry=runtime_telemetry,
        )
        # Register position commands here so order_manager is available for both modes.
        register_position_commands(
            cmd_handler=telegram_cmd_handler,
            session_factory=session_factory,
            config=config,
            mode=mode,
            order_manager=order_manager,
        )

        # Embed the API server inside run so it shares the live OrderManager.
        if config.dashboard.api_enabled:
            import freqpred.alerts.models     # noqa: F401
            import freqpred.ingestion.models  # noqa: F401
            import freqpred.llm.models        # noqa: F401
            import freqpred.rag.models        # noqa: F401
            import freqpred.signal.models     # noqa: F401
            import freqpred.strategy.models   # noqa: F401
            from freqpred.dashboard.api.app import create_app as _create_app  # noqa: PLC0415
            import uvicorn as _uvicorn  # noqa: PLC0415

            _dash_app = _create_app(
                session_factory=session_factory,
                daily_cap_usd=config.risk.max_daily_llm_spend_usd,
                risk_config=config.risk,
                bankroll_usd=config.trading.bankroll_usd,
                signal_pipeline=pipeline,
                order_manager=order_manager,
                runtime_telemetry=runtime_telemetry,
                kalshi_base_url=config.kalshi.base_url,
            )
            _dash_server = _uvicorn.Server(
                _uvicorn.Config(
                    _dash_app,
                    host=config.dashboard.host,
                    port=config.dashboard.port,
                    log_level="warning",
                )
            )
            tasks.append(asyncio.create_task(_dash_server.serve(), name="dashboard_api"))
            import structlog as _sl
            _sl.get_logger("freqpred.cli").info(
                "dashboard.api_started",
                host=config.dashboard.host,
                port=config.dashboard.port,
            )

        tasks.append(asyncio.create_task(watcher.run(), name="market_watcher"))
        tasks.append(
            asyncio.create_task(
                run_scheduler(
                    session_factory=session_factory,
                    embedder=embedder,
                    interval_seconds=config.ingestion.schedule_interval_seconds,
                    strategy=strategy,
                    llm_client=llm_client,
                    cheap_model=config.anthropic.cheap_model,
                    tavily_api_key=config.tavily.api_key,
                    tavily_daily_cap=config.tavily.daily_cap,
                    tavily_min_fetch_interval_hours=config.tavily.min_fetch_interval_hours,
                    newsapi_api_key=config.newsapi.api_key,
                    newsapi_enabled=config.newsapi.enabled,
                    newsapi_max_window_requests=config.newsapi.max_window_requests,
                    newsapi_min_fetch_interval_hours=config.newsapi.min_fetch_interval_hours,
                    guardian_api_key=config.guardian.api_key,
                    guardian_enabled=config.guardian.enabled,
                    guardian_daily_cap=config.guardian.daily_cap,
                    guardian_min_fetch_interval_hours=config.guardian.min_fetch_interval_hours,
                    telemetry=runtime_telemetry,
                ),
                name="ingestion_scheduler",
            )
        )
        tasks.append(
            asyncio.create_task(
                run_realtime_scheduler(
                    session_factory=session_factory,
                    embedder=embedder,
                    interval_seconds=config.ingestion.realtime_interval_seconds,
                    tv_chyron_enabled=config.ingestion.tv_chyron_enabled,
                    truthsocial_enabled=config.ingestion.truthsocial.enabled,
                    truthsocial_username=config.truthsocial.username,
                    truthsocial_password=config.truthsocial.password,
                    truthsocial_accounts=config.ingestion.truthsocial.accounts,
                    telemetry=runtime_telemetry,
                ),
                name="realtime_scheduler",
            )
        )

        import structlog as _sl
        _sl.get_logger("freqpred.cli").info(
            "startup.factbase_scheduler",
            allowlist=list(_factbase_allowlist),
            enabled=bool(_factbase_allowlist),
        )
        if _factbase_allowlist:
            tasks.append(
                asyncio.create_task(
                    run_factbase_scheduler(
                        session_factory=session_factory,
                        series_allowlist=_factbase_allowlist,
                        phrase_cache=phrase_cache,
                        llm_client=llm_client,
                        interval_seconds=300,
                        telemetry=runtime_telemetry,
                    ),
                    name="factbase_scheduler",
                )
            )

        tasks.append(asyncio.create_task(signal_loop(), name="signal_loop"))
        tasks.append(asyncio.create_task(position_monitor.run(), name="position_monitor"))
        if position_watcher is not None:
            tasks.append(
                asyncio.create_task(position_watcher.run(), name="position_watcher")
            )
        tasks.append(
            asyncio.create_task(telegram_cmd_handler.run(), name="telegram_commands")
        )

        from freqpred.metrics.reporting import run_digest_scheduler
        tasks.append(
            asyncio.create_task(
                run_digest_scheduler(
                    session_factory=session_factory,
                    llm_client=llm_client,
                    alert_dispatcher=alert_dispatcher,
                    digest_time=config.alerts.digest_time,
                    digest_timezone=config.alerts.digest_timezone,
                    trading_mode=mode,
                    bankroll=config.trading.bankroll_usd,
                    model=config.anthropic.cheap_model,
                ),
                name="digest_scheduler",
            )
        )
        from freqpred.metrics.scheduler import run_source_quality_scheduler
        tasks.append(
            asyncio.create_task(
                run_source_quality_scheduler(
                    session_factory=session_factory,
                    refresh_time=config.alerts.digest_time,
                    refresh_timezone=config.alerts.digest_timezone,
                    kalshi_client=kalshi_client,
                    telemetry=runtime_telemetry,
                ),
                name="source_quality_scheduler",
            )
        )
        tasks.append(
            asyncio.create_task(
                run_stale_service_watchdog(
                    session_factory=session_factory,
                    telemetry=runtime_telemetry,
                    alert_dispatcher=alert_dispatcher,
                    started_at=_process_started_at,
                ),
                name="stale_service_watchdog",
            )
        )
        from freqpred.ingestion.kalshi_changelog import run_changelog_monitor  # noqa: PLC0415
        tasks.append(
            asyncio.create_task(
                run_changelog_monitor(
                    session_factory=session_factory,
                    dispatcher=alert_dispatcher,
                    telemetry=runtime_telemetry,
                ),
                name="kalshi_changelog_monitor",
            )
        )

        async with session_factory() as _startup_session:
            _startup_state = await get_run_state(_startup_session)
            from freqpred.alerts.run_state import get_strategy_name  # noqa: PLC0415
            from freqpred.strategy.config_store import save_overrides  # noqa: PLC0415
            _prev_strategy = await get_strategy_name(_startup_session)
            if _prev_strategy is not None and _prev_strategy != strategy_name:
                log.info(
                    "startup.strategy_changed",
                    prev=_prev_strategy,
                    current=strategy_name,
                )
                await save_overrides(_startup_session, _prev_strategy, {})
            await set_strategy_name(_startup_session, strategy_name)
            await set_mode(_startup_session, mode)
        if _startup_state != "running":
            click.echo(
                f"\n*** WARNING: run_state='{_startup_state}' — "
                "signal loop is INACTIVE. Use /start on Telegram to resume. ***\n",
                err=True,
            )
        await alert_dispatcher.startup_alert(strategy_name, mode, _startup_state)

        click.echo(f"Running {len(tasks)} task(s). Press Ctrl+C to stop.")

        loop = asyncio.get_event_loop()
        loop.add_signal_handler(_signal.SIGTERM, lambda: [t.cancel() for t in tasks])

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            open_live_positions = 0
            if mode == "live":
                try:
                    from freqpred.markets.models import PositionRow
                    from sqlalchemy import func as _func
                    async with session_factory() as _shutdown_session:
                        _result = await _shutdown_session.execute(
                            select(_func.count()).select_from(PositionRow).where(
                                PositionRow.status == "open", PositionRow.mode == "live"
                            )
                        )
                        raw = _result.scalar_one()
                        open_live_positions = int(raw) if raw is not None else 0
                except Exception:
                    pass
            await alert_dispatcher.shutdown_alert(strategy_name, mode, open_live_positions)
            await engine.dispose()
            click.echo("Shutdown complete.")


@main.group()
def markets() -> None:
    """Manage and inspect Kalshi markets."""


@markets.command(name="list")
@click.option(
    "--category",
    default=None,
    help="Filter by Kalshi category string (e.g. Elections, Sports, World). Case-sensitive.",
)
@click.option(
    "--min-volume",
    default=None,
    type=float,
    help="Only show markets with volume_24h >= this value.",
)
@click.option(
    "--max-days",
    default=None,
    type=float,
    help="Only show markets closing within this many days.",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="Skip writing results to the database.",
)
@click.pass_context
def markets_list(
    ctx: click.Context,
    category: str | None,
    min_volume: float | None,
    max_days: float | None,
    no_db: bool,
) -> None:
    """Fetch active Kalshi markets and write them to the database."""
    config = ctx.obj["config"]
    asyncio.run(_markets_list(config, category=category, min_volume=min_volume, max_days=max_days, skip_db=no_db))


async def _markets_list(
    config: object,
    category: str | None,
    min_volume: float | None,
    max_days: float | None,
    skip_db: bool,
) -> None:
    from freqpred.db import make_engine, make_session_factory
    from freqpred.markets.kalshi import KalshiClient
    from freqpred.markets.repository import upsert_markets
    import freqpred.signal.models  # noqa: F401 — register SignalRow with SQLAlchemy mapper
    import freqpred.rag.models  # noqa: F401 — register DocumentMarketLinkRow with SQLAlchemy mapper

    async with KalshiClient(
        api_key=config.kalshi.api_key,
        base_url=config.kalshi.base_url,
        private_key_path=config.kalshi.private_key_path,
    ) as client:
        click.echo(
            f"Fetching markets from Kalshi"
            + (f" [category={category}]" if category else "")
            + " ..."
        )
        market_list = await client.list_markets(category=category)

    if not market_list:
        click.echo("No markets found.")
        return

    # Apply optional client-side filters
    from datetime import datetime, timezone
    now = datetime.now(tz=timezone.utc)
    if min_volume is not None:
        market_list = [m for m in market_list if m.volume_24h >= min_volume]
    if max_days is not None:
        market_list = [m for m in market_list if (m.close_time - now).total_seconds() / 86400 <= max_days]

    if not market_list:
        click.echo("No markets matched the filters.")
        return

    # Write to DB (pre-filter full list already written above if applicable)
    if not skip_db and config.database.url:
        engine = make_engine(config.database.url)
        session_factory = make_session_factory(engine)
        async with session_factory() as session:
            written = await upsert_markets(session, market_list)
        await engine.dispose()
        click.echo(f"Wrote {written} market(s) to database.")

    # Print table to stdout
    header = (
        f"{'TICKER':<30} {'CATEGORY':<20} {'SERIES':<14} "
        f"{'VOL_24H':>10} {'DAYS':>5} {'BID':>6} {'ASK':>6} {'MID':>6}  QUESTION"
    )
    click.echo(header)
    click.echo("-" * min(160, len(header) + 20))
    for m in market_list:
        days_to_close = (m.close_time - now).total_seconds() / 86400
        question_preview = m.question[:55] + "…" if len(m.question) > 55 else m.question
        click.echo(
            f"{m.id:<30} {m.category:<20} {(m.series_ticker or ''):<14} "
            f"{m.volume_24h:>10.0f} {days_to_close:>5.1f} "
            f"{m.yes_bid:>6.3f} {m.yes_ask:>6.3f} {m.mid_price:>6.3f}  {question_preview}"
        )
    click.echo(f"\nTotal: {len(market_list)} market(s)")


@main.group()
def ingestion() -> None:
    """Ingestion pipeline commands."""


@ingestion.command(name="run")
@click.option(
    "--category",
    default=None,
    help="Only process markets in this category (e.g. politics, economics).",
)
@click.option(
    "--limit",
    default=3,
    show_default=True,
    help="Maximum number of markets to process.",
)
@click.option(
    "--min-volume",
    default=0.0,
    show_default=True,
    help="Minimum 24h volume filter.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Generate catalysts but skip the news fetching step.",
)
@click.pass_context
def ingestion_run(
    ctx: click.Context,
    category: str | None,
    limit: int,
    min_volume: float,
    dry_run: bool,
) -> None:
    """Generate catalysts for selected markets then fetch targeted news.

    Pulls active markets from the DB, generates 3-5 targeted search queries
    per market (via Claude Haiku), then runs Tavily + NewsAPI + Reddit
    fetchers against those queries and stores results in the document store.

    Use --dry-run to generate and print catalysts without fetching news.
    """
    config = ctx.obj["config"]
    asyncio.run(_ingestion_run(config, category, limit, min_volume, dry_run))


async def _ingestion_run(
    config: object,
    category: str | None,
    limit: int,
    min_volume: float,
    dry_run: bool,
) -> None:
    import freqpred.ingestion.models  # noqa: F401
    import freqpred.metrics.models  # noqa: F401
    import freqpred.signal.models     # noqa: F401
    import freqpred.rag.models        # noqa: F401

    import anthropic
    from datetime import UTC, datetime

    from sqlalchemy import select

    from freqpred.db import make_engine, make_session_factory
    from freqpred.ingestion.catalyst_generator import CatalystGenerationError, generate_catalysts
    from freqpred.llm.client import LLMClient
    from freqpred.ingestion.models import CatalystQueryRow, CatalystRunRow
    from freqpred.ingestion.store import RawDocument, upsert_document
    from freqpred.markets.models import MarketRow
    from freqpred.rag.embedder import LocalEmbedder

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return

    anthropic_api_key = config.anthropic.api_key
    if not anthropic_api_key:
        click.echo("ERROR: ANTHROPIC_API_KEY not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)
    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=anthropic_api_key),
        session_factory,
        prompt_version="catalyst-v1",
    )

    embedder = LocalEmbedder()

    now = datetime.now(UTC)

    async with session_factory() as session:
        # Fetch non-closed markets from DB.
        stmt = select(MarketRow).where(MarketRow.close_time > now)
        if category:
            stmt = stmt.where(MarketRow.category == category)
        if min_volume > 0:
            stmt = stmt.where(MarketRow.volume_24h >= min_volume)
        stmt = stmt.order_by(MarketRow.volume_24h.desc()).limit(limit)

        result = await session.execute(stmt)
        market_rows = result.scalars().all()

    if not market_rows:
        click.echo("No markets found in DB. Run `freqpred markets list` first.")
        await engine.dispose()
        return

    click.echo(f"Processing {len(market_rows)} market(s)...")

    total_docs = 0

    for mrow in market_rows:
        from freqpred.markets.models import Market
        market = Market(
            id=mrow.id,
            platform=mrow.platform,
            question=mrow.question,
            category=mrow.category,
            status=mrow.status,
            result=mrow.result,
            close_time=mrow.close_time,
            yes_bid=mrow.yes_bid,
            yes_ask=mrow.yes_ask,
            mid_price=mrow.mid_price,
            last_price=mrow.last_price,
            volume_24h=mrow.volume_24h,
            open_interest=mrow.open_interest,
            yes_bid_size=mrow.yes_bid_size,
            yes_ask_size=mrow.yes_ask_size,
            last_fetched_at=mrow.last_fetched_at,
            price_updated_at=mrow.price_updated_at,
            metadata_fetched_at=mrow.metadata_fetched_at,
            current_signal_id=str(mrow.current_signal_id) if mrow.current_signal_id else None,
            metadata=dict(mrow.metadata_),
            open_time=mrow.open_time,
        )

        click.echo(f"\n{'─'*70}")
        click.echo(f"Market : {market.id}")
        click.echo(f"Question: {market.question}")
        click.echo(f"Category: {market.category}  |  Volume: {market.volume_24h:.0f}  |  Mid: {market.mid_price:.3f}")
        click.echo(f"Closes : {market.close_time.strftime('%Y-%m-%d')}")

        # Generate catalysts.
        async with session_factory() as session:
            try:
                run = await generate_catalysts(
                    market,
                    session,
                    llm_client,
                    embedder,
                    model=config.anthropic.cheap_model,
                )
                await session.commit()

                # Fetch the query texts we just wrote.
                q_result = await session.execute(
                    select(CatalystQueryRow).where(CatalystQueryRow.run_id == run.id)
                )
                query_rows = q_result.scalars().all()
                queries = [q.query_text for q in query_rows]
            except CatalystGenerationError as exc:
                click.echo(f"  ✗ Catalyst generation failed: {exc}", err=True)
                continue

        click.echo(f"\nCatalysts (generation {run.generation}):")
        for i, q in enumerate(queries, 1):
            click.echo(f"  {i}. {q}")

        if dry_run:
            click.echo("  (dry-run: skipping news fetch)")
            continue

        if not queries:
            continue

        # Run fetchers against each catalyst query.
        click.echo("\nFetching news...")
        from freqpred.ingestion.fetchers import tavily as tavily_fetcher
        from freqpred.ingestion.fetchers import newsapi as newsapi_fetcher
        from freqpred.ingestion.fetchers import reddit as reddit_fetcher
        from datetime import timedelta

        raw_docs: list[RawDocument] = []

        for query in queries:
            # Tavily
            if config.tavily.api_key:
                tavily_docs = await tavily_fetcher.fetch(
                    api_key=config.tavily.api_key,
                    query=query,
                    max_results=5,
                )
                raw_docs.extend(tavily_docs)

            # NewsAPI
            if config.newsapi.api_key:
                newsapi_docs = await newsapi_fetcher.fetch(
                    api_key=config.newsapi.api_key,
                    query=query,
                    from_date=datetime.now(UTC) - timedelta(days=7),
                    max_results=5,
                )
                raw_docs.extend(newsapi_docs)

            # Reddit — use category to pick subreddits
            subreddits = _subreddits_for_category(market.category)
            reddit_docs = await reddit_fetcher.fetch(
                subreddits=subreddits,
                query=query,
                limit=20,
            )
            raw_docs.extend(reddit_docs)

        click.echo(f"  Fetched {len(raw_docs)} raw document(s) across all sources.")

        if not raw_docs:
            continue

        # Upsert into document store.
        stored = 0
        skipped = 0
        async with session_factory() as session:
            for raw_doc in raw_docs:
                raw_doc.category = market.category
                try:
                    doc, _status = await upsert_document(
                        session,
                        embedder,
                        raw_doc,
                        llm_client=llm_client,
                        query_text=query,
                        market_question=market.question,
                        summary_model=config.anthropic.cheap_model,
                    )
                    stored += 1
                except Exception as exc:
                    click.echo(f"  ✗ Store error: {exc}", err=True)
                    skipped += 1
            await session.commit()

        click.echo(f"  Stored {stored} doc(s) ({skipped} errors).")
        total_docs += stored

    click.echo(f"\n{'═'*70}")
    click.echo(f"Done. Total documents stored: {total_docs}")
    await engine.dispose()


def _subreddits_for_category(category: str) -> list[str]:
    _MAP = {
        "politics":    ["politics", "PoliticalDiscussion", "neutralpolitics"],
        "technology":  ["technology", "MachineLearning", "singularity"],
        "economics":   ["economics", "investing", "stocks"],
        "fintech":     ["investing", "wallstreetbets", "stocks", "fintech"],
        "sports":      ["sports"],
        "crypto":      ["CryptoCurrency", "Bitcoin"],
        "climate":     ["climate", "environment"],
    }
    return _MAP.get(category.lower(), ["news"])


@main.group()
def signal() -> None:
    """Signal pipeline commands."""


@signal.command(name="analyze")
@click.option("--market-id", required=True, help="Kalshi market ID to analyze.")
@click.option("--force", is_flag=True, default=False, help="Bypass hash deduplication and force a new LLM call.")
@click.option("--strategy", "strategy_name", default="PoliticsEdgeStrategy", show_default=True, help="Strategy to load (determines factbase allowlist and other config).")
@click.pass_context
def signal_analyze(ctx: click.Context, market_id: str, force: bool, strategy_name: str) -> None:
    """One-shot signal analysis for a specific market."""
    config = ctx.obj["config"]
    asyncio.run(_signal_analyze(config, market_id, force=force, strategy_name=strategy_name))


async def _signal_analyze(config: object, market_id: str, *, force: bool = False, strategy_name: str = "PoliticsEdgeStrategy") -> None:
    import anthropic

    import freqpred.signal.models  # noqa: F401
    import freqpred.rag.models  # noqa: F401

    from sqlalchemy import select

    from freqpred.db import make_engine, make_session_factory
    from freqpred.llm.client import LLMClient
    from freqpred.markets.models import Market, MarketRow
    from freqpred.rag.embedder import LocalEmbedder
    from freqpred.signal.pipeline import SignalPipeline

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return
    if not config.anthropic.api_key:
        click.echo("ERROR: ANTHROPIC_API_KEY not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    try:
        async with session_factory() as session:
            result = await session.execute(
                select(MarketRow).where(MarketRow.id == market_id)
            )
            row = result.scalar_one_or_none()

        if row is None:
            click.echo(f"ERROR: Market {market_id!r} not found in DB. Run `freqpred markets list` first.", err=True)
            return

        market = Market(
            id=row.id,
            platform=row.platform,
            question=row.question,
            category=row.category,
            status=row.status,
            result=row.result,
            close_time=row.close_time,
            yes_bid=row.yes_bid,
            yes_ask=row.yes_ask,
            mid_price=row.mid_price,
            last_price=row.last_price,
            volume_24h=row.volume_24h,
            open_interest=row.open_interest,
            yes_bid_size=row.yes_bid_size,
            yes_ask_size=row.yes_ask_size,
            last_fetched_at=row.last_fetched_at,
            price_updated_at=row.price_updated_at,
            metadata_fetched_at=row.metadata_fetched_at,
            current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
            metadata=dict(row.metadata_),
            open_time=row.open_time,
            series_ticker=row.series_ticker,
        )

        click.echo(f"Analyzing: {market.question}")
        click.echo(f"Category : {market.category}  |  Mid: {market.mid_price:.3f}")

        embedder = LocalEmbedder()
        from freqpred.signal.llm import PROMPT_VERSION  # noqa: PLC0415
        llm_client = LLMClient(
            anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
            session_factory,
            prompt_version=PROMPT_VERSION,
        )
        from freqpred.strategy.loader import load_strategy as _load_strategy  # noqa: PLC0415
        _cli_strategy = _load_strategy(strategy_name)
        _cli_factbase_allowlist = frozenset(
            getattr(getattr(_cli_strategy, "config", None), "factbase_series_allowlist", [])
        )
        pipeline = SignalPipeline(
            session_factory=session_factory,
            embedder=embedder,
            llm_client=llm_client,
            model=config.anthropic.primary_model,
            top_k=config.signal.top_k_documents,
            factbase_series_allowlist=_cli_factbase_allowlist,
            max_scheduled_interval_hours=config.signal.max_scheduled_interval_hours,
        )

        signal = await pipeline.analyze(market, trigger="manual", force=force)

        if signal is None:
            click.echo("No new signal generated (evidence unchanged or LLM error).")
        else:
            click.echo(f"\nSignal ID  : {signal.id}")
            click.echo(f"Probability: {signal.estimated_probability:.3f}")
            click.echo(f"Edge       : {signal.edge:+.3f}")
            click.echo(f"Confidence : {signal.confidence:.2f}")
            click.echo(f"Direction  : {signal.direction}")
            click.echo(f"\nReasoning:\n{signal.reasoning}")
    finally:
        await engine.dispose()


@main.group()
def positions() -> None:
    """Inspect trading positions."""


@positions.command(name="list")
@click.option(
    "--status",
    type=click.Choice(["open", "closed", "all"]),
    default="all",
    show_default=True,
    help="Filter by position status.",
)
@click.option(
    "--limit",
    default=50,
    show_default=True,
    help="Maximum number of positions to display.",
)
@click.option(
    "--strategy",
    default=None,
    help="Filter by strategy name (exact match).",
)
@click.option(
    "--days",
    default=None,
    type=float,
    help="Only show positions entered within the last N days (e.g. 1 = last 24 hours).",
)
@click.pass_context
def positions_list(ctx: click.Context, status: str, limit: int, strategy: str | None, days: float | None) -> None:
    """Print positions from the database."""
    config = ctx.obj["config"]
    asyncio.run(_positions_list(config, status, limit, strategy, days))


async def _positions_list(
    config: object,
    status: str,
    limit: int,
    strategy: str | None,
    days: float | None,
) -> None:
    from datetime import UTC, datetime as _datetime, timedelta  # noqa: PLC0415
    import freqpred.signal.models  # noqa: F401
    import freqpred.rag.models     # noqa: F401

    from sqlalchemy import select

    from freqpred.db import make_engine, make_session_factory
    from freqpred.markets.models import PositionRow

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    try:
        async with session_factory() as session:
            stmt = select(PositionRow).order_by(PositionRow.entry_time.desc()).limit(limit)
            if status != "all":
                stmt = stmt.where(PositionRow.status == status)
            if strategy is not None:
                stmt = stmt.where(PositionRow.strategy_name == strategy)
            if days is not None:
                cutoff = _datetime.now(tz=UTC) - timedelta(days=days)
                stmt = stmt.where(PositionRow.entry_time >= cutoff)
            result = await session.execute(stmt)
            rows = result.scalars().all()
    finally:
        await engine.dispose()

    if not rows:
        click.echo("No positions found.")
        return

    now = _datetime.now(tz=UTC)

    def _fmt_held(r: "PositionRow") -> str:
        end = r.exit_time if r.exit_time is not None else now
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        start = r.entry_time
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        secs = max(0, int((end - start).total_seconds()))
        days, rem = divmod(secs, 86400)
        hours, rem = divmod(rem, 3600)
        mins = rem // 60
        if days > 0:
            return f"{days}d{hours:02d}h"
        if hours > 0:
            return f"{hours}h{mins:02d}m"
        return f"{mins}m"

    header = (
        f"{'ID':<38} {'MARKET':<28} {'STRATEGY':<20} {'DIR':<4} {'CTRCTS':>6} "
        f"{'ENTRY':>6} {'EDGE':>6} {'STATUS':<7} {'HELD':>7} {'MODE':<6} {'PNL':>8} {'PNL%':>7} {'MAE':>7} {'MFE':>7}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for r in rows:
        pnl_str = f"{r.pnl:+.4f}" if r.pnl is not None else "      -"
        pnl_pct_str = f"{r.pnl_pct:+.1%}" if r.pnl_pct is not None else "     -"
        edge_str = f"{r.signal_edge:+.3f}" if r.signal_edge is not None else "     -"
        mae_str = f"{r.mae:+.4f}" if r.mae is not None else "      -"
        mfe_str = f"{r.mfe:+.4f}" if r.mfe is not None else "      -"
        strategy_str = (r.strategy_name or "")[:20]
        click.echo(
            f"{str(r.id):<38} {r.market_id:<28} {strategy_str:<20} {r.direction:<4} "
            f"{r.contracts:>6} {r.entry_price:>6.3f} {edge_str:>6} {r.status:<7} "
            f"{_fmt_held(r):>7} {r.mode:<6} {pnl_str:>8} {pnl_pct_str:>7} {mae_str:>7} {mfe_str:>7}"
        )
    click.echo(f"\nTotal: {len(rows)} position(s)")


@positions.command(name="resolve")
@click.option("--position-id", required=True, help="UUID of the position to resolve.")
@click.option(
    "--resolution",
    type=click.Choice(["yes", "no"]),
    required=True,
    help="Market resolution outcome (yes = event happened, no = event did not happen).",
)
@click.pass_context
def positions_resolve(ctx: click.Context, position_id: str, resolution: str) -> None:
    """Close a position and calculate P&L based on market resolution."""
    config = ctx.obj["config"]
    asyncio.run(_positions_resolve(config, position_id, resolution))


async def _positions_resolve(config: object, position_id: str, resolution: str) -> None:
    import freqpred.signal.models  # noqa: F401
    import freqpred.rag.models     # noqa: F401

    import uuid as _uuid
    from sqlalchemy import select

    from freqpred.db import make_engine, make_session_factory
    from freqpred.markets.models import PositionRow
    from freqpred.trading import ledger

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    try:
        async with session_factory() as session:
            result = await session.execute(
                select(PositionRow).where(PositionRow.id == _uuid.UUID(position_id))
            )
            row = result.scalar_one_or_none()

        if row is None:
            click.echo(f"ERROR: Position {position_id!r} not found.", err=True)
            return

        if row.status == "closed":
            click.echo(f"Position {position_id} is already closed.", err=True)
            return

        # Determine exit price: YES contracts pay 1.0 if YES resolves, 0.0 otherwise.
        # NO contracts pay 1.0 if NO resolves (i.e., YES did NOT happen), 0.0 otherwise.
        yes_resolved = resolution == "yes"
        if row.direction == "YES":
            exit_price = 1.0 if yes_resolved else 0.0
        else:  # NO
            exit_price = 0.0 if yes_resolved else 1.0
        resolution_int = 1 if yes_resolved else 0

        async with session_factory() as session:
            position = await ledger.close_position(
                session,
                position_id,
                exit_price=exit_price,
                resolution=resolution_int,
            )

        pnl_str = f"{position.pnl:+.4f}" if position.pnl is not None else "N/A"
        pnl_pct_str = f"{position.pnl_pct:+.2%}" if position.pnl_pct is not None else "N/A"

        click.echo(f"Position resolved: {position.id}")
        click.echo(f"Direction  : {position.direction}")
        click.echo(f"Resolution : {resolution.upper()}")
        click.echo(f"Entry price: {position.entry_price:.4f}")
        click.echo(f"Exit price : {exit_price:.4f}")
        click.echo(f"Contracts  : {position.contracts}")
        click.echo(f"P&L        : {pnl_str} ({pnl_pct_str})")
    finally:
        await engine.dispose()


@main.group()
def metrics() -> None:
    """Calibration and performance metrics."""


@metrics.command(name="calibration")
@click.option("--days", type=int, default=None, help="Lookback window in days (e.g. 7, 30). Default: all time.")
@click.option(
    "--period",
    type=click.Choice(["day", "week", "month"]),
    default=None,
    help="Convenience alias: day=1, week=7, month=30. Mutually exclusive with --days.",
)
@click.pass_context
def metrics_calibration(ctx: click.Context, days: int | None, period: str | None) -> None:
    """Print Brier score, market baseline, and calibration buckets.

    Scores every signal against the final market result — not just traded markets.
    Use --days or --period to filter by when the signal was generated.
    """
    if days is not None and period is not None:
        raise click.UsageError("--days and --period are mutually exclusive.")
    _period_map = {"day": 1, "week": 7, "month": 30}
    lookback_days = days if days is not None else (_period_map[period] if period else None)
    config = ctx.obj["config"]
    asyncio.run(_metrics_calibration(config, lookback_days=lookback_days))


async def _metrics_calibration(config: object, lookback_days: int | None = None) -> None:
    import freqpred.signal.models  # noqa: F401
    import freqpred.rag.models     # noqa: F401

    from freqpred.db import make_engine, make_session_factory
    from freqpred.metrics.calibration import compute_calibration

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    try:
        async with session_factory() as session:
            report = await compute_calibration(session, lookback_days=lookback_days)
    finally:
        await engine.dispose()

    period_label = f"last {report.lookback_days}d" if report.lookback_days else "all time"
    if report.n_samples == 0:
        click.echo(f"No resolved signals yet ({period_label}).")
        return

    improvement = report.market_brier_score - report.brier_score
    direction = "better" if improvement > 0 else "worse"
    click.echo(f"Period: {period_label}  |  Signals scored: {report.n_samples}")
    click.echo(f"Brier Score:     {report.brier_score:.3f}  (market baseline: {report.market_brier_score:.3f})")
    click.echo(f"Improvement vs market: {improvement:+.3f} ({direction})")
    click.echo("")
    header = f"{'Probability Bucket':<22} {'Count':>6} {'Mean Est.':>10} {'Resolution Rate':>16}"
    click.echo(header)
    click.echo("-" * len(header))
    for b in report.buckets:
        if b.count == 0:
            continue
        click.echo(
            f"{b.lower:.2f} \u2013 {b.upper:.2f}              "
            f"{b.count:>6} {b.mean_estimated_prob:>10.3f} {b.actual_resolution_rate:>16.3f}"
        )


@metrics.command(name="source-calibration")
@click.option("--days", type=int, default=None, help="Lookback window in days. Default: all time.")
@click.option(
    "--period",
    type=click.Choice(["day", "week", "month"]),
    default=None,
    help="Convenience alias: day=1, week=7, month=30. Mutually exclusive with --days.",
)
@click.option(
    "--min-docs",
    type=int,
    default=50,
    show_default=True,
    help="Hide sources whose total document appearances are below this threshold.",
)
@click.pass_context
def metrics_source_calibration(
    ctx: click.Context, days: int | None, period: str | None, min_docs: int
) -> None:
    """Print weighted Brier score per document source name.

    For each resolved signal, the Brier loss is distributed across its evidence
    documents proportionally by source name.  The weighted average gives a
    per-source quality score — lower is better.

    Only signals that have at least one linked document are included.
    Use --min-docs 0 to show all sources including the long tail.
    """
    if days is not None and period is not None:
        raise click.UsageError("--days and --period are mutually exclusive.")
    _period_map = {"day": 1, "week": 7, "month": 30}
    lookback_days = days if days is not None else (_period_map[period] if period else None)
    config = ctx.obj["config"]
    asyncio.run(_metrics_source_calibration(config, lookback_days=lookback_days, min_docs=min_docs))


async def _metrics_source_calibration(
    config: object,
    lookback_days: int | None = None,
    min_docs: int = 50,
) -> None:
    import freqpred.signal.models  # noqa: F401
    import freqpred.rag.models     # noqa: F401

    from freqpred.db import make_engine, make_session_factory
    from freqpred.metrics.calibration import compute_calibration, compute_source_brier_scores

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    try:
        async with session_factory() as session:
            scores = await compute_source_brier_scores(
                session, lookback_days=lookback_days, min_docs=min_docs
            )
            calibration = await compute_calibration(session, lookback_days=lookback_days)
    finally:
        await engine.dispose()

    period_label = f"last {lookback_days}d" if lookback_days else "all time"
    min_docs_label = f", min {min_docs} doc appearances" if min_docs > 0 else ""
    if not scores:
        click.echo(
            f"No qualifying sources ({period_label}{min_docs_label}). "
            "Try --min-docs 0 to include all sources."
        )
        return

    overall = calibration.brier_score
    click.echo(f"Source-weighted Brier scores ({period_label}{min_docs_label})  — lower is better")
    click.echo(f"Overall Brier: {overall:.4f}  (+ above overall = hurting, - = helping)")
    click.echo("")
    header = f"{'Source Name':<32} {'Wtd Brier':>10} {'vs Overall':>11} {'Signals':>8} {'Doc Uses':>9}"
    click.echo(header)
    click.echo("-" * len(header))
    for s in scores:
        delta = s.weighted_brier_score - overall
        click.echo(
            f"{s.source_name:<32} {s.weighted_brier_score:>10.4f}"
            f" {delta:>+11.4f} {s.n_signals:>8} {s.total_doc_appearances:>9}"
        )


@main.group()
def report() -> None:
    """Reporting commands."""


@report.command(name="digest")
@click.option(
    "--send",
    is_flag=True,
    default=False,
    help="Send digest via Telegram/Discord alert (requires T23).",
)
@click.option(
    "--mode",
    default="paper",
    show_default=True,
    type=click.Choice(["paper", "live", "signal-only"]),
    help="Trading mode to display in digest.",
)
@click.pass_context
def report_digest(ctx: click.Context, send: bool, mode: str) -> None:
    """Generate and print a daily digest summary."""
    config = ctx.obj["config"]
    asyncio.run(_report_digest(config, send=send, trading_mode=mode))


async def _report_digest(config: object, *, send: bool, trading_mode: str = "paper") -> None:
    import anthropic

    import freqpred.ingestion.models  # noqa: F401
    import freqpred.metrics.models  # noqa: F401
    import freqpred.signal.models     # noqa: F401
    import freqpred.rag.models        # noqa: F401

    from freqpred.db import make_engine, make_session_factory
    from freqpred.llm.client import LLMClient
    from freqpred.metrics.reporting import generate_daily_digest

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return
    if not config.anthropic.api_key:
        click.echo("ERROR: ANTHROPIC_API_KEY not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        session_factory,
        prompt_version="digest-v1",
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
    )

    try:
        async with session_factory() as session:
            digest = await generate_daily_digest(
                session,
                llm_client,
                trading_mode=trading_mode,
                model=config.anthropic.cheap_model,
            )
    finally:
        await engine.dispose()

    click.echo(digest)

    if send:
        click.echo("\n(--send flag set but alert integrations not yet implemented; T23)", err=True)


@main.group()
def alerts() -> None:
    """Alert channel commands."""


@alerts.command(name="test")
@click.option(
    "--channel",
    type=click.Choice(["telegram", "discord", "all"]),
    default="all",
    show_default=True,
    help="Which alert channel(s) to test.",
)
@click.pass_context
def alerts_test(ctx: click.Context, channel: str) -> None:
    """Send a test message to verify alert credentials."""
    config = ctx.obj["config"]
    asyncio.run(_alerts_test(config, channel))


async def _alerts_test(config: object, channel: str) -> None:
    from freqpred.alerts.telegram import TelegramSender
    from freqpred.alerts.discord import DiscordSender

    test_msg = "freqpred alert test — if you see this, alerts are working."

    if channel in ("telegram", "all"):
        sender = TelegramSender(
            bot_token=config.alerts.telegram_bot_token,
            chat_id=config.alerts.telegram_chat_id,
        )
        try:
            await sender.send(test_msg)
            click.echo("Telegram: OK")
        except Exception as exc:
            click.echo(f"Telegram: FAILED — {exc}", err=True)

    if channel in ("discord", "all"):
        sender = DiscordSender(webhook_url=config.alerts.discord_webhook_url)
        try:
            await sender.send(test_msg)
            click.echo("Discord: OK")
        except Exception as exc:
            click.echo(f"Discord: FAILED — {exc}", err=True)


@main.command()
@click.pass_context
def dashboard(ctx: click.Context) -> None:
    """Start the Vite dev server for dashboard UI development.

    The API server runs inside `freqpred run`. Start that first, then run this
    command in a separate terminal for hot-reload UI at http://localhost:5173.
    """
    config = ctx.obj["config"]
    asyncio.run(_dashboard(config.dashboard.port))


async def _dashboard(api_port: int) -> None:
    import os

    ui_dir = Path(__file__).parent / "dashboard" / "ui"
    click.echo("Starting Vite dev server on http://localhost:5173")
    click.echo(f"Requires: freqpred run (serves the API on port {api_port})")
    vite_env = os.environ.copy()
    vite_env["FREQPRED_API_PORT"] = str(api_port)
    vite_proc = await asyncio.create_subprocess_exec(
        "npm",
        "run",
        "dev",
        cwd=str(ui_dir),
        env=vite_env,
    )
    try:
        await vite_proc.wait()
    finally:
        if vite_proc.returncode is None:
            vite_proc.terminate()
            await vite_proc.wait()


@main.group()
def db() -> None:
    """Database management commands."""


@db.command(name="migrate")
def db_migrate() -> None:
    """Apply all pending Alembic migrations (upgrade head)."""
    import subprocess
    click.echo("Running: alembic upgrade head")
    result = subprocess.run(["uv", "run", "alembic", "upgrade", "head"])  # noqa: S603
    raise SystemExit(result.returncode)
