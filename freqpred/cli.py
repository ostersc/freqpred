"""freqpred CLI entry point."""
from __future__ import annotations

import asyncio
import logging

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

    # Console handler — colours on
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processor=structlog.dev.ConsoleRenderer(colors=True),
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
                processor=structlog.dev.ConsoleRenderer(colors=False),
            )
        )
        root.addHandler(file_handler)

    # Suppress chatty libraries unless debugging
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

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
    import anthropic

    import freqpred.ingestion.models  # noqa: F401
    import freqpred.signal.models  # noqa: F401
    import freqpred.rag.models  # noqa: F401

    from freqpred.db import make_engine, make_session_factory
    from freqpred.ingestion.scheduler import run_scheduler
    from freqpred.llm.client import LLMClient
    from freqpred.markets.kalshi import KalshiClient
    from freqpred.markets.models import Market, MarketRow
    from freqpred.markets.watcher import MarketWatcher
    from freqpred.rag.embedder import LocalEmbedder
    from freqpred.signal.pipeline import SignalPipeline
    from freqpred.strategy.loader import load_strategy

    from sqlalchemy import select

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return
    if not config.anthropic.api_key:
        click.echo("ERROR: ANTHROPIC_API_KEY not configured.", err=True)
        return

    strategy = load_strategy(strategy_name)
    click.echo(f"Loaded strategy: {strategy.config.name}")
    click.echo(f"Starting freqpred | strategy={strategy_name} | mode={mode}")

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    embedder = LocalEmbedder()
    llm_client = LLMClient(
        anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
        session_factory,
        prompt_version="signal-v1",
        daily_spend_cap_usd=config.risk.max_daily_llm_spend_usd,
    )
    pipeline = SignalPipeline(
        session_factory=session_factory,
        embedder=embedder,
        llm_client=llm_client,
        top_k=config.signal.top_k_documents,
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
    await alert_dispatcher.send(f"freqpred started | strategy={strategy_name} | mode={mode}")

    telegram_cmd_handler = TelegramCommandHandler(
        bot_token=config.alerts.telegram_bot_token,
        authorized_users=config.alerts.telegram_authorized_users,
    )

    import freqpred.alerts.models  # noqa: F401 — register RunStateRow
    from freqpred.alerts.command_handlers import register_system_commands
    from freqpred.alerts.metrics_handlers import register_metrics_commands
    from freqpred.alerts.position_handlers import register_position_commands
    from freqpred.alerts.run_state import get_run_state

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
    register_position_commands(
        cmd_handler=telegram_cmd_handler,
        session_factory=session_factory,
        config=config,
        mode=mode,
    )

    from freqpred.trading.risk import RiskEngine, TradingCircuitBreakerError
    from freqpred.trading.order_manager import OrderManager

    order_manager = None
    if mode == "paper":
        risk_engine = RiskEngine(config.risk)
        order_manager = OrderManager(
            risk=risk_engine,
            session_factory=session_factory,
            bankroll=config.trading.bankroll_usd,
            mode="paper",
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
    )

    async def signal_loop() -> None:
        import structlog
        log = structlog.get_logger("freqpred.cli.signal_loop")
        log.info("signal_loop.started")
        while True:
            try:
                # Check run-loop state; skip cycle if paused/stopped.
                async with session_factory() as rs_session:
                    run_state = await get_run_state(rs_session)

                if run_state == "stopped":
                    log.debug("signal_loop.skipped", reason="stopped")
                    await asyncio.sleep(config.signal.interval_seconds)
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
                        liquidity=row.liquidity,
                        last_fetched_at=row.last_fetched_at,
                        price_updated_at=row.price_updated_at,
                        metadata_fetched_at=row.metadata_fetched_at,
                        current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
                        metadata=dict(row.metadata_),
                        open_time=row.open_time,
                    )
                    for row in market_rows
                ]

                interesting = strategy.filter_markets(markets)
                log.info("signal_loop.cycle", total_markets=len(markets), selected=len(interesting))

                # Circuit breaker check at the top of each cycle
                circuit_breaker_active = False
                if order_manager is not None:
                    try:
                        async with session_factory() as cb_session:
                            await order_manager._risk.check_circuit_breakers(
                                cb_session, order_manager._bankroll, mode=order_manager._mode
                            )
                    except TradingCircuitBreakerError as exc:
                        log.warning("signal_loop.circuit_breaker_fired", reason=str(exc))
                        circuit_breaker_active = True
                        await alert_dispatcher.circuit_breaker_alert(str(exc))

                for market in interesting:
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
                            else:
                                strategy.on_order_failed(market)
            except asyncio.CancelledError:
                raise
            except Exception:
                import structlog
                structlog.get_logger("freqpred.cli.signal_loop").exception("signal_loop.error")

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
                    f"Startup aborted: Kalshi balance ${balance:.2f} < "
                    f"bankroll ${config.trading.bankroll_usd:.2f}"
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
            )

        position_monitor._kalshi_client = kalshi_client

        watcher = MarketWatcher(
            client=kalshi_client,
            session_factory=session_factory,
            polling_interval=config.kalshi.polling_interval_seconds,
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
                    tavily_api_key=config.tavily.api_key,
                    newsapi_api_key=config.newsapi.api_key,
                    newsapi_enabled=config.newsapi.enabled,
                    newsapi_max_window_requests=config.newsapi.max_window_requests,
                    truthsocial_enabled=config.ingestion.truthsocial.enabled,
                    truthsocial_username=config.truthsocial.username,
                    truthsocial_password=config.truthsocial.password,
                    truthsocial_accounts=config.ingestion.truthsocial.accounts,
                ),
                name="ingestion_scheduler",
            )
        )

        tasks.append(asyncio.create_task(signal_loop(), name="signal_loop"))
        tasks.append(asyncio.create_task(position_monitor.run(), name="position_monitor"))
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
                ),
                name="digest_scheduler",
            )
        )

        click.echo(f"Running {len(tasks)} task(s). Press Ctrl+C to stop.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await engine.dispose()
            click.echo("Shutdown complete.")


@main.group()
def markets() -> None:
    """Manage and inspect Kalshi markets."""


@markets.command(name="list")
@click.option(
    "--category",
    default=None,
    help="Filter by category (e.g. politics, technology).",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="Skip writing results to the database.",
)
@click.pass_context
def markets_list(ctx: click.Context, category: str | None, no_db: bool) -> None:
    """Fetch active Kalshi markets and write them to the database."""
    config = ctx.obj["config"]
    asyncio.run(_markets_list(config, category=category, skip_db=no_db))


async def _markets_list(config: object, category: str | None, skip_db: bool) -> None:
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

    # Write to DB
    if not skip_db and config.database.url:
        engine = make_engine(config.database.url)
        session_factory = make_session_factory(engine)
        async with session_factory() as session:
            written = await upsert_markets(session, market_list)
        await engine.dispose()
        click.echo(f"Wrote {written} market(s) to database.")

    # Print table to stdout
    header = f"{'TICKER':<30} {'CATEGORY':<14} {'BID':>6} {'ASK':>6} {'MID':>6}  QUESTION"
    click.echo(header)
    click.echo("-" * min(120, len(header) + 40))
    for m in market_list:
        question_preview = m.question[:60] + "…" if len(m.question) > 60 else m.question
        click.echo(
            f"{m.id:<30} {m.category:<14} "
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
            liquidity=mrow.liquidity,
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
                run = await generate_catalysts(market, session, llm_client, embedder)
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
                    doc = await upsert_document(session, embedder, raw_doc)
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
@click.pass_context
def signal_analyze(ctx: click.Context, market_id: str) -> None:
    """One-shot signal analysis for a specific market."""
    config = ctx.obj["config"]
    asyncio.run(_signal_analyze(config, market_id))


async def _signal_analyze(config: object, market_id: str) -> None:
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
            liquidity=row.liquidity,
            last_fetched_at=row.last_fetched_at,
            price_updated_at=row.price_updated_at,
            metadata_fetched_at=row.metadata_fetched_at,
            current_signal_id=str(row.current_signal_id) if row.current_signal_id else None,
            metadata=dict(row.metadata_),
            open_time=row.open_time,
        )

        click.echo(f"Analyzing: {market.question}")
        click.echo(f"Category : {market.category}  |  Mid: {market.mid_price:.3f}")

        embedder = LocalEmbedder()
        llm_client = LLMClient(
            anthropic.AsyncAnthropic(api_key=config.anthropic.api_key),
            session_factory,
            prompt_version="signal-v1",
        )
        pipeline = SignalPipeline(
            session_factory=session_factory,
            embedder=embedder,
            llm_client=llm_client,
            top_k=config.signal.top_k_documents,
        )

        signal = await pipeline.analyze(market, trigger="manual")

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
@click.pass_context
def positions_list(ctx: click.Context, status: str, limit: int) -> None:
    """Print positions from the database."""
    config = ctx.obj["config"]
    asyncio.run(_positions_list(config, status, limit))


async def _positions_list(config: object, status: str, limit: int) -> None:
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
            result = await session.execute(stmt)
            rows = result.scalars().all()
    finally:
        await engine.dispose()

    if not rows:
        click.echo("No positions found.")
        return

    header = (
        f"{'ID':<38} {'MARKET':<28} {'STRATEGY':<20} {'DIR':<4} {'CTRCTS':>6} "
        f"{'ENTRY':>6} {'EDGE':>6} {'STATUS':<7} {'MODE':<6} {'PNL':>8} {'MAE':>7} {'MFE':>7}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for r in rows:
        pnl_str = f"{r.pnl:+.4f}" if r.pnl is not None else "      -"
        edge_str = f"{r.signal_edge:+.3f}" if r.signal_edge is not None else "     -"
        mae_str = f"{r.mae:+.4f}" if r.mae is not None else "      -"
        mfe_str = f"{r.mfe:+.4f}" if r.mfe is not None else "      -"
        strategy_str = (r.strategy_name or "")[:20]
        click.echo(
            f"{str(r.id):<38} {r.market_id:<28} {strategy_str:<20} {r.direction:<4} "
            f"{r.contracts:>6} {r.entry_price:>6.3f} {edge_str:>6} {r.status:<7} "
            f"{r.mode:<6} {pnl_str:>8} {mae_str:>7} {mfe_str:>7}"
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
@click.pass_context
def metrics_calibration(ctx: click.Context) -> None:
    """Print Brier score, naive baseline, and calibration buckets."""
    config = ctx.obj["config"]
    asyncio.run(_metrics_calibration(config))


async def _metrics_calibration(config: object) -> None:
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
            report = await compute_calibration(session)
    finally:
        await engine.dispose()

    if report.n_samples == 0:
        click.echo("No resolved positions yet.")
        return

    improvement = report.naive_brier_score - report.brier_score
    direction = "better" if improvement > 0 else "worse"
    click.echo(f"Brier Score:     {report.brier_score:.3f}  (naive baseline: {report.naive_brier_score:.3f})")
    click.echo(f"Improvement vs baseline: {improvement:+.3f} ({direction})")
    click.echo(f"Samples: {report.n_samples}")
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
            digest = await generate_daily_digest(session, llm_client, trading_mode=trading_mode)
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
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind.")
@click.option("--port", default=8000, show_default=True, help="Port to listen on.")
@click.option("--mode", default="paper", type=click.Choice(["paper", "live"]), show_default=True, help="Trading mode to display.")
@click.pass_context
def dashboard(ctx: click.Context, host: str, port: int, mode: str) -> None:
    """Start the dashboard API server (read-only JSON API)."""
    config = ctx.obj["config"]
    asyncio.run(_dashboard(config, host, port, mode))


async def _dashboard(config: object, host: str, port: int, mode: str = "paper") -> None:
    import uvicorn

    import freqpred.ingestion.models  # noqa: F401
    import freqpred.llm.models        # noqa: F401
    import freqpred.rag.models        # noqa: F401
    import freqpred.signal.models     # noqa: F401

    from freqpred.dashboard.api.app import create_app
    from freqpred.db import make_engine, make_session_factory

    if not config.database.url:
        click.echo("ERROR: DATABASE_URL not configured.", err=True)
        return

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    app = create_app(
        session_factory=session_factory,
        daily_cap_usd=config.risk.max_daily_llm_spend_usd,
        mode=mode,
    )

    click.echo(f"Starting dashboard on http://{host}:{port}")
    server_config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(server_config)
    try:
        await server.serve()
    finally:
        await engine.dispose()


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
