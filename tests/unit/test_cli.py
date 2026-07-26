"""Unit tests for freqpred CLI commands (T13)."""
from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from freqpred.cli import main

if TYPE_CHECKING:
    from freqpred.signal.models import Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=UTC)


def _make_config(
    database_url: str = "postgresql+asyncpg://x:y@localhost/test",
    anthropic_api_key: str = "sk-test",
) -> MagicMock:
    cfg = MagicMock()
    cfg.log_level = "INFO"
    cfg.log_file = ""        # disable file logging in tests
    cfg.log_backup_days = 14
    cfg.database.url = database_url
    cfg.anthropic.api_key = anthropic_api_key
    cfg.kalshi.api_key = ""
    cfg.kalshi.base_url = "https://api.kalshi.com"
    cfg.kalshi.private_key_path = ""
    cfg.kalshi.polling_interval_seconds = 300
    cfg.ingestion.schedule_interval_seconds = 1800
    cfg.ingestion.realtime_interval_seconds = 300
    cfg.tavily.api_key = ""
    cfg.newsapi.api_key = ""
    cfg.signal.top_k_documents = 10
    cfg.signal.interval_seconds = 1800
    cfg.trading.bankroll_usd = 1000.0
    cfg.risk.max_daily_llm_spend_usd = 10.0
    return cfg


def _make_fake_signal(direction: str = "YES") -> Signal:
    from freqpred.signal.models import Signal
    return Signal(
        id=str(uuid.uuid4()),
        market_id="MKT-1",
        estimated_probability=0.65,
        confidence=0.82,
        edge=0.23,
        market_mid_at_signal=0.42,
        direction=direction,
        reasoning="test reasoning",
        sources=[],
        retrieval_hash="abc" * 21,
        model_used="claude-sonnet-4-6",
        prompt_version="v1",
        trigger="scheduled",
        created_at=NOW,
        raw_context="{}",
    )


def _make_position_row(market_id: str = "MKT-1", direction: str = "YES") -> MagicMock:
    """A PositionRow stand-in that survives ledger._row_to_position()."""
    row = MagicMock()
    row.id = uuid.uuid4()
    row.market_id = market_id
    row.signal_id = uuid.uuid4()
    row.strategy_name = "TestStrategy"
    row.strategy_version = "1"
    row.signal_confidence = 0.85
    row.signal_edge = 0.15
    row.signal_estimated_prob = 0.70
    row.direction = direction
    row.contracts = 3
    row.entry_price = 0.40
    row.entry_time = NOW
    row.mode = "paper"
    row.status = "open"
    row.exit_price = None
    row.exit_time = None
    row.exit_reason = None
    row.resolution = None
    row.pnl = None
    row.pnl_pct = None
    row.mae = None
    row.mfe = None
    row.exchange_order_id = None
    row.entry_fee_usd = 0.0
    row.exit_order_id = None
    row.exit_fee_usd = 0.0
    row.exit_requested_contracts = None
    row.exit_filled_contracts = None
    row.realized_pnl_accumulator = 0.0
    row.exchange_stoploss_order_id = None
    return row


def _make_run_mocks(
    market_row: MagicMock,
    signal: Signal | None,
    position_rows: list[MagicMock] | None = None,
):
    """Build the full set of mocks needed to run _run_main for one signal loop cycle."""
    # DB session returns the market row
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [market_row]
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    # get_run_state uses scalar_one_or_none; None → default "running" state
    mock_result.scalar_one_or_none.return_value = None
    mock_result.all.return_value = []

    # The open-positions query selects PositionRow entities, same call shape as
    # the markets query, so route by target table rather than call order.
    pos_scalars = MagicMock()
    pos_scalars.all.return_value = list(position_rows or [])
    pos_result = MagicMock()
    pos_result.scalars.return_value = pos_scalars
    pos_result.scalar_one_or_none.return_value = None
    pos_result.all.return_value = list(position_rows or [])

    async def _execute(stmt, *args, **kwargs):
        if "FROM positions" in str(stmt):
            return pos_result
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=_execute)
    ctx_mgr = MagicMock()
    ctx_mgr.__aenter__ = AsyncMock(return_value=mock_session)
    ctx_mgr.__aexit__ = AsyncMock(return_value=False)
    mock_factory = MagicMock(return_value=ctx_mgr)

    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock()

    # Pipeline returns the signal (or None)
    mock_pipeline_instance = AsyncMock()
    mock_pipeline_instance.analyze = AsyncMock(return_value=signal)
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)

    # Strategy passes all markets through
    mock_strategy = MagicMock()
    mock_strategy.config.name = "TestStrategy"
    mock_strategy.config.min_edge = 0.10
    mock_strategy.filter_markets = MagicMock(side_effect=lambda markets: markets)

    # KalshiClient as async context manager
    mock_kalshi_ctx = AsyncMock()
    mock_kalshi_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    mock_kalshi_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_kalshi_cls = MagicMock(return_value=mock_kalshi_ctx)

    return {
        "factory": mock_factory,
        "engine": mock_engine,
        "pipeline_cls": mock_pipeline_cls,
        "pipeline_instance": mock_pipeline_instance,
        "strategy": mock_strategy,
        "kalshi_cls": mock_kalshi_cls,
        "session_result": mock_result,
        "position_result": pos_result,
    }


def _make_market_row(market_id: str = "MKT-1") -> MagicMock:
    row = MagicMock()
    row.id = market_id
    row.platform = "kalshi"
    row.question = "Will X happen?"
    row.category = "politics"
    row.close_time = NOW + timedelta(days=14)
    row.yes_bid = 0.40
    row.yes_ask = 0.44
    row.mid_price = 0.42
    row.volume_24h = 1000.0
    row.open_interest = 500.0
    row.last_fetched_at = NOW
    row.price_updated_at = NOW
    row.metadata_fetched_at = NOW
    row.current_signal_id = None
    row.metadata_ = {}
    return row


# ---------------------------------------------------------------------------
# `freqpred --help`
# ---------------------------------------------------------------------------


class TestHelp:
    def test_main_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "freqpred" in result.output.lower()

    def test_run_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.output
        assert "--mode" in result.output

    def test_markets_list_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["markets", "list", "--help"])
        assert result.exit_code == 0
        assert "--category" in result.output

    def test_signal_analyze_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["signal", "analyze", "--help"])
        assert result.exit_code == 0
        assert "--market-id" in result.output

    def test_db_migrate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["db", "migrate", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# `freqpred run` — validates config and loads strategy
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_exits_cleanly_on_keyboard_interrupt(self) -> None:
        """run should catch KeyboardInterrupt from asyncio.run."""
        runner = CliRunner()
        config = _make_config()

        with patch("freqpred.cli.load_config", return_value=config), \
             patch("freqpred.db.run_migrations"), \
             patch("freqpred.cli._run_main", new_callable=AsyncMock, side_effect=KeyboardInterrupt):
            result = runner.invoke(main, ["run", "--strategy", "ConservativeDefault", "--mode", "signal-only"])
        # KeyboardInterrupt is caught; exit code should be 0
        assert result.exit_code == 0

    def test_run_requires_strategy_option(self) -> None:
        runner = CliRunner()
        with patch("freqpred.cli.load_config", return_value=_make_config()):
            result = runner.invoke(main, ["run"])
        assert result.exit_code != 0
        assert "strategy" in result.output.lower() or "missing" in result.output.lower()

    def test_run_rejects_invalid_mode(self) -> None:
        runner = CliRunner()
        with patch("freqpred.cli.load_config", return_value=_make_config()):
            result = runner.invoke(main, ["run", "--strategy", "ConservativeDefault", "--mode", "invalid"])
        assert result.exit_code != 0

    def test_run_aborts_without_database_url(self) -> None:
        runner = CliRunner()
        config = _make_config(database_url="")

        async def fake_run(cfg, strat, mode):
            from freqpred.cli import _run_main
            await _run_main(cfg, strat, mode)

        with patch("freqpred.cli.load_config", return_value=config):
            result = runner.invoke(
                main,
                ["run", "--strategy", "ConservativeDefault", "--mode", "signal-only"],
            )
        # Should not raise — will call _run_main which prints error and returns
        # (we don't block on asyncio details, just verify no unhandled exception)
        assert result.exit_code == 0  # click doesn't set non-zero for early-return

    def test_default_mode_is_paper(self) -> None:
        runner = CliRunner()
        config = _make_config()
        captured_mode: list[str] = []

        async def capture(cfg, strat, mode):
            captured_mode.append(mode)

        with patch("freqpred.cli.load_config", return_value=config), \
             patch("freqpred.db.run_migrations"), \
             patch("freqpred.cli._run_main", new=capture):
            runner.invoke(main, ["run", "--strategy", "ConservativeDefault"])

        assert captured_mode == ["paper"]


# ---------------------------------------------------------------------------
# `_run_main` — async internals
# ---------------------------------------------------------------------------


class TestRunMainAsync:
    @pytest.mark.asyncio
    async def test_aborts_without_database_url(self) -> None:
        from freqpred.cli import _run_main
        config = _make_config(database_url="")
        # Should return early without raising
        await _run_main(config, "ConservativeDefault", "signal-only")

    @pytest.mark.asyncio
    async def test_aborts_without_anthropic_key(self) -> None:
        from freqpred.cli import _run_main
        config = _make_config(anthropic_api_key="")
        await _run_main(config, "ConservativeDefault", "signal-only")

    @pytest.mark.asyncio
    async def test_invalid_strategy_raises(self) -> None:
        from freqpred.cli import _run_main
        config = _make_config()
        with pytest.raises(ValueError, match="Unknown strategy"):
            await _run_main(config, "NoSuchStrategy", "signal-only")


# ---------------------------------------------------------------------------
# `freqpred signal analyze` — one-shot analysis
# ---------------------------------------------------------------------------


class TestSignalAnalyzeCommand:
    def test_requires_market_id(self) -> None:
        runner = CliRunner()
        with patch("freqpred.cli.load_config", return_value=_make_config()):
            result = runner.invoke(main, ["signal", "analyze"])
        assert result.exit_code != 0

    def test_prints_signal_on_success(self) -> None:
        runner = CliRunner()
        config = _make_config()

        with patch("freqpred.cli.load_config", return_value=config), \
             patch("freqpred.cli._signal_analyze", new_callable=AsyncMock) as mock_analyze:
            result = runner.invoke(main, ["signal", "analyze", "--market-id", "MKT-1"])
        mock_analyze.assert_called_once_with(config, "MKT-1", force=False, strategy_name="PoliticsEdgeStrategy")
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# `_signal_analyze` — async internals
# ---------------------------------------------------------------------------


class TestSignalAnalyzeAsync:
    @pytest.mark.asyncio
    async def test_aborts_without_database_url(self) -> None:
        from freqpred.cli import _signal_analyze
        config = _make_config(database_url="")
        # Should return early without raising
        await _signal_analyze(config, "MKT-1")

    @pytest.mark.asyncio
    async def test_prints_error_when_market_not_found(self) -> None:
        from freqpred.cli import _signal_analyze
        config = _make_config()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        # Imports inside _signal_analyze use `from freqpred.db import ...`,
        # so we patch at the source module, not at freqpred.cli.
        with patch("freqpred.db.make_engine", return_value=mock_engine), \
             patch("freqpred.db.make_session_factory", return_value=mock_factory), \
             patch("freqpred.rag.embedder.LocalEmbedder"), \
             patch("freqpred.llm.client.LLMClient"), \
             patch("freqpred.signal.pipeline.SignalPipeline"), \
             patch("anthropic.AsyncAnthropic"):
            # Should return without raising
            await _signal_analyze(config, "MISSING-MKT")

    @pytest.mark.asyncio
    async def test_calls_pipeline_analyze_on_found_market(self) -> None:
        from freqpred.cli import _signal_analyze
        config = _make_config()

        row = _make_market_row("MKT-TEST")

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = row
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_pipeline_instance = AsyncMock()
        mock_pipeline_instance.analyze = AsyncMock(return_value=None)
        mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)

        with patch("freqpred.db.make_engine", return_value=mock_engine), \
             patch("freqpred.db.make_session_factory", return_value=mock_factory), \
             patch("freqpred.rag.embedder.LocalEmbedder"), \
             patch("freqpred.llm.client.LLMClient"), \
             patch("freqpred.signal.pipeline.SignalPipeline", mock_pipeline_cls), \
             patch("anthropic.AsyncAnthropic"):
            await _signal_analyze(config, "MKT-TEST")

        mock_pipeline_instance.analyze.assert_called_once()
        call_kwargs = mock_pipeline_instance.analyze.call_args
        # trigger="manual" may be positional or keyword
        assert "manual" in (call_kwargs.args + tuple(call_kwargs.kwargs.values()))


# ---------------------------------------------------------------------------
# `freqpred db migrate`
# ---------------------------------------------------------------------------


class TestDbMigrate:
    def test_runs_alembic_upgrade_head(self) -> None:
        runner = CliRunner()

        with patch("freqpred.cli.load_config", return_value=_make_config()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            runner.invoke(main, ["db", "migrate"])

        mock_run.assert_called_once()
        cmd_args = mock_run.call_args.args[0]
        assert "alembic" in cmd_args
        assert "upgrade" in cmd_args
        assert "head" in cmd_args

    def test_propagates_alembic_exit_code(self) -> None:
        runner = CliRunner()

        with patch("freqpred.cli.load_config", return_value=_make_config()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = runner.invoke(main, ["db", "migrate"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# T20: Paper trading wired into signal loop
# ---------------------------------------------------------------------------


class TestPaperTradingSignalLoop:
    @pytest.mark.asyncio
    async def test_run_paper_mode_calls_order_manager(self) -> None:
        """paper mode + non-SKIP signal → order_manager.submit() called."""
        from freqpred.cli import _run_main

        config = _make_config()
        market_row = _make_market_row("MKT-1")
        fake_signal = _make_fake_signal(direction="YES")
        mocks = _make_run_mocks(market_row, fake_signal)

        mock_om_instance = AsyncMock()
        mock_om_instance.submit = AsyncMock(return_value=None)
        mock_om_instance._risk = AsyncMock()
        mock_om_instance._risk.check_circuit_breakers = AsyncMock(return_value=None)
        mock_om_instance._risk.check_entry_capacity = AsyncMock(return_value=(False, ""))
        mock_om_instance._risk.pre_signal_gate = AsyncMock(return_value=(False, ""))
        mock_om_instance._bankroll = 1000.0
        mock_om_instance._mode = "paper"
        mock_om_cls = MagicMock(return_value=mock_om_instance)

        mock_risk_cls = MagicMock(return_value=MagicMock())

        async def _cancel_on_sleep(_: float) -> None:
            raise asyncio.CancelledError()

        with patch("freqpred.db.make_engine", return_value=mocks["engine"]), \
             patch("freqpred.db.make_session_factory", return_value=mocks["factory"]), \
             patch("freqpred.strategy.loader.load_strategy", return_value=mocks["strategy"]), \
             patch("freqpred.signal.pipeline.SignalPipeline", mocks["pipeline_cls"]), \
             patch("freqpred.rag.embedder.LocalEmbedder"), \
             patch("freqpred.llm.client.LLMClient"), \
             patch("freqpred.markets.kalshi.KalshiClient", mocks["kalshi_cls"]), \
             patch("freqpred.trading.risk.RiskEngine", mock_risk_cls), \
             patch("freqpred.trading.order_manager.OrderManager", mock_om_cls), \
             patch("asyncio.sleep", side_effect=_cancel_on_sleep), \
             patch("anthropic.AsyncAnthropic"):
            await _run_main(config, "TestStrategy", "paper")

        mock_om_instance.submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_signal_only_does_not_call_order_manager(self) -> None:
        """signal-only mode → OrderManager never constructed or called."""
        from freqpred.cli import _run_main

        config = _make_config()
        market_row = _make_market_row("MKT-1")
        fake_signal = _make_fake_signal(direction="YES")
        mocks = _make_run_mocks(market_row, fake_signal)

        mock_om_cls = MagicMock()

        async def _cancel_on_sleep(_: float) -> None:
            raise asyncio.CancelledError()

        with patch("freqpred.db.make_engine", return_value=mocks["engine"]), \
             patch("freqpred.db.make_session_factory", return_value=mocks["factory"]), \
             patch("freqpred.strategy.loader.load_strategy", return_value=mocks["strategy"]), \
             patch("freqpred.signal.pipeline.SignalPipeline", mocks["pipeline_cls"]), \
             patch("freqpred.rag.embedder.LocalEmbedder"), \
             patch("freqpred.llm.client.LLMClient"), \
             patch("freqpred.markets.kalshi.KalshiClient", mocks["kalshi_cls"]), \
             patch("freqpred.trading.order_manager.OrderManager", mock_om_cls), \
             patch("asyncio.sleep", side_effect=_cancel_on_sleep), \
             patch("anthropic.AsyncAnthropic"):
            await _run_main(config, "TestStrategy", "signal-only")

        mock_om_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_signal_flip_calls_check_all_positions_before_submit(self) -> None:
        """When a market has an open position and the signal direction flips,
        check_all_positions(fresh_signals=...) must be called before submit().

        This is the wiring test: it verifies that the signal loop actually passes
        the fresh signal to the position monitor so should_exit() can fire, not
        just that should_exit() works when called in isolation.
        """
        from freqpred.cli import _run_main

        config = _make_config()
        market_row = _make_market_row("MKT-1")
        fake_signal = _make_fake_signal(direction="NO")  # flipped from existing YES position
        # One open position for MKT-1 so open_market_ids = {"MKT-1"} and the
        # monitor call fires.
        mocks = _make_run_mocks(
            market_row, fake_signal, position_rows=[_make_position_row("MKT-1", "YES")]
        )

        mock_om_instance = AsyncMock()
        mock_om_instance._risk = AsyncMock()
        mock_om_instance._risk.check_circuit_breakers = AsyncMock(return_value=None)
        mock_om_instance._risk.check_entry_capacity = AsyncMock(return_value=(False, ""))
        mock_om_instance._risk.pre_signal_gate = AsyncMock(return_value=(False, ""))
        mock_om_instance._bankroll = 1000.0
        mock_om_instance._mode = "paper"
        mock_om_cls = MagicMock(return_value=mock_om_instance)

        mock_pm_instance = AsyncMock()
        mock_pm_cls = MagicMock(return_value=mock_pm_instance)

        call_order: list[str] = []

        async def _check_all(*, fresh_signals=None) -> list:
            call_order.append("check_all_positions")
            return []

        async def _submit(*args, **kwargs):
            call_order.append("submit")
            return None

        mock_pm_instance.check_all_positions = AsyncMock(side_effect=_check_all)
        mock_om_instance.submit = AsyncMock(side_effect=_submit)

        mock_risk_cls = MagicMock(return_value=MagicMock())

        async def _cancel_on_sleep(_: float) -> None:
            raise asyncio.CancelledError()

        with patch("freqpred.db.make_engine", return_value=mocks["engine"]), \
             patch("freqpred.db.make_session_factory", return_value=mocks["factory"]), \
             patch("freqpred.strategy.loader.load_strategy", return_value=mocks["strategy"]), \
             patch("freqpred.signal.pipeline.SignalPipeline", mocks["pipeline_cls"]), \
             patch("freqpred.rag.embedder.LocalEmbedder"), \
             patch("freqpred.llm.client.LLMClient"), \
             patch("freqpred.markets.kalshi.KalshiClient", mocks["kalshi_cls"]), \
             patch("freqpred.trading.risk.RiskEngine", mock_risk_cls), \
             patch("freqpred.trading.order_manager.OrderManager", mock_om_cls), \
             patch("freqpred.trading.position_monitor.PositionMonitor", mock_pm_cls), \
             patch("asyncio.sleep", side_effect=_cancel_on_sleep), \
             patch("anthropic.AsyncAnthropic"):
            await _run_main(config, "TestStrategy", "paper")

        mock_pm_instance.check_all_positions.assert_called_once_with(
            fresh_signals={"MKT-1": fake_signal}
        )
        mock_om_instance.submit.assert_called_once()
        assert call_order == ["check_all_positions", "submit"], (
            "check_all_positions must be called before submit on a direction flip"
        )


class TestDecidedPositionAnalysisGate:
    """Wiring for the decided-position gate.

    Proves the run loop actually consults strategy.should_analyze_position()
    for markets kept alive by an open position, and skips pipeline.analyze()
    when it declines. Uses the real default implementation rather than a stub
    so the configured bounds are exercised end to end.
    """

    @staticmethod
    async def _run_cycle(mid_price: float, position_rows: list[MagicMock]):
        from freqpred.cli import _run_main
        from freqpred.strategy.defaults.conservative import ConservativeDefault

        config = _make_config()
        market_row = _make_market_row("MKT-1")
        market_row.mid_price = mid_price
        market_row.yes_bid = max(mid_price - 0.01, 0.0)
        market_row.yes_ask = min(mid_price + 0.01, 1.0)

        mocks = _make_run_mocks(
            market_row, _make_fake_signal(direction="NO"), position_rows=position_rows
        )
        # filter_markets rejects the extreme price, exactly as in production —
        # the market is only in play because of the open-position bypass.
        mocks["strategy"].filter_markets = MagicMock(return_value=[])
        mocks["strategy"].should_analyze_position = MagicMock(
            side_effect=ConservativeDefault().should_analyze_position
        )

        mock_om_instance = AsyncMock()
        mock_om_instance.submit = AsyncMock(return_value=None)
        mock_om_instance._risk = AsyncMock()
        mock_om_instance._risk.check_circuit_breakers = AsyncMock(return_value=None)
        mock_om_instance._risk.check_entry_capacity = AsyncMock(return_value=(False, ""))
        mock_om_instance._risk.pre_signal_gate = AsyncMock(return_value=(False, ""))
        mock_om_instance._bankroll = 1000.0
        mock_om_instance._mode = "paper"

        async def _cancel_on_sleep(_: float) -> None:
            raise asyncio.CancelledError()

        with patch("freqpred.db.make_engine", return_value=mocks["engine"]), \
             patch("freqpred.db.make_session_factory", return_value=mocks["factory"]), \
             patch("freqpred.strategy.loader.load_strategy", return_value=mocks["strategy"]), \
             patch("freqpred.signal.pipeline.SignalPipeline", mocks["pipeline_cls"]), \
             patch("freqpred.rag.embedder.LocalEmbedder"), \
             patch("freqpred.llm.client.LLMClient"), \
             patch("freqpred.markets.kalshi.KalshiClient", mocks["kalshi_cls"]), \
             patch("freqpred.trading.risk.RiskEngine", MagicMock(return_value=MagicMock())), \
             patch("freqpred.trading.order_manager.OrderManager",
                   MagicMock(return_value=mock_om_instance)), \
             patch("freqpred.trading.position_monitor.PositionMonitor",
                   MagicMock(return_value=AsyncMock())), \
             patch("asyncio.sleep", side_effect=_cancel_on_sleep), \
             patch("anthropic.AsyncAnthropic"):
            await _run_main(config, "TestStrategy", "paper")

        return mocks

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("direction", "mid_price"),
        [
            ("NO", 0.99),   # NO side ~0.01 — dead loss, settlement pays 0
            ("YES", 0.99),  # YES side ~0.99 — fee-free settlement beats selling
            ("YES", 0.01),  # YES side ~0.01
            ("NO", 0.01),   # NO side ~0.99
        ],
    )
    async def test_skips_llm_for_decided_position(
        self, direction: str, mid_price: float
    ) -> None:
        mocks = await self._run_cycle(
            mid_price, [_make_position_row("MKT-1", direction)]
        )
        mocks["strategy"].should_analyze_position.assert_called_once()
        mocks["pipeline_instance"].analyze.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("direction", ["YES", "NO"])
    async def test_analyzes_undecided_position(self, direction: str) -> None:
        # mid 0.42 → YES side 0.42, NO side 0.58; neither bound is hit.
        mocks = await self._run_cycle(
            0.42, [_make_position_row("MKT-1", direction)]
        )
        mocks["pipeline_instance"].analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyzes_when_only_one_of_two_positions_is_decided(self) -> None:
        """A market is skipped only when *every* open position in it is decided.

        At mid 0.97 the NO side is exactly on min_analysis_price (decided) while
        a hypothetical position whose side is still live must keep the market
        analysed. Here the second position is undecided by construction, so the
        any() in the run loop must keep the LLM call.
        """
        undecided = _make_position_row("MKT-1", "YES")
        decided = _make_position_row("MKT-1", "NO")
        mocks = await self._run_cycle(0.90, [decided, undecided])
        mocks["pipeline_instance"].analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_gate_not_consulted_for_markets_without_positions(self) -> None:
        """Markets with no open position go through the risk gate, not this one."""
        mocks = await self._run_cycle(0.42, [])
        mocks["strategy"].should_analyze_position.assert_not_called()


# ---------------------------------------------------------------------------
# T37: Live mode startup balance guard
# ---------------------------------------------------------------------------


class TestLiveModeStartupGuard:
    @pytest.mark.asyncio
    async def test_startup_aborts_if_balance_below_bankroll(self) -> None:
        """mode='live' + balance < bankroll → _run_main returns early, no tasks started."""
        from freqpred.cli import _run_main

        config = _make_config()
        config.trading.bankroll_usd = 1000.0
        mocks = _make_run_mocks(_make_market_row(), None)

        mock_kalshi_instance = AsyncMock()
        mock_kalshi_instance.get_balance = AsyncMock(return_value=500.0)
        mock_kalshi_ctx = MagicMock()
        mock_kalshi_ctx.__aenter__ = AsyncMock(return_value=mock_kalshi_instance)
        mock_kalshi_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_kalshi_cls = MagicMock(return_value=mock_kalshi_ctx)

        mock_om_cls = MagicMock()

        with patch("freqpred.db.make_engine", return_value=mocks["engine"]), \
             patch("freqpred.db.make_session_factory", return_value=mocks["factory"]), \
             patch("freqpred.strategy.loader.load_strategy", return_value=mocks["strategy"]), \
             patch("freqpred.signal.pipeline.SignalPipeline", mocks["pipeline_cls"]), \
             patch("freqpred.rag.embedder.LocalEmbedder"), \
             patch("freqpred.llm.client.LLMClient"), \
             patch("freqpred.markets.kalshi.KalshiClient", mock_kalshi_cls), \
             patch("freqpred.trading.order_manager.OrderManager", mock_om_cls), \
             patch("anthropic.AsyncAnthropic"):
            await _run_main(config, "TestStrategy", "live")

        # OrderManager never constructed — startup aborted before task creation
        mock_om_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_startup_logs_balance_and_bankroll_when_ok(self) -> None:
        """mode='live' + balance >= bankroll → OrderManager created with mode='live'."""
        from freqpred.cli import _run_main

        config = _make_config()
        config.trading.bankroll_usd = 1000.0
        mocks = _make_run_mocks(_make_market_row(), None)

        mock_kalshi_instance = AsyncMock()
        mock_kalshi_instance.get_balance = AsyncMock(return_value=2000.0)
        mock_kalshi_ctx = MagicMock()
        mock_kalshi_ctx.__aenter__ = AsyncMock(return_value=mock_kalshi_instance)
        mock_kalshi_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_kalshi_cls = MagicMock(return_value=mock_kalshi_ctx)

        mock_om_instance = AsyncMock()
        mock_om_instance.submit = AsyncMock(return_value=None)
        mock_om_instance._risk = AsyncMock()
        mock_om_instance._risk.check_circuit_breakers = AsyncMock(return_value=None)
        mock_om_instance._bankroll = 1000.0
        mock_om_cls = MagicMock(return_value=mock_om_instance)

        mock_risk_cls = MagicMock(return_value=MagicMock())

        async def _cancel_on_sleep(_: float) -> None:
            raise asyncio.CancelledError()

        with patch("freqpred.db.make_engine", return_value=mocks["engine"]), \
             patch("freqpred.db.make_session_factory", return_value=mocks["factory"]), \
             patch("freqpred.strategy.loader.load_strategy", return_value=mocks["strategy"]), \
             patch("freqpred.signal.pipeline.SignalPipeline", mocks["pipeline_cls"]), \
             patch("freqpred.rag.embedder.LocalEmbedder"), \
             patch("freqpred.llm.client.LLMClient"), \
             patch("freqpred.markets.kalshi.KalshiClient", mock_kalshi_cls), \
             patch("freqpred.trading.risk.RiskEngine", mock_risk_cls), \
             patch("freqpred.trading.order_manager.OrderManager", mock_om_cls), \
             patch("asyncio.sleep", side_effect=_cancel_on_sleep), \
             patch("anthropic.AsyncAnthropic"):
            await _run_main(config, "TestStrategy", "live")

        mock_om_cls.assert_called_once()
        call_kwargs = mock_om_cls.call_args.kwargs
        assert call_kwargs["mode"] == "live"
        assert call_kwargs["kalshi_client"] is mock_kalshi_instance


# ---------------------------------------------------------------------------
# T20: positions resolve command
# ---------------------------------------------------------------------------


class TestPositionsResolve:
    def test_positions_resolve_help(self) -> None:
        runner = CliRunner()
        with patch("freqpred.cli.load_config", return_value=_make_config()):
            result = runner.invoke(main, ["positions", "resolve", "--help"])
        assert result.exit_code == 0
        assert "--position-id" in result.output
        assert "--resolution" in result.output

    @pytest.mark.asyncio
    async def test_positions_resolve_updates_db(self) -> None:
        """resolve calls ledger.close_position with correct exit_price."""
        from freqpred.cli import _positions_resolve

        config = _make_config()
        pos_id = str(uuid.uuid4())

        mock_row = MagicMock()
        mock_row.status = "open"
        mock_row.direction = "YES"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=mock_session)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=ctx_mgr)

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        from freqpred.markets.models import Position
        mock_closed_position = MagicMock(spec=Position)
        mock_closed_position.id = pos_id
        mock_closed_position.direction = "YES"
        mock_closed_position.entry_price = 0.44
        mock_closed_position.contracts = 100
        mock_closed_position.pnl = 56.0
        mock_closed_position.pnl_pct = 1.272727

        with patch("freqpred.db.make_engine", return_value=mock_engine), \
             patch("freqpred.db.make_session_factory", return_value=mock_factory), \
             patch("freqpred.trading.ledger.close_position",
                   new_callable=AsyncMock,
                   return_value=mock_closed_position) as mock_close:
            await _positions_resolve(config, pos_id, "yes")

        mock_close.assert_called_once()
        call_kwargs = mock_close.call_args.kwargs
        assert call_kwargs["exit_price"] == 1.0  # YES direction + yes resolution
        assert call_kwargs["resolution"] == 1

    @pytest.mark.asyncio
    async def test_positions_resolve_prints_pnl(self) -> None:
        """resolve prints P&L after closing the position."""
        from freqpred.cli import _positions_resolve

        config = _make_config()
        pos_id = str(uuid.uuid4())

        mock_row = MagicMock()
        mock_row.status = "open"
        mock_row.direction = "NO"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=mock_session)
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=ctx_mgr)

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        from freqpred.markets.models import Position
        mock_closed_position = MagicMock(spec=Position)
        mock_closed_position.id = pos_id
        mock_closed_position.direction = "NO"
        mock_closed_position.entry_price = 0.50
        mock_closed_position.contracts = 80
        mock_closed_position.pnl = 40.0
        mock_closed_position.pnl_pct = 1.0

        output_lines: list[str] = []

        import freqpred.cli as cli_mod

        def capture_echo(msg: str = "", **kwargs) -> None:
            if not kwargs.get("err"):
                output_lines.append(str(msg))

        with patch("freqpred.db.make_engine", return_value=mock_engine), \
             patch("freqpred.db.make_session_factory", return_value=mock_factory), \
             patch("freqpred.trading.ledger.close_position",
                   new_callable=AsyncMock,
                   return_value=mock_closed_position), \
             patch.object(cli_mod.click, "echo", side_effect=capture_echo):
            await _positions_resolve(config, pos_id, "no")

        full_output = "\n".join(output_lines)
        assert "P&L" in full_output
        assert pos_id in full_output
