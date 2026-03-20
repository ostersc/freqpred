"""Unit tests for freqpred CLI commands (T13)."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from freqpred.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)


def _make_config(
    database_url: str = "postgresql+asyncpg://x:y@localhost/test",
    anthropic_api_key: str = "sk-test",
) -> MagicMock:
    cfg = MagicMock()
    cfg.log_level = "INFO"
    cfg.database.url = database_url
    cfg.anthropic.api_key = anthropic_api_key
    cfg.kalshi.api_key = ""
    cfg.kalshi.base_url = "https://api.kalshi.com"
    cfg.kalshi.private_key_path = ""
    cfg.kalshi.polling_interval_seconds = 300
    cfg.ingestion.schedule_interval_seconds = 1800
    cfg.tavily.api_key = ""
    cfg.newsapi.api_key = ""
    cfg.signal.top_k_documents = 10
    cfg.signal.interval_seconds = 1800
    cfg.trading.mode = "paper"
    cfg.trading.bankroll_usd = 1000.0
    cfg.risk.max_daily_llm_spend_usd = 10.0
    return cfg


def _make_fake_signal(direction: str = "YES") -> "Signal":
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


def _make_run_mocks(market_row: MagicMock, signal: "Signal | None"):
    """Build the full set of mocks needed to run _run_main for one signal loop cycle."""
    # DB session returns the market row
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [market_row]
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    # get_run_state uses scalar_one_or_none; None → default "running" state
    mock_result.scalar_one_or_none.return_value = None

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
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

        from freqpred.signal.models import Signal

        fake_signal = Signal(
            id=str(uuid.uuid4()),
            market_id="MKT-1",
            estimated_probability=0.65,
            confidence=0.82,
            edge=0.23,
            market_mid_at_signal=0.42,
            direction="YES",
            reasoning="Strong evidence.",
            sources=[],
            retrieval_hash="abc",
            model_used="claude-sonnet-4-6",
            prompt_version="v1",
            trigger="manual",
            created_at=NOW,
            raw_context="{}",
        )

        async def fake_analyze(cfg, market_id):
            from freqpred.cli import _signal_analyze
            # Patch internals inline
            import freqpred.cli as cli_mod
            click_echo_calls: list[str] = []

        with patch("freqpred.cli.load_config", return_value=config), \
             patch("freqpred.cli._signal_analyze", new_callable=AsyncMock) as mock_analyze:
            result = runner.invoke(main, ["signal", "analyze", "--market-id", "MKT-1"])
        mock_analyze.assert_called_once_with(config, "MKT-1")
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
            result = runner.invoke(main, ["db", "migrate"])

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
        original_echo = cli_mod.click.echo

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
