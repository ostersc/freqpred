"""Unit tests for the strategy plugin system (T12)."""
from __future__ import annotations

import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqpred.markets.models import Market
from freqpred.signal.models import Signal
from freqpred.strategy.base import IPredictionStrategy
from freqpred.strategy.config import StrategyConfig
from freqpred.strategy.defaults.conservative import ConservativeDefault
from freqpred.strategy.loader import load_strategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)


def _market(
    category: str = "politics",
    volume_24h: float = 1000.0,
    days_to_close: int = 14,
) -> Market:
    return Market(
        id=str(uuid.uuid4()),
        platform="kalshi",
        question="Will X happen?",
        category=category,
        close_time=NOW + timedelta(days=days_to_close),
        yes_bid=0.40,
        yes_ask=0.44,
        mid_price=0.42,
        volume_24h=volume_24h,
        open_interest=500.0,
        last_fetched_at=NOW,
        price_updated_at=NOW,
        metadata_fetched_at=NOW,
    )


def _signal(
    edge: float = 0.15,
    confidence: float = 0.85,
    estimated_probability: float = 0.60,
) -> Signal:
    return Signal(
        id=str(uuid.uuid4()),
        market_id="MKT-1",
        estimated_probability=estimated_probability,
        confidence=confidence,
        edge=edge,
        market_mid_at_signal=0.42,
        direction="YES",
        reasoning="test",
        sources=[],
        retrieval_hash="abc123",
        model_used="claude-sonnet-4-6",
        prompt_version="v1",
        trigger="manual",
        created_at=NOW,
        raw_context="{}",
    )


# ---------------------------------------------------------------------------
# ConservativeDefault — should_trade threshold logic
# ---------------------------------------------------------------------------

class TestConservativeDefaultShouldTrade:
    strategy = ConservativeDefault()

    def test_passes_when_above_thresholds(self) -> None:
        assert self.strategy.should_trade(
            _signal(edge=0.15, confidence=0.85), _market()
        )

    def test_rejects_insufficient_edge(self) -> None:
        assert not self.strategy.should_trade(
            _signal(edge=0.11, confidence=0.85), _market()
        )

    def test_rejects_at_exact_edge_boundary(self) -> None:
        # min_edge is 0.12, signal with edge exactly 0.12 should pass
        assert self.strategy.should_trade(
            _signal(edge=0.12, confidence=0.85), _market()
        )

    def test_rejects_below_edge_boundary(self) -> None:
        assert not self.strategy.should_trade(
            _signal(edge=0.119, confidence=0.85), _market()
        )

    def test_rejects_insufficient_confidence(self) -> None:
        assert not self.strategy.should_trade(
            _signal(edge=0.15, confidence=0.79), _market()
        )

    def test_rejects_at_confidence_boundary(self) -> None:
        # min_confidence is 0.80, exactly 0.80 should pass
        assert self.strategy.should_trade(
            _signal(edge=0.15, confidence=0.80), _market()
        )

    def test_rejects_both_below_threshold(self) -> None:
        assert not self.strategy.should_trade(
            _signal(edge=0.05, confidence=0.50), _market()
        )

    def test_config_values(self) -> None:
        assert self.strategy.config.min_edge == 0.12
        assert self.strategy.config.min_confidence == 0.80
        assert self.strategy.config.kelly_fraction == 0.15


# ---------------------------------------------------------------------------
# ConservativeDefault — position_size Kelly math + cap
# ---------------------------------------------------------------------------

class TestConservativeDefaultPositionSize:
    strategy = ConservativeDefault()

    def test_basic_kelly_sizing(self) -> None:
        # edge=0.20, estimated_prob=0.60 → kelly = 0.20 / 0.40 = 0.50
        # raw = 1000 * 0.50 * 0.15 = 75.0
        # cap = 1000 * 0.02 = 20.0 → capped at 20.0
        sig = _signal(edge=0.20, estimated_probability=0.60)
        result = self.strategy.position_size(sig, bankroll=1000.0)
        assert result == pytest.approx(20.0)

    def test_capped_at_max_exposure(self) -> None:
        # Very high edge/prob → raw Kelly would exceed 2% cap
        sig = _signal(edge=0.40, estimated_probability=0.80)
        result = self.strategy.position_size(sig, bankroll=10_000.0)
        cap = 10_000.0 * 0.02
        assert result <= cap

    def test_not_capped_when_small_edge(self) -> None:
        # edge=0.13, estimated_prob=0.50 → kelly = 0.13 / 0.50 = 0.26
        # raw = 1000 * 0.26 * 0.15 = 39.0; cap = 20.0 → capped
        sig = _signal(edge=0.13, estimated_probability=0.50)
        result = self.strategy.position_size(sig, bankroll=1000.0)
        assert result == pytest.approx(
            min(1000.0 * (0.13 / 0.50) * 0.15, 1000.0 * 0.02)
        )

    def test_never_exceeds_cap_for_any_signal(self) -> None:
        bankroll = 5000.0
        cap = bankroll * 0.02
        for edge, prob in [(0.12, 0.55), (0.30, 0.70), (0.50, 0.90)]:
            sig = _signal(edge=edge, estimated_probability=prob)
            assert self.strategy.position_size(sig, bankroll) <= cap + 1e-9

    def test_scales_with_bankroll(self) -> None:
        sig = _signal(edge=0.13, estimated_probability=0.55)
        r1 = self.strategy.position_size(sig, bankroll=1000.0)
        r2 = self.strategy.position_size(sig, bankroll=2000.0)
        assert r2 == pytest.approx(r1 * 2, rel=1e-6)


# ---------------------------------------------------------------------------
# is_market_interesting — default implementation
# ---------------------------------------------------------------------------

class TestIsMarketInteresting:
    strategy = ConservativeDefault()

    def test_passes_with_good_market(self) -> None:
        assert self.strategy.is_market_interesting(
            _market(category="politics", volume_24h=1000.0, days_to_close=14)
        )

    def test_passes_all_categories_when_categories_empty(self) -> None:
        # ConservativeDefault has categories=[] meaning all categories
        assert self.strategy.is_market_interesting(
            _market(category="sports", volume_24h=1000.0, days_to_close=14)
        )
        assert self.strategy.is_market_interesting(
            _market(category="economics", volume_24h=1000.0, days_to_close=14)
        )

    def test_rejects_below_volume_floor(self) -> None:
        assert not self.strategy.is_market_interesting(
            _market(volume_24h=100.0, days_to_close=14)
        )

    def test_rejects_above_max_days_to_close(self) -> None:
        assert not self.strategy.is_market_interesting(
            _market(days_to_close=120)  # clearly above 60-day max
        )

    def test_rejects_below_min_days_to_close(self) -> None:
        assert not self.strategy.is_market_interesting(
            _market(days_to_close=0)  # clearly below 2-day min
        )

    def test_passes_within_days_window(self) -> None:
        assert self.strategy.is_market_interesting(_market(days_to_close=10))
        assert self.strategy.is_market_interesting(_market(days_to_close=30))

    def test_category_filtered_strategy_rejects_wrong_category(self) -> None:
        from freqpred.strategy.config import StrategyConfig

        class PoliticsOnly(IPredictionStrategy):
            config = StrategyConfig(
                name="PoliticsOnly",
                min_edge=0.10,
                min_confidence=0.70,
                max_exposure_per_market=0.05,
                kelly_fraction=0.25,
                categories=["politics"],
                min_volume_24h=0.0,
                max_days_to_close=90,
                min_days_to_close=1,
            )

            def should_trade(self, signal, market):  # type: ignore[override]
                return True

            def position_size(self, signal, bankroll):  # type: ignore[override]
                return 0.0

        strat = PoliticsOnly()
        assert strat.is_market_interesting(_market(category="politics"))
        assert not strat.is_market_interesting(_market(category="sports"))


# ---------------------------------------------------------------------------
# filter_markets delegates to is_market_interesting
# ---------------------------------------------------------------------------

class TestFilterMarkets:
    def test_delegates_to_is_market_interesting(self) -> None:
        strategy = ConservativeDefault()

        good = _market(volume_24h=1000.0, days_to_close=14)
        bad_volume = _market(volume_24h=10.0, days_to_close=14)
        bad_days = _market(volume_24h=1000.0, days_to_close=90)

        result = strategy.filter_markets([good, bad_volume, bad_days])
        assert result == [good]

    def test_empty_list(self) -> None:
        assert ConservativeDefault().filter_markets([]) == []

    def test_all_pass(self) -> None:
        markets = [_market() for _ in range(3)]
        result = ConservativeDefault().filter_markets(markets)
        assert result == markets


# ---------------------------------------------------------------------------
# load_strategy — built-in names
# ---------------------------------------------------------------------------

class TestLoadStrategyBuiltins:
    def test_load_conservative_default(self) -> None:
        strat = load_strategy("ConservativeDefault")
        assert isinstance(strat, ConservativeDefault)

    def test_load_returns_instantiated_strategy(self) -> None:
        strat = load_strategy("ConservativeDefault")
        assert isinstance(strat, IPredictionStrategy)

    def test_unknown_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            load_strategy("NonExistentStrategy")


# ---------------------------------------------------------------------------
# load_strategy — file path loading
# ---------------------------------------------------------------------------

class TestLoadStrategyFromFile:
    def test_load_custom_strategy_from_file(self, tmp_path: Path) -> None:
        strategy_code = textwrap.dedent("""\
            from freqpred.strategy.base import IPredictionStrategy
            from freqpred.strategy.config import StrategyConfig

            class MyCustomStrategy(IPredictionStrategy):
                config = StrategyConfig(
                    name="MyCustomStrategy",
                    min_edge=0.05,
                    min_confidence=0.60,
                    max_exposure_per_market=0.10,
                    kelly_fraction=0.50,
                    categories=["politics"],
                    min_volume_24h=100.0,
                    max_days_to_close=30,
                    min_days_to_close=1,
                )

                def should_trade(self, signal, market):
                    return signal.edge >= self.config.min_edge

                def position_size(self, signal, bankroll):
                    return bankroll * 0.01
        """)
        strategy_file = tmp_path / "my_strategy.py"
        strategy_file.write_text(strategy_code)

        strat = load_strategy(str(strategy_file))
        assert isinstance(strat, IPredictionStrategy)
        assert strat.config.name == "MyCustomStrategy"

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_strategy("/nonexistent/path/strategy.py")

    def test_file_with_no_strategy_raises(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("x = 1\n")
        with pytest.raises(ValueError, match="No concrete IPredictionStrategy"):
            load_strategy(str(empty_file))

    def test_loaded_file_strategy_behaves_like_builtin(self, tmp_path: Path) -> None:
        strategy_code = textwrap.dedent("""\
            from freqpred.strategy.base import IPredictionStrategy
            from freqpred.strategy.config import StrategyConfig

            class FileStrategy(IPredictionStrategy):
                config = StrategyConfig(
                    name="FileStrategy",
                    min_edge=0.10,
                    min_confidence=0.70,
                    max_exposure_per_market=0.05,
                    kelly_fraction=0.25,
                    categories=[],
                    min_volume_24h=0.0,
                    max_days_to_close=365,
                    min_days_to_close=0,
                )

                def should_trade(self, signal, market):
                    return signal.edge >= self.config.min_edge

                def position_size(self, signal, bankroll):
                    return bankroll * self.config.kelly_fraction * signal.edge
        """)
        f = tmp_path / "file_strategy.py"
        f.write_text(strategy_code)

        strat = load_strategy(str(f))
        market = _market()
        sig_pass = _signal(edge=0.15, confidence=0.75)
        sig_fail = _signal(edge=0.05, confidence=0.75)

        assert strat.should_trade(sig_pass, market)
        assert not strat.should_trade(sig_fail, market)
        assert strat.is_market_interesting(market)
