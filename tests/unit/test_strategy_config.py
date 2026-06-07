"""Unit tests for StrategyConfig exit fields (T25)."""
from __future__ import annotations

import pytest

from freqpred.strategy.config import StrategyConfig
from freqpred.strategy.defaults.conservative import ConservativeDefault
from freqpred.strategy.defaults.politics import PoliticsEdgeStrategy
from freqpred.strategy.defaults.tech import TechNewsStrategy


def _minimal_config(**overrides) -> StrategyConfig:
    defaults = dict(
        name="TestStrategy",
        min_edge=0.10,
        min_confidence=0.70,
        max_exposure_per_market=0.05,
        kelly_fraction=0.25,
        categories=[],
        min_volume_24h=0.0,
        max_days_to_close=90,
        min_days_to_close=1,
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


class TestOrderTypesDefaults:
    def test_entry_default_is_market(self) -> None:
        from freqpred.strategy.config import OrderTypes
        assert OrderTypes().entry == "market"

    def test_exit_default_is_market(self) -> None:
        from freqpred.strategy.config import OrderTypes
        assert OrderTypes().exit == "market"

    def test_emergency_exit_default_is_market(self) -> None:
        from freqpred.strategy.config import OrderTypes
        assert OrderTypes().emergency_exit == "market"

    def test_stoploss_on_exchange_default_false(self) -> None:
        from freqpred.strategy.config import OrderTypes
        assert OrderTypes().stoploss_on_exchange is False

    def test_strategy_config_order_types_defaults_all_market(self) -> None:
        config = _minimal_config()
        assert config.order_types.entry == "market"
        assert config.order_types.exit == "market"
        assert config.order_types.emergency_exit == "market"

    def test_strategy_config_limit_order_timeout_hours_default(self) -> None:
        config = _minimal_config()
        assert config.limit_order_timeout_hours == pytest.approx(4.0)

    def test_strategy_config_order_types_can_be_set_to_limit(self) -> None:
        from freqpred.strategy.config import OrderTypes
        config = _minimal_config(order_types=OrderTypes(entry="limit"))
        assert config.order_types.entry == "limit"


class TestStrategyConfigDefaults:
    def test_stoploss_default_is_negative(self) -> None:
        config = _minimal_config()
        assert config.stoploss < 0

    def test_trailing_stop_default_is_false(self) -> None:
        config = _minimal_config()
        assert config.trailing_stop is False

    def test_trailing_stop_positive_default_is_none(self) -> None:
        config = _minimal_config()
        assert config.trailing_stop_positive is None

    def test_trailing_stop_positive_offset_default(self) -> None:
        config = _minimal_config()
        assert config.trailing_stop_positive_offset == pytest.approx(0.02)

class TestConservativeDefaultExitConfig:
    config = ConservativeDefault().config

    def test_stoploss(self) -> None:
        assert self.config.stoploss == pytest.approx(-0.15)

    def test_trailing_stop_enabled(self) -> None:
        assert self.config.trailing_stop is True

    def test_trailing_stop_positive(self) -> None:
        assert self.config.trailing_stop_positive == pytest.approx(0.15)

class TestPoliticsExitConfig:
    config = PoliticsEdgeStrategy().config

    def test_stoploss_disabled(self) -> None:
        assert self.config.stoploss == pytest.approx(-1.0)

    def test_trailing_stop_disabled(self) -> None:
        assert self.config.trailing_stop is False

class TestTechExitConfig:
    config = TechNewsStrategy().config

    def test_stoploss(self) -> None:
        assert self.config.stoploss == pytest.approx(-0.15)

    def test_trailing_stop_enabled(self) -> None:
        assert self.config.trailing_stop is True

