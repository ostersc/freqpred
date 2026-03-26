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

    def test_stoploss(self) -> None:
        assert self.config.stoploss == pytest.approx(-0.15)

    def test_trailing_stop_disabled(self) -> None:
        assert self.config.trailing_stop is False

class TestTechExitConfig:
    config = TechNewsStrategy().config

    def test_stoploss(self) -> None:
        assert self.config.stoploss == pytest.approx(-0.15)

    def test_trailing_stop_enabled(self) -> None:
        assert self.config.trailing_stop is True

