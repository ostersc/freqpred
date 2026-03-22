"""Unit tests for freqpred.config."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from freqpred.config import Settings, load_config


def test_default_settings_are_valid() -> None:
    settings = Settings()
    assert settings.trading.bankroll_usd == 1000.0
    assert settings.risk.max_position_pct == 0.05
    assert settings.risk.max_open_positions == 20
    assert settings.signal.top_k_documents == 10
    assert settings.ingestion.categories == ["politics", "technology"]


def test_load_config_no_file_returns_defaults(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.yaml"
    settings = load_config(missing)
    assert isinstance(settings, Settings)
    assert settings.trading.bankroll_usd == 1000.0


def test_load_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "trading": {"bankroll_usd": 500.0},
        "risk": {"max_open_positions": 5},
        "signal": {"top_k_documents": 20},
    }))

    settings = load_config(config_file)
    assert settings.trading.bankroll_usd == 500.0
    assert settings.risk.max_open_positions == 5
    assert settings.signal.top_k_documents == 20


def test_env_var_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"database": {"url": "from-yaml"}}))

    monkeypatch.setenv("DATABASE_URL", "from-env")
    settings = load_config(config_file)
    assert settings.database.url == "from-env"


def test_env_var_sets_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    settings = load_config(tmp_path / "missing.yaml")
    assert settings.anthropic.api_key == "sk-test-123"



def test_kalshi_base_url_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo_url = "https://demo-api.kalshi.co/trade-api/v2"
    monkeypatch.setenv("KALSHI_BASE_URL", demo_url)
    settings = load_config(tmp_path / "missing.yaml")
    assert settings.kalshi.base_url == demo_url


def test_kalshi_demo_credentials_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KALSHI_DEMO_API_KEY", "demo-key-id")
    monkeypatch.setenv("KALSHI_DEMO_PRIVATE_KEY_PATH", "/tmp/demo.pem")
    settings = load_config(tmp_path / "missing.yaml")
    assert settings.kalshi.demo_api_key == "demo-key-id"
    assert settings.kalshi.demo_private_key_path == "/tmp/demo.pem"
