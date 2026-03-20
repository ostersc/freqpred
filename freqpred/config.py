"""Config loading: YAML file + environment variable overrides.

Usage:
    from freqpred.config import load_config

    settings = load_config()  # reads config/config.yaml by default
    settings = load_config(Path("custom/path.yaml"))
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    url: str = Field(default="")


class KalshiConfig(BaseModel):
    api_key: str = Field(default="")
    private_key_path: str = Field(default="")
    base_url: str = Field(default="https://api.elections.kalshi.com/trade-api/v2")
    polling_interval_seconds: int = Field(default=300)


class AnthropicConfig(BaseModel):
    api_key: str = Field(default="")
    primary_model: str = Field(default="claude-sonnet-4-6")
    cheap_model: str = Field(default="claude-haiku-4-5-20251001")


class TavilyConfig(BaseModel):
    api_key: str = Field(default="")


class NewsAPIConfig(BaseModel):
    api_key: str = Field(default="")
    enabled: bool = Field(default=True)
    max_daily_requests: int = Field(default=90)


class RedditConfig(BaseModel):
    user_agent: str = Field(default="freqpred/0.1")


class TruthSocialConfig(BaseModel):
    username: str = Field(default="")
    password: str = Field(default="")


class TruthSocialIngestionConfig(BaseModel):
    enabled: bool = Field(default=False)
    accounts: list[str] = Field(default_factory=list)


class IngestionConfig(BaseModel):
    schedule_interval_seconds: int = Field(default=1800)
    categories: list[str] = Field(default_factory=lambda: ["politics", "technology"])
    truthsocial: TruthSocialIngestionConfig = Field(default_factory=TruthSocialIngestionConfig)


class SignalConfig(BaseModel):
    top_k_documents: int = Field(default=10)
    staleness_multiplier: int = Field(default=3)
    # How often the signal loop re-analyzes markets (seconds).
    # Separate from the watcher poll interval — no need to embed market
    # questions every 5 minutes when the ingestion scheduler only runs every 30.
    interval_seconds: int = Field(default=1800)


class RiskConfig(BaseModel):
    max_position_pct: float = Field(default=0.05)
    max_daily_loss_pct: float = Field(default=0.15)
    max_total_exposure_pct: float = Field(default=0.40)
    min_edge_floor: float = Field(default=0.10)
    max_open_positions: int = Field(default=20)
    max_daily_llm_spend_usd: float = Field(default=10.0)


class TradingConfig(BaseModel):
    mode: str = Field(default="paper")
    bankroll_usd: float = Field(default=1000.0)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got: {v!r}")
        return v


class DashboardConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)


class AlertsConfig(BaseModel):
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    telegram_authorized_users: list[str] = Field(default_factory=list)
    discord_webhook_url: str = Field(default="")


class Settings(BaseModel):
    log_level: str = Field(default="INFO")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    kalshi: KalshiConfig = Field(default_factory=KalshiConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    newsapi: NewsAPIConfig = Field(default_factory=NewsAPIConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    truthsocial: TruthSocialConfig = Field(default_factory=TruthSocialConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got: {v!r}")
        return upper


# Maps environment variable name → (section, field) path in Settings
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "DATABASE_URL": ("database", "url"),
    "KALSHI_API_KEY": ("kalshi", "api_key"),
    "KALSHI_PRIVATE_KEY_PATH": ("kalshi", "private_key_path"),
    "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
    "TAVILY_API_KEY": ("tavily", "api_key"),
    "NEWSAPI_KEY": ("newsapi", "api_key"),
    "TRUTHSOCIAL_USERNAME": ("truthsocial", "username"),
    "TRUTHSOCIAL_PASSWORD": ("truthsocial", "password"),
    "TELEGRAM_BOT_TOKEN": ("alerts", "telegram_bot_token"),
    "TELEGRAM_CHAT_ID": ("alerts", "telegram_chat_id"),
    "DISCORD_WEBHOOK_URL": ("alerts", "discord_webhook_url"),
}

# Env vars whose values are comma-separated lists.
_ENV_LIST_OVERRIDES: dict[str, tuple[str, str]] = {
    "TELEGRAM_AUTHORIZED_USERS": ("alerts", "telegram_authorized_users"),
}


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            if section not in data:
                data[section] = {}
            data[section][key] = value
    for env_var, (section, key) in _ENV_LIST_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            if section not in data:
                data[section] = {}
            data[section][key] = [v.strip() for v in value.split(",") if v.strip()]
    return data


def load_config(config_path: Path | None = None) -> Settings:
    """Load config from YAML file, then apply env var overrides.

    Loads .env from the current working directory (or any parent) before
    applying env var overrides, so local .env files work out of the box.

    Args:
        config_path: Path to config YAML. Defaults to config/config.yaml
                     relative to the current working directory.

    Returns:
        Validated Settings instance.
    """
    load_dotenv()  # no-op if no .env present

    if config_path is None:
        config_path = Path("config/config.yaml")

    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open() as f:
            loaded = yaml.safe_load(f)
            if loaded and isinstance(loaded, dict):
                data = loaded

    data = _apply_env_overrides(data)
    return Settings.model_validate(data)
