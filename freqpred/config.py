"""Config loading: YAML file + environment variable overrides.

Usage:
    from freqpred.config import load_config

    settings = load_config()  # reads config/config.yaml by default
    settings = load_config(Path("custom/path.yaml"))
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    url: str = Field(default="")


class RedisConfig(BaseModel):
    url: str = Field(default="")


class KalshiConfig(BaseModel):
    api_key: str = Field(default="")
    base_url: str = Field(default="https://trading-api.kalshi.com/trade-api/v2")
    polling_interval_seconds: int = Field(default=300)


class AnthropicConfig(BaseModel):
    api_key: str = Field(default="")
    primary_model: str = Field(default="claude-sonnet-4-6")
    cheap_model: str = Field(default="claude-haiku-4-5-20251001")


class VoyageConfig(BaseModel):
    api_key: str = Field(default="")
    model: str = Field(default="voyage-3")
    embedding_dim: int = Field(default=1024)


class TavilyConfig(BaseModel):
    api_key: str = Field(default="")


class NewsAPIConfig(BaseModel):
    api_key: str = Field(default="")


class RedditConfig(BaseModel):
    client_id: str = Field(default="")
    client_secret: str = Field(default="")
    user_agent: str = Field(default="freqpred/0.1")


class IngestionConfig(BaseModel):
    schedule_interval_seconds: int = Field(default=1800)
    categories: list[str] = Field(default_factory=lambda: ["politics", "technology"])


class SignalConfig(BaseModel):
    top_k_documents: int = Field(default=10)
    staleness_multiplier: int = Field(default=3)


class RiskConfig(BaseModel):
    max_position_pct: float = Field(default=0.05)
    max_daily_loss_pct: float = Field(default=0.15)
    max_total_exposure_pct: float = Field(default=0.40)
    min_edge_floor: float = Field(default=0.10)
    max_open_positions: int = Field(default=20)
    max_daily_llm_spend_usd: float = Field(default=10.0)


class TradingConfig(BaseModel):
    mode: str = Field(default="paper")

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
    discord_webhook_url: str = Field(default="")


class Settings(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kalshi: KalshiConfig = Field(default_factory=KalshiConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    voyage: VoyageConfig = Field(default_factory=VoyageConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    newsapi: NewsAPIConfig = Field(default_factory=NewsAPIConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)


# Maps environment variable name → (section, field) path in Settings
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "DATABASE_URL": ("database", "url"),
    "REDIS_URL": ("redis", "url"),
    "KALSHI_API_KEY": ("kalshi", "api_key"),
    "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
    "VOYAGE_API_KEY": ("voyage", "api_key"),
    "TAVILY_API_KEY": ("tavily", "api_key"),
    "NEWSAPI_KEY": ("newsapi", "api_key"),
    "REDDIT_CLIENT_ID": ("reddit", "client_id"),
    "REDDIT_CLIENT_SECRET": ("reddit", "client_secret"),
    "TELEGRAM_BOT_TOKEN": ("alerts", "telegram_bot_token"),
    "TELEGRAM_CHAT_ID": ("alerts", "telegram_chat_id"),
    "DISCORD_WEBHOOK_URL": ("alerts", "discord_webhook_url"),
}


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            if section not in data:
                data[section] = {}
            data[section][key] = value
    return data


def load_config(config_path: Path | None = None) -> Settings:
    """Load config from YAML file, then apply env var overrides.

    Args:
        config_path: Path to config YAML. Defaults to config/config.yaml
                     relative to the current working directory.

    Returns:
        Validated Settings instance.
    """
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
