"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from freqpred.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    """Return a default Settings instance with no external dependencies."""
    return Settings()
