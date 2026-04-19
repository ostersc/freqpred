"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from freqpred.config import Settings

# Ensure all ORM models are registered with SQLAlchemy before any test runs.
# Without these imports, forward-reference relationships (e.g. "SignalRow" in
# DocumentMarketLinkRow) fail to resolve when test modules are run in isolation.
import freqpred.ingestion.models  # noqa: F401
import freqpred.llm.models  # noqa: F401
import freqpred.markets.models  # noqa: F401
import freqpred.metrics.models  # noqa: F401
import freqpred.rag.models  # noqa: F401
import freqpred.signal.models  # noqa: F401


@pytest.fixture
def default_settings() -> Settings:
    """Return a default Settings instance with no external dependencies."""
    return Settings()
