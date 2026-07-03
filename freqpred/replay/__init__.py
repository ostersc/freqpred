"""Deterministic replay harness for frozen signal-decision fixtures (T66)."""
from freqpred.replay.engine import (
    ReplayCheck,
    ReplayError,
    ReplayResult,
    compute_expectations,
    replay_fixture,
)
from freqpred.replay.fixtures import (
    DEFAULT_FIXTURE_DIR,
    SCHEMA_VERSION,
    ReplayFixture,
    load_fixture,
    save_fixture,
)
from freqpred.replay.recorder import RecordingError, record_fixture

__all__ = [
    "DEFAULT_FIXTURE_DIR",
    "SCHEMA_VERSION",
    "RecordingError",
    "ReplayCheck",
    "ReplayError",
    "ReplayFixture",
    "ReplayResult",
    "compute_expectations",
    "load_fixture",
    "record_fixture",
    "replay_fixture",
    "save_fixture",
]
