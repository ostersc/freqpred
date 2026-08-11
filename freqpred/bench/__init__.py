"""Scenario-bank benchmark harness for model/prompt candidates (T93)."""
from freqpred.bench.report import (
    ARTIFACT_SCHEMA_VERSION,
    build_artifact,
    cost_summary,
    format_summary,
    score_run,
)
from freqpred.bench.runner import BenchmarkRun, ScenarioRun, estimate_run, run_benchmark
from freqpred.bench.scenarios import (
    ModelOutput,
    Scenario,
    apply_extraction,
    build_db_scenarios,
    build_fixture_scenarios,
    filter_contaminated,
    sample_markets,
    sample_per_market,
    scenario_from_fixture,
    scenario_from_signal,
)
from freqpred.bench.scoring import PairScore, aggregate, score_pair, trade_decision

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BenchmarkRun",
    "ModelOutput",
    "PairScore",
    "Scenario",
    "ScenarioRun",
    "aggregate",
    "build_artifact",
    "build_db_scenarios",
    "apply_extraction",
    "build_fixture_scenarios",
    "cost_summary",
    "estimate_run",
    "filter_contaminated",
    "format_summary",
    "run_benchmark",
    "sample_markets",
    "sample_per_market",
    "scenario_from_fixture",
    "scenario_from_signal",
    "score_pair",
    "score_run",
    "trade_decision",
]
