"""Per-fetcher freshness specs — every ingestion fetcher must have a heartbeat spec."""
from __future__ import annotations

from freqpred.runtime.telemetry import build_freshness_specs


def _specs():
    return build_freshness_specs(
        ingestion_interval_seconds=1800,
        realtime_interval_seconds=300,
        signal_interval_seconds=1800,
        market_watcher_interval_seconds=60,
    )


def test_every_main_scheduler_fetcher_has_a_freshness_spec() -> None:
    """The spec list must cover every fetcher the ingestion scheduler runs,
    using the scheduler's own service-name mapping. A fetcher without a spec
    can die silently — that's how the Reddit JSON shutdown went unnoticed."""
    from freqpred.ingestion.scheduler import _MAIN_SCHEDULER_SERVICES, _fetcher_service

    specs = _specs()
    for fetcher in _MAIN_SCHEDULER_SERVICES:
        assert _fetcher_service(fetcher) in specs, f"missing FreshnessSpec for {fetcher}"


def test_fetcher_specs_are_alertable_with_24h_threshold() -> None:
    from freqpred.ingestion.scheduler import _MAIN_SCHEDULER_SERVICES, _fetcher_service

    specs = _specs()
    for fetcher in _MAIN_SCHEDULER_SERVICES:
        spec = specs[_fetcher_service(fetcher)]
        assert spec.alertable
        assert spec.stale_after_seconds == 24 * 3600
