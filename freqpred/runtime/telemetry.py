"""Runtime heartbeat persistence, websocket state, and stale-service alerts."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from freqpred.runtime.models import RuntimeEventRow, ServiceHeartbeatRow

log = structlog.get_logger(__name__)

SERVICE_INGESTION_SCHEDULER = "ingestion_scheduler"
SERVICE_REALTIME_SCHEDULER = "realtime_scheduler"
SERVICE_SIGNAL_LOOP = "signal_loop"
SERVICE_SOURCE_QUALITY_SCHEDULER = "source_quality_scheduler"
SERVICE_SERIES_HISTORY_SCHEDULER = "series_history_scheduler"
SERVICE_FACTBASE_SCHEDULER = "factbase_scheduler"
SERVICE_POSITION_WATCHER_LAST_MESSAGE = "position_watcher_last_message"
SERVICE_POSITION_WATCHER_RECONCILE = "position_watcher_reconcile"
SERVICE_PENDING_ORDER_RECONCILE = "pending_order_reconcile"
SERVICE_MARKET_WATCHER = "market_watcher"
SERVICE_KALSHI_CHANGELOG = "kalshi_changelog"

# Per-fetcher heartbeats. Each ingestion fetcher reports independently (the
# scheduler loop marks these at cycle end) so a single dead source surfaces as
# stale even while the scheduler heartbeat stays green. Names follow the
# "fetcher_<name>" pattern produced by the ingestion scheduler.
SERVICE_FETCHER_REDDIT = "fetcher_reddit"
SERVICE_FETCHER_GDELT = "fetcher_gdelt"
SERVICE_FETCHER_TAVILY = "fetcher_tavily"
SERVICE_FETCHER_NEWSAPI = "fetcher_newsapi"
SERVICE_FETCHER_GUARDIAN = "fetcher_guardian"
SERVICE_FETCHER_TV_ARCHIVE = "fetcher_tv_archive"

EVENT_CATEGORY_KALSHI_API = "kalshi_api"
EVENT_CATEGORY_STALE_SERVICE = "stale_service"
EVENT_CATEGORY_WEBSOCKET = "websocket"


@dataclass(frozen=True)
class FreshnessSpec:
    service_name: str
    label: str
    stale_after_seconds: int
    alertable: bool = True


@dataclass(frozen=True)
class ServiceFreshnessState:
    service_name: str
    label: str
    status: str
    last_success_at: datetime | None
    last_error_at: datetime | None
    last_error_message: str | None
    stale_after_seconds: int
    age_seconds: int | None
    alertable: bool


def build_freshness_specs(
    *,
    ingestion_interval_seconds: int,
    realtime_interval_seconds: int,
    signal_interval_seconds: int,
    market_watcher_interval_seconds: int,
) -> dict[str, FreshnessSpec]:
    """Build dynamic freshness thresholds from runtime intervals."""
    return {
        SERVICE_INGESTION_SCHEDULER: FreshnessSpec(
            service_name=SERVICE_INGESTION_SCHEDULER,
            label="Ingestion scheduler",
            stale_after_seconds=max(ingestion_interval_seconds * 2, 900),
        ),
        SERVICE_REALTIME_SCHEDULER: FreshnessSpec(
            service_name=SERVICE_REALTIME_SCHEDULER,
            label="Realtime scheduler",
            stale_after_seconds=max(realtime_interval_seconds * 2, 300),
        ),
        SERVICE_SIGNAL_LOOP: FreshnessSpec(
            service_name=SERVICE_SIGNAL_LOOP,
            label="Signal loop",
            stale_after_seconds=max(signal_interval_seconds * 2, 300),
        ),
        SERVICE_SOURCE_QUALITY_SCHEDULER: FreshnessSpec(
            service_name=SERVICE_SOURCE_QUALITY_SCHEDULER,
            label="Source quality refresh",
            stale_after_seconds=36 * 3600,
        ),
        SERVICE_SERIES_HISTORY_SCHEDULER: FreshnessSpec(
            service_name=SERVICE_SERIES_HISTORY_SCHEDULER,
            label="Series option history refresh",
            stale_after_seconds=36 * 3600,
        ),
        SERVICE_FACTBASE_SCHEDULER: FreshnessSpec(
            service_name=SERVICE_FACTBASE_SCHEDULER,
            label="FactBase phrase refresh",
            stale_after_seconds=36 * 3600,
        ),
        SERVICE_POSITION_WATCHER_LAST_MESSAGE: FreshnessSpec(
            service_name=SERVICE_POSITION_WATCHER_LAST_MESSAGE,
            label="Position watcher feed",
            stale_after_seconds=10 * 60,
        ),
        SERVICE_PENDING_ORDER_RECONCILE: FreshnessSpec(
            service_name=SERVICE_PENDING_ORDER_RECONCILE,
            label="Pending-order reconcile",
            # Reconcile runs every 30s by default; tolerate 4x the interval
            # before flagging stale so a brief Kalshi outage doesn't page.
            stale_after_seconds=4 * 60,
        ),
        SERVICE_MARKET_WATCHER: FreshnessSpec(
            service_name=SERVICE_MARKET_WATCHER,
            label="Market watcher",
            stale_after_seconds=max(market_watcher_interval_seconds * 2, 900),
        ),
        SERVICE_KALSHI_CHANGELOG: FreshnessSpec(
            service_name=SERVICE_KALSHI_CHANGELOG,
            label="Kalshi changelog monitor",
            stale_after_seconds=36 * 3600,
        ),
        # Individual ingestion fetchers. 24h staleness is generous enough not
        # to flap on quiet news days or short backoffs, while still surfacing
        # a dead source within a day (the Reddit JSON shutdown went unnoticed
        # for 12 days because only the scheduler-level heartbeat existed).
        # Fetchers disabled by config never write a heartbeat and stay
        # "unknown", which the watchdog does not alert on.
        SERVICE_FETCHER_REDDIT: FreshnessSpec(
            service_name=SERVICE_FETCHER_REDDIT,
            label="Reddit fetcher",
            stale_after_seconds=24 * 3600,
        ),
        SERVICE_FETCHER_GDELT: FreshnessSpec(
            service_name=SERVICE_FETCHER_GDELT,
            label="GDELT fetcher",
            stale_after_seconds=24 * 3600,
        ),
        SERVICE_FETCHER_TAVILY: FreshnessSpec(
            service_name=SERVICE_FETCHER_TAVILY,
            label="Tavily fetcher",
            stale_after_seconds=24 * 3600,
        ),
        SERVICE_FETCHER_NEWSAPI: FreshnessSpec(
            service_name=SERVICE_FETCHER_NEWSAPI,
            label="NewsAPI fetcher",
            stale_after_seconds=24 * 3600,
        ),
        SERVICE_FETCHER_GUARDIAN: FreshnessSpec(
            service_name=SERVICE_FETCHER_GUARDIAN,
            label="Guardian fetcher",
            stale_after_seconds=24 * 3600,
        ),
        SERVICE_FETCHER_TV_ARCHIVE: FreshnessSpec(
            service_name=SERVICE_FETCHER_TV_ARCHIVE,
            label="TV Archive fetcher",
            stale_after_seconds=24 * 3600,
        ),
    }


async def list_service_heartbeats(session: AsyncSession) -> dict[str, ServiceHeartbeatRow]:
    """Return all heartbeat rows keyed by service name."""
    result = await session.execute(select(ServiceHeartbeatRow))
    rows = result.scalars().all()
    return {row.service_name: row for row in rows}


class RuntimeTelemetry:
    """Shared runtime telemetry manager for loops, API, and alerts."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        freshness_specs: dict[str, FreshnessSpec],
        websocket_persist_interval_seconds: int = 30,
    ) -> None:
        self._session_factory = session_factory
        self._freshness_specs = freshness_specs
        self._websocket_persist_interval_seconds = websocket_persist_interval_seconds
        self._websocket_connected: bool | None = None
        self._websocket_subscribed_markets: int | None = None
        self._websocket_last_message_at: datetime | None = None
        self._websocket_last_reconcile_at: datetime | None = None
        self._last_persisted_websocket_message_at: datetime | None = None

    @property
    def freshness_specs(self) -> dict[str, FreshnessSpec]:
        return self._freshness_specs

    def websocket_state(self) -> dict[str, Any]:
        """Return the latest in-memory websocket state."""
        return {
            "connected": self._websocket_connected,
            "subscribed_markets": self._websocket_subscribed_markets,
            "last_message_at": self._websocket_last_message_at,
            "last_reconcile_at": self._websocket_last_reconcile_at,
        }

    async def mark_success(
        self,
        service_name: str,
        *,
        details: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Persist a success heartbeat for a service."""
        ts = now or datetime.now(UTC)
        payload = dict(details or {})
        async with self._session_factory() as session:
            await self._upsert_heartbeat(
                session,
                service_name,
                last_success_at=ts,
                details=payload,
            )
            await session.commit()

    async def mark_error(
        self,
        service_name: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Persist the latest error heartbeat for a service."""
        ts = now or datetime.now(UTC)
        payload = dict(details or {})
        async with self._session_factory() as session:
            await self._upsert_heartbeat(
                session,
                service_name,
                last_error_at=ts,
                last_error_message=message[:1000],
                details=payload,
            )
            await session.commit()

    async def record_event(
        self,
        *,
        service_name: str,
        category: str,
        level: str,
        message: str,
        details: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Insert a timestamped runtime event."""
        ts = now or datetime.now(UTC)
        async with self._session_factory() as session:
            session.add(
                RuntimeEventRow(
                    service_name=service_name,
                    category=category,
                    level=level,
                    message=message[:2000],
                    details=dict(details or {}),
                    created_at=ts,
                )
            )
            await session.commit()

    async def record_kalshi_error(
        self,
        service_name: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Record a Kalshi-facing error event and update the latest error heartbeat."""
        await self.record_event(
            service_name=service_name,
            category=EVENT_CATEGORY_KALSHI_API,
            level="error",
            message=message,
            details=details,
            now=now,
        )
        await self.mark_error(service_name, message, details=details, now=now)

    async def set_websocket_connected(
        self,
        connected: bool,
        *,
        subscribed_markets: int | None = None,
    ) -> None:
        """Update in-memory websocket connection state."""
        self._websocket_connected = connected
        if subscribed_markets is not None:
            self._websocket_subscribed_markets = subscribed_markets

    async def set_websocket_subscribed_markets(self, subscribed_markets: int) -> None:
        """Update the in-memory subscribed market count."""
        self._websocket_subscribed_markets = subscribed_markets

    async def note_websocket_message(
        self,
        *,
        now: datetime | None = None,
        subscribed_markets: int | None = None,
    ) -> None:
        """Track the latest websocket message and persist it on a bounded cadence."""
        ts = now or datetime.now(UTC)
        self._websocket_last_message_at = ts
        if subscribed_markets is not None:
            self._websocket_subscribed_markets = subscribed_markets

        should_persist = (
            self._last_persisted_websocket_message_at is None
            or (ts - self._last_persisted_websocket_message_at).total_seconds()
            >= self._websocket_persist_interval_seconds
        )
        if should_persist:
            self._last_persisted_websocket_message_at = ts
            await self.mark_success(
                SERVICE_POSITION_WATCHER_LAST_MESSAGE,
                details={
                    "connected": bool(self._websocket_connected),
                    "subscribed_markets": self._websocket_subscribed_markets or 0,
                },
                now=ts,
            )

    async def note_websocket_reconcile(self, *, now: datetime | None = None) -> None:
        """Track and persist the latest successful websocket reconciliation."""
        ts = now or datetime.now(UTC)
        self._websocket_last_reconcile_at = ts
        await self.mark_success(
            SERVICE_POSITION_WATCHER_RECONCILE,
            details={"subscribed_markets": self._websocket_subscribed_markets or 0},
            now=ts,
        )

    def evaluate_service_states(
        self,
        heartbeats: dict[str, ServiceHeartbeatRow],
        *,
        run_state: str,
        now: datetime | None = None,
    ) -> list[ServiceFreshnessState]:
        """Compute freshness state for each major long-running service."""
        current = now or datetime.now(UTC)
        states: list[ServiceFreshnessState] = []

        for service_name, spec in self._freshness_specs.items():
            row = heartbeats.get(service_name)
            last_success_at = row.last_success_at if row is not None else None
            last_error_at = row.last_error_at if row is not None else None
            last_error_message = row.last_error_message if row is not None else None
            age_seconds = (
                int((current - last_success_at).total_seconds())
                if last_success_at is not None
                else None
            )

            status = "unknown"
            if service_name == SERVICE_SIGNAL_LOOP and run_state != "running":
                status = "idle"
            elif service_name == SERVICE_POSITION_WATCHER_LAST_MESSAGE:
                status = self._websocket_service_status(
                    current=current,
                    stale_after_seconds=spec.stale_after_seconds,
                )
                if self._websocket_last_message_at is not None:
                    last_success_at = self._websocket_last_message_at
                    age_seconds = int((current - last_success_at).total_seconds())
            elif last_success_at is None:
                status = "unknown"
            elif age_seconds is not None and age_seconds > spec.stale_after_seconds:
                status = "stale"
            else:
                status = "ok"

            states.append(
                ServiceFreshnessState(
                    service_name=service_name,
                    label=spec.label,
                    status=status,
                    last_success_at=last_success_at,
                    last_error_at=last_error_at,
                    last_error_message=last_error_message,
                    stale_after_seconds=spec.stale_after_seconds,
                    age_seconds=age_seconds,
                    alertable=spec.alertable,
                )
            )

        return sorted(states, key=lambda s: s.stale_after_seconds)

    def _websocket_service_status(self, *, current: datetime, stale_after_seconds: int) -> str:
        if self._websocket_connected is None:
            return "unknown"
        if not self._websocket_connected:
            return "stale"
        if not self._websocket_subscribed_markets:
            return "idle"
        if self._websocket_last_message_at is None:
            return "stale"
        age_seconds = int((current - self._websocket_last_message_at).total_seconds())
        return "stale" if age_seconds > stale_after_seconds else "ok"

    async def _upsert_heartbeat(
        self,
        session: AsyncSession,
        service_name: str,
        *,
        last_success_at: datetime | None = None,
        last_error_at: datetime | None = None,
        last_error_message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        ts = datetime.now(UTC)
        # Only include non-null timestamp fields so mark_error doesn't wipe
        # last_success_at and mark_success doesn't wipe last_error_*.
        update_fields: dict[str, Any] = {"details": dict(details or {}), "updated_at": ts}
        if last_success_at is not None:
            update_fields["last_success_at"] = last_success_at
        if last_error_at is not None:
            update_fields["last_error_at"] = last_error_at
        if last_error_message is not None:
            update_fields["last_error_message"] = last_error_message

        stmt = insert(ServiceHeartbeatRow).values(
            service_name=service_name,
            **update_fields,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ServiceHeartbeatRow.service_name],
            set_=update_fields,
        )
        await session.execute(stmt)


async def run_stale_service_watchdog(
    session_factory: async_sessionmaker[AsyncSession],
    telemetry: RuntimeTelemetry,
    alert_dispatcher: Any,
    *,
    interval_seconds: int = 60,
    started_at: datetime | None = None,
) -> None:
    """Periodically send bounded alerts when a critical service becomes stale.

    After a restart the existing heartbeat rows are stale by definition — every
    service needs at least one full interval to complete its first run.  We
    suppress alerts for any service whose ``stale_after_seconds`` has not yet
    elapsed since ``started_at``, giving each service a per-service grace window
    before it can page anyone.
    """
    from freqpred.alerts.run_state import get_run_state  # noqa: PLC0415

    process_started_at = started_at or datetime.now(UTC)
    active_alerts: set[str] = set()
    log.info("runtime_telemetry.watchdog_started", interval_seconds=interval_seconds)

    while True:
        try:
            now = datetime.now(UTC)
            uptime_seconds = (now - process_started_at).total_seconds()

            async with session_factory() as session:
                run_state = await get_run_state(session)
                heartbeats = await list_service_heartbeats(session)

            stale_states = telemetry.evaluate_service_states(
                heartbeats,
                run_state=run_state,
                now=now,
            )
            if run_state != "running":
                active_alerts.clear()
            else:
                for state in stale_states:
                    truly_stale = (
                        state.status == "stale"
                        and state.age_seconds is not None
                        and state.age_seconds > state.stale_after_seconds
                    )
                    # Suppress alerts during the per-service startup grace window.
                    # Each service is allowed one full stale_after_seconds interval
                    # after process start before it can fire an alert.
                    in_grace_period = uptime_seconds < state.stale_after_seconds
                    if truly_stale and state.alertable and not in_grace_period:
                        if state.service_name not in active_alerts:
                            reason = (
                                f"{state.label} stale: no successful progress for "
                                f"{state.age_seconds or 'unknown'}s "
                                f"(threshold {state.stale_after_seconds}s)"
                            )
                            await alert_dispatcher.send(reason)
                            await telemetry.record_event(
                                service_name=state.service_name,
                                category=EVENT_CATEGORY_STALE_SERVICE,
                                level="warning",
                                message=reason,
                            )
                            active_alerts.add(state.service_name)
                    if not truly_stale:
                        active_alerts.discard(state.service_name)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("runtime_telemetry.watchdog_error")

        await asyncio.sleep(interval_seconds)

