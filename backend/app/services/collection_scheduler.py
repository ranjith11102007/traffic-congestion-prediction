"""Opt-in background collection of live traffic readings.

Why a scheduler
---------------
Model version 4.0.0 builds lag and rolling features from prior observations and
needs six of them per segment before it will score. TomTom publishes current
conditions only, with no historical feed, so a real deployment has to accumulate
its own history over time. Somebody has to do that collecting.

Why it is off by default
------------------------
``COLLECTION_ENABLED`` defaults to false. Every enabled cycle spends vendor quota
and writes rows to the database, so it must be an explicit operational decision
rather than a side effect of starting the API.

Why it never blocks the API
--------------------------
The loop runs in an :mod:`asyncio` task and hands the synchronous provider and
SQLAlchemy work to a worker thread. FastAPI's endpoints are also synchronous and
run in a thread pool, so a collection cycle can occupy a worker without starving
request handling. The database session is created inside the worker, never shared
across threads.

Failure handling
----------------
One failed cycle must not stop the loop. An exception is logged with its class and
message, and the next cycle runs at the next interval. ``asyncio.CancelledError``
is re-raised, because that is how shutdown is signalled and swallowing it would
make the task unkillable.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import TrafficPredictionError
from app.services.providers import TrafficDataProvider
from app.services.traffic_collector import collect_and_predict, get_provider

logger = logging.getLogger(__name__)

#: The scheduler the running process owns, so a status endpoint can report it
#: without the lifespan having to thread the instance through every dependency.
#: ``None`` when collection is disabled, which is exactly what the dashboard
#: reports as ``scheduler_enabled=false``.
_active_scheduler: "CollectionScheduler | None" = None


def set_active_scheduler(scheduler: "CollectionScheduler | None") -> None:
    """Register the process-wide scheduler instance, or clear it."""

    global _active_scheduler
    _active_scheduler = scheduler


def get_active_scheduler() -> "CollectionScheduler | None":
    """Return the running scheduler, or ``None`` when collection is disabled."""

    return _active_scheduler


@dataclass
class CollectionScheduler:
    """Runs :func:`collect_current` on a fixed interval in the background.

    Attributes:
        settings: runtime configuration, read at start-up.
        session_factory: a zero-argument callable returning a
            :class:`~sqlalchemy.orm.Session`. Used instead of a live session
            because a single session cannot be shared across threads, and the
            loop runs in a worker.
        provider: overrides the configured provider. Tests inject a fake here.
        interval_seconds: overrides ``COLLECTION_INTERVAL_SECONDS``.
    """

    settings: Settings
    session_factory: Callable[[], Session]
    provider: TrafficDataProvider | None = None
    interval_seconds: int | None = None
    predict: bool | None = None

    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    cycles_completed: int = field(default=0, init=False)
    cycles_failed: int = field(default=0, init=False)
    observations_stored: int = field(default=0, init=False)
    predictions_stored: int = field(default=0, init=False)
    predictions_skipped: int = field(default=0, init=False)

    #: When the last cycle finished, when the last one succeeded, and why the last
    #: one failed. Reported by ``GET /api/dashboard/status`` so an operator can see
    #: that collection stopped without reading logs. The error string is the
    #: domain message only — never a traceback, a DSN or a vendor response body.
    last_cycle_at: datetime | None = field(default=None, init=False)
    last_success_at: datetime | None = field(default=None, init=False)
    last_error: str | None = field(default=None, init=False)

    @property
    def interval(self) -> int:
        return self.interval_seconds or self.settings.COLLECTION_INTERVAL_SECONDS

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> bool:
        """Schedule the loop. Returns false if already running.

        The first cycle waits one interval rather than running immediately:
        start-up should not depend on an external API being reachable, and a
        fresh process has just restarted, so the history it needs is already in
        the database.
        """

        if self.is_running:
            logger.warning("Collection scheduler is already running; not starting again.")
            return False

        self._task = asyncio.create_task(self._run(), name="traffic-collection-scheduler")
        logger.info(
            "Collection scheduler started: every %ds via the %s provider",
            self.interval,
            (self.provider or get_provider(self.settings)).name,
        )
        return True

    async def stop(self) -> None:
        """Cancel the loop and wait for it to acknowledge cancellation."""

        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
        logger.info(
            "Collection scheduler stopped after %d cycle(s): %d succeeded, %d failed",
            self.cycles_completed + self.cycles_failed,
            self.cycles_completed,
            self.cycles_failed,
        )

    def run_once(self) -> int:
        """Run one collection cycle synchronously and return rows stored.

        Exposed separately from the loop so a test can exercise the real
        provider/collector/session path without waiting on a timer, and so an
        operator can trigger one cycle on demand.

        Prediction runs in the same cycle, straight after the rows are stored.
        An observation with too little history is kept and reported as unpredicted,
        so a fresh location converges on its own: the readings this cycle stores
        are the history the next one needs.

        Errors are logged and counted rather than raised, matching the loop's
        behaviour: a scheduled cycle has no caller to report to.
        """

        provider = self.provider or get_provider(self.settings)
        should_predict = (
            self.settings.AUTO_PREDICT_AFTER_COLLECTION
            if self.predict is None
            else self.predict
        )
        session: Session = self.session_factory()
        try:
            stored, outcomes = collect_and_predict(
                session,
                provider,
                persist=True,
                predict=should_predict,
                settings=self.settings,
            )
        except Exception as exc:  # noqa: BLE001 - a cycle must never kill the loop
            self.cycles_failed += 1
            self.last_cycle_at = _now()
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Collection cycle failed: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=not isinstance(exc, TrafficPredictionError),
            )
            return 0
        finally:
            session.close()

        self.cycles_completed += 1
        self.observations_stored += len(stored)
        predicted = [outcome for outcome in outcomes if outcome.stored]
        self.predictions_stored += len(predicted)
        self.predictions_skipped += len(outcomes) - len(predicted)
        self.last_cycle_at = _now()
        self.last_success_at = self.last_cycle_at
        self.last_error = None
        logger.info(
            "Collection cycle stored %d observation(s) from the %s provider and "
            "%d prediction(s); %d observation(s) not yet scoreable",
            len(stored),
            provider.name,
            len(predicted),
            len(outcomes) - len(predicted),
        )
        return len(stored)

    async def _run(self) -> None:
        """The loop: wait an interval, collect, repeat until cancelled."""

        while True:
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                raise
            # Off the event loop so provider HTTP calls and the database write do
            # not block request handling.
            await asyncio.to_thread(self.run_once)

    def status(self) -> dict[str, object]:
        """Return a status payload for the health or collector endpoint."""

        return {
            "enabled": self.settings.COLLECTION_ENABLED,
            "running": self.is_running,
            "interval_seconds": self.interval,
            "cycles_completed": self.cycles_completed,
            "cycles_failed": self.cycles_failed,
            "observations_stored": self.observations_stored,
            "predictions_stored": self.predictions_stored,
            "predictions_skipped": self.predictions_skipped,
            "last_cycle_at": self.last_cycle_at,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "provider": (
                self.provider.name
                if self.provider
                else self.settings.TRAFFIC_PROVIDER
            ),
        }


def _now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)


def build_scheduler(settings: Settings | None = None) -> CollectionScheduler:
    """Construct a scheduler wired to the application's session factory.

    The database session factory is imported lazily. Importing it at module load
    would make this module depend on the database engine even in a test that only
    wants to check the interval arithmetic.
    """

    from app.database.session import get_session_factory

    active = settings or get_settings()
    return CollectionScheduler(
        settings=active,
        session_factory=get_session_factory(),
    )


__all__ = [
    "CollectionScheduler",
    "build_scheduler",
    "get_active_scheduler",
    "set_active_scheduler",
]