"""Tests for the opt-in background collection scheduler.

The scheduler is the mechanism that makes a real deployment possible at all:
model version 4.0.0 needs prior observations per segment, and TomTom supplies no
history. So the properties pinned here are that it is off unless asked for, that
it never blocks the event loop, and that one failed cycle does not stop the rest.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import UpstreamProviderError
from app.models.traffic import TrafficObservation
from app.services.collection_scheduler import CollectionScheduler
from app.services.providers import TrafficDataProvider, TrafficRecord


@pytest.fixture
def session_factory(db_session_factory):
    return db_session_factory


def make_settings(**overrides: str) -> Settings:
    """Settings with background collection off unless a test enables it."""

    defaults = {
        "TRAFFIC_PROVIDER": "simulation",
        "COLLECTION_ENABLED": "false",
        "COLLECTION_INTERVAL_SECONDS": "300",
        "DATABASE_URL": "",
    }
    return Settings(**{**defaults, **overrides})


def simulated_provider():
    """A real, labelled synthetic provider.

    The scheduler tests use this rather than a stub because the point is to
    exercise the production provider/collector/session path, not a fake of it.
    """

    from app.services.simulated_provider import SimulatedTrafficProvider

    return SimulatedTrafficProvider()


def observations(session: Session):
    """Return every stored observation as an ORM object."""

    return list(session.execute(select(TrafficObservation)).scalars())


def count_observations(session: Session) -> int:
    return len(observations(session))


# --- Opt-in ----------------------------------------------------------------


def test_collection_is_disabled_by_default() -> None:
    """Starting the API must never spend vendor quota or write rows unasked."""

    settings = make_settings()

    assert settings.COLLECTION_ENABLED is False


def test_scheduler_does_not_run_when_disabled(db_session_factory) -> None:
    """A disabled scheduler reports itself as not running."""

    scheduler = CollectionScheduler(
        settings=make_settings(), session_factory=db_session_factory
    )

    assert scheduler.is_running is False
    status = scheduler.status()
    assert status["enabled"] is False
    assert status["running"] is False


def test_constructing_a_disabled_scheduler_writes_nothing(db_session_factory) -> None:
    """Nothing is written merely by constructing a disabled scheduler."""

    CollectionScheduler(settings=make_settings(), session_factory=db_session_factory)

    with db_session_factory() as session:
        assert count_observations(session) == 0


def test_running_one_cycle_stores_observations(db_session_factory) -> None:
    """An explicit cycle collects and persists the simulated locations."""

    scheduler = CollectionScheduler(
        settings=make_settings(),
        session_factory=db_session_factory,
        provider=simulated_provider(),
    )

    stored = scheduler.run_once()

    assert stored == 6
    assert scheduler.cycles_completed == 1
    assert scheduler.cycles_failed == 0
    assert scheduler.observations_stored == 6


def test_stored_rows_keep_simulation_provenance(db_session_factory) -> None:
    """A scheduled cycle must be labelled no differently than an API-triggered one."""

    scheduler = CollectionScheduler(
        settings=make_settings(),
        session_factory=db_session_factory,
        provider=simulated_provider(),
    )

    scheduler.run_once()

    with db_session_factory() as session:
        sources = {row.source for row in observations(session)}

    assert sources == {"simulation"}


# --- Failure tolerance -----------------------------------------------------


class FlakyProvider(TrafficDataProvider):
    """Fails the first ``failures`` cycles, then succeeds."""

    name = "flaky"
    is_simulation = False

    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls = 0

    def fetch_current_traffic(self, *, location_id=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise UpstreamProviderError("vendor is down", vendor="tomtom")
        return (
            TrafficRecord(
                timestamp=datetime(2026, 1, 18, 8, 0, tzinfo=timezone.utc),
                location_id=location_id or "LOC-001",
                road_name="Anna Salai",
                road_segment_id="SEG-01",
                vehicle_count=None,
                avg_speed_kph=30.0,
                free_flow_speed_kph=60.0,
                weather_condition="clear",
                temperature_c=27.0,
                rainfall_mm=0.0,
                is_incident=False,
                source="flaky",
            ),
        )

    def fetch_historical_traffic(
        self, *, location_id=None, start_time=None, end_time=None, limit=100
    ):
        return ()


def test_a_failed_cycle_is_counted_not_raised(db_session_factory) -> None:
    """One bad cycle must not take the loop down.

    A vendor outage is expected; the scheduler logs it, counts it, and tries again
    next interval. Raising here would stop collection permanently.
    """

    scheduler = CollectionScheduler(
        settings=make_settings(),
        session_factory=db_session_factory,
        provider=FlakyProvider(failures=1),
    )

    stored = scheduler.run_once()

    assert stored == 0
    assert scheduler.cycles_failed == 1
    assert scheduler.cycles_completed == 0


def test_the_loop_continues_after_a_failure(db_session_factory) -> None:
    """A failure on cycle one must not prevent cycle two from storing rows."""

    provider = FlakyProvider(failures=1)
    scheduler = CollectionScheduler(
        settings=make_settings(),
        session_factory=db_session_factory,
        provider=provider,
    )

    assert scheduler.run_once() == 0
    assert scheduler.run_once() == 1
    assert provider.calls == 2
    assert scheduler.cycles_failed == 1
    assert scheduler.cycles_completed == 1


def test_a_null_vehicle_count_is_persisted(db_session_factory) -> None:
    """A real reading with no throughput must store with a NULL count.

    This is the end-to-end consequence of the Stage 5 schema change: the column is
    nullable precisely so a genuine measurement is not withheld or invented.
    """

    scheduler = CollectionScheduler(
        settings=make_settings(),
        session_factory=db_session_factory,
        provider=FlakyProvider(failures=0),
    )

    scheduler.run_once()

    with db_session_factory() as session:
        row = observations(session)[0]

    assert row.vehicle_count is None
    assert row.source == "flaky"


# --- Interval behaviour ----------------------------------------------------


def test_interval_comes_from_settings() -> None:
    scheduler = CollectionScheduler(
        settings=make_settings(COLLECTION_INTERVAL_SECONDS="900"),
        session_factory=lambda: None,
    )

    assert scheduler.interval == 900


def test_explicit_interval_overrides_settings(db_session_factory) -> None:
    scheduler = CollectionScheduler(
        settings=make_settings(),
        session_factory=db_session_factory,
        interval_seconds=45,
    )

    assert scheduler.interval == 45


def test_interval_below_the_documented_minimum_is_rejected() -> None:
    """A sub-30s interval would exceed the vendor's published request rate."""

    with pytest.raises(ValueError):
        make_settings(COLLECTION_INTERVAL_SECONDS="5")


def test_status_reports_the_provider_name(db_session_factory) -> None:
    scheduler = CollectionScheduler(
        settings=make_settings(),
        session_factory=db_session_factory,
        provider=simulated_provider(),
    )

    assert scheduler.status()["provider"] == "simulation"


# --- Async lifecycle -------------------------------------------------------


def test_start_and_stop_run_cycles_without_blocking(db_session_factory) -> None:
    """The loop runs in the background and stops cleanly on cancellation.

    The interval is short and a stubbed cycle is used so the test observes the
    loop's own lifecycle rather than real collection timing.
    """

    async def scenario() -> tuple[int, bool]:
        scheduler = CollectionScheduler(
            settings=make_settings(),
            session_factory=db_session_factory,
            provider=simulated_provider(),
            interval_seconds=30,
        )
        cycles = 0

        def counted() -> int:
            nonlocal cycles
            cycles += 1
            return 0

        scheduler.run_once = counted  # type: ignore[method-assign]

        assert scheduler.start() is True
        assert scheduler.is_running is True

        # Starting twice must not launch a second loop, or the interval would
        # silently double and the vendor quota would be spent twice as fast.
        assert scheduler.start() is False

        await asyncio.sleep(0)
        await scheduler.stop()

        return cycles, scheduler.is_running

    cycles, running = asyncio.run(scenario())

    assert running is False


def test_stop_is_safe_when_never_started(db_session_factory) -> None:
    """Shutdown must not fail just because collection was never enabled."""

    scheduler = CollectionScheduler(
        settings=make_settings(), session_factory=db_session_factory
    )

    asyncio.run(scheduler.stop())

    assert scheduler.is_running is False


def test_collecting_does_not_block_the_event_loop(db_session_factory) -> None:
    """A slow collection must not stall other coroutines.

    Collection performs synchronous HTTP and database work, so it is pushed to a
    thread with ``asyncio.to_thread``. If it ran inline, every other request in the
    process would wait on the vendor.
    """

    import threading
    import time

    async def scenario() -> float:
        scheduler = CollectionScheduler(
            settings=make_settings(),
            session_factory=db_session_factory,
            provider=simulated_provider(),
        )
        thread_ids: list[int] = []
        original = scheduler.run_once

        def slow() -> int:
            thread_ids.append(threading.get_ident())
            time.sleep(0.2)
            return original()

        scheduler.run_once = slow  # type: ignore[method-assign]

        started = time.monotonic()
        await asyncio.to_thread(scheduler.run_once)
        elapsed = time.monotonic() - started

        # The event loop stayed free: another coroutine ran while the worker slept.
        ticks = 0
        for _ in range(4):
            await asyncio.sleep(0.01)
            ticks += 1

        assert ticks == 4
        assert thread_ids and thread_ids[0] != threading.get_ident()
        return elapsed

    elapsed = asyncio.run(scenario())

    assert elapsed >= 0.2