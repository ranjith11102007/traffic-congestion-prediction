"""Traffic collector: the seam between data providers and the database.

The collector owns the one piece of policy that both the API-triggered and
scheduled ingestion paths share — translating provider records into stored rows
and refusing to store anything mislabelled. Routes call the collector; the
collector calls a provider; it does not know which provider is in use.

Two invariants are enforced here rather than trusted:

* a record from a simulation provider must be stamped ``source="simulation"``,
  and the stored value is taken from the provider's own flag rather than from the
  record's self-reported field, so a provider cannot mislabel itself;
* duplicates are skipped rather than treated as errors during a collection run,
  because a collector that re-reads the same hour should not fail — whereas a
  single explicit POST of a duplicate is a client mistake and raises.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, DuplicateObservationError
from app.models.traffic import SIMULATION_SOURCE, TrafficObservation
from app.repositories import traffic_repository as repository
from app.services.providers import TrafficDataProvider, TrafficRecord
from app.services.real_provider import RealTrafficProvider
from app.services.simulated_provider import SimulatedTrafficProvider

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from app.services.prediction_service import PredictionOutcome

logger = logging.getLogger(__name__)


def get_provider(settings: Settings | None = None) -> TrafficDataProvider:
    """Return the provider selected by ``TRAFFIC_PROVIDER``.

    Raises:
        ConfigurationError: if the configured name is not implemented. Rejecting
            it matters: falling back to simulation would let fabricated records
            pass under a real provider's name.
    """

    active = settings or get_settings()

    if active.TRAFFIC_PROVIDER == "real":
        return RealTrafficProvider(active)
    if active.TRAFFIC_PROVIDER == "simulation":
        return SimulatedTrafficProvider()

    # Unreachable while the TRAFFIC_PROVIDER validator holds. Kept as a hard
    # failure anyway: silently substituting the simulated provider here would
    # label fabricated traffic with the configured provider's name.
    raise ConfigurationError(
        f"TRAFFIC_PROVIDER={active.TRAFFIC_PROVIDER!r} names no implemented "
        "provider. Use 'simulation' or 'real'."
    )


def record_to_observation(record: TrafficRecord, source: str) -> TrafficObservation:
    """Convert a provider record into an unsaved ORM row.

    Args:
        record: the provider's normalised reading.
        source: provenance label taken from the provider, not from the record.
    """

    return TrafficObservation(
        timestamp=record.timestamp,
        location_id=record.location_id,
        road_name=record.road_name,
        road_segment_id=record.road_segment_id,
        latitude=record.latitude,
        longitude=record.longitude,
        vehicle_count=record.vehicle_count,
        avg_speed_kph=record.avg_speed_kph,
        free_flow_speed_kph=record.free_flow_speed_kph,
        weather_condition=record.weather_condition,
        temperature_c=record.temperature_c,
        rainfall_mm=record.rainfall_mm,
        is_incident=record.is_incident,
        source=source,
    )


def effective_source(provider: TrafficDataProvider, record: TrafficRecord) -> str:
    """Return the provenance label to store.

    A provider flagged as simulating always wins, even if the record claims
    otherwise. That inversion is deliberate: the trust boundary runs from the
    provider to the database, never the other way.
    """

    if provider.is_simulation:
        return SIMULATION_SOURCE
    return record.source or provider.name


def store_records(
    session: Session,
    records: Sequence[TrafficRecord],
    provider: TrafficDataProvider,
    *,
    skip_duplicates: bool = True,
) -> list[TrafficObservation]:
    """Persist provider records and return the rows actually stored.

    Args:
        skip_duplicates: when true (the collector default) a row that already
            exists is logged and skipped; when false a duplicate raises
            :class:`DuplicateObservationError`, which is what the single-record
            POST path wants.

    Returns:
        The stored observations, in input order. Skipped duplicates are absent.
    """

    source = effective_source(provider, records[0]) if records else provider.name
    stored: list[TrafficObservation] = []

    for record in records:
        row = record_to_observation(record, source)
        session.add(row)
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
            if not skip_duplicates:
                raise DuplicateObservationError(
                    "An observation already exists for this location and timestamp.",
                    details={
                        "location_id": record.location_id,
                        "timestamp": record.timestamp.isoformat(),
                    },
                ) from None
            logger.info(
                "Skipping duplicate observation for %s at %s",
                record.location_id,
                record.timestamp.isoformat(),
            )
            continue
        session.refresh(row)
        stored.append(row)

    session.commit()
    logger.info(
        "Stored %d of %d records from the %s provider",
        len(stored),
        len(records),
        provider.name,
    )
    return stored


def collect_current(
    session: Session,
    provider: TrafficDataProvider,
    *,
    location_id: str | None = None,
    persist: bool = True,
) -> list[TrafficObservation]:
    """Fetch current readings and store them.

    Args:
        persist: when false the readings are generated and returned but not
            written, which is what a dry-run check needs.

    Returns:
        The stored observations, or unsaved rows when ``persist`` is false.
    """

    records = provider.fetch_current_traffic(location_id=location_id)
    if not persist:
        source = effective_source(provider, records[0]) if records else provider.name
        return [record_to_observation(record, source) for record in records]
    return store_records(session, records, provider)


def collect_and_predict(
    session: Session,
    provider: TrafficDataProvider,
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
    persist: bool = True,
    predict: bool = True,
    settings: Settings | None = None,
) -> tuple[list[TrafficObservation], list["PredictionOutcome"]]:
    """Collect, store, then try to score — the full Stage 6 pipeline in one call.

    Order is deliberate: the rows are stored **before** any scoring is attempted, so
    a prediction failure can never cost an observation. History is the scarce
    resource in this system, and losing a reading to make room for a forecast would
    be exactly backwards.

    ``road_segment_id`` is applied after the rows exist and before scoring, because
    the segment decides which history the lag features are built from. A caller
    that supplies one can therefore score a location the provider did not name.

    Returns:
        The stored observations and one outcome per observation describing whether
        a forecast was produced and, when not, why. An empty prediction list is a
        normal result for the first six readings of a new location.

    Raises:
        Whatever the provider raises. Upstream failures are *not* swallowed here:
            collection failing and prediction failing mean different things, and an
            operator must be able to tell them apart.
    """

    active = settings or get_settings()
    stored = collect_current(session, provider, location_id=location_id, persist=persist)

    if road_segment_id is not None:
        for observation in stored:
            observation.road_segment_id = road_segment_id
        if persist:
            session.commit()

    if not predict or not persist or not stored:
        return stored, []

    # Imported here rather than at module scope: the collector is the lowest layer
    # of the ingestion path and must stay usable when no model artifact exists.
    from app.services.prediction_service import predict_collected_observations

    return stored, predict_collected_observations(session, stored, settings=active)


def collect_historical(
    session: Session,
    provider: TrafficDataProvider,
    *,
    location_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 24,
    persist: bool = True,
) -> list[TrafficObservation]:
    """Fetch a historical range and store it.

    A history this long is what makes a location predictable: the model's lag and
    rolling features need several prior observations per segment.
    """

    records = provider.fetch_historical_traffic(
        location_id=location_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    if not persist:
        source = effective_source(provider, records[0]) if records else provider.name
        return [record_to_observation(record, source) for record in records]
    return store_records(session, records, provider)


def seed_history_for_prediction(
    session: Session,
    provider: TrafficDataProvider,
    *,
    location_id: str,
    road_segment_id: str | None,
    limit: int,
) -> list[TrafficObservation]:
    """Backfill enough history for a location to become predictable.

    Only acts when the location has no stored readings at all. An existing history
    is never overwritten: silently replacing real observations with synthetic ones
    would be the worst possible outcome here, so this refuses to run once any
    data exists for that location.
    """

    if repository.has_enough_history(session, location_id=location_id, required=1):
        logger.info(
            "Location %s already has stored observations; not seeding history",
            location_id,
        )
        return repository.get_recent_history(session, location_id=location_id, limit=limit)

    known = provider.list_locations()
    if location_id not in known:
        return []

    stored = collect_historical(
        session, provider, location_id=location_id, limit=limit, persist=True
    )
    if road_segment_id is not None:
        for row in stored:
            row.road_segment_id = road_segment_id
        session.commit()
    return stored


__all__ = [
    "collect_and_predict",
    "collect_current",
    "collect_historical",
    "effective_source",
    "get_provider",
    "record_to_observation",
    "seed_history_for_prediction",
    "store_records",
]