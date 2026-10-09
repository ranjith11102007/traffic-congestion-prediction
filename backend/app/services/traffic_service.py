"""Traffic service: storage-side use cases for observations.

Sits between the routes and the repository. Its job is to turn a validated
request into stored rows and back, and to keep the routes free of both ORM
objects and query construction.

Nothing here talks to a data provider or to the ML model; those are
:mod:`app.services.traffic_collector` and :mod:`app.services.prediction_service`
respectively. Keeping them separate means storing an observation never depends on
a model artifact being present.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.exceptions import ResourceNotFoundError
from app.models.traffic import API_SOURCE, TrafficObservation
from app.repositories import traffic_repository as repository
from app.schemas.traffic import (
    MAX_VEHICLE_COUNT,
    TrafficObservationCreate,
    TrafficObservationPage,
)

#: Upper bound on a single page, so one request cannot pull the whole table.
MAX_PAGE_SIZE = 500


def clamp_limit(limit: int | None, default: int = 50) -> int:
    """Bound a caller-supplied ``limit`` to a sane range.

    A client asking for 10_000 rows would otherwise stream the entire table; the
    cap turns that into a bounded response instead of an outage.
    """

    if limit is None:
        return default
    return max(1, min(int(limit), MAX_PAGE_SIZE))


def create_observation(
    session: Session, payload: TrafficObservationCreate
) -> TrafficObservation:
    """Store one observation from an API client.

    Raises:
        DuplicateObservationError: if the location already has a reading at that
            timestamp.
    """

    row = TrafficObservation(
        timestamp=payload.timestamp,
        location_id=payload.location_id,
        road_name=payload.road_name,
        road_segment_id=payload.road_segment_id,
        latitude=payload.latitude,
        longitude=payload.longitude,
        vehicle_count=payload.vehicle_count,
        avg_speed_kph=payload.avg_speed_kph,
        free_flow_speed_kph=payload.free_flow_speed_kph,
        weather_condition=payload.weather_condition,
        temperature_c=payload.temperature_c,
        rainfall_mm=payload.rainfall_mm,
        is_incident=payload.is_incident,
        source=payload.source or API_SOURCE,
    )
    return repository.add_observation(session, row)


def create_observations(
    session: Session, payloads: list[TrafficObservationCreate]
) -> list[TrafficObservation]:
    """Store several observations in one transaction."""

    rows = [
        TrafficObservation(
            timestamp=payload.timestamp,
            location_id=payload.location_id,
            road_name=payload.road_name,
            road_segment_id=payload.road_segment_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            vehicle_count=payload.vehicle_count,
            avg_speed_kph=payload.avg_speed_kph,
            free_flow_speed_kph=payload.free_flow_speed_kph,
            weather_condition=payload.weather_condition,
            temperature_c=payload.temperature_c,
            rainfall_mm=payload.rainfall_mm,
            is_incident=payload.is_incident,
            source=payload.source or API_SOURCE,
        )
        for payload in payloads
    ]
    return [repository.add_observation(session, row) for row in rows]


def list_observations(
    session: Session,
    *,
    location_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int | None = None,
) -> TrafficObservationPage:
    """Return a filtered page of observations, newest first.

    The response echoes the filters applied so a client can confirm the query
    it got matches the one it asked for.
    """

    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError("start_time must not be after end_time")

    effective_limit = clamp_limit(limit)
    rows = repository.list_observations(
        session,
        location_id=location_id,
        start_time=start_time,
        end_time=end_time,
        limit=effective_limit,
    )
    filters: dict[str, object] = {"limit": effective_limit}
    if location_id is not None:
        filters["location_id"] = location_id
    if start_time is not None:
        filters["start_time"] = start_time.isoformat()
    if end_time is not None:
        filters["end_time"] = end_time.isoformat()

    return TrafficObservationPage(
        observations=rows, count=len(rows), limit=effective_limit, filters=filters
    )


def get_observation(session: Session, observation_id: int) -> TrafficObservation:
    """Return one observation.

    Raises:
        ResourceNotFoundError: if no row has that id.
    """

    row = repository.get_observation(session, observation_id)
    if row is None:
        raise ResourceNotFoundError(
            f"No traffic observation with id {observation_id}.",
            details={"observation_id": observation_id},
        )
    return row


def get_latest_observation(
    session: Session,
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
) -> TrafficObservation:
    """Return the most recent observation for a location.

    Raises:
        ResourceNotFoundError: if nothing has been recorded yet. An empty result
            would be indistinguishable from a filter that matched nothing, so a
            missing "latest" is reported as a 404.
    """

    row = repository.get_latest_observation(
        session, location_id=location_id, road_segment_id=road_segment_id
    )
    if row is None:
        raise ResourceNotFoundError(
            "No traffic observation has been recorded yet."
            + (f" None for location_id={location_id!r}." if location_id else ""),
            details={
                "location_id": location_id,
                "road_segment_id": road_segment_id,
            },
        )
    return row


def observation_count(session: Session) -> int:
    """Return the total number of stored observations."""

    return repository.count_observations(session)


__all__ = [
    "MAX_PAGE_SIZE",
    "MAX_VEHICLE_COUNT",
    "clamp_limit",
    "create_observation",
    "create_observations",
    "get_latest_observation",
    "get_observation",
    "list_observations",
    "observation_count",
]