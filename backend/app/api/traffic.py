"""Traffic observation routes.

Thin by design: every handler validates via Pydantic, then delegates to a service.
No SQL, no ORM objects and no business rules live here.

Response models are explicit rather than inferred from ORM classes, so an added
column cannot silently start appearing in API responses.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.traffic import TrafficObservation
from app.schemas.traffic import (
    TrafficObservationCreate,
    TrafficObservationPage,
    TrafficObservationResponse,
)
from app.services import traffic_service

router = APIRouter(tags=["traffic"])


@router.post(
    "/traffic/observations",
    response_model=TrafficObservationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a traffic observation",
    description=(
        "Stores one traffic reading and returns it with its assigned id. "
        "A second reading for the same location and timestamp is rejected with "
        "409 Conflict rather than silently overwriting or duplicating."
    ),
)
async def create_observation(
    payload: TrafficObservationCreate,
    session: Session = Depends(get_db),
) -> TrafficObservation:
    """Store a single observation supplied by a client."""

    return traffic_service.create_observation(session, payload)


@router.get(
    "/traffic/observations",
    response_model=TrafficObservationPage,
    summary="List recent traffic observations",
    description=(
        "Returns stored observations newest first, optionally filtered by "
        "location and time window. The response echoes the filters that were "
        "applied so the caller can confirm the query it received."
    ),
)
async def list_observations(
    location_id: str | None = Query(
        default=None, description="Restrict to one sensor or junction."
    ),
    start_time: datetime | None = Query(
        default=None, description="Inclusive lower bound on the observation time."
    ),
    end_time: datetime | None = Query(
        default=None, description="Inclusive upper bound on the observation time."
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum rows to return."),
    session: Session = Depends(get_db),
) -> TrafficObservationPage:
    """Return a filtered page of observations."""

    return traffic_service.list_observations(
        session,
        location_id=location_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


@router.get(
    "/traffic/observations/{observation_id}",
    response_model=TrafficObservationResponse,
    summary="Fetch one traffic observation",
    description=(
        "Returns a single observation by id. Returns 404 with a structured error "
        "envelope if no such observation exists."
    ),
)
async def get_observation(
    observation_id: int,
    session: Session = Depends(get_db),
) -> TrafficObservation:
    """Return one observation by primary key."""

    return traffic_service.get_observation(session, observation_id)


@router.get(
    "/traffic/latest",
    response_model=TrafficObservationResponse,
    summary="Fetch the latest observation",
    description=(
        "Returns the most recent observation, optionally for a specific "
        "location_id. Returns 404 when nothing has been recorded yet, which is "
        "deliberately different from an empty list."
    ),
)
async def get_latest_observation(
    location_id: str | None = Query(
        default=None, description="Restrict to one sensor or junction."
    ),
    road_segment_id: str | None = Query(
        default=None, description="Restrict to one ML road segment, e.g. SEG-01."
    ),
    session: Session = Depends(get_db),
) -> TrafficObservation:
    """Return the most recent observation."""

    return traffic_service.get_latest_observation(
        session, location_id=location_id, road_segment_id=road_segment_id
    )