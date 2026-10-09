"""Congestion prediction routes.

Two entry points, both backed by the Stage 3 model:

``POST /api/traffic/predict``
    Receive an observation, store it, score it against stored history, store the
    forecast, and return both. This is the complete flow the specification calls
    for, and it is the only path that can produce a forecast without a scheduler.

``GET /api/traffic/predictions/status``
    Report whether a model is present, which version, and what it needs. Without
    this, an untrained deployment would only discover the problem by getting a
    503 from a prediction call.

``GET /api/traffic/predictions``
    Read stored forecasts. A dashboard needs history to draw a chart, and that
    history exists in the database whether or not anyone re-scores anything.

``GET /api/traffic/predictions/latest``
    Read the newest forecast for a location or the whole system.

Both routes fail explicitly rather than degrading. A missing model is a 503, a
location with too little history is a 422, and neither ever returns an invented
congestion level. The two read routes are the exception: they have no prediction to
make, so they report what is stored and use 404 only when there is genuinely
nothing to report.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Path, Query, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import NoPredictionAvailableError
from app.database.session import get_db
from app.schemas.traffic import (
    PredictionHistoryPage,
    PredictionRequest,
    PredictionResponse,
    PredictionRunStatus,
    PredictionSummary,
    TrafficObservationCreate,
)
from app.services import prediction_service, traffic_service

router = APIRouter(tags=["prediction"])


def _settings_dependency() -> Settings:
    """Expose settings to the routes.

    A plain function rather than a lambda so the dependency is importable and
    overridable in tests.
    """

    return get_settings()


@router.get(
    "/traffic/predictions",
    response_model=PredictionHistoryPage,
    summary="List stored predictions, newest first",
    description=(
        "Returns stored forecasts joined to the observation each describes, so a "
        "chart can plot a forecast against the traffic state it was made for.\n\n"
        "`start_time` and `end_time` filter on the **observation** timestamp, not "
        "the moment the forecast was generated. A start after the end is a 422 "
        "rather than an empty list, so a mistyped range is not mistaken for a road "
        "with no traffic.\n\n"
        "`limit` defaults to 50 and is capped, because this endpoint has no "
        "pagination cursor and an unbounded read would pull the whole forecast "
        "table into one response."
    ),
)
async def list_predictions(
    location_id: str | None = Query(
        default=None,
        description="Restrict to one location. Omit for every location.",
        max_length=64,
    ),
    start_time: datetime | None = Query(
        default=None, description="Earliest observation time to include."
    ),
    end_time: datetime | None = Query(
        default=None, description="Latest observation time to include."
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=prediction_service.MAX_PREDICTION_PAGE_SIZE,
        description="Maximum rows to return.",
    ),
    session: Session = Depends(get_db),
) -> PredictionHistoryPage:
    """Return a page of stored forecasts."""

    return prediction_service.list_prediction_history(
        session,
        location_id=location_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


@router.get(
    "/traffic/predictions/latest",
    response_model=PredictionSummary,
    summary="Return the most recent stored prediction",
    description=(
        "Returns the newest forecast for one location, or for the whole system "
        "when `location_id` is omitted.\n\n"
        "Returns 404 when nothing has been predicted yet. That is a real answer "
        "about a young database — a new location needs six prior readings — so it "
        "is reported as such rather than as an error, and no placeholder "
        "congestion level is ever substituted."
    ),
)
async def latest_prediction(
    location_id: str | None = Query(
        default=None,
        description="Restrict to one location. Omit for the newest overall.",
        max_length=64,
    ),
    session: Session = Depends(get_db),
) -> PredictionSummary:
    """Return the newest stored forecast."""

    summary = prediction_service.get_latest_prediction(session, location_id=location_id)
    if summary is None:
        raise NoPredictionAvailableError(
            "No prediction has been stored yet"
            + (f" for location_id={location_id!r}" if location_id else "")
            + ". A location needs six prior readings before it can be scored; "
            "collect more observations, then POST /api/traffic/predict.",
            details={"location_id": location_id},
        )
    return summary


@router.get(
    "/traffic/predictions/status",
    response_model=PredictionRunStatus,
    summary="Report ML prediction availability",
    description=(
        "Reports whether a trained model was found, its version, whether it was "
        "trained on simulated data, the raw columns it requires, the road segments "
        "and weather labels it knows, and how much prior history each location "
        "needs before it can be scored. Never raises for a missing model: the "
        "point is to report the state."
    ),
)
async def prediction_status(
    settings: Settings = Depends(_settings_dependency),
) -> PredictionRunStatus:
    """Return the health of the ML integration."""

    return prediction_service.model_status(settings)


@router.post(
    "/traffic/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store an observation and predict its congestion level",
    description=(
        "Persists the supplied observation, scores it with the trained model "
        "using that location's stored history, persists the forecast, and returns "
        "both. Requires road_segment_id, temperature_c and rainfall_mm, because "
        "the model cannot score without them.\n\n"
        "Returns 422 with a clear message when the location has too few prior "
        "observations: the model's lag and rolling features read backwards, so a "
        "location needs a run of readings before any forecast is meaningful. "
        "Derived values such as speed_ratio are never accepted as input."
    ),
)
async def predict_congestion(
    payload: PredictionRequest,
    session: Session = Depends(get_db),
    settings: Settings = Depends(_settings_dependency),
) -> PredictionResponse:
    """Store an observation, predict its congestion level, and persist both."""

    stored = traffic_service.create_observation(
        session, TrafficObservationCreate(**payload.model_dump())
    )
    return prediction_service.predict_for_observation(
        session, stored, settings=settings, store=True
    )


@router.post(
    "/traffic/observations/{observation_id}/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Predict from an already stored observation",
    description=(
        "Scores an observation that is already in the database, which avoids "
        "storing a duplicate when the reading arrived through the collector or a "
        "previous POST. The forecast is stored against that existing row."
    ),
)
async def predict_existing_observation(
    observation_id: int,
    session: Session = Depends(get_db),
    settings: Settings = Depends(_settings_dependency),
) -> PredictionResponse:
    """Score and persist a forecast for a stored observation."""

    observation = traffic_service.get_observation(session, observation_id)
    return prediction_service.predict_for_observation(
        session, observation, settings=settings, store=True
    )


__all__ = ["router"]