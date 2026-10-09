"""Dashboard API routes.

Five read-only endpoints shaped for a browser rather than for a data analyst.
The Stage 7 frontend polls them on a timer, so the design pressure is different
from the rest of the API: a single poll must cover every monitored location, and a
partial outage must still render something useful.

The split follows what a page needs to ask:

``GET /api/dashboard/current``
    One call for the whole overview panel — every location, its newest reading,
    its newest forecast and its freshness.
``GET /api/dashboard/trends``
    One chart series, optionally for a single location.
``GET /api/dashboard/locations``
    Per-location operational state, including *why* a location has no forecast.
``GET /api/dashboard/status``
    Readiness of the database, model, providers and scheduler.
``GET /api/dashboard/model``
    The loaded artifact: features, provenance, metrics and known limitations.

These routes contain no query construction and no aggregation. That is
:mod:`app.services.dashboard_service`, so the same responses can be produced from
a script, a test or a future WebSocket push without duplicating any logic.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.database.session import get_db
from app.schemas.dashboard import (
    DashboardCurrentResponse,
    LocationsResponse,
    ModelInfoResponse,
    SystemStatusResponse,
    TrendsResponse,
)
from app.services import dashboard_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _settings_dependency() -> Settings:
    """Expose settings to the routes.

    A named function rather than a lambda so tests can override the dependency
    with ``app.dependency_overrides``.
    """

    return get_settings()


@router.get(
    "/current",
    response_model=DashboardCurrentResponse,
    summary="Current traffic and congestion state for every monitored location",
    description=(
        "Returns the newest observation, newest stored forecast and a freshness "
        "verdict for every location in `config/monitored_locations.json`, plus any "
        "other location that has stored data.\n\n"
        "A location that has never been collected is still listed, with null "
        "measurements and `freshness: \"unavailable\"`. It is deliberately absent "
        "of a zero speed: a road nobody has measured is not a road with no "
        "traffic.\n\n"
        "`prediction_status` distinguishes `insufficient_history` (fewer than six "
        "prior readings) from `model_unavailable` and `unscoreable_location`, "
        "because those three need different things done about them.\n\n"
        "Resolves to a fixed number of queries regardless of how many locations "
        "are monitored."
    ),
)
async def dashboard_current(
    session: Session = Depends(get_db),
    settings: Settings = Depends(_settings_dependency),
) -> DashboardCurrentResponse:
    """Return the current state of every monitored location."""

    return dashboard_service.build_current(session, settings=settings)


@router.get(
    "/trends",
    response_model=TrendsResponse,
    summary="Speed and forecast series for charting",
    description=(
        "Returns observations oldest-first so a time axis needs no re-sorting, "
        "each carrying the forecast made for it.\n\n"
        "An observation with no forecast contributes a null `congestion_prediction`, "
        "which renders as a gap. Interpolating across it would draw a congestion "
        "trend that was never predicted.\n\n"
        "The window comes from `hours` (default 24), or from `start_time` and "
        "`end_time` when given — explicit bounds win. A start after the end is a "
        "422.\n\n"
        "With `require_fresh=true` a 409 is raised when the newest reading in the "
        "window is older than `DATA_FRESHNESS_THRESHOLD_SECONDS`. That is opt-in: "
        "the default returns the stale series with `freshness: \"stale\"`, because "
        "history stays useful after it stops being current.\n\n"
        "`location_id` naming neither a configured nor an observed location is a "
        "404, not an empty series."
    ),
)
async def dashboard_trends(
    location_id: str | None = Query(
        default=None,
        description="Restrict to one location. Omit to include every location.",
        max_length=64,
    ),
    hours: int | None = Query(
        default=None,
        ge=1,
        le=24 * 365,
        description="Window length in hours when no explicit bounds are given.",
    ),
    start_time: datetime | None = Query(
        default=None, description="Inclusive window start. Takes precedence over hours."
    ),
    end_time: datetime | None = Query(
        default=None, description="Inclusive window end. Takes precedence over hours."
    ),
    limit: int | None = Query(
        default=None,
        ge=1,
        le=dashboard_service.MAX_TREND_POINTS,
        description="Maximum points to return. The newest matching points win.",
    ),
    require_fresh: bool = Query(
        default=False,
        description="Return 409 instead of stale data when the newest reading is old.",
    ),
    session: Session = Depends(get_db),
    settings: Settings = Depends(_settings_dependency),
) -> TrendsResponse:
    """Return a charting series of speed and forecast over time."""

    return dashboard_service.build_trends(
        session,
        location_id=location_id,
        hours=hours,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        require_fresh=require_fresh,
        settings=settings,
    )


@router.get(
    "/locations",
    response_model=LocationsResponse,
    summary="Per-location data state and prediction readiness",
    description=(
        "Returns each location with its stored observation count, freshness and "
        "`prediction_status`, which is what lets a client show progress such as "
        "\"5 of 6 readings\" instead of a bare \"no prediction\".\n\n"
        "`available_readings` counts the readings ahead of the newest one, since "
        "that is the number the model's lag features are built from.\n\n"
        "Configured locations come first, in file order, followed by any location "
        "that has data but is not in the file."
    ),
)
async def dashboard_locations(
    session: Session = Depends(get_db),
    settings: Settings = Depends(_settings_dependency),
) -> LocationsResponse:
    """Return the operational state of every location."""

    return dashboard_service.build_locations(session, settings=settings)


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Readiness of the database, model, providers and scheduler",
    description=(
        "Reports whether each dependency is usable, plus how much data is stored "
        "and how fresh it is. Never raises for a broken dependency: a database "
        "outage is reported as `database_status: \"unavailable\"`, because this is "
        "the endpoint a client calls to find out why everything else is failing.\n\n"
        "Exposes no configuration. No DSN, no API key and no vendor response body "
        "appears here, because this payload is read by a browser.\n\n"
        "`status` is `healthy`, `degraded` (usable but impaired — no model, an "
        "unconfigured provider, stale data, or failing collection cycles) or "
        "`unhealthy` (the database is unreachable, so there is nothing to serve)."
    ),
)
async def dashboard_status(
    session: Session = Depends(get_db),
    settings: Settings = Depends(_settings_dependency),
) -> SystemStatusResponse:
    """Return system readiness for the dashboard status panel."""

    return dashboard_service.build_system_status(session, settings=settings)


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Loaded model artifact: features, provenance, metrics and limitations",
    description=(
        "Returns the model version, class labels, every input feature with where "
        "it comes from, how much history a location needs, the training dataset's "
        "provenance, the held-out metrics, and an explicit list of limitations.\n\n"
        "`dataset_is_simulated` and `validated_on_real_traffic` are the two fields "
        "a client should surface next to any forecast it renders. The current "
        "artifact was trained on simulated data, so its predictions demonstrate "
        "the pipeline; they are not field-validated.\n\n"
        "`majority_class_baseline_accuracy` is included beside the model's accuracy "
        "because that accuracy means very little without it.\n\n"
        "Returns 503 when no artifact is present rather than describing a model "
        "that does not exist."
    ),
)
async def dashboard_model(
    settings: Settings = Depends(_settings_dependency),
) -> ModelInfoResponse:
    """Return metadata about the loaded model."""

    return dashboard_service.build_model_info(settings=settings)


__all__ = ["router"]