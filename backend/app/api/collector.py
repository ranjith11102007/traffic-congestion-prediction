"""Collector routes: trigger and inspect traffic data collection.

These endpoints exist so the ingestion path can be exercised without a scheduler,
and so no client can mistake synthetic development data for live traffic. Every
collection response carries ``provider`` and ``is_simulation``, and the status
endpoint describes what the configured provider can actually do.

``POST /api/traffic/collect`` with the default ``TRAFFIC_PROVIDER=real`` and no
credentials returns 503 rather than simulated readings. That refusal is the whole
point: fabricated traffic stored under a real provider's name is the failure this
design exists to prevent.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.database.session import get_db
from app.schemas.traffic import (
    CollectRequest,
    CollectResponse,
    CollectorStatusResponse,
)
from app.services import dashboard_service, traffic_collector

router = APIRouter(tags=["collector"])


def _settings_dependency() -> Settings:
    """Expose settings to the routes as an importable, overridable dependency."""

    return get_settings()


@router.get(
    "/traffic/collector/status",
    response_model=CollectorStatusResponse,
    summary="Report which traffic provider is configured",
    description=(
        "Names the active provider and states explicitly whether it serves real "
        "traffic or synthetic development data, along with whether it is usable. "
        "The real provider reports that no vendor is connected yet instead of "
        "quietly serving simulated readings."
    ),
)
async def collector_status(
    settings: Settings = Depends(_settings_dependency),
) -> CollectorStatusResponse:
    """Describe the configured provider without contacting it."""

    provider = traffic_collector.get_provider(settings)
    return CollectorStatusResponse(
        provider=provider.name,
        is_simulation=provider.is_simulation,
        configured=provider.is_configured,
        detail=provider.describe(),
    )


@router.post(
    "/traffic/collect",
    response_model=CollectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run one collection from the configured provider, then predict",
    description=(
        "Fetches current readings from the configured provider, stores them, and "
        "then scores each new reading with the trained model.\n\n"
        "Collection happens first and predictions are attempted second, so a "
        "prediction failure can never cost an observation. Each reading is "
        "reported in `predictions` with its outcome: `predicted` with the forecast, "
        "or a reason it was not — `insufficient_history` for the first six readings "
        "of a new location, `model_unavailable` when no artifact is present, "
        "`unscoreable_location` when the road has no ML segment. "
        "`predictions_stored` counts only the forecasts that were persisted.\n\n"
        "This endpoint does not fail when prediction fails; only when the provider "
        "or the database does.\n\n"
        "Set `predict=false` on the request body, or "
        "`AUTO_PREDICT_AFTER_COLLECTION=false` in configuration, to store readings "
        "without scoring them.\n\n"
        "Set persist=false for a dry run that generates data without writing to the "
        "database. Returns 503 if the configured provider is not usable, for example "
        "a real provider with no API key. It never falls back to simulated data."
    ),
)
async def collect_traffic(
    payload: CollectRequest,
    session: Session = Depends(get_db),
    settings: Settings = Depends(_settings_dependency),
) -> CollectResponse:
    """Fetch and store one round of readings, then predict from them."""

    provider = traffic_collector.get_provider(settings)
    observations, outcomes = traffic_collector.collect_and_predict(
        session,
        provider,
        location_id=payload.location_id,
        # The caller's segment wins, so a location the provider did not name can
        # still be scored once the reading exists.
        road_segment_id=payload.road_segment_id,
        persist=payload.persist,
        predict=payload.predict and settings.AUTO_PREDICT_AFTER_COLLECTION,
        settings=settings,
    )

    reports = dashboard_service.prediction_reports(outcomes)

    return CollectResponse(
        provider=provider.name,
        is_simulation=provider.is_simulation,
        observations=observations,
        count=len(observations),
        persisted=payload.persist,
        predictions=reports,
        predictions_stored=sum(1 for item in reports if item["status"] == "predicted"),
    )


@router.post(
    "/traffic/collect/history",
    response_model=CollectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Backfill history so a location becomes predictable",
    description=(
        "Stores a run of prior observations for a location. This is what makes a "
        "fresh location scorable: the model's lag and rolling features need several "
        "readings before a forecast is meaningful.\n\n"
        "Refuses to overwrite an existing history. Silently replacing stored "
        "observations with synthetic ones would destroy real data, so the endpoint "
        "only seeds a location that has nothing recorded yet."
    ),
)
async def collect_history(
    payload: CollectRequest,
    session: Session = Depends(get_db),
    settings: Settings = Depends(_settings_dependency),
) -> CollectResponse:
    """Backfill enough history for prediction to be possible."""

    provider = traffic_collector.get_provider(settings)
    if payload.location_id is None:
        payload = CollectRequest(
            location_id=provider.list_locations()[0],
            road_segment_id=payload.road_segment_id,
            persist=payload.persist,
        )

    observations = traffic_collector.seed_history_for_prediction(
        session,
        provider,
        location_id=payload.location_id,
        road_segment_id=payload.road_segment_id,
        limit=settings.ML_PREDICTION_HISTORY,
    )
    return CollectResponse(
        provider=provider.name,
        is_simulation=provider.is_simulation,
        observations=observations,
        count=len(observations),
        persisted=payload.persist,
        predictions=[],
        predictions_stored=0,
    )


__all__ = ["router"]