"""Dashboard aggregation: the read side of the Stage 6 prediction engine.

What this module is for
-----------------------
A traffic dashboard polls one endpoint and renders every monitored road at once.
That shape dictates three rules, and they are why this is a service rather than
three more routes:

**Two queries, not two hundred.** ``GET /api/dashboard/current`` used as a naive
implementation would ask for the newest observation per location in a loop. A
browser refreshing every few seconds would then issue one query per road per
refresh, which is how a modest deployment turns a dashboard into a load test. Here
the newest observation and the newest forecast for *all* locations come back in two
window-function queries, and every remaining figure is derived in Python from those
rows.

**Absent is not zero.** Every measurement is nullable here, and the services fill
it from what is actually stored. A road that has never been collected reports
``null`` speeds, a ``freshness`` of ``unavailable`` and a prediction status that
explains itself. It never reports 0 km/h, which would render as "a standstill".

**Freshness is measured, not asserted.** ``fresh``/``stale`` is computed from the
stored observation timestamp against a configurable threshold, and the threshold
travels with the response so a client reaches the same verdict instead of guessing.
The word "real-time" is never used for data that is ten minutes old.

Database failures
-----------------
:func:`build_system_status` is the one function that must not raise, because it is
what a client polls to find out *why* everything else is failing. It reports
``database_status="unavailable"`` and continues.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConfigurationError,
    InvalidTimeRangeError,
    StaleDataError,
    UnknownLocationError,
)
from app.models.traffic import SIMULATION_SOURCE
from app.repositories import traffic_repository as repository
from app.schemas.dashboard import (
    DashboardCurrentResponse,
    LatestPredictionBrief,
    LocationCurrent,
    LocationsResponse,
    LocationSummary,
    ModelFeature,
    ModelInfoResponse,
    PredictionAvailability,
    SystemStatusResponse,
    TrendPoint,
    TrendsResponse,
)
from app.services import prediction_service
from app.services.collection_scheduler import get_active_scheduler
from app.services.locations import MonitoredLocation, get_monitored_locations
from app.services.traffic_collector import get_provider

logger = logging.getLogger(__name__)

#: Hard ceiling on one trend response. ``hours`` bounds the window; this bounds
#: the payload, because a five-minute resolution over a year is a million points.
MAX_TREND_POINTS = 2000

#: Provider name reported for weather. Open-Meteo's free tier needs no key, so
#: "configured" here means enabled with a usable base URL.
WEATHER_PROVIDER = "open-meteo"

#: Forecast statuses that mean "the system tried and cannot yet".
_UNSCORABLE = (
    prediction_service.SKIPPED_INSUFFICIENT_HISTORY,
    prediction_service.SKIPPED_NO_SEGMENT,
)

#: Features whose name alone reveals how they were produced. Used to annotate the
#: model's feature list rather than describing it in prose only.
_FEATURE_ORIGINS = (
    ("_lag_", "lag", "This road's own speed, or speed ratio, N readings ago."),
    (
        "_rolling_mean_",
        "rolling",
        "Mean of this road's speed ratio over the last N readings.",
    ),
    ("__", "one_hot", "One-of-N indicator for a category the model was trained on."),
)

_CALENDAR_ORIGINS = {
    "hour": "Local hour of day, derived from the timestamp.",
    "day_of_week": "Day of week, derived from the timestamp.",
    "day_of_month": "Day of month, derived from the timestamp.",
    "is_weekend": "1 for Saturday or Sunday.",
    "month": "Calendar month, derived from the timestamp.",
}

_MEASUREMENT_ORIGINS = {
    "avg_speed_kph": "Mean observed speed, measured by the traffic provider.",
    "free_flow_speed_kph": "Uncongested reference speed for the road.",
    "temperature_c": "Air temperature in Celsius, from the weather provider.",
    "precipitation_mm": "Rainfall in millimetres, from the weather provider.",
    "is_incident": "1 when the traffic provider reports an incident or closure.",
}

#: Stated rather than omitted. A client that renders these is telling the truth
#: about what the numbers mean; one that renders nothing implies validation.
KNOWN_LIMITATIONS = (
    "Trained on simulated/development traffic data, not on measured ground truth. "
    "The congestion label is derived from the ratio of observed speed to free-flow "
    "speed, so the model has learned that relationship, not real-world congestion "
    "dynamics.",
    "Real-world predictive performance has not been scientifically validated. Live "
    "TomTom/Open-Meteo readings feed this unvalidated model.",
    "vehicles counted over time (flow) is not used: no self-serve traffic API "
    "publishes throughput, so the feature was dropped in model version 4.0.0.",
    "A location needs six prior readings on its own road segment before any "
    "forecast exists. New locations show no prediction until then.",
    "No vendor supplies historical traffic, so history is whatever this deployment "
    "has collected itself.",
    "Confidence is the estimator's highest class probability on this observation. "
    "It is not a calibrated probability of being correct in the field.",
)


def _now(now: datetime | None = None) -> datetime:
    """Return a timezone-aware UTC timestamp, defaulting to the wall clock."""

    if now is not None:
        return now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """Force a naive value read from SQLite's DATETIME into UTC.

    SQLite has no timestamp type, so SQLAlchemy returns naive datetimes there even
    for ``TIMESTAMP WITH TIME ZONE`` columns. Comparing one of those with an aware
    ``datetime.now()`` raises, and every freshness calculation would fail on an
    otherwise healthy SQLite-backed test run. The stored values are UTC by
    construction, so attaching the zone is correct rather than a guess.
    """

    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def classify_freshness(
    timestamp: datetime | None,
    *,
    now: datetime | None = None,
    threshold_seconds: int,
) -> tuple[str, float | None]:
    """Classify how recent a stored observation is.

    Args:
        timestamp: the observation time, or ``None`` when nothing is stored.
        now: reference time; injectable so tests are not clock-dependent.
        threshold_seconds: age beyond which data is stale.

    Returns:
        ``(freshness, age_seconds)``. ``unavailable`` means there is no
        observation at all, which is a different problem from being late and is
        reported separately.
    """

    if timestamp is None:
        return "unavailable", None
    observed = _aware(timestamp)
    assert observed is not None  # narrowed by the None check above
    age = (_now(now) - observed).total_seconds()
    return ("fresh" if age <= threshold_seconds else "stale"), max(age, 0.0)


def _configured_locations(settings: Settings) -> tuple[MonitoredLocation, ...]:
    """Return the monitored locations, tolerating an unusable file.

    A dashboard must render even when the locations file is missing or malformed:
    that is exactly when an operator needs to see it. The failure is logged and the
    locations are reported as empty rather than the whole endpoint returning 500.
    """

    try:
        return get_monitored_locations(settings)
    except ConfigurationError as error:
        logger.warning("Monitored locations unavailable: %s", error.message)
        return ()


def _provider_snapshot(settings: Settings) -> tuple[str, bool, bool]:
    """Return ``(provider_name, is_configured, is_simulation)`` without fetching."""

    try:
        provider = get_provider(settings)
    except ConfigurationError as error:
        logger.warning("Traffic provider unavailable: %s", error.message)
        return settings.TRAFFIC_PROVIDER, False, settings.uses_simulation

    return provider.name, provider.is_configured, provider.is_simulation


def _prediction_availability(
    has_prediction: bool,
    *,
    observation_count: int,
    required_readings: int,
    has_segment: bool,
    model_available: bool,
) -> PredictionAvailability:
    """Explain the absence of a forecast instead of hiding it.

    The order matters: "the model is missing" explains every location at once, so
    it is checked first, then an unmappable location, then not-enough-history. A
    caller gets the most actionable reason rather than an arbitrary one.
    """

    if has_prediction:
        return "available"
    if not model_available:
        return "model_unavailable"
    if not has_segment:
        return "unscoreable_location"
    if observation_count <= required_readings:
        # The newest observation is the one being scored, so the readings *ahead*
        # of it are total minus one.
        return "insufficient_history"
    return "unavailable"


def _model_snapshot(settings: Settings) -> Any:
    """Return the model status, never raising.

    ``model_status`` already swallows a missing artifact, so this only guards
    against something unexpected. Either way the dashboard renders, because a
    broken model must not take down the page that explains it is broken.
    """

    try:
        return prediction_service.model_status(settings)
    except Exception:  # noqa: BLE001 - status must never break the dashboard
        logger.exception("Could not read model status")
        return None


def _brief_prediction(row: Mapping[str, Any] | None) -> LatestPredictionBrief | None:
    """Render the newest forecast row for the current-traffic panel."""

    if row is None:
        return None
    return LatestPredictionBrief(
        predicted_congestion=row["predicted_congestion"],
        predicted_congestion_level=row["predicted_congestion_level"],
        prediction_timestamp=row["prediction_timestamp"],
        observation_timestamp=row["observation_timestamp"],
        confidence=prediction_service.validated_confidence(row["confidence"]),
        model_version=row["model_version"],
    )


# ---------------------------------------------------------------------------
# GET /api/dashboard/current
# ---------------------------------------------------------------------------


def build_current(
    session: Session,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> DashboardCurrentResponse:
    """Assemble the current picture for every monitored location.

    Configured locations are always included, even with nothing collected, because
    a dashboard should show that a road is being watched but has no data yet.
    Locations that have observations but are not in the file are appended, so a
    road seeded through the API is never invisible.
    """

    active = settings or get_settings()
    reference = _now(now)
    threshold = active.DATA_FRESHNESS_THRESHOLD_SECONDS
    configured = _configured_locations(active)

    status = _model_snapshot(active)
    model_available = bool(status and status.model_available)
    required_readings = (status.min_history_per_segment if status else 0) or 0

    # Deliberately unfiltered. The configured list decides which never-collected
    # locations are *added*, but an observed location absent from the file still has
    # a real reading and must be shown; restricting these to the file would silently
    # drop it. Both queries stay single round trips because each returns one row per
    # location regardless of how many there are.
    latest = repository.latest_observations_by_location(session)
    forecasts = repository.latest_predictions_by_location(session)
    # One extra aggregate, not one query per location: the panel needs to say
    # "5 of 6 readings so far", which is only knowable from the total.
    counts = repository.observation_counts_by_location(session)

    rows: list[LocationCurrent] = []
    seen: set[str] = set()
    with_data = 0
    with_prediction = 0

    for location in configured:
        seen.add(location.location_id)
        entry = _current_for(
            location.location_id,
            location.road_name,
            latitude=location.latitude,
            longitude=location.longitude,
            segment=location.road_segment_id,
            observation=latest.get(location.location_id),
            forecast=forecasts.get(location.location_id),
            observation_count=counts.get(location.location_id, 0),
            required_readings=required_readings,
            model_available=model_available,
            threshold=threshold,
            reference=reference,
        )
        rows.append(entry)
        with_data += entry.observation_timestamp is not None
        with_prediction += entry.prediction is not None

    # Anything observed but not configured still has a truthful reading to show.
    for location_id in sorted(set(latest) - seen):
        entry = _current_for(
            location_id,
            location_id,
            latitude=None,
            longitude=None,
            segment=None,
            observation=latest.get(location_id),
            forecast=forecasts.get(location_id),
            observation_count=counts.get(location_id, 0),
            required_readings=required_readings,
            model_available=model_available,
            threshold=threshold,
            reference=reference,
        )
        rows.append(entry)
        with_data += entry.observation_timestamp is not None
        with_prediction += entry.prediction is not None

    return DashboardCurrentResponse(
        generated_at=reference,
        freshness_threshold_seconds=threshold,
        locations=rows,
        location_count=len(rows),
        locations_with_data=with_data,
        predictions_available=with_prediction,
    )


def _current_for(
    location_id: str,
    road_name: str,
    *,
    latitude: float | None,
    longitude: float | None,
    segment: str | None,
    observation: Mapping[str, Any] | None,
    forecast: Mapping[str, Any] | None,
    observation_count: int,
    required_readings: int,
    model_available: bool,
    threshold: int,
    reference: datetime,
) -> LocationCurrent:
    """Build one location's current state from already-fetched rows."""

    if observation is None:
        return LocationCurrent(
            location_id=location_id,
            road_name=road_name,
            road_segment_id=segment,
            latitude=latitude,
            longitude=longitude,
            prediction=_brief_prediction(forecast),
            prediction_status=_prediction_availability(
                False,
                observation_count=observation_count,
                required_readings=required_readings,
                has_segment=segment is not None,
                model_available=model_available,
            ),
            freshness="unavailable",
            data_source=None,
            is_simulation=False,
        )

    observed_at = _aware(observation["timestamp"])
    freshness, age = classify_freshness(
        observed_at, now=reference, threshold_seconds=threshold
    )
    source = observation["source"]
    stored_segment = observation["road_segment_id"] or segment

    return LocationCurrent(
        location_id=location_id,
        road_name=observation["road_name"] or road_name,
        road_segment_id=stored_segment,
        # Stored coordinates win: they came back with the reading. The configured
        # coordinate is only a fallback for a location with no data yet.
        latitude=observation["latitude"] if observation["latitude"] is not None else latitude,
        longitude=observation["longitude"]
        if observation["longitude"] is not None
        else longitude,
        observation_timestamp=observed_at,
        avg_speed_kph=observation["avg_speed_kph"],
        free_flow_speed_kph=observation["free_flow_speed_kph"],
        weather_condition=observation["weather_condition"],
        temperature_c=observation["temperature_c"],
        rainfall_mm=observation["rainfall_mm"],
        prediction=_brief_prediction(forecast),
        prediction_status=_prediction_availability(
            forecast is not None,
            observation_count=observation_count,
            required_readings=required_readings,
            has_segment=stored_segment is not None,
            model_available=model_available,
        ),
        freshness=freshness,
        observation_age_seconds=age,
        data_source=source,
        is_simulation=source == SIMULATION_SOURCE,
    )


# ---------------------------------------------------------------------------
# GET /api/dashboard/trends
# ---------------------------------------------------------------------------


def build_trends(
    session: Session,
    *,
    location_id: str | None = None,
    hours: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int | None = None,
    require_fresh: bool = False,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> TrendsResponse:
    """Return a speed-and-forecast series for charting.

    Explicit bounds win over ``hours``: a caller asking for 24 hours *and* a start
    time gets the window they spelled out. Missing points are gaps, not zeroes —
    a straight line drawn across an hour with no readings would invent data.

    Raises:
        UnknownLocationError: ``location_id`` names nothing that exists.
        InvalidTimeRangeError: start is after end.
        StaleDataError: ``require_fresh`` was set and the newest point is too old.
    """

    active = settings or get_settings()
    reference = _now(now)
    threshold = active.DATA_FRESHNESS_THRESHOLD_SECONDS
    window = _resolve_window(
        hours=hours, start_time=start_time, end_time=end_time, now=reference, settings=active
    )
    resolved_start, resolved_end, resolved_hours = window

    if location_id is not None:
        _assert_known_location(session, location_id, settings=active)

    cap = max(1, min(int(limit or MAX_TREND_POINTS), MAX_TREND_POINTS))
    series = repository.observation_series(
        session,
        location_id=location_id,
        start_time=resolved_start,
        end_time=resolved_end,
        limit=cap + 1,
    )
    truncated = len(series) > cap
    if truncated:
        series = series[-cap:]

    points: list[TrendPoint] = []
    if series:
        forecasts = repository.predictions_for_observations(
            session, [row["id"] for row in series]
        )
        for row in series:
            forecast = forecasts.get(row["id"])
            points.append(
                TrendPoint(
                    observation_id=row["id"],
                    timestamp=_aware(row["timestamp"]),
                    location_id=row["location_id"],
                    avg_speed_kph=row["avg_speed_kph"],
                    free_flow_speed_kph=row["free_flow_speed_kph"],
                    congestion_prediction=forecast["predicted_congestion"] if forecast else None,
                    congestion_level=forecast["predicted_congestion_level"] if forecast else None,
                    weather_condition=row["weather_condition"],
                    data_source=row["source"],
                )
            )

    newest = points[-1].timestamp if points else None
    freshness, _ = classify_freshness(
        newest, now=reference, threshold_seconds=threshold
    )

    if require_fresh and freshness != "fresh":
        raise StaleDataError(
            "The newest reading is not within the freshness threshold"
            + (
                f" (it is {freshness})."
                if freshness == "stale"
                else ", and no readings exist in this window."
            )
            + " Drop require_fresh to receive the stale series anyway.",
            details={
                "location_id": location_id,
                "freshness": freshness,
                "freshness_threshold_seconds": threshold,
                "newest_observation_time": newest.isoformat() if newest else None,
            },
        )

    return TrendsResponse(
        generated_at=reference,
        location_id=location_id,
        start_time=resolved_start,
        end_time=resolved_end,
        hours=resolved_hours,
        point_count=len(points),
        truncated=truncated,
        freshness=freshness,
        points=points,
        note=_trend_note(len(points), location_id, freshness),
    )


def _trend_note(point_count: int, location_id: str | None, freshness: str) -> str | None:
    """Explain an empty or partial series rather than returning it unexplained."""

    if point_count:
        if freshness == "stale":
            return (
                "The newest reading in this window is older than "
                "DATA_FRESHNESS_THRESHOLD_SECONDS. The series is accurate but not "
                "current; collection may have stopped."
            )
        return None
    scope = f" for location {location_id}" if location_id else ""
    if freshness == "unavailable":
        return (
            f"No observations have been collected{scope}. Start collection with "
            "COLLECTION_ENABLED=true or POST observations; nothing is invented here."
        )
    return (
        f"No observations were stored{scope} in the requested window. Widen the "
        "window, or check the collection schedule."
    )


def _resolve_window(
    *,
    hours: int | None,
    start_time: datetime | None,
    end_time: datetime | None,
    now: datetime,
    settings: Settings,
) -> tuple[datetime, datetime, int]:
    """Work out the window a trends request covers.

    Explicit bounds take precedence over ``hours`` so that a caller who spelled out
    a range gets it, while a caller who only wants "the last 24 hours" gets that.
    """

    if start_time is not None and end_time is not None and start_time > end_time:
        raise InvalidTimeRangeError(
            f"start_time ({start_time.isoformat()}) must not be after end_time "
            f"({end_time.isoformat()}).",
            details={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
        )

    if start_time is None and end_time is None:
        span = max(1, min(int(hours or 24), settings.DASHBOARD_TREND_MAX_HOURS))
        resolved_end = now
        resolved_start = now - timedelta(hours=span)
    else:
        resolved_start = start_time or now - timedelta(hours=max(1, hours or 24))
        resolved_end = end_time or now
        span = max(1, min(int(hours or 24), settings.DASHBOARD_TREND_MAX_HOURS))

    if resolved_start > resolved_end:
        raise InvalidTimeRangeError(
            f"start_time ({resolved_start.isoformat()}) must not be after end_time "
            f"({resolved_end.isoformat()}).",
            details={
                "start_time": resolved_start.isoformat(),
                "end_time": resolved_end.isoformat(),
            },
        )

    return resolved_start, resolved_end, span


def _assert_known_location(
    session: Session, location_id: str, *, settings: Settings
) -> None:
    """Reject a location that is neither configured nor observed.

    Answering with an empty series instead would be indistinguishable from "this
    road is quiet today", which is precisely the mistake this project keeps
    refusing to make.
    """

    configured = {location.location_id for location in _configured_locations(settings)}
    if location_id in configured:
        return
    observed = repository.observation_counts_by_location(session)
    if location_id in observed:
        return
    raise UnknownLocationError(
        f"Location {location_id!r} is not configured and has no stored observations. "
        f"Configured locations: {sorted(configured) or 'none'}.",
        details={
            "location_id": location_id,
            "configured_locations": sorted(configured)[:25],
        },
    )


# ---------------------------------------------------------------------------
# GET /api/dashboard/locations
# ---------------------------------------------------------------------------


def build_locations(
    session: Session,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> LocationsResponse:
    """Report each location's data state, including why it has no prediction.

    ``available_readings`` counts the readings *ahead* of the newest one, because
    that is the number that determines whether the next cycle can produce a
    forecast. It is what lets a panel show "5 of 6 readings" instead of just
    "no prediction".
    """

    active = settings or get_settings()
    reference = _now(now)
    threshold = active.DATA_FRESHNESS_THRESHOLD_SECONDS
    configured = _configured_locations(active)

    status = _model_snapshot(active)

    model_available = bool(status and status.model_available)
    required_readings = (status.min_history_per_segment if status else 0) or 0

    latest = repository.latest_observations_by_location(session)
    forecasts = repository.latest_predictions_by_location(session)
    counts = repository.observation_counts_by_location(session)

    rows: list[LocationSummary] = []
    seen: set[str] = set()
    for location in configured:
        seen.add(location.location_id)
        rows.append(
            _summary_for(
                location.location_id,
                location.road_name,
                latitude=location.latitude,
                longitude=location.longitude,
                segment=location.road_segment_id,
                configured=True,
                observation=latest.get(location.location_id),
                forecast=forecasts.get(location.location_id),
                observation_count=counts.get(location.location_id, 0),
                required_readings=required_readings,
                model_available=model_available,
                threshold=threshold,
                reference=reference,
            )
        )

    for location_id in sorted(set(latest) - seen):
        rows.append(
            _summary_for(
                location_id,
                location_id,
                latitude=None,
                longitude=None,
                segment=None,
                configured=False,
                observation=latest.get(location_id),
                forecast=forecasts.get(location_id),
                observation_count=counts.get(location_id, 0),
                required_readings=required_readings,
                model_available=model_available,
                threshold=threshold,
                reference=reference,
            )
        )

    return LocationsResponse(
        generated_at=reference,
        locations=rows,
        location_count=len(rows),
        monitored_location_count=len(configured),
    )


def _summary_for(
    location_id: str,
    road_name: str,
    *,
    latitude: float | None,
    longitude: float | None,
    segment: str | None,
    configured: bool,
    observation: Mapping[str, Any] | None,
    forecast: Mapping[str, Any] | None,
    observation_count: int,
    required_readings: int,
    model_available: bool,
    threshold: int,
    reference: datetime,
) -> LocationSummary:
    """Build one row of the locations panel."""

    newest = _aware(observation["timestamp"]) if observation else None
    freshness, _ = classify_freshness(newest, now=reference, threshold_seconds=threshold)
    stored_segment = (observation["road_segment_id"] if observation else None) or segment
    source = observation["source"] if observation else None
    status = _prediction_availability(
        forecast is not None,
        observation_count=observation_count,
        required_readings=required_readings,
        has_segment=stored_segment is not None,
        model_available=model_available,
    )

    return LocationSummary(
        location_id=location_id,
        road_name=observation["road_name"] if observation else road_name,
        road_segment_id=stored_segment,
        latitude=(observation["latitude"] if observation else None) or latitude,
        longitude=(observation["longitude"] if observation else None) or longitude,
        configured=configured,
        observation_count=observation_count,
        latest_observation_time=newest,
        latest_prediction_time=forecast["prediction_timestamp"] if forecast else None,
        prediction_status=status,
        required_readings=required_readings,
        available_readings=max(observation_count - 1, 0),
        source=source,
        is_simulation=source == SIMULATION_SOURCE,
        freshness=freshness,
        data_status=_data_status(status, freshness, source),
    )


def _data_status(status: str, freshness: str, source: str | None) -> str:
    """One short phrase for a status column."""

    if freshness == "unavailable":
        return "never collected"
    if source == SIMULATION_SOURCE:
        return "simulated development data"
    if freshness == "stale":
        return f"stale ({status.replace('_', ' ')})" if status != "available" else "stale"
    if status == "available":
        return "live"
    return status.replace("_", " ")


# ---------------------------------------------------------------------------
# GET /api/dashboard/status
# ---------------------------------------------------------------------------


def build_system_status(
    session: Session,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> SystemStatusResponse:
    """Report operational readiness without exposing any configuration.

    This is the endpoint a client calls when something is wrong, so it must not be
    the thing that raises: a database outage is reported as a field, not as a 500.
    Every value here is either a flag or a timestamp. No DSN, no key, no vendor
    response body and no exception text from a third party.
    """

    active = settings or get_settings()
    reference = _now(now)
    threshold = active.DATA_FRESHNESS_THRESHOLD_SECONDS

    database_status: str
    observation_count = 0
    prediction_count = 0
    newest_observation: datetime | None = None
    newest_write: datetime | None = None
    newest_prediction: datetime | None = None

    # Probed rather than inferred from configuration. The session arrives through a
    # dependency, so by the time this runs a usable database may already exist even
    # when no DATABASE_URL is set in the environment — a test override, or an
    # application wiring its own engine. Reporting `not_configured` there would be
    # declaring the service unhealthy while it is demonstrably answering queries.
    try:
        session.execute(text("SELECT 1"))
        database_status = "connected"
    except SQLAlchemyError:
        # The session may be unusable after a connection failure; roll it back so
        # the request can still close cleanly.
        logger.exception("Dashboard status could not reach the database")
        try:
            session.rollback()
        except SQLAlchemyError:  # pragma: no cover - already failing
            logger.exception("Could not roll back after a database failure")
        # Distinguishing the two tells an operator whether to fix a DSN or chase a
        # down server.
        database_status = (
            "unavailable" if active.has_database_configured else "not_configured"
        )

    if database_status == "connected":
        observation_count = repository.count_observations(session)
        prediction_count = repository.count_predictions(session)
        newest_observation = _aware(repository.latest_observation_time(session))
        newest_write = _aware(repository.latest_observation_write_time(session))
        newest_prediction = _aware(repository.latest_prediction_time(session))

    freshness, _ = classify_freshness(
        newest_observation, now=reference, threshold_seconds=threshold
    )

    provider_name, provider_configured, provider_is_simulation = _provider_snapshot(active)

    model_status = prediction_service.model_status(active)
    model_available = bool(model_status.model_available)

    scheduler = get_active_scheduler()
    scheduler_enabled = bool(active.COLLECTION_ENABLED)
    scheduler_running = bool(scheduler.is_running) if scheduler is not None else False

    monitored = _configured_locations(active)

    state = _overall_status(
        database_status=database_status,
        model_available=model_available,
        provider_configured=provider_configured,
        freshness=freshness,
        scheduler_failed=bool(scheduler and scheduler.cycles_failed),
    )

    return SystemStatusResponse(
        status=state,
        service=active.SERVICE_NAME,
        environment=active.ENVIRONMENT,
        database_status=database_status,
        model_version=model_status.model_version,
        model_status="available" if model_available else "not_available",
        model_trained_on_simulated_data=model_status.dataset_is_simulated,
        traffic_provider=provider_name,
        traffic_provider_configured=provider_configured,
        traffic_provider_is_simulation=provider_is_simulation,
        weather_provider=WEATHER_PROVIDER,
        weather_provider_configured=bool(
            active.WEATHER_ENABLED and active.WEATHER_API_BASE_URL.strip()
        ),
        scheduler_enabled=scheduler_enabled,
        scheduler_running=scheduler_running,
        scheduler_interval_seconds=active.COLLECTION_INTERVAL_SECONDS
        if scheduler_enabled
        else None,
        collection_cycles_completed=scheduler.cycles_completed if scheduler else 0,
        collection_cycles_failed=scheduler.cycles_failed if scheduler else 0,
        last_collection_time=newest_write,
        last_successful_collection_time=scheduler.last_success_at
        if scheduler
        else newest_observation,
        last_prediction_time=newest_prediction,
        last_error=scheduler.last_error if scheduler else None,
        observation_count=observation_count,
        prediction_count=prediction_count,
        monitored_location_count=len(monitored),
        freshness_threshold_seconds=threshold,
        data_freshness=freshness,
    )


def _overall_status(
    *,
    database_status: str,
    model_available: bool,
    provider_configured: bool,
    freshness: str,
    scheduler_failed: bool,
) -> str:
    """Roll the individual states into one verdict.

    ``degraded`` rather than ``unhealthy`` for anything recoverable: the API is
    still serving real data, so a dashboard can render it. Only an unusable
    database is ``unhealthy``, because then there is nothing left to show.
    """

    if database_status != "connected":
        return "unhealthy"
    if not model_available or not provider_configured:
        return "degraded"
    if freshness == "stale" or scheduler_failed:
        return "degraded"
    return "healthy"


# ---------------------------------------------------------------------------
# GET /api/dashboard/model
# ---------------------------------------------------------------------------


def build_model_info(*, settings: Settings | None = None) -> ModelInfoResponse:
    """Report the loaded artifact: features, metrics, provenance and limits.

    Reads ``model_metadata.json`` rather than restating anything here, so retraining
    cannot leave this endpoint describing a model that no longer exists.

    Raises:
        ModelNotAvailableError: no artifact is present, which is a 503. An endpoint
            that invented a version string for a missing model would be worse than
            refusing.
    """

    active = settings or get_settings()
    bundle = prediction_service.load_model_bundle(active)
    metadata = bundle.metadata

    features = [str(name) for name in metadata.get("features", [])]
    return ModelInfoResponse(
        model_version=bundle.model_version,
        model_type=str(metadata.get("model_type", "unknown")),
        model_display_name=str(
            metadata.get("model_display_name", metadata.get("model_name", "unknown"))
        ),
        trained_at=_parse_timestamp(metadata.get("trained_at")),
        target_name=str(metadata.get("target_name", "congestion_level")),
        class_labels=bundle.label_mapping,
        feature_count=int(metadata.get("feature_count", len(features))),
        features=features,
        feature_details=[_describe_feature(name) for name in features],
        required_input_columns=list(bundle.required_inputs),
        min_history_per_segment=bundle.min_history,
        lag_steps=list(getattr(bundle.schema, "lag_steps", ()) or ()),
        rolling_windows=list(getattr(bundle.schema, "rolling_windows", ()) or ()),
        dataset_name=str(metadata.get("dataset_name", "unknown")),
        dataset_is_simulated=bool(metadata.get("dataset_is_simulated")),
        dataset_rows_total=int(metadata.get("dataset_rows_total", 0)),
        split_strategy=str(metadata.get("split_strategy", "unknown")),
        split_ratio=float(metadata.get("split_ratio", 0.0)),
        train_records=int(metadata.get("training_records", 0)),
        test_records=int(metadata.get("test_records", 0)),
        train_time_range=[str(value) for value in metadata.get("train_time_range", [])],
        test_time_range=[str(value) for value in metadata.get("test_time_range", [])],
        metrics=_scalable_metrics(metadata.get("training_run_metrics") or {}),
        primary_metric=_primary_metric(metadata.get("training_run_metrics") or {}),
        majority_class_baseline_accuracy=majority_baseline_accuracy(metadata),
        known_limitations=list(KNOWN_LIMITATIONS),
        validated_on_real_traffic=not bool(metadata.get("dataset_is_simulated")),
    )


def _scalable_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep the numeric headline metrics, dropping nested reports.

    The full classification report and confusion matrix live in
    ``ml/models/model_report.md`` and ``evaluation_results.json``. Returning them
    as JSON here would bury the four numbers a reader actually needs.
    """

    scalars: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
    return scalars


def _primary_metric(metrics: Mapping[str, Any]) -> str | None:
    """Name the metric the training run treated as primary, when it said."""

    if metrics.get("macro_f1_is_primary"):
        return "f1_macro"
    return None


def _describe_feature(name: str) -> ModelFeature:
    """Explain one feature column from its name and the ML contract."""

    if name in _CALENDAR_ORIGINS:
        return ModelFeature(
            name=name, origin="calendar", description=_CALENDAR_ORIGINS[name]
        )
    for marker, origin, description in _FEATURE_ORIGINS:
        if marker in name:
            if origin == "one_hot":
                source, _, level = name.partition("__")
                return ModelFeature(
                    name=name,
                    origin="one_hot",
                    description=f"1 when {source.replace('_', ' ')} is {level.replace('_', ' ')}, else 0.",
                )
            return ModelFeature(name=name, origin=origin, description=description)
    return ModelFeature(
        name=name,
        origin="measurement",
        description=_MEASUREMENT_ORIGINS.get(
            name, "Measured value taken from the observation record."
        ),
    )


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a metadata timestamp, tolerating the formats the trainer emits."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Model metadata has an unparseable trained_at: %r", value)
        return None
    return _aware(parsed)


def majority_baseline_accuracy(metadata: Mapping[str, Any] | None = None) -> float | None:
    """Return the accuracy of always predicting the most common training class.

    Computed from the recorded class distribution rather than hard-coded, so the
    comparison stays true across retraining. Reported next to the model's accuracy
    because an accuracy means very little without it.
    """

    source = metadata or {}
    if "majority_class_baseline_accuracy" in source:
        value = source["majority_class_baseline_accuracy"]
        return float(value) if isinstance(value, (int, float)) else None

    distribution = source.get("train_class_distribution") or {}
    total = sum(int(count) for count in distribution.values())
    if not total:
        return None
    return max(int(count) for count in distribution.values()) / total


def freshness_threshold(settings: Settings | None = None) -> int:
    """Expose the configured freshness threshold in seconds."""

    return (settings or get_settings()).DATA_FRESHNESS_THRESHOLD_SECONDS


def prediction_reports(
    outcomes: Iterable[Any],
) -> list[dict[str, Any]]:
    """Convert prediction outcomes into the API response shape.

    Lives here so both the collector route and the scheduler report identical
    payloads, and so the mapping is written once.
    """

    from app.schemas.traffic import AutoPredictionReport

    reports: list[dict[str, Any]] = []
    for outcome in outcomes:
        prediction = getattr(outcome, "prediction", None)
        reports.append(
            AutoPredictionReport(
                observation_id=getattr(outcome, "observation_id", None) or None,
                location_id=outcome.location_id,
                status=outcome.status,
                predicted_congestion=prediction.predicted_congestion if prediction else None,
                confidence=prediction.confidence if prediction else None,
                model_version=prediction.model_version if prediction else None,
                prediction_id=prediction.id if prediction else None,
                required_readings=outcome.required_readings,
                available_readings=outcome.available_readings,
                reason=outcome.reason,
            ).model_dump()
        )
    return reports


__all__ = [
    "KNOWN_LIMITATIONS",
    "MAX_TREND_POINTS",
    "WEATHER_PROVIDER",
    "build_current",
    "build_locations",
    "build_model_info",
    "build_system_status",
    "build_trends",
    "classify_freshness",
    "freshness_threshold",
    "majority_baseline_accuracy",
    "prediction_reports",
]