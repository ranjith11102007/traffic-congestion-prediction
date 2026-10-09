"""Response schemas for the dashboard API.

These contracts are what the Stage 7 frontend will consume, so two rules apply
throughout:

**A field is present only when it is real.** A location that has never been
collected reports ``null`` measurements and a ``freshness`` of ``"unavailable"``;
it does not report a zero speed or a "normal" congestion level. A missing value is
visually distinct from a measured one, which is the only way a dashboard can
avoid implying data that was never fetched.

**Nothing internal leaks.** No ids that only mean something to the database, no
column names, no row counts, no DSN, no vendor error bodies, no stack traces. The
dashboard is a product surface, not a database browser.

Freshness and prediction availability are modelled as closed literals rather than
booleans, because "there is no reading" and "the last reading is old" call for
different user interfaces.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: How recent the newest observation is. ``unavailable`` means nothing has been
#: collected for the location at all, which is not the same problem as being late.
Freshness = Literal["fresh", "stale", "unavailable"]

#: Whether a forecast exists for a location, and why not when it does not.
PredictionAvailability = Literal[
    "available",
    "insufficient_history",
    "unscoreable_location",
    "model_unavailable",
    "unavailable",
]


class LatestPredictionBrief(BaseModel):
    """The newest forecast for a location, as the dashboard shows it."""

    predicted_congestion: str = Field(
        description="Predicted level name, e.g. 'moderate'."
    )
    predicted_congestion_level: int = Field(
        description="Predicted class id: 0=free_flow, 1=moderate, 2=heavy, 3=severe."
    )
    prediction_timestamp: datetime = Field(
        description="When the forecast was generated."
    )
    observation_timestamp: datetime = Field(
        description="Time of the traffic state the forecast describes."
    )
    confidence: float | None = Field(
        default=None,
        description=(
            "Highest predicted class probability, or null when the estimator "
            "cannot report one. Not a substitute for validation."
        ),
    )
    model_version: str = Field(description="Model artifact that produced this.")


class LocationCurrent(BaseModel):
    """Everything the dashboard shows for one monitored road right now."""

    location_id: str = Field(description="Configured location identifier.")
    road_name: str = Field(description="Road this location represents.")
    road_segment_id: str | None = Field(
        default=None,
        description="ML road segment used for scoring, or null when unknown.",
    )
    latitude: float | None = Field(
        default=None,
        description="Latitude in degrees, or null when the location has never been "
        "collected and has no configured coordinate.",
    )
    longitude: float | None = Field(
        default=None, description="Longitude in degrees, or null when unavailable."
    )

    # --- Latest traffic reading. Null when nothing has been collected. --------
    observation_timestamp: datetime | None = Field(
        default=None, description="Time of the newest observation, or null if none."
    )
    avg_speed_kph: float | None = Field(
        default=None, description="Mean observed speed in km/h, or null if uncollected."
    )
    free_flow_speed_kph: float | None = Field(
        default=None, description="Uncongested reference speed in km/h, or null."
    )

    # --- Weather, from the same observation. Never fetched separately, so it is
    # # absent exactly when the observation is. ---------------------------------
    weather_condition: str | None = Field(
        default=None, description="Weather label, or null when unavailable."
    )
    temperature_c: float | None = Field(
        default=None, description="Air temperature in Celsius, or null when the "
        "provider did not report one.",
    )
    rainfall_mm: float | None = Field(
        default=None, description="Rainfall in millimetres, or null when unavailable."
    )

    # --- Forecast --------------------------------------------------------------
    prediction: LatestPredictionBrief | None = Field(
        default=None,
        description=(
            "Newest stored forecast, or null when none exists. Never a placeholder "
            "congestion level."
        ),
    )
    prediction_status: PredictionAvailability = Field(
        description="Why a forecast is or is not available for this location."
    )
    freshness: Freshness = Field(
        description="Whether the newest observation is recent enough to trust as "
        "current, judged against DATA_FRESHNESS_THRESHOLD_SECONDS."
    )
    observation_age_seconds: float | None = Field(
        default=None,
        description="Seconds between the newest observation and now, or null when "
        "nothing has been collected. Lets a client age the display itself.",
    )
    data_source: str | None = Field(
        default=None,
        description=(
            "Provenance of the newest observation: 'simulation' for synthetic "
            "development data, 'tomtom' for a live reading."
        ),
    )
    is_simulation: bool = Field(
        default=False,
        description="True when the newest observation is synthetic rather than real "
        "traffic. A dashboard must be able to say so.",
    )


class DashboardCurrentResponse(BaseModel):
    """One response covering every monitored location, for a single poll."""

    generated_at: datetime = Field(
        description="When this response was assembled."
    )
    freshness_threshold_seconds: int = Field(
        description="Threshold used to classify freshness, so the client can render "
        "the same verdict without guessing."
    )
    locations: list[LocationCurrent] = Field(
        default_factory=list,
        description="One entry per monitored location, in configured order."
    )
    location_count: int = Field(description="Number of locations reported.")
    locations_with_data: int = Field(
        description="How many of them have at least one stored observation."
    )
    predictions_available: int = Field(
        description="How many of them have a stored forecast."
    )


class TrendPoint(BaseModel):
    """One point on a congestion trend chart.

    Matches what Chart.js needs: a label, two comparable numeric series, and the
    forecast for that moment. Points with no forecast carry ``null`` for it — the
    chart shows a gap rather than a straight line through it.
    """

    observation_id: int = Field(description="Observation this point describes.")
    timestamp: datetime = Field(description="Time of the observation.")
    location_id: str = Field(description="Location the reading belongs to.")
    avg_speed_kph: float = Field(description="Mean observed speed in km/h.")
    free_flow_speed_kph: float = Field(
        description="Uncongested reference speed in km/h."
    )
    congestion_prediction: str | None = Field(
        default=None,
        description="Forecast label for this observation, or null when none exists.",
    )
    congestion_level: int | None = Field(
        default=None, description="Forecast class id, or null when none exists."
    )
    weather_condition: str | None = Field(
        default=None, description="Weather label at that reading, or null."
    )
    data_source: str | None = Field(
        default=None, description="Provenance of the observation."
    )


class TrendsResponse(BaseModel):
    """A time series ready for charting."""

    generated_at: datetime = Field(description="When this response was assembled.")
    location_id: str | None = Field(
        default=None, description="Location the series covers, or null for all."
    )
    start_time: datetime = Field(description="Inclusive start of the window.")
    end_time: datetime = Field(description="Inclusive end of the window.")
    hours: int = Field(description="Window length in hours, as requested.")
    point_count: int = Field(description="Number of points returned.")
    truncated: bool = Field(
        default=False,
        description=(
            "True when more readings matched than the response limit allows, so the "
            "chart is showing the most recent slice of the window."
        ),
    )
    freshness: Freshness = Field(
        description="Freshness of the newest point in the series."
    )
    points: list[TrendPoint] = Field(
        default_factory=list, description="Series points, oldest first."
    )
    note: str | None = Field(
        default=None,
        description="Set when no readings matched, explaining why rather than "
        "returning an empty chart with no explanation.",
    )


class LocationSummary(BaseModel):
    """Per-location operational state for the locations panel."""

    location_id: str = Field(description="Configured location identifier.")
    road_name: str = Field(description="Road this location represents.")
    road_segment_id: str | None = Field(default=None, description="ML road segment.")
    latitude: float | None = Field(default=None, description="Latitude in degrees.")
    longitude: float | None = Field(default=None, description="Longitude in degrees.")
    configured: bool = Field(
        description="True when the location comes from the monitored-locations file."
    )
    observation_count: int = Field(
        default=0, description="Stored observations for this location."
    )
    latest_observation_time: datetime | None = Field(
        default=None, description="Newest observation time, or null when none."
    )
    latest_prediction_time: datetime | None = Field(
        default=None, description="Newest forecast generation time, or null."
    )
    prediction_status: PredictionAvailability = Field(
        description="Whether a forecast exists for this location, and why not."
    )
    required_readings: int | None = Field(
        default=None,
        description="Prior readings the model needs before it will score.",
    )
    available_readings: int = Field(
        default=0,
        description="Readings that exist ahead of the newest one. Lets a client show "
        "progress towards a first prediction.",
    )
    source: str | None = Field(
        default=None, description="Provenance of the newest observation."
    )
    is_simulation: bool = Field(
        default=False, description="True when the newest observation is synthetic."
    )
    freshness: Freshness = Field(description="Freshness of the newest observation.")
    data_status: str = Field(
        description=(
            "Short human-readable state, e.g. 'live', 'simulated development data', "
            "'stale', 'never collected'."
        )
    )


class LocationsResponse(BaseModel):
    """Configured and observed locations with their current state."""

    generated_at: datetime = Field(description="When this response was assembled.")
    locations: list[LocationSummary] = Field(default_factory=list)
    location_count: int = Field(description="Number of locations reported.")
    monitored_location_count: int = Field(
        description="How many come from the monitored-locations file."
    )


class SystemStatusResponse(BaseModel):
    """Operational status for a dashboard status panel.

    Reports readiness only — whether a dependency is usable — and never any of its
    configuration. A DSN, an API key or a vendor error body echoed here would put
    credentials into a browser, where they are readable by anything on the page.
    """

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall state: degraded when a dependency is usable but impaired."
    )
    service: str = Field(description="Service identifier from configuration.")
    environment: str = Field(description="Environment label from configuration.")
    database_status: Literal["connected", "unavailable", "not_configured"] = Field(
        description="Database reachability. Never includes the DSN."
    )
    model_version: str | None = Field(
        default=None,
        description="Loaded model artifact version, or null when none is available.",
    )
    model_status: Literal["available", "not_available"] = Field(
        description="Whether the trained artifact could be loaded."
    )
    model_trained_on_simulated_data: bool | None = Field(
        default=None,
        description=(
            "True when the artifact was trained on simulated data. Predictions are "
            "demonstrative until a real dataset retrains the model."
        ),
    )
    traffic_provider: str = Field(description="Configured traffic provider name.")
    traffic_provider_configured: bool = Field(
        description="Whether the provider has the credentials it needs to fetch."
    )
    traffic_provider_is_simulation: bool = Field(
        description="True when the provider serves synthetic data."
    )
    weather_provider: str = Field(
        description="Weather provider name. Open-Meteo's free tier needs no key."
    )
    weather_provider_configured: bool = Field(
        description="Whether weather fetching is enabled and has a usable base URL."
    )
    scheduler_enabled: bool = Field(description="Whether background collection is on.")
    scheduler_running: bool = Field(
        description="Whether the background loop is currently scheduled."
    )
    scheduler_interval_seconds: int | None = Field(
        default=None, description="Configured seconds between collection cycles."
    )
    collection_cycles_completed: int = Field(
        default=0, description="Collection cycles that succeeded in this process."
    )
    collection_cycles_failed: int = Field(
        default=0, description="Collection cycles that failed in this process."
    )
    last_collection_time: datetime | None = Field(
        default=None,
        description="Newest observation write time, the closest thing stored to "
        "'the collector last ran'.",
    )
    last_successful_collection_time: datetime | None = Field(
        default=None, description="Newest observation timestamp from a completed cycle."
    )
    last_prediction_time: datetime | None = Field(
        default=None, description="Newest forecast generation time, or null."
    )
    last_error: str | None = Field(
        default=None,
        description=(
            "Most recent collection error, safe to show: the exception type and the "
            "domain message, never a traceback, DSN or vendor response body."
        ),
    )
    observation_count: int = Field(description="Stored observations in total.")
    prediction_count: int = Field(description="Stored forecasts in total.")
    monitored_location_count: int = Field(
        description="Locations in the monitored-locations file, or 0 when none load."
    )
    freshness_threshold_seconds: int = Field(
        description="Threshold used to classify data freshness."
    )
    data_freshness: Freshness = Field(
        description="Freshness of the newest observation in the whole system."
    )


class ModelFeature(BaseModel):
    """One model input, with its origin.

    The origin is what makes the feature list auditable: a reviewer can see which
    columns are measured, which are derived from the timestamp, and which come from
    the vendor's own identifiers.
    """

    name: str = Field(description="Feature column name.")
    origin: Literal["measurement", "lag", "rolling", "calendar", "one_hot"] = Field(
        description="How the feature is produced."
    )
    description: str = Field(description="Human-readable explanation.")


class ModelInfoResponse(BaseModel):
    """What the model is, how good it is, and what it is not good for."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str = Field(description="Version recorded in model_metadata.json.")
    model_type: str = Field(description="Estimator class name.")
    model_display_name: str = Field(description="Human-readable model name.")
    trained_at: datetime | None = Field(
        default=None, description="When the artifact was trained, or null if unknown."
    )
    target_name: str = Field(
        description="Predicted target. Derived from the speed ratio, never an input."
    )
    class_labels: dict[str, str] = Field(
        description="Class id to label mapping."
    )
    feature_count: int = Field(description="Number of model inputs.")
    features: list[str] = Field(
        default_factory=list, description="Model inputs, in training order."
    )
    feature_details: list[ModelFeature] = Field(
        default_factory=list,
        description="Model inputs with their origin and meaning.",
    )
    required_input_columns: list[str] = Field(
        default_factory=list,
        description="Raw observation columns a caller must supply per prediction."
    )
    min_history_per_segment: int = Field(
        default=0,
        description="Prior observations per road segment needed before scoring.",
    )
    lag_steps: list[int] = Field(
        default_factory=list, description="Backward-looking lags the model uses."
    )
    rolling_windows: list[int] = Field(
        default_factory=list, description="Rolling windows the model uses."
    )

    # --- Provenance -----------------------------------------------------------
    dataset_name: str = Field(description="Training dataset file name.")
    dataset_is_simulated: bool = Field(
        description=(
            "True when the artifact was trained on synthetic data. Predictions from "
            "such a model are demonstrative, not validated real-world forecasts."
        )
    )
    dataset_rows_total: int = Field(default=0, description="Rows in the dataset.")

    # --- Evaluation -----------------------------------------------------------
    split_strategy: str = Field(
        description="How train and test were separated, e.g. 'chronological'."
    )
    split_ratio: float = Field(
        description="Fraction held out for evaluation."
    )
    train_records: int = Field(default=0, description="Rows used for training.")
    test_records: int = Field(default=0, description="Rows used for evaluation.")
    train_time_range: list[str] = Field(
        default_factory=list, description="Training window."
    )
    test_time_range: list[str] = Field(default_factory=list, description="Test window.")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Held-out metrics, verbatim from the training run. Empty when no "
            "evaluation was recorded."
        ),
    )
    primary_metric: str | None = Field(
        default=None, description="Which metric the training run treated as primary."
    )
    majority_class_baseline_accuracy: float | None = Field(
        default=None,
        description=(
            "Accuracy of always predicting the most common training class. The "
            "comparison a reader needs before trusting the accuracy above."
        ),
    )

    # --- Limits ----------------------------------------------------------------
    known_limitations: list[str] = Field(
        default_factory=list,
        description=(
            "What this model cannot do. Populated explicitly rather than left "
            "empty so a client can render the caveats instead of implying the "
            "forecast is validated."
        ),
    )
    validated_on_real_traffic: bool = Field(
        default=False,
        description=(
            "Always false while the training data is simulated. Exposed so no "
            "consumer can accidentally present the forecast as field-validated."
        )
    )


__all__ = [
    "DashboardCurrentResponse",
    "Freshness",
    "LatestPredictionBrief",
    "LocationCurrent",
    "LocationSummary",
    "LocationsResponse",
    "ModelFeature",
    "ModelInfoResponse",
    "PredictionAvailability",
    "SystemStatusResponse",
    "TrendPoint",
    "TrendsResponse",
]