"""Pydantic request and response schemas for the traffic endpoints.

Validation happens here rather than in the routes so it is testable without HTTP
and impossible to bypass. The rules that matter:

* coordinates must fall inside the real globe;
* counts, speeds and rainfall cannot be negative, and speeds are capped at a
  physically plausible maximum so a typo cannot become a model input of 9000 kph;
* ``timestamp`` must be timezone-aware, because a naive timestamp has no
  unambiguous place on the timeline and lag features are built from ordering;
* ``free_flow_speed_kph`` must exceed ``avg_speed_kph``, since free-flow speed is
  the uncongested reference and a lower value is a contradiction;
* weather and road segment values are checked against the levels the trained
  model actually knows, because an unseen category produces a silently wrong
  prediction rather than an error.

Derived target quantities are deliberately **not** accepted. ``speed_ratio`` and
``congestion_level`` are computed from the model's own inputs; taking them from a
caller would reintroduce the leakage Stage 3 removed, so extra fields are
rejected rather than ignored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Physically plausible bounds. A bus on a motorway does not exceed ~200 km/h, and
#: no road segment in the model has a free-flow reference above 150 km/h.
MIN_SPEED_KPH = 0.0
MAX_SPEED_KPH = 200.0
MAX_FREE_FLOW_KPH = 150.0
MAX_RAINFALL_MM = 1_000.0
MIN_TEMPERATURE_C = -100.0
MAX_TEMPERATURE_C = 70.0
MAX_VEHICLE_COUNT = 200_000

NonNegativeFloat = Annotated[float, Field(ge=0)]
SpeedKph = Annotated[float, Field(ge=MIN_SPEED_KPH, le=MAX_SPEED_KPH)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0, description="Latitude in degrees.")]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0, description="Longitude in degrees.")]


class TrafficObservationBase(BaseModel):
    """Fields shared by the create request and the stored-observation response."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    timestamp: datetime = Field(
        description="When the traffic state was observed. Must include a timezone offset.",
    )
    location_id: Annotated[str, Field(min_length=1, max_length=64)] = Field(
        description="Stable sensor or junction identifier.",
    )
    road_name: Annotated[str, Field(min_length=1, max_length=255)] = Field(
        description="Human-readable road name.",
    )
    road_segment_id: Annotated[str, Field(min_length=1, max_length=64)] | None = Field(
        default=None,
        description=(
            "Road segment for ML scoring, e.g. SEG-01. Required to predict."
        ),
    )
    latitude: Latitude | None = Field(default=None, description="Latitude in degrees.")
    longitude: Longitude | None = Field(default=None, description="Longitude in degrees.")

    vehicle_count: Annotated[int, Field(ge=0, le=MAX_VEHICLE_COUNT)] | None = Field(
        default=None,
        description=(
            "Vehicles passing during the interval. Optional: no self-serve traffic "
            "API publishes throughput, so real observations record null rather than "
            "an invented count."
        ),
    )
    avg_speed_kph: SpeedKph = Field(description="Mean observed speed in km/h.")
    free_flow_speed_kph: Annotated[
        float, Field(ge=MIN_SPEED_KPH, le=MAX_FREE_FLOW_KPH)
    ] = Field(description="Uncongested reference speed for this road in km/h.")

    weather_condition: Annotated[str, Field(min_length=1, max_length=32)] = Field(
        description="Weather label matching the model's trained levels.",
    )
    temperature_c: Annotated[
        float, Field(ge=MIN_TEMPERATURE_C, le=MAX_TEMPERATURE_C)
    ] | None = Field(default=None, description="Air temperature in Celsius.")
    rainfall_mm: Annotated[
        float, Field(ge=0, le=MAX_RAINFALL_MM)
    ] | None = Field(default=None, description="Rainfall in millimetres.")
    is_incident: bool | None = Field(
        default=None,
        description="True if an incident or roadworks affected the road.",
    )

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject naive timestamps.

        A timestamp without an offset cannot be ordered unambiguously against
        other readings across a daylight-saving boundary, and every lag feature
        the model uses is built from that ordering.
        """

        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(
                "timestamp must include a timezone offset, e.g. "
                "'2026-01-18T08:00:00+00:00'. A naive timestamp cannot be ordered "
                "reliably against other observations."
            )
        return value

    @field_validator("avg_speed_kph", "free_flow_speed_kph")
    @classmethod
    def reject_non_finite(cls, value: float) -> float:
        """Reject NaN and infinity.

        ``float('nan')`` satisfies every comparison, so it would slip past a
        range check and reach the estimator as a silent no-op.
        """

        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("speed must be a finite number")
        return value

    @model_validator(mode="after")
    def check_speed_relationship(self) -> "TrafficObservationBase":
        """Free-flow speed must exceed observed speed.

        Free-flow speed is the uncongested reference. If a reading says traffic is
        currently faster than free flow, either the reference is wrong or the
        reading is, and either way the label derived from the ratio is meaningless.
        """

        if self.free_flow_speed_kph <= self.avg_speed_kph:
            raise ValueError(
                f"free_flow_speed_kph ({self.free_flow_speed_kph}) must be greater "
                f"than avg_speed_kph ({self.avg_speed_kph}). Free-flow speed is "
                "the uncongested reference, so traffic cannot exceed it."
            )
        return self


class TrafficObservationCreate(TrafficObservationBase):
    """Request body for ``POST /api/traffic/observations``."""

    source: Annotated[str, Field(min_length=1, max_length=64)] | None = Field(
        default=None,
        description=(
            "Origin label. Defaults to 'api'. Use 'simulation' only for synthetic "
            "development data."
        ),
    )


class TrafficObservationResponse(BaseModel):
    """A stored observation as returned by the read endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Database identifier.")
    timestamp: datetime = Field(description="When the state was observed.")
    location_id: str
    road_name: str
    road_segment_id: str | None
    latitude: float | None
    longitude: float | None
    vehicle_count: int | None
    avg_speed_kph: float
    free_flow_speed_kph: float
    weather_condition: str
    temperature_c: float | None
    rainfall_mm: float | None
    is_incident: bool | None
    source: str = Field(
        description=(
            "Origin of the record. 'simulation' means synthetic development data, "
            "not real traffic."
        ),
    )
    created_at: datetime = Field(description="When the row was written.")


class TrafficObservationPage(BaseModel):
    """A page of observations plus the filters applied, so the caller can verify them."""

    observations: list[TrafficObservationResponse] = Field(
        default_factory=list,
        description="Matching observations, newest first.",
    )
    count: int = Field(description="Number of observations in this response.")
    limit: int = Field(description="Maximum rows the query was allowed to return.")
    filters: dict[str, Any] = Field(
        description="Echo of the filters applied, for verification.",
    )


class PredictionRequest(TrafficObservationBase):
    """Request body for ``POST /api/traffic/predict``.

    Narrows three fields from the shared base:

    ``road_segment_id`` becomes required, because the model scores per known road
        segment and silently scoring an unseen one would return a meaningless
        number rather than an error.
    ``temperature_c`` and ``rainfall_mm`` become required. The Stage 3 model takes
        both as pass-through features, so a request that omits them would reach the
        estimator with missing values. Rejecting that at validation time is far
        better than a prediction built on an imputation the model never saw.
    ``source`` becomes explicit. Synthetic development data must declare itself, so
        a stored forecast can never be mistaken for a real observation.

    Unlike the observation endpoints this request always stores what it receives:
    a prediction without a persisted observation cannot be audited or re-scored, so
    there is no "dry run" flag.
    """

    road_segment_id: Annotated[str, Field(min_length=1, max_length=64)] = Field(
        description="Road segment the model was trained on, e.g. SEG-01.",
    )
    temperature_c: Annotated[
        float, Field(ge=MIN_TEMPERATURE_C, le=MAX_TEMPERATURE_C)
    ] = Field(description="Air temperature in Celsius. Required by the model.")
    rainfall_mm: Annotated[float, Field(ge=0, le=MAX_RAINFALL_MM)] = Field(
        description="Rainfall in millimetres. Required by the model."
    )
    source: Annotated[str, Field(min_length=1, max_length=64)] = Field(
        default="api",
        description=(
            "Origin label stored with the observation and the forecast. Use "
            "'simulation' for synthetic development data."
        ),
    )


class PredictionResponse(BaseModel):
    """A stored congestion forecast."""

    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(
        default=None,
        description="Prediction row id, or null when nothing was stored.",
    )
    observation_id: int | None = Field(
        default=None,
        description="Observation the forecast came from.",
    )
    road_segment_id: str | None = None
    observation_timestamp: datetime | None = Field(
        default=None, description="Time of the observation this forecast describes."
    )
    prediction_timestamp: datetime | None = Field(
        default=None,
        description="When the forecast was generated. Shown separately from the "
        "observation time so the lead time is visible.",
    )
    predicted_congestion_level: int = Field(
        description="Predicted class id: 0=free_flow, 1=moderate, 2=heavy, 3=severe."
    )
    predicted_congestion: str = Field(description="Human-readable predicted level.")
    confidence: float | None = Field(
        default=None,
        description=(
            "Highest predicted class probability, or null when the estimator "
            "cannot report probabilities."
        ),
    )
    class_probabilities: dict[str, float] | None = Field(
        default=None,
        description="Full per-class probability distribution.",
    )
    model_version: str = Field(
        description="Version of the Stage 3 model artifact that produced this."
    )
    source: str = Field(
        description="Origin of the underlying observation. 'simulation' is synthetic."
    )
    note: str | None = Field(
        default=None,
        description="Diagnostic recorded with the prediction, if any.",
    )
    created_at: datetime | None = Field(
        default=None, description="When the forecast row was written."
    )


class CollectorStatusResponse(BaseModel):
    """Reports which data provider the collector is using.

    Exists so a client can never mistake synthetic development data for live
    traffic: the provider name and an explicit flag travel with every response.
    """

    provider: str = Field(description="Configured provider name.")
    is_simulation: bool = Field(
        description="True when serving synthetic data that is not real traffic."
    )
    configured: bool = Field(
        description=(
            "False when the selected provider lacks credentials or a base URL. "
            "The collector will raise a configuration error rather than fabricate data."
        ),
    )
    detail: str = Field(description="Human-readable description of the state.")


class CollectRequest(BaseModel):
    """Request body for ``POST /api/traffic/collect``.

    Triggers one fetch from the configured provider and stores the result, which
    is how the collection pipeline is exercised without a scheduler.
    """

    model_config = ConfigDict(extra="forbid")

    location_id: Annotated[str, Field(min_length=1, max_length=64)] | None = Field(
        default=None,
        description="Location to fetch. Defaults to the provider's first location.",
    )
    road_segment_id: Annotated[str, Field(min_length=1, max_length=64)] | None = Field(
        default=None,
        description=(
            "Road segment to attribute the reading to. Required for prediction."
        ),
    )
    persist: bool = Field(
        default=True,
        description="Store the fetched observation. Set false for a dry run.",
    )
    predict: bool = Field(
        default=True,
        description=(
            "Score each stored reading with the trained model. Set false to collect "
            "without predicting; AUTO_PREDICT_AFTER_COLLECTION=false also turns "
            "this off. A dry run (persist=false) never predicts, because there is "
            "no stored row to hang a forecast on."
        ),
    )


class CollectedObservation(BaseModel):
    """One observation returned by a collection run.

    Separate from :class:`TrafficObservationResponse` because a dry run
    (``persist=false``) returns readings that were never written: they have no id
    and no ``created_at``. Reusing the stored-row schema would force those fields
    to be invented, and reporting a fake id for data that does not exist is worse
    than reporting none.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(
        default=None, description="Database id, or null for a dry-run reading."
    )
    timestamp: datetime
    location_id: str
    road_name: str
    road_segment_id: str | None
    vehicle_count: int | None = Field(
        description=(
            "Vehicles counted in the interval, or null when the provider does not "
            "report throughput. No self-serve traffic API does, so a live reading "
            "from TomTom always has null here. Model version 4.0.0 does not use "
            "this field."
        )
    )
    avg_speed_kph: float
    free_flow_speed_kph: float
    weather_condition: str
    temperature_c: float | None
    rainfall_mm: float | None
    is_incident: bool | None
    source: str = Field(
        description=(
            "Origin of the record. 'simulation' means synthetic development data, "
            "not real traffic."
        ),
    )
    created_at: datetime | None = Field(
        default=None, description="Null for a dry-run reading."
    )


class CollectResponse(BaseModel):
    """Outcome of one collection run."""

    provider: str
    is_simulation: bool = Field(
        description="True when the returned record is synthetic, not real traffic."
    )
    observations: list[CollectedObservation] = Field(
        default_factory=list,
        description="Observations that were collected and, unless persist=false, stored.",
    )
    count: int = Field(description="Number of observations in this response.")
    persisted: bool = Field(
        description="True when the readings were written to the database."
    )
    predictions: list[AutoPredictionReport] = Field(
        default_factory=list,
        description=(
            "What automatic prediction did with each stored observation. Empty when "
            "prediction is disabled or nothing was stored."
        ),
    )
    predictions_stored: int = Field(
        default=0,
        description="How many forecasts were generated and stored by this run.",
    )


class PredictionSummary(BaseModel):
    """One stored forecast, shaped for a list or a map tooltip.

    Carries the location and the observation it describes alongside the forecast,
    so a client never has to join tables itself to render a row.
    """

    id: int = Field(description="Prediction row id.")
    observation_id: int = Field(description="Observation this forecast came from.")
    location_id: str = Field(description="Location the underlying observation belongs to.")
    road_name: str = Field(description="Road the observation was recorded on.")
    road_segment_id: str | None = Field(
        default=None, description="ML road segment used for scoring."
    )
    observation_timestamp: datetime = Field(
        description="Time of the traffic state this forecast describes."
    )
    prediction_timestamp: datetime = Field(
        description="When the forecast was generated. Kept separate from the "
        "observation time so the lead time is visible."
    )
    predicted_congestion_level: int = Field(
        description="Predicted class id: 0=free_flow, 1=moderate, 2=heavy, 3=severe."
    )
    predicted_congestion: str = Field(description="Human-readable predicted level.")
    confidence: float | None = Field(
        default=None,
        description=(
            "Highest predicted class probability, or null when the estimator cannot "
            "report one. Never a placeholder."
        ),
    )
    model_version: str = Field(description="Model artifact that produced this.")
    source: str = Field(
        description="Provenance of the underlying observation. 'simulation' is "
        "synthetic development data, not real traffic."
    )
    created_at: datetime = Field(description="When the forecast row was written.")


class PredictionHistoryPage(BaseModel):
    """A page of stored forecasts plus the filters that produced it."""

    predictions: list[PredictionSummary] = Field(
        default_factory=list,
        description="Matching forecasts, newest first.",
    )
    count: int = Field(description="Number of forecasts in this response.")
    limit: int = Field(description="Maximum rows the query was allowed to return.")
    filters: dict[str, Any] = Field(
        description="Echo of the filters applied, for verification."
    )


class AutoPredictionReport(BaseModel):
    """What automatic prediction did with one freshly collected observation.

    Reported per observation rather than as a single summary so a client can tell
    "not enough history yet" apart from "this location is not scorable at all".
    """

    observation_id: int | None = Field(
        default=None, description="Observation the outcome refers to."
    )
    location_id: str = Field(description="Location that was collected.")
    status: str = Field(
        description=(
            "One of 'predicted', 'insufficient_history', 'unscoreable_location' or "
            "'failed'."
        )
    )
    predicted_congestion: str | None = Field(
        default=None, description="Stored forecast label, when one was produced."
    )
    confidence: float | None = Field(
        default=None, description="Model confidence, or null when unavailable."
    )
    model_version: str | None = Field(
        default=None, description="Model that produced the forecast."
    )
    prediction_id: int | None = Field(
        default=None, description="Stored forecast id, when a forecast was produced."
    )
    required_readings: int | None = Field(
        default=None, description="Prior readings the model needs before it will score."
    )
    available_readings: int | None = Field(
        default=None, description="Prior readings that existed when scoring was tried."
    )
    reason: str | None = Field(
        default=None,
        description="Why no forecast exists. Never replaced with a placeholder value.",
    )


class PredictionRunStatus(BaseModel):
    """Health of the ML prediction integration."""

    model_available: bool = Field(description="Whether the Stage 3 artifacts were found.")
    model_version: str | None = Field(
        default=None, description="Version recorded in model_metadata.json."
    )
    dataset_is_simulated: bool | None = Field(
        default=None,
        description=(
            "Whether the model was trained on simulated data. When true, "
            "predictions are demonstrative only."
        ),
    )
    required_input_columns: list[str] = Field(
        default_factory=list,
        description="Raw columns the model needs for each prediction.",
    )
    known_road_segments: list[str] = Field(
        default_factory=list,
        description="Road segments the model can score.",
    )
    known_weather_conditions: list[str] = Field(
        default_factory=list,
        description="Weather labels the model was trained on.",
    )
    min_history_per_segment: int | None = Field(
        default=None,
        description="Prior observations per segment needed before scoring is possible.",
    )


CongestionLiteral = Literal["free_flow", "moderate", "heavy", "severe"]

__all__ = [
    "AutoPredictionReport",
    "CollectedObservation",
    "CollectRequest",
    "CollectResponse",
    "CollectorStatusResponse",
    "MAX_FREE_FLOW_KPH",
    "MAX_RAINFALL_MM",
    "MAX_SPEED_KPH",
    "MAX_TEMPERATURE_C",
    "MAX_VEHICLE_COUNT",
    "MIN_SPEED_KPH",
    "MIN_TEMPERATURE_C",
    "PredictionHistoryPage",
    "PredictionRequest",
    "PredictionResponse",
    "PredictionRunStatus",
    "PredictionSummary",
    "TrafficObservationCreate",
    "TrafficObservationPage",
    "TrafficObservationResponse",
    "utc_now",
]


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)