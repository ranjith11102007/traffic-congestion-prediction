"""Tests for the traffic Pydantic schemas.

These cover the validation rules that exist to stop bad data reaching the model or
the database: out-of-range values, naive timestamps, contradictory speeds,
unknown categories, and any attempt to smuggle a derived target column past the
API.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.traffic import (
    MAX_SPEED_KPH,
    MAX_TEMPERATURE_C,
    MAX_VEHICLE_COUNT,
    PredictionRequest,
    TrafficObservationCreate,
    utc_now,
)


def test_valid_observation_is_accepted(observation_payload: dict) -> None:
    """The baseline payload must be accepted; everything else builds on it."""

    created = TrafficObservationCreate(**observation_payload)

    assert created.location_id == "LOC-001"
    assert created.road_segment_id == "SEG-01"
    assert created.timestamp.tzinfo is not None


def test_naive_timestamp_is_rejected(observation_payload: dict) -> None:
    """A timestamp without an offset cannot be ordered against other readings.

    Lag features are built from that ordering, so an ambiguous timestamp is a
    correctness problem, not a cosmetic one.
    """

    payload = {**observation_payload, "timestamp": "2026-01-18T08:00:00"}

    with pytest.raises(ValidationError, match="timezone"):
        TrafficObservationCreate(**payload)


def test_free_flow_speed_must_exceed_observed_speed(observation_payload: dict) -> None:
    """Traffic cannot be faster than the uncongested reference."""

    payload = {**observation_payload, "avg_speed_kph": 70.0, "free_flow_speed_kph": 60.0}

    with pytest.raises(ValidationError, match="free_flow_speed_kph"):
        TrafficObservationCreate(**payload)


def test_equal_speeds_are_rejected(observation_payload: dict) -> None:
    """Equality is a contradiction too, not a boundary case to allow."""

    payload = {**observation_payload, "avg_speed_kph": 60.0, "free_flow_speed_kph": 60.0}

    with pytest.raises(ValidationError, match="free_flow_speed_kph"):
        TrafficObservationCreate(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vehicle_count", -1),
        ("vehicle_count", MAX_VEHICLE_COUNT + 1),
        ("avg_speed_kph", -0.1),
        ("avg_speed_kph", MAX_SPEED_KPH + 1),
        ("free_flow_speed_kph", 151.0),
        ("latitude", 91.0),
        ("latitude", -91.0),
        ("longitude", 181.0),
        ("longitude", -181.0),
        ("rainfall_mm", -0.5),
        ("temperature_c", MAX_TEMPERATURE_C + 1),
        ("temperature_c", -101.0),
    ],
)
def test_out_of_range_values_are_rejected(
    observation_payload: dict, field: str, value: float
) -> None:
    """Physical and geographic bounds are enforced before storage."""

    payload = copy.deepcopy(observation_payload)
    payload[field] = value

    with pytest.raises(ValidationError):
        TrafficObservationCreate(**payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_speeds_are_rejected(observation_payload: dict, value: float) -> None:
    """NaN satisfies every comparison, so it must be rejected explicitly.

    Left alone it would pass a naive range check and reach the estimator as a
    silent no-op.
    """

    payload = {**observation_payload, "avg_speed_kph": 1.0, "free_flow_speed_kph": 100.0}
    payload["avg_speed_kph"] = value

    with pytest.raises(ValidationError):
        TrafficObservationCreate(**payload)


def test_unknown_field_is_rejected(observation_payload: dict) -> None:
    """Extra fields are refused rather than ignored.

    Silently dropping ``speed_ratio`` or ``congestion_level`` would let a caller
    believe they had supplied a value the model actually computes itself.
    """

    payload = {**observation_payload, "speed_ratio": 0.53}

    with pytest.raises(ValidationError, match="speed_ratio"):
        TrafficObservationCreate(**payload)


def test_derived_target_cannot_be_supplied(observation_payload: dict) -> None:
    """``congestion_level`` is the ML target and must never be an API input."""

    payload = {**observation_payload, "congestion_level": 2}

    with pytest.raises(ValidationError, match="congestion_level"):
        TrafficObservationCreate(**payload)


def test_prediction_request_requires_road_segment(observation_payload: dict) -> None:
    """Scoring an unknown segment would return a meaningless number."""

    payload = {**observation_payload}
    payload.pop("road_segment_id")

    with pytest.raises(ValidationError, match="road_segment_id"):
        PredictionRequest(**payload)


def test_prediction_request_requires_model_features(observation_payload: dict) -> None:
    """Temperature and rainfall are model inputs, so a prediction cannot omit them.

    Without this the estimator would receive missing values and the API would have
    to impute something the model was never trained on.
    """

    for field in ("temperature_c", "rainfall_mm"):
        payload = {**observation_payload}
        payload.pop(field)
        with pytest.raises(ValidationError, match=field):
            PredictionRequest(**payload)


def test_prediction_request_always_persists(observation_payload: dict) -> None:
    """There is no dry-run flag on the prediction endpoint.

    A forecast with no stored observation cannot be audited or re-scored, so the
    option was removed rather than defaulted.
    """

    request = PredictionRequest(**observation_payload)

    assert not hasattr(request, "store_observation")
    assert "store_observation" not in request.model_dump()


def test_optional_observation_fields_default_to_none() -> None:
    """Context fields stay optional on the storage path.

    Storing an observation does not require a weather reading, whereas predicting
    does. Both are the same schema with different strictness.
    """

    payload = {
        "timestamp": "2026-01-18T08:00:00+00:00",
        "location_id": "LOC-009",
        "road_name": "Unmeasured Road",
        "vehicle_count": 100,
        "avg_speed_kph": 40.0,
        "free_flow_speed_kph": 60.0,
        "weather_condition": "clear",
    }

    created = TrafficObservationCreate(**payload)

    assert created.temperature_c is None
    assert created.rainfall_mm is None
    assert created.is_incident is None
    assert created.latitude is None
    assert created.source is None


def test_utc_now_is_timezone_aware() -> None:
    """The helper used for prediction timestamps must not return a naive value."""

    now = utc_now()

    assert isinstance(now, datetime)
    assert now.tzinfo is not None
    assert now.utcoffset() == timezone.utc.utcoffset(None)