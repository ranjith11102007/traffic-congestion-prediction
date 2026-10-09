"""End-to-end tests for the traffic and prediction HTTP API.

These exercise the full stack against the real Stage 3 model artifact and a real
database schema: a request is validated, stored, scored, and the forecast is
persisted and readable. Nothing here mocks the estimator, because a mocked
prediction would prove nothing about whether the integration actually works.

Every observation used here is labelled ``source="simulation"`` and the tests
assert that label survives into the prediction response. That is the guarantee
that development data cannot be mistaken for real traffic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services.simulated_provider import SimulatedTrafficProvider

API = "/api"


def post_observation(client, payload: dict) -> dict:
    """POST an observation and assert it was accepted."""

    response = client.post(f"{API}/traffic/observations", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def seed_history(client, location_id: str = "LOC-001", count: int = 8) -> None:
    """Store ``count`` prior observations for a location.

    The model needs six prior readings to build its lag and rolling features, so
    history has to exist before a prediction is meaningful. A 400 from the strict
    CHECK constraint would mean the generated speeds violated the domain rules.
    """

    provider = SimulatedTrafficProvider()
    base = datetime(2026, 1, 18, 6, 0, tzinfo=timezone.utc)

    for offset in range(count):
        record = provider._generate(
            location_id, "SEG-01", "Anna Salai", 60.0, base + timedelta(hours=offset)
        )
        response = client.post(
            f"{API}/traffic/observations",
            json={
                "timestamp": record.timestamp.isoformat(),
                "location_id": record.location_id,
                "road_name": record.road_name,
                "road_segment_id": record.road_segment_id,
                "vehicle_count": record.vehicle_count,
                "avg_speed_kph": record.avg_speed_kph,
                "free_flow_speed_kph": record.free_flow_speed_kph,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "rainfall_mm": record.rainfall_mm,
                "is_incident": record.is_incident,
                "source": "simulation",
            },
        )
        assert response.status_code == 201, response.text


# --- Observation storage ----------------------------------------------------


def test_create_observation_returns_stored_row(db_client, observation_payload) -> None:
    """A stored observation comes back with an id and a creation timestamp."""

    created = post_observation(db_client, observation_payload)

    assert isinstance(created["id"], int)
    assert created["location_id"] == "LOC-001"
    assert created["source"] == "simulation"
    assert created["created_at"] is not None
    assert created["is_incident"] is False


def test_observation_defaults_source_to_api(db_client, observation_payload) -> None:
    """A client that does not declare provenance is recorded as ``api``."""

    payload = {**observation_payload}
    payload.pop("source")

    created = post_observation(db_client, payload)

    assert created["source"] == "api"


def test_duplicate_observation_returns_409(db_client, observation_payload) -> None:
    """The same location and timestamp is a conflict, not a silent overwrite."""

    post_observation(db_client, observation_payload)
    response = db_client.post(f"{API}/traffic/observations", json=observation_payload)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "duplicate_observation"


def test_invalid_observation_returns_structured_422(
    db_client, observation_payload
) -> None:
    """Validation failures use the shared error envelope, not FastAPI's default.

    A client should not have to parse two different error shapes depending on
    whether the failure came from Pydantic or from the domain layer.
    """

    payload = {**observation_payload, "avg_speed_kph": 70.0}
    response = db_client.post(f"{API}/traffic/observations", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert isinstance(body["error"]["details"], list)


def test_derived_target_field_is_rejected(db_client, observation_payload) -> None:
    """A caller cannot supply the ML target as an input."""

    payload = {**observation_payload, "congestion_level": 3}

    response = db_client.post(f"{API}/traffic/observations", json=payload)

    assert response.status_code == 422


def test_get_observation_by_id(db_client, observation_payload) -> None:
    """A stored observation is retrievable by primary key."""

    created = post_observation(db_client, observation_payload)

    response = db_client.get(f"{API}/traffic/observations/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_unknown_observation_returns_404(db_client) -> None:
    """A missing row is a 404, not an empty 200."""

    response = db_client.get(f"{API}/traffic/observations/999999")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_list_observations_is_newest_first(db_client, observation_payload) -> None:
    """"What is happening now" is the common question, so ordering is newest first."""

    post_observation(db_client, {**observation_payload, "timestamp": "2026-01-18T08:00:00+00:00"})
    post_observation(db_client, {**observation_payload, "timestamp": "2026-01-18T09:00:00+00:00"})

    response = db_client.get(f"{API}/traffic/observations")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["observations"][0]["timestamp"] > body["observations"][1]["timestamp"]


def test_list_observations_honours_location_filter(
    db_client, observation_payload
) -> None:
    """Filters must actually restrict the result set."""

    post_observation(db_client, observation_payload)
    post_observation(db_client, {**observation_payload, "location_id": "LOC-002"})

    response = db_client.get(f"{API}/traffic/observations", params={"location_id": "LOC-001"})

    body = response.json()
    assert body["count"] == 1
    assert body["observations"][0]["location_id"] == "LOC-001"
    assert body["filters"]["location_id"] == "LOC-001"


def test_list_observations_echoes_filters(db_client, observation_payload) -> None:
    """The echoed filters let a client verify the query it received."""

    post_observation(db_client, observation_payload)

    response = db_client.get(
        f"{API}/traffic/observations", params={"limit": 5, "location_id": "LOC-001"}
    )

    assert response.json()["filters"] == {"limit": 5, "location_id": "LOC-001"}


def test_latest_returns_most_recent(db_client, observation_payload) -> None:
    """The latest endpoint picks the newest row, not an arbitrary one."""

    post_observation(db_client, {**observation_payload, "timestamp": "2026-01-18T08:00:00+00:00"})
    post_observation(db_client, {**observation_payload, "timestamp": "2026-01-18T11:00:00+00:00"})

    response = db_client.get(f"{API}/traffic/latest")

    assert response.status_code == 200
    assert response.json()["timestamp"].startswith("2026-01-18T11:00:00")


def test_latest_returns_404_when_empty(db_client) -> None:
    """An empty table is a 404, so it cannot be mistaken for an empty match."""

    response = db_client.get(f"{API}/traffic/latest")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# --- Prediction -------------------------------------------------------------


def test_predict_without_history_returns_422(
    db_client, observation_payload
) -> None:
    """Too little history is a clear 422, not a fabricated forecast.

    This is the case that matters most: a prediction here would look identical to a
    real one.
    """

    payload = {**observation_payload, "timestamp": "2026-01-18T08:00:00+00:00"}
    response = db_client.post(f"{API}/traffic/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "insufficient_history"
    # Named for what the model actually needs: readings ahead of the one being
    # scored. Zero-filled counts would let a caller believe a forecast is coming.
    assert body["error"]["details"]["required_readings"] >= 1
    assert body["error"]["details"]["available_readings"] == 0


def test_predict_stores_observation_and_forecast(db_client, observation_payload) -> None:
    """The complete flow: store, predict, store, return.

    Asserts against the real model artifact, so a broken ML integration fails here
    rather than being masked by a mock.
    """

    seed_history(db_client, count=8)

    payload = {**observation_payload, "timestamp": "2026-01-18T14:00:00+00:00"}
    response = db_client.post(f"{API}/traffic/predict", json=payload)

    assert response.status_code == 201, response.text
    body = response.json()

    # Prediction details
    assert body["observation_id"] is not None
    assert body["predicted_congestion_level"] in (0, 1, 2, 3)
    assert body["predicted_congestion"] in ("free_flow", "moderate", "heavy", "severe")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "4.0.0"

    # Provenance survives into the forecast
    assert body["source"] == "simulation"
    assert "simulated" in (body["note"] or "").lower()

    # Both timestamps are present and distinct in meaning
    assert body["observation_timestamp"].startswith("2026-01-18T14:00:00")
    assert body["prediction_timestamp"] is not None

    # Both rows really exist in the database
    assert body["id"] is not None
    stored = db_client.get(f"{API}/traffic/observations/{body['observation_id']}")
    assert stored.status_code == 200


def test_predict_requires_model_features(db_client, observation_payload) -> None:
    """Temperature and rainfall cannot be omitted from a prediction request."""

    payload = {**observation_payload}
    payload.pop("rainfall_mm")

    response = db_client.post(f"{API}/traffic/predict", json=payload)

    assert response.status_code == 422


def test_predict_existing_observation(db_client, observation_payload) -> None:
    """A stored observation can be scored without posting it again."""

    seed_history(db_client, count=8)
    payload = {**observation_payload, "timestamp": "2026-01-18T14:00:00+00:00"}
    created = post_observation(db_client, payload)

    response = db_client.post(
        f"{API}/traffic/observations/{created['id']}/predict"
    )

    assert response.status_code == 201, response.text
    assert response.json()["observation_id"] == created["id"]


def test_prediction_status_reports_the_real_contract(db_client) -> None:
    """The status endpoint reports what the model actually needs.

    Read from the saved artifact rather than hard-coded, so it stays truthful
    after a retrain.
    """

    response = db_client.get(f"{API}/traffic/predictions/status")

    assert response.status_code == 200
    body = response.json()
    assert body["model_available"] is True
    assert body["model_version"] == "4.0.0"
    assert body["dataset_is_simulated"] is True
    assert body["min_history_per_segment"] >= 1

    for column in (
        "timestamp",
        "road_segment_id",
        "avg_speed_kph",
        "free_flow_speed_kph",
        "weather_condition",
        "temperature_c",
        "precipitation_mm",
    ):
        assert column in body["required_input_columns"]

    assert "speed_ratio" not in body["required_input_columns"]
    assert "flow_veh_per_hr" not in body["required_input_columns"], (
        "4.0.0 dropped vehicle throughput: no self-serve traffic API publishes it, "
        "so requiring it would make real observations unscorable"
    )
    assert "SEG-01" in body["known_road_segments"]


# --- Collector --------------------------------------------------------------


def test_collector_status_declares_simulation(db_client) -> None:
    """The provider must identify itself, so synthetic data is never ambiguous."""

    response = db_client.get(f"{API}/traffic/collector/status")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "simulation"
    assert body["is_simulation"] is True
    assert "not real traffic" in body["detail"].lower()


def test_collect_stores_labelled_observations(db_client) -> None:
    """A collection run stores rows stamped ``simulation``."""

    response = db_client.post(f"{API}/traffic/collect", json={"location_id": "LOC-001"})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["is_simulation"] is True
    assert body["count"] == 1
    assert body["observations"][0]["source"] == "simulation"


def test_collect_dry_run_persists_nothing(db_client) -> None:
    """``persist=false`` generates data without writing it."""

    response = db_client.post(
        f"{API}/traffic/collect", json={"location_id": "LOC-001", "persist": False}
    )

    assert response.status_code == 201
    assert response.json()["count"] == 1

    listing = db_client.get(f"{API}/traffic/observations")
    assert listing.json()["count"] == 0


def test_collect_unknown_location_returns_503(db_client) -> None:
    """An unmodelled location fails with a 503 and a usable message."""

    response = db_client.post(f"{API}/traffic/collect", json={"location_id": "LOC-999"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"


def test_collect_history_seeds_and_then_declines_to_overwrite(db_client) -> None:
    """Seeding fills an empty location once, then refuses to touch it.

    Overwriting existing observations with synthetic ones would destroy real data,
    so the second call must be a no-op.
    """

    first = db_client.post(
        f"{API}/traffic/collect/history", json={"location_id": "LOC-001"}
    )
    assert first.status_code == 201, first.text
    assert first.json()["count"] > 0

    seeded_count = db_client.get(f"{API}/traffic/observations").json()["count"]

    db_client.post(f"{API}/traffic/collect/history", json={"location_id": "LOC-001"})

    assert db_client.get(f"{API}/traffic/observations").json()["count"] == seeded_count


def test_collect_history_makes_a_location_predictable(db_client) -> None:
    """After backfill, a fresh observation can actually be scored.

    This is the integration path that ties the provider, the database history and
    the model together.
    """

    seeded = db_client.post(
        f"{API}/traffic/collect/history", json={"location_id": "LOC-002"}
    )
    assert seeded.status_code == 201, seeded.text

    stored = seeded.json()["observations"]
    latest = max(stored, key=lambda item: item["timestamp"])

    response = db_client.post(
        f"{API}/traffic/observations/{latest['id']}/predict"
    )

    assert response.status_code == 201, response.text
    assert response.json()["predicted_congestion_level"] in (0, 1, 2, 3)


# --- Database-level integrity ----------------------------------------------


def test_database_rejects_free_flow_below_observed(db_session) -> None:
    """The CHECK constraint holds even for writes that bypass Pydantic.

    Validation protects only requests that pass through the API. This proves the
    schema itself refuses a contradictory row, which is what stops a future import
    job from writing a reading whose derived label would be meaningless.
    """

    from sqlalchemy.exc import IntegrityError

    from app.models.traffic import TrafficObservation

    db_session.add(
        TrafficObservation(
            timestamp=datetime(2026, 1, 18, 8, 0, tzinfo=timezone.utc),
            location_id="LOC-CHECK",
            road_name="Constraint Test Road",
            road_segment_id="SEG-01",
            vehicle_count=100,
            avg_speed_kph=90.0,  # faster than the free-flow reference
            free_flow_speed_kph=60.0,
            weather_condition="clear",
            source="api",
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_database_rejects_out_of_range_coordinates(db_session) -> None:
    """Latitude and longitude bounds are enforced by the schema too."""

    from sqlalchemy.exc import IntegrityError

    from app.models.traffic import TrafficObservation

    db_session.add(
        TrafficObservation(
            timestamp=datetime(2026, 1, 18, 8, 0, tzinfo=timezone.utc),
            location_id="LOC-CHECK",
            road_name="Constraint Test Road",
            road_segment_id="SEG-01",
            latitude=120.0,  # not a real latitude
            longitude=0.0,
            vehicle_count=100,
            avg_speed_kph=30.0,
            free_flow_speed_kph=60.0,
            weather_condition="clear",
            source="api",
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_deleting_an_observation_cascades_to_predictions(
    db_client, db_engine, observation_payload
) -> None:
    """A forecast has no meaning without its observation, so it must not orphan."""

    seed_history(db_client, count=8)
    payload = {**observation_payload, "timestamp": "2026-01-18T14:00:00+00:00"}
    predicted = db_client.post(f"{API}/traffic/predict", json=payload).json()
    observation_id = predicted["observation_id"]

    from sqlalchemy.orm import Session

    from app.models.traffic import TrafficObservation, TrafficPrediction
    from app.repositories import traffic_repository as repository

    # ``db_client`` and ``db_engine`` are the same instance here: the client
    # fixture is built on the engine fixture.
    with Session(db_engine, expire_on_commit=False) as session:
        assert repository.count_predictions_for_observation(session, observation_id) == 1
        session.delete(session.get(TrafficObservation, observation_id))
        session.commit()

        assert session.query(TrafficPrediction).count() == 0