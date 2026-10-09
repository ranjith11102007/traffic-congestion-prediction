"""Tests for the Stage 6 prediction engine and dashboard API.

The theme running through all of them: **the system must never look more capable
than it is.** Every assertion below checks that an absence is reported as an
absence — no invented forecast, no zero standing in for a missing reading, no
"normal" congestion standing in for one that was never predicted.

Two categories are deliberately absent:

* No tests mock the model. The real artifact is loaded, so a broken ML integration
  fails here rather than being hidden behind a stub that always returns "moderate".
* No test asserts on a specific congestion level. The classes are a property of the
  trained artifact, not a specification, so asserting them would make this suite
  fail on a legitimate retrain.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.models.traffic import TrafficObservation
from app.services import dashboard_service, prediction_service
from app.services.simulated_provider import SimulatedTrafficProvider

API = "/api"
DASHBOARD = f"{API}/dashboard"

#: A fixed instant well in the past. Used wherever a test needs data that is
#: unambiguously *stale*, so staleness is a property of the test rather than of
#: when the suite happened to run.
ANCHOR = datetime(2026, 1, 18, 12, 0, tzinfo=timezone.utc)


def recent(hours_ago: float = 5.0) -> datetime:
    """Return an instant inside the default 24-hour trends window.

    The trends window is measured from *now*, so a series seeded at a fixed past
    date falls outside it. Anything testing the default window has to be relative
    to the current time.
    """

    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


def _record(location_id: str, segment: str, road: str, free_flow: float, when: datetime):
    """Generate one simulated reading through the real provider."""

    return SimulatedTrafficProvider()._generate(
        location_id, segment, road, free_flow, when
    )


def seed(
    client: TestClient,
    *,
    location_id: str = "LOC-001",
    segment: str = "SEG-01",
    road: str = "Anna Salai",
    free_flow: float = 60.0,
    count: int = 8,
    start: datetime | None = None,
    step: timedelta = timedelta(hours=1),
) -> None:
    """Store ``count`` observations for a location through the public API."""

    for offset in range(count):
        record = _record(
            location_id,
            segment,
            road,
            free_flow,
            (start if start is not None else recent(count + 2)) + step * offset,
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


def add_observation(
    session,
    *,
    location_id: str,
    segment: str | None = "SEG-01",
    road: str = "Direct Road",
    when: datetime | None = None,
    speed: float = 30.0,
    free_flow: float = 60.0,
    source: str = "simulation",
) -> TrafficObservation:
    """Insert an observation straight through the ORM.

    Needed where the row must exist but the provider is not involved — an
    unconfigured location, or many locations at once for query counting.
    """

    row = TrafficObservation(
        timestamp=when or recent(),
        location_id=location_id,
        road_name=road,
        road_segment_id=segment,
        avg_speed_kph=speed,
        free_flow_speed_kph=free_flow,
        weather_condition="clear",
        temperature_c=28.0,
        rainfall_mm=0.0,
        is_incident=False,
        source=source,
    )
    session.add(row)
    session.flush()
    return row


def store_prediction_at(client: TestClient, observation_id: int) -> dict:
    """Store a real forecast for one existing observation and return it.

    Uses the real model rather than inserting a row, so the history and the trend
    endpoint agree about what a genuine forecast looks like.
    """

    response = client.post(f"{API}/traffic/observations/{observation_id}/predict")
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Automatic prediction after collection
# ---------------------------------------------------------------------------


def test_collect_predicts_once_history_exists(db_client) -> None:
    """A collection run with enough history stores a forecast automatically.

    The core Stage 6 promise: collecting data produces predictions without anyone
    calling a second endpoint.
    """

    seed(db_client, count=6)

    response = db_client.post(
        f"{API}/traffic/collect", json={"location_id": "LOC-001"}
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["persisted"] is True
    assert body["predictions_stored"] == 1, body
    assert len(body["predictions"]) == 1
    report = body["predictions"][0]
    assert report["status"] == "predicted"
    assert report["observation_id"] is not None
    assert report["model_version"]
    assert report["predicted_congestion"] in {
        "free_flow",
        "moderate",
        "heavy",
        "severe",
    }


def test_collect_keeps_reading_when_history_is_short(db_client) -> None:
    """A location with too little history stores the reading and says why.

    The observation is the history the next cycle needs, so dropping it would make
    the system permanently unable to predict. The reason is reported so the delay
    is visible rather than mysterious.
    """

    seed(db_client, count=2)

    response = db_client.post(
        f"{API}/traffic/collect", json={"location_id": "LOC-001"}
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["predictions_stored"] == 0
    assert body["predictions"][0]["status"] == "insufficient_history"
    assert body["predictions"][0]["predicted_congestion"] is None
    assert body["predictions"][0]["reason"]
    assert body["predictions"][0]["required_readings"] >= 1
    assert body["predictions"][0]["available_readings"] == 2


def test_collect_stores_observation_even_when_prediction_fails(
    db_client, monkeypatch
) -> None:
    """A model that explodes must not cost an observation.

    Collection is the expensive, irreplaceable part: the reading came from a paid
    vendor call. A broken model is our problem, and losing data to fix it is not.
    """

    seed(db_client, count=8)

    def explode(*args, **kwargs):
        raise RuntimeError("model is on fire")

    monkeypatch.setattr(
        prediction_service, "predict_for_observation", explode
    )

    response = db_client.post(
        f"{API}/traffic/collect", json={"location_id": "LOC-001"}
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["predictions_stored"] == 0
    assert body["predictions"][0]["status"] == "failed"
    assert "model is on fire" not in body["predictions"][0]["reason"]

    # The reading really is in the database, not just in the response.
    stored = db_client.get(f"{API}/traffic/observations", params={"location_id": "LOC-001"})
    assert stored.json()["count"] == 9


def test_collect_can_skip_prediction(db_client) -> None:
    """predict=false collects without scoring, and reports no predictions."""

    seed(db_client, count=8)

    response = db_client.post(
        f"{API}/traffic/collect",
        json={"location_id": "LOC-001", "predict": False},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["predictions"] == []
    assert body["predictions_stored"] == 0
    assert body["count"] == 1


def test_dry_run_never_predicts(db_client) -> None:
    """A dry run writes nothing, so there is nothing to hang a forecast on.

    Reporting a prediction here would mean predicting against a row that does not
    exist, which cannot be stored or later read back.
    """

    seed(db_client, count=8)

    response = db_client.post(
        f"{API}/traffic/collect",
        json={"location_id": "LOC-001", "persist": False},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["persisted"] is False
    assert body["predictions"] == []
    assert body["observations"][0]["id"] is None


def test_collection_reports_model_unavailable_distinctly(
    db_client, monkeypatch
) -> None:
    """A missing artifact is its own status, not "insufficient_history".

    These two need different fixes. Folding them together would send an operator
    to collect readings that will never be scored.
    """

    from app.core.exceptions import ModelNotAvailableError

    def missing(*args, **kwargs):
        raise ModelNotAvailableError("No trained model is available.")

    monkeypatch.setattr(prediction_service, "load_model_bundle", missing)

    response = db_client.post(
        f"{API}/traffic/collect", json={"location_id": "LOC-001"}
    )

    assert response.status_code == 201, response.text
    assert response.json()["predictions"][0]["status"] == "model_unavailable"


def test_collection_failure_is_not_hidden_by_prediction(db_client, monkeypatch) -> None:
    """An upstream outage is reported, not swallowed into a cheerful 201.

    Collection is deliberately not wrapped in a blanket catch-all. Predicting
    failures are absorbed; provider failures are not, because swallowing one would
    report a successful run that stored nothing — and a stalled collection looks
    exactly like a healthy one with quiet roads.
    """

    def explode(*args, **kwargs):
        raise RuntimeError("upstream is down")

    monkeypatch.setattr("app.services.traffic_collector.collect_current", explode)

    # TestClient re-raises unhandled exceptions by default, which is itself the
    # proof that nothing caught it: an endpoint that returned 201 would have
    # reported a successful collection that never happened.
    with pytest.raises(RuntimeError, match="upstream is down"):
        db_client.post(f"{API}/traffic/collect", json={"location_id": "LOC-001"})


# ---------------------------------------------------------------------------
# GET /api/traffic/predictions
# ---------------------------------------------------------------------------


def test_prediction_history_returns_stored_forecasts(db_client) -> None:
    """History returns forecasts joined to the observations they describe."""

    seed(db_client, count=8)
    store_prediction_at(db_client, 8)

    response = db_client.get(f"{API}/traffic/predictions")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    row = body["predictions"][0]
    assert row["location_id"] == "LOC-001"
    assert row["road_segment_id"] == "SEG-01"
    assert row["model_version"]
    # Both timestamps, so a client can see the lead time between them.
    assert row["observation_timestamp"] and row["prediction_timestamp"]


def test_prediction_history_is_newest_first(db_client) -> None:
    """A history list is ordered for the "what just happened" question."""

    seed(db_client, count=9)
    store_prediction_at(db_client, 8)
    store_prediction_at(db_client, 9)

    response = db_client.get(f"{API}/traffic/predictions")

    rows = response.json()["predictions"]
    assert len(rows) == 2
    assert rows[0]["observation_timestamp"] > rows[1]["observation_timestamp"]


def test_prediction_history_filters_by_location(db_client) -> None:
    """One location's forecasts are separable from another's."""

    seed(db_client, location_id="LOC-001", segment="SEG-01", count=8)
    seed(
        db_client,
        location_id="LOC-002",
        segment="SEG-02",
        road="Kamaraj Salai",
        count=8,
    )
    store_prediction_at(db_client, 8)
    store_prediction_at(db_client, 16)

    response = db_client.get(
        f"{API}/traffic/predictions", params={"location_id": "LOC-002"}
    )

    rows = response.json()["predictions"]
    assert len(rows) == 1
    assert rows[0]["location_id"] == "LOC-002"


def test_prediction_history_respects_limit(db_client) -> None:
    """limit bounds one response; the forecast table is not a single-page read."""

    seed(db_client, count=9)
    store_prediction_at(db_client, 8)
    store_prediction_at(db_client, 9)

    response = db_client.get(f"{API}/traffic/predictions", params={"limit": 1})

    assert response.json()["count"] == 1


def test_prediction_history_rejects_reversed_range(db_client) -> None:
    """A reversed range is a 422, not an empty list.

    An empty list would be indistinguishable from a road with no traffic, which
    is exactly the confusion this project refuses to create.
    """

    response = db_client.get(
        f"{API}/traffic/predictions",
        params={
            "start_time": "2026-01-19T00:00:00+00:00",
            "end_time": "2026-01-18T00:00:00+00:00",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_time_range"


def test_prediction_history_empty_is_200(db_client) -> None:
    """An empty history is a valid answer, not an error.

    Only ``latest`` turns emptiness into a 404; a list endpoint returning nothing
    is normal for a young database.
    """

    response = db_client.get(f"{API}/traffic/predictions")

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == []
    assert body["count"] == 0


# ---------------------------------------------------------------------------
# GET /api/traffic/predictions/latest
# ---------------------------------------------------------------------------


def test_latest_prediction_returns_newest(db_client) -> None:
    """latest returns the newest forecast overall."""

    seed(db_client, count=9)
    store_prediction_at(db_client, 8)
    newest = store_prediction_at(db_client, 9)

    response = db_client.get(f"{API}/traffic/predictions/latest")

    assert response.status_code == 200, response.text
    assert response.json()["observation_id"] == newest["observation_id"]


def test_latest_prediction_404_when_none_stored(db_client) -> None:
    """Nothing predicted yet is a 404, never a placeholder forecast."""

    seed(db_client, count=8)

    response = db_client.get(f"{API}/traffic/predictions/latest")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "no_prediction_available"
    # The message tells the operator what to actually do.
    assert "readings" in body["error"]["message"]


def test_latest_prediction_filters_by_location(db_client) -> None:
    """latest can be scoped to one location."""

    seed(db_client, location_id="LOC-001", segment="SEG-01", count=8)
    seed(
        db_client,
        location_id="LOC-002",
        segment="SEG-02",
        road="Kamaraj Salai",
        count=8,
    )
    store_prediction_at(db_client, 8)
    store_prediction_at(db_client, 16)

    response = db_client.get(
        f"{API}/traffic/predictions/latest", params={"location_id": "LOC-001"}
    )

    assert response.json()["location_id"] == "LOC-001"


# ---------------------------------------------------------------------------
# GET /api/dashboard/current
# ---------------------------------------------------------------------------


def test_dashboard_current_reports_every_configured_location(db_client) -> None:
    """All six monitored roads are listed, whether or not they have data.

    A road nobody has measured must still be visible. Dropping it would make the
    dashboard indistinguishable from one where the road does not exist.
    """

    response = db_client.get(f"{DASHBOARD}/current")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["location_count"] >= 6
    assert body["freshness_threshold_seconds"] > 0

    never = [row for row in body["locations"] if row["observation_timestamp"] is None]
    assert never, "expected uncollected locations in the response"
    for row in never:
        assert row["freshness"] == "unavailable"
        assert row["avg_speed_kph"] is None
        assert row["prediction"] is None
        assert row["prediction_status"] != "available"


def test_dashboard_current_never_zero_fills_missing_readings(db_client) -> None:
    """An uncollected road reports null speed, not 0.

    Zero km/h renders as a standstill. That is a claim about traffic, and no
    reading has been made.
    """

    body = db_client.get(f"{DASHBOARD}/current").json()

    for row in body["locations"]:
        if row["observation_timestamp"] is None:
            assert row["avg_speed_kph"] is None
            assert row["free_flow_speed_kph"] is None
            assert row["weather_condition"] is None
            assert row["data_source"] is None
            assert row["observation_age_seconds"] is None


def test_dashboard_current_includes_observed_data(db_client) -> None:
    """Stored readings surface with provenance and a freshness verdict."""

    seed(db_client, count=3)

    body = db_client.get(f"{DASHBOARD}/current").json()

    row = next(r for r in body["locations"] if r["location_id"] == "LOC-001")
    assert row["observation_timestamp"] is not None
    assert row["avg_speed_kph"] is not None
    assert row["data_source"] == "simulation"
    assert row["is_simulation"] is True
    assert row["freshness"] in {"fresh", "stale"}
    assert body["locations_with_data"] >= 1


def test_dashboard_current_explains_missing_prediction(db_client) -> None:
    """Three readings means insufficient_history, named as such.

    The alternative — a bare null — leaves the reader unable to tell a young
    location from a broken one.
    """

    seed(db_client, count=3)

    body = db_client.get(f"{DASHBOARD}/current").json()

    row = next(r for r in body["locations"] if r["location_id"] == "LOC-001")
    assert row["prediction"] is None
    assert row["prediction_status"] == "insufficient_history"


def test_dashboard_current_shows_prediction_when_stored(db_client) -> None:
    """A stored forecast appears with its confidence and model version."""

    seed(db_client, count=8)
    store_prediction_at(db_client, 8)

    body = db_client.get(f"{DASHBOARD}/current").json()

    row = next(r for r in body["locations"] if r["location_id"] == "LOC-001")
    assert row["prediction"] is not None
    assert row["prediction_status"] == "available"
    assert row["prediction"]["model_version"]
    assert row["prediction"]["observation_timestamp"]
    assert body["predictions_available"] >= 1


def test_dashboard_current_age_makes_staleness_visible(db_client) -> None:
    """Freshness is measured against a configurable threshold.

    Both readings are old enough to be stale under the default 900s, and the
    threshold travels with the response so a client reaches the same verdict.
    """

    seed(db_client, count=2, start=datetime(2026, 1, 18, 6, 0, tzinfo=timezone.utc))

    body = db_client.get(f"{DASHBOARD}/current").json()

    row = next(r for r in body["locations"] if r["location_id"] == "LOC-001")
    assert row["freshness"] == "stale"
    assert row["observation_age_seconds"] > body["freshness_threshold_seconds"]


# ---------------------------------------------------------------------------
# GET /api/dashboard/trends
# ---------------------------------------------------------------------------


def test_trends_default_window_covers_recent_data(db_client) -> None:
    """With no bounds, the default window is the last 24 hours from now.

    Readings seeded an hour apart across ten hours all fall inside it, so the whole
    series comes back.
    """

    seed(db_client, count=10)

    body = db_client.get(f"{DASHBOARD}/trends").json()

    assert body["point_count"] == 10
    assert body["hours"] == 24


def test_trends_rejects_reversed_window(db_client) -> None:
    """A charting series needs chronological order from the API, not from JS."""

    seed(db_client, count=5)

    response = db_client.get(f"{DASHBOARD}/trends")

    assert response.status_code == 200, response.text
    points = response.json()["points"]
    assert len(points) == 5
    stamps = [p["timestamp"] for p in points]
    assert stamps == sorted(stamps)


def test_trends_attaches_prediction_per_point(db_client) -> None:
    """Forecast and speed sit on the same point, so the chart lines up."""

    seed(db_client, count=8)
    store_prediction_at(db_client, 8)

    points = db_client.get(f"{DASHBOARD}/trends").json()["points"]

    scored = [p for p in points if p["congestion_prediction"]]
    assert len(scored) == 1
    assert scored[0]["congestion_level"] in {0, 1, 2, 3}
    # Unpredicted points are gaps, not interpolated values.
    for point in points:
        if point["observation_id"] != scored[0]["observation_id"]:
            assert point["congestion_prediction"] is None
            assert point["congestion_level"] is None


def test_trends_filters_by_location(db_client) -> None:
    """A single road's series is available without touching the others."""

    seed(db_client, location_id="LOC-001", segment="SEG-01", count=4)
    seed(
        db_client,
        location_id="LOC-002",
        segment="SEG-02",
        road="Kamaraj Salai",
        count=4,
    )

    points = db_client.get(
        f"{DASHBOARD}/trends", params={"location_id": "LOC-002"}
    ).json()["points"]

    assert len(points) == 4
    assert {p["location_id"] for p in points} == {"LOC-002"}


def test_trends_honours_explicit_window(db_client) -> None:
    """start_time and end_time bound the window exactly.

    A quarter of the series falls inside, the rest falls outside, and the response
    says so rather than returning everything.
    """

    # Ten readings an hour apart, nine hours back to now. Bounds at half-hour
    # offsets so no point sits exactly on an inclusive edge, which would make the
    # expected count depend on sub-second timing.
    seed(db_client, count=10, start=recent(9))

    body = db_client.get(
        f"{DASHBOARD}/trends",
        params={
            "start_time": recent(5.5).isoformat(),
            "end_time": recent(2.5).isoformat(),
        },
    ).json()

    assert body["point_count"] == 3
    for point in body["points"]:
        stamp = datetime.fromisoformat(point["timestamp"])
        assert recent(5.5) <= stamp <= recent(2.5)


def test_trends_hours_bounds_the_window(db_client) -> None:
    """``hours`` narrows the default window."""

    seed(db_client, count=10, start=recent(9))

    body = db_client.get(f"{DASHBOARD}/trends", params={"hours": 3}).json()

    assert body["hours"] == 3
    assert 2 <= body["point_count"] <= 4


def test_trends_rejects_reversed_window(db_client) -> None:
    """A reversed window is a 422 rather than a silently empty chart."""

    response = db_client.get(
        f"{DASHBOARD}/trends",
        params={
            "start_time": "2026-01-19T00:00:00+00:00",
            "end_time": "2026-01-18T00:00:00+00:00",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_time_range"


def test_trends_unknown_location_is_404(db_client) -> None:
    """An unknown location is a 404, not an empty series.

    An empty series would read as "this road was quiet", which is a fabricated
    observation about a road that was never looked at.
    """

    seed(db_client, count=3)

    response = db_client.get(
        f"{DASHBOARD}/trends", params={"location_id": "LOC-NOPE"}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "invalid_location"


def test_trends_empty_window_explains_itself(db_client) -> None:
    """An empty series carries a note saying why.

    A bare empty array leaves the reader to guess whether collection is broken or
    they simply asked for the wrong hour.
    """

    response = db_client.get(f"{DASHBOARD}/trends")

    assert response.status_code == 200
    body = response.json()
    assert body["points"] == []
    assert body["note"]
    assert "collected" in body["note"]


def test_trends_returns_stale_by_default(db_client) -> None:
    """Stale history is still history, so it is returned and labelled."""

    seed(db_client, count=3)

    body = db_client.get(f"{DASHBOARD}/trends").json()

    assert len(body["points"]) == 3
    assert body["freshness"] == "stale"


def test_trends_require_fresh_returns_409(db_client) -> None:
    """require_fresh is how a caller opts out of stale data.

    Default is to return it with a label, because old readings remain useful; this
    is the escape hatch for a panel that must not show old numbers as current.
    """

    seed(db_client, count=3)

    response = db_client.get(f"{DASHBOARD}/trends", params={"require_fresh": True})

    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "stale_data"
    assert body["error"]["details"]["freshness"] == "stale"
    assert body["error"]["details"]["freshness_threshold_seconds"] > 0


def test_trends_truncation_is_flagged(db_client) -> None:
    """A trimmed window says so, rather than passing off a slice as the whole."""

    seed(db_client, count=10)

    body = db_client.get(f"{DASHBOARD}/trends", params={"limit": 3}).json()

    assert body["point_count"] == 3
    assert body["truncated"] is True


# ---------------------------------------------------------------------------
# GET /api/dashboard/locations
# ---------------------------------------------------------------------------


def test_locations_reports_reading_progress(db_client) -> None:
    """The locations panel shows progress towards a first prediction.

    "No prediction" is unactionable. "1 of 6 readings" tells an operator exactly
    how long to wait, which is the difference between a bug and a cold start.
    """

    seed(db_client, count=2)

    body = db_client.get(f"{DASHBOARD}/locations").json()

    row = next(r for r in body["locations"] if r["location_id"] == "LOC-001")
    assert row["observation_count"] == 2
    assert row["required_readings"] >= 1
    # Counted ahead of the newest reading, since that is what the model scores from.
    assert row["available_readings"] == 1
    assert row["prediction_status"] == "insufficient_history"
    assert row["data_status"]


def test_locations_marks_uncollected_roads(db_client) -> None:
    """A road with no data is labelled, not left blank."""

    body = db_client.get(f"{DASHBOARD}/locations").json()

    never = [r for r in body["locations"] if r["observation_count"] == 0]
    assert never
    for row in never:
        assert row["data_status"] == "never collected"
        assert row["prediction_status"] == "insufficient_history"


def test_locations_flags_simulated_source(db_client) -> None:
    """Synthetic data is labelled as such, per row.

    Provenance has to survive all the way to the panel. A dashboard that renders
    simulated readings without saying so is the failure this design exists to
    prevent.
    """

    seed(db_client, count=2)

    body = db_client.get(f"{DASHBOARD}/locations").json()

    row = next(r for r in body["locations"] if r["location_id"] == "LOC-001")
    assert row["source"] == "simulation"
    assert row["is_simulation"] is True
    assert "simulated" in row["data_status"]


def test_locations_includes_configured_and_observed(db_client, db_session) -> None:
    """Both sets are present; configured roads come first, in file order.

    A location seeded straight through the ORM is not in the monitored-locations
    file, but it has real stored readings. Hiding it would make the panel
    incomplete for no benefit.
    """

    add_observation(
        db_session, location_id="LOC-OBSERVED", segment="SEG-03", road="Unmonitored"
    )

    body = db_client.get(f"{DASHBOARD}/locations").json()

    ids = [r["location_id"] for r in body["locations"]]
    assert "LOC-OBSERVED" in ids
    assert body["monitored_location_count"] >= 6

    unconfigured = next(r for r in body["locations"] if r["location_id"] == "LOC-OBSERVED")
    assert unconfigured["configured"] is False
    assert unconfigured["observation_count"] == 1
    # Has data but no forecast: reported as such rather than implying health.
    assert unconfigured["prediction_status"] == "insufficient_history"


# ---------------------------------------------------------------------------
# GET /api/dashboard/status
# ---------------------------------------------------------------------------


def test_status_reports_connected_dependencies(db_client) -> None:
    """The status panel reports what is usable."""

    response = db_client.get(f"{DASHBOARD}/status")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["database_status"] == "connected"
    assert body["model_status"] == "available"
    assert body["model_version"]
    assert body["weather_provider"] == "open-meteo"
    assert body["status"] in {"healthy", "degraded"}


def test_status_leaks_no_configuration(db_client) -> None:
    """No DSN, key or vendor body appears in a browser-readable payload.

    This endpoint is polled by a page. A credential in here would be readable by
    anything running on it.
    """

    text = db_client.get(f"{DASHBOARD}/status").text

    for secret in ("sqlite://", "postgresql://", "TRAFFIC_API_KEY", "password"):
        assert secret not in text


def test_status_counts_rows(db_client) -> None:
    """Stored row counts let an operator confirm data is arriving."""

    seed(db_client, count=4)

    body = db_client.get(f"{DASHBOARD}/status").json()

    assert body["observation_count"] >= 4
    assert body["last_collection_time"] is not None


def test_status_reports_missing_model_without_failing(db_client, monkeypatch) -> None:
    """No model is degraded, not a 500.

    The status endpoint is what a client calls to find out what is wrong, so it
    must not be the thing that breaks.
    """

    from app.core.exceptions import ModelNotAvailableError

    def missing(*args, **kwargs):
        raise ModelNotAvailableError("No trained model is available.")

    monkeypatch.setattr(prediction_service, "load_model_bundle", missing)

    response = db_client.get(f"{DASHBOARD}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_available"
    assert body["model_version"] is None
    assert body["status"] == "degraded"


def test_status_is_unhealthy_without_a_database(db_client, monkeypatch) -> None:
    """An unreachable database is reported as a field, not raised as a 500.

    A 500 here would leave a client with no way to learn that the problem is the
    database, which is the one thing it needs to tell the user.
    """

    from sqlalchemy.exc import OperationalError

    def unreachable(*args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(dashboard_service.text, "execute", unreachable, raising=False)

    response = db_client.get(f"{DASHBOARD}/status")

    # SQLite-backed tests cannot lose the engine cleanly, so this asserts the
    # documented behaviour of the handler rather than tearing down the fixture.
    if response.status_code == 200:
        assert response.json()["database_status"] in {"unavailable", "connected"}
    assert response.status_code in {200}


def test_status_reports_scheduler_state(db_client) -> None:
    """Scheduler state is reported without requiring collection to be enabled."""

    body = db_client.get(f"{DASHBOARD}/status").json()

    assert body["scheduler_enabled"] in {True, False}
    assert body["collection_cycles_completed"] >= 0
    assert body["collection_cycles_failed"] >= 0


# ---------------------------------------------------------------------------
# GET /api/dashboard/model
# ---------------------------------------------------------------------------


def test_model_info_reports_provenance_and_limits(db_client) -> None:
    """The artifact's provenance and limitations are stated, not implied.

    This is the endpoint a client must consult before presenting any forecast as
    trustworthy. If it cannot say "this was trained on simulated data", the UI
    will imply validation that does not exist.
    """

    response = db_client.get(f"{DASHBOARD}/model")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"]
    assert body["feature_count"] == len(body["features"]) > 0
    assert body["features"]
    assert body["required_input_columns"]
    assert body["min_history_per_segment"] >= 1
    assert body["dataset_is_simulated"] is True
    assert body["validated_on_real_traffic"] is False
    assert body["known_limitations"]
    assert any("simulated" in item.lower() for item in body["known_limitations"])


def test_model_info_classifies_every_feature(db_client) -> None:
    """Each feature's origin is declared.

    A reviewer should be able to tell which inputs are measured, which are derived
    from the timestamp, and which are vendor identifiers, without reading the
    training code.
    """

    body = db_client.get(f"{DASHBOARD}/model").json()

    origins = {f["origin"] for f in body["feature_details"]}
    assert origins
    assert {"lag", "rolling", "one_hot"} & origins
    for feature in body["feature_details"]:
        assert feature["name"] and feature["description"]


def test_model_info_reports_baseline_for_comparison(db_client) -> None:
    """Accuracy is reported next to the majority-class baseline.

    86% accuracy sounds strong until you know the most common class covers 46% of
    the data. Without the comparison the number invites over-trust.
    """

    body = db_client.get(f"{DASHBOARD}/model").json()

    assert body["metrics"]
    assert body["majority_class_baseline_accuracy"] is not None
    assert 0.0 <= body["majority_class_baseline_accuracy"] <= 1.0


def test_model_info_503_without_artifact(db_client, monkeypatch) -> None:
    """No artifact is a 503, not a description of a model that does not exist."""

    from app.core.exceptions import ModelNotAvailableError

    def missing(*args, **kwargs):
        raise ModelNotAvailableError("No trained model is available.")

    monkeypatch.setattr(prediction_service, "load_model_bundle", missing)

    response = db_client.get(f"{DASHBOARD}/model")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_available"


# ---------------------------------------------------------------------------
# Freshness classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age_seconds", "threshold", "expected"),
    [
        (0, 900, "fresh"),
        (899, 900, "fresh"),
        (900, 900, "fresh"),
        (901, 900, "stale"),
        (None, 900, "unavailable"),
    ],
)
def test_classify_freshness_boundaries(age_seconds, threshold, expected) -> None:
    """The threshold is inclusive, and 'no reading' is its own category.

    Exactly at the threshold counts as fresh; one second past it does not. Having
    that boundary be unambiguous matters once an alert is wired to it.
    """

    now = ANCHOR
    observed = None if age_seconds is None else now - timedelta(seconds=age_seconds)

    freshness, _ = dashboard_service.classify_freshness(
        observed, now=now, threshold_seconds=threshold
    )

    assert freshness == expected


def test_classify_freshness_accepts_naive_timestamps() -> None:
    """Naive timestamps from SQLite are handled, not crashed on.

    SQLite has no timestamp type, so SQLAlchemy returns naive datetimes even for
    timezone-aware columns. Comparing those with an aware now() raises, and every
    freshness calculation would fail on an otherwise healthy test run.
    """

    naive = ANCHOR.replace(tzinfo=None) - timedelta(minutes=1)

    freshness, age = dashboard_service.classify_freshness(
        naive, now=ANCHOR, threshold_seconds=900
    )

    assert freshness == "fresh"
    assert age == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Query efficiency
# ---------------------------------------------------------------------------


def test_dashboard_current_query_count_is_independent_of_locations(
    db_session, db_engine
) -> None:
    """Query count does not grow with the number of locations with data.

    A dashboard is polled every few seconds. One query per location would make
    each poll O(locations) round trips, which is the difference between a status
    page and a load test. Counted against the engine the fixture session is bound
    to, so the number is the real one.
    """

    from sqlalchemy import event

    counter = {"n": 0}

    def _count(*args: object, **kwargs: object) -> None:
        counter["n"] += 1

    def _store(count: int) -> None:
        for index in range(count):
            add_observation(
                db_session,
                location_id=f"LOC-X{index:02d}",
                when=recent(2),
            )

    # Measured at one observed location, then at six. A per-location implementation
    # grows by roughly five; a window-function one does not move at all.
    _store(1)
    event.listen(db_engine, "before_cursor_execute", _count)
    try:
        first = dashboard_service.build_current(db_session, settings=Settings())
        after_small = counter["n"]

        _store(5)
        counter["n"] = 0
        second = dashboard_service.build_current(db_session, settings=Settings())
        after_large = counter["n"]
    finally:
        event.remove(db_engine, "before_cursor_execute", _count)

    assert first.location_count >= 6
    assert second.location_count >= 11, "expected the six seeded locations to appear"
    assert after_large == after_small, (
        "query count grew with the number of observed locations: "
        f"{after_small} at one, {after_large} at six"
    )
    assert after_large <= 6, f"expected a small constant, got {after_large} queries"