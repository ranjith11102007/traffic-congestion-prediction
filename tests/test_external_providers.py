"""Tests for the Open-Meteo weather client and the locations file.

Both modules decide what gets written into a real observation, so both are tested
against the failure case as much as the happy path. A weather parser that defaults
a missing temperature to 0, or a locations loader that accepts an unknown road
segment, would store something plausible and wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    ProviderRateLimitedError,
    ProviderResponseError,
    UpstreamProviderError,
    UpstreamTimeoutError,
)
from app.services.locations import (
    KNOWN_ROAD_SEGMENTS,
    load_monitored_locations,
)
from app.services.weather_client import (
    OpenMeteoClient,
    WMO_CODE_TO_CONDITION,
    classify_weather_code,
)


def make_settings(**overrides: str) -> Settings:
    defaults = {
        "WEATHER_API_BASE_URL": "https://api.open-meteo.com/v1",
        "DATABASE_URL": "",
    }
    return Settings(**{**defaults, **overrides})


def client_for(handler, **overrides: str) -> OpenMeteoClient:
    """Build a client whose every request is served by ``handler``."""

    return OpenMeteoClient(
        make_settings(**overrides),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def weather_body(
    *, code: int = 0, temperature: object = 31.2, precipitation: object = 0.0
) -> dict:
    body: dict = {"current": {"time": "2026-10-05T12:00", "weather_code": code}}
    if temperature is not None:
        body["current"]["temperature_2m"] = temperature
    if precipitation is not None:
        body["current"]["precipitation"] = precipitation
    return body


# --- Request shape ---------------------------------------------------------


def test_request_uses_the_documented_parameters() -> None:
    """Only the three model inputs are requested, in UTC.

    ``timezone=GMT`` matters: without it the returned local time is +05:30 for
    Chennai, and every stored observation would be shifted by half a day.
    """

    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=weather_body())

    client_for(handler).fetch_current(latitude=13.0569, longitude=80.2425)

    url = seen[0]
    assert url.path == "/v1/forecast"
    assert url.params["latitude"] == "13.0569"
    assert url.params["longitude"] == "80.2425"
    assert url.params["current"] == "temperature_2m,precipitation,weather_code"
    assert url.params["timezone"] == "GMT"
    # The free tier needs no key, so none is sent.
    assert "key" not in url.params


# --- WMO code mapping ------------------------------------------------------


@pytest.mark.parametrize(
    "code, expected",
    [
        (0, "clear"),
        (1, "clear"),
        (2, "cloudy"),
        (3, "cloudy"),
        (45, "cloudy"),
        (48, "cloudy"),
        (61, "rain"),
        (63, "rain"),
        (95, "rain"),
        (99, "rain"),
        (71, "rain"),
        (85, "rain"),
    ],
)
def test_documented_codes_map_to_model_labels(code: int, expected: str) -> None:
    """Only the three labels the model was trained on are ever produced.

    Model version 4.0.0 knows ``clear``/``cloudy``/``rain``. Returning a finer
    label would score the observation under a category it never saw.
    """

    assert classify_weather_code(code) == expected
    assert set(WMO_CODE_TO_CONDITION.values()) == {"clear", "cloudy", "rain"}


def test_unknown_code_is_recorded_as_a_fallback() -> None:
    """An undocumented code is not an error, but it is not silently accepted."""

    assert classify_weather_code(7) == "cloudy"
    assert 7 not in WMO_CODE_TO_CONDITION


@pytest.mark.parametrize("value", [None, "3", 3.5, True, [3]])
def test_non_integer_code_is_rejected(value: object) -> None:
    """A shape change must fail loudly rather than look like ordinary weather."""

    with pytest.raises(ProviderResponseError):
        classify_weather_code(value)


# --- Response validation ---------------------------------------------------


def test_missing_weather_code_is_an_error() -> None:
    """Without a code there is no condition, and the column is NOT NULL."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = weather_body()
        del payload["current"]["weather_code"]
        return httpx.Response(200, json=payload)

    with pytest.raises(ProviderResponseError, match="weather_code"):
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"current": "not-an-object"},
        [1, 2, 3],
        "a string",
    ],
)
def test_malformed_body_is_rejected(payload: object) -> None:
    """A 2xx body without a usable ``current`` object is a failure, not a default."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(ProviderResponseError):
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)


def test_missing_temperature_stays_none() -> None:
    """A missing temperature is not zero degrees.

    A missing value and a genuine zero are different facts, and the model should
    not be told they are the same.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=weather_body(temperature=None, precipitation=None))

    reading = client_for(handler).fetch_current(latitude=13.0, longitude=80.0)

    assert reading.temperature_c is None
    assert reading.precipitation_mm is None


@pytest.mark.parametrize(
    "literal",
    ['"warm"', "true", "NaN", "Infinity"],
)
def test_non_numeric_temperature_is_rejected(literal: str) -> None:
    """A non-numeric value means the shape changed; it is not coerced.

    ``NaN``/``Infinity`` are included because JSON permits them as bare literals
    while ``json.dumps`` refuses to write them, so the body is built as raw text.
    Both would otherwise reach the model as a float and poison the feature.
    """

    body = (
        '{"current": {"time": "2026-10-05T12:00", "weather_code": 0, '
        f'"temperature_2m": {literal}}}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    with pytest.raises(ProviderResponseError):
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)


def test_non_json_body_is_rejected() -> None:
    """A proxy returning an HTML error page must not parse as weather."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    with pytest.raises(ProviderResponseError, match="JSON"):
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)


def test_time_is_parsed_as_utc() -> None:
    """``timezone=GMT`` was requested, so the naive value is made aware as UTC."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=weather_body())

    reading = client_for(handler).fetch_current(latitude=13.0, longitude=80.0)

    assert reading.observed_at.tzinfo is not None
    assert reading.observed_at.utcoffset().total_seconds() == 0


def test_unparseable_time_is_rejected() -> None:
    """A timestamp that cannot be read would be stored as-is and corrupt history."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = weather_body()
        payload["current"]["time"] = "not-a-timestamp"
        return httpx.Response(200, json=payload)

    with pytest.raises(ProviderResponseError, match="current.time"):
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)


# --- Upstream failures -----------------------------------------------------


def test_rate_limit_maps_to_503() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"reason": "daily limit"})

    with pytest.raises(ProviderRateLimitedError) as caught:
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)

    assert caught.value.status_code == 503


def test_server_error_maps_to_502() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    with pytest.raises(UpstreamProviderError) as caught:
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)

    assert caught.value.status_code == 502


def test_timeout_maps_to_504() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(UpstreamTimeoutError) as caught:
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)

    assert caught.value.status_code == 504


def test_vendor_error_text_is_not_forwarded() -> None:
    """The vendor's body is not echoed, because it can contain the request URL.

    The traffic request URL carries ``key=...``; forwarding a vendor body verbatim
    would put a credential into an API response or log line.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="failed for https://api.tomtom.com/?key=SECRET")

    with pytest.raises(UpstreamProviderError) as caught:
        client_for(handler).fetch_current(latitude=13.0, longitude=80.0)

    assert "SECRET" not in caught.value.message


# --- Locations file --------------------------------------------------------

VALID_LOCATIONS = {
    "version": 1,
    "locations": [
        {
            "location_id": "LOC-001",
            "road_segment_id": "SEG-01",
            "road_name": "Anna Salai",
            "latitude": 13.0569,
            "longitude": 80.2425,
            "free_flow_speed_kph_max": 60.0,
        },
        {
            "location_id": "LOC-002",
            "road_segment_id": "SEG-02",
            "road_name": "Raj Bhavan Road",
            "latitude": 12.9758,
            "longitude": 77.7497,
            "free_flow_speed_kph_max": 60.0,
        },
    ],
}


def write_locations(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "locations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_file_loads(tmp_path: Path) -> None:
    locations = load_monitored_locations(write_locations(tmp_path, VALID_LOCATIONS))

    assert len(locations) == 2
    first = locations[0]
    assert first.location_id == "LOC-001"
    assert first.road_segment_id == "SEG-01"
    # TomTom's `point` parameter takes "lat,lon" in that order.
    assert first.coordinates == "13.05690,80.24250"


def test_shipped_file_is_valid() -> None:
    """The file committed to the repository must load.

    A broken default file means the real provider is unconfigured for everyone who
    clones the project, and the failure only appears on the first fetch.
    """

    root = Path(__file__).resolve().parents[1]
    locations = load_monitored_locations(root / "config" / "monitored_locations.json")

    assert locations
    for location in locations:
        assert location.road_segment_id in KNOWN_ROAD_SEGMENTS


def test_missing_file_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_monitored_locations(tmp_path / "absent.json")


def test_invalid_json_is_a_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="not valid JSON"):
        load_monitored_locations(path)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"locations": []},
        {"locations": "not-a-list"},
    ],
)
def test_unusable_container_is_rejected(tmp_path: Path, payload: object) -> None:
    """A real provider has no built-in locations, so an empty file is an error."""

    with pytest.raises(ConfigurationError):
        load_monitored_locations(write_locations(tmp_path, payload))


def test_unknown_road_segment_is_rejected(tmp_path: Path) -> None:
    """A segment the model never learned cannot be scored, so it is refused."""

    payload = json.loads(json.dumps(VALID_LOCATIONS))
    payload["locations"][0]["road_segment_id"] = "SEG-99"

    with pytest.raises(ConfigurationError, match="SEG-99"):
        load_monitored_locations(write_locations(tmp_path, payload))


def test_duplicate_location_id_is_rejected(tmp_path: Path) -> None:
    """Identifiers key stored observations, so they must be unique."""

    payload = json.loads(json.dumps(VALID_LOCATIONS))
    payload["locations"][1]["location_id"] = "LOC-001"

    with pytest.raises(ConfigurationError, match="repeats"):
        load_monitored_locations(write_locations(tmp_path, payload))


@pytest.mark.parametrize(
    "field, value",
    [
        ("latitude", 95.0),
        ("latitude", "north"),
        ("longitude", 200.0),
        ("free_flow_speed_kph_max", 0.0),
        ("free_flow_speed_kph_max", 900.0),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path: Path, field: str, value) -> None:
    """Bad coordinates or bounds fail at load, not twenty minutes into a cycle."""

    payload = json.loads(json.dumps(VALID_LOCATIONS))
    payload["locations"][0][field] = value

    with pytest.raises(ConfigurationError):
        load_monitored_locations(write_locations(tmp_path, payload))


@pytest.mark.parametrize("field", ["location_id", "road_segment_id", "road_name", "latitude"])
def test_missing_required_field_is_rejected(tmp_path: Path, field: str) -> None:
    payload = json.loads(json.dumps(VALID_LOCATIONS))
    del payload["locations"][0][field]

    with pytest.raises(ConfigurationError, match=field):
        load_monitored_locations(write_locations(tmp_path, payload))


def test_max_locations_is_enforced(tmp_path: Path) -> None:
    """A cap stops a config mistake from spending a month's quota in one cycle."""

    with pytest.raises(ConfigurationError, match="TRAFFIC_MAX_LOCATIONS"):
        load_monitored_locations(
            write_locations(tmp_path, VALID_LOCATIONS), max_locations=1
        )


def test_free_flow_bound_defaults_when_absent(tmp_path: Path) -> None:
    """Omitting the bound is allowed; a sane ceiling is applied instead."""

    payload = json.loads(json.dumps(VALID_LOCATIONS))
    del payload["locations"][0]["free_flow_speed_kph_max"]

    locations = load_monitored_locations(write_locations(tmp_path, payload))

    assert locations[0].free_flow_speed_kph_max > 0