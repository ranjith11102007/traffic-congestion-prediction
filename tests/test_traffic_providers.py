"""Tests for the data-provider abstraction and the simulated provider.

The behaviour worth pinning down here is provenance and refusal. Simulated data
must always be labelled, and the real provider must fail rather than substitute
synthetic readings for a missing API key.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    ProviderNotConfiguredError,
    ProviderRateLimitedError,
    ProviderResponseError,
    UpstreamProviderError,
    UpstreamTimeoutError,
)
from app.models.traffic import SIMULATION_SOURCE
from app.services.locations import MonitoredLocation
from app.services.providers import TrafficDataProvider, TrafficRecord
from app.services.real_provider import RealTrafficProvider
from app.services.weather_client import OpenMeteoClient
from app.services.simulated_provider import (
    SIMULATED_LOCATIONS,
    SimulatedTrafficProvider,
)
from app.services.traffic_collector import (
    effective_source,
    get_provider,
    record_to_observation,
    store_records,
)


def make_settings(**overrides: str) -> Settings:
    """Build settings directly, bypassing the environment and any ``.env`` file.

    Constructing ``Settings`` rather than mutating ``os.environ`` keeps these tests
    from leaking configuration into the rest of the suite through the cached
    ``get_settings`` singleton.
    """

    defaults = {
        "TRAFFIC_PROVIDER": "simulation",
        "TRAFFIC_API_KEY": "",
        "TRAFFIC_API_BASE_URL": "",
        "DATABASE_URL": "",
    }
    return Settings(**{**defaults, **overrides})


#: Locations mirroring config/monitored_locations.json. Injected so these tests
#: never depend on that file being present, nor on the working directory.
_LOCATIONS: tuple[MonitoredLocation, ...] = (
    MonitoredLocation(
        location_id="LOC-001",
        road_segment_id="SEG-01",
        road_name="Anna Salai",
        latitude=13.0569,
        longitude=80.2425,
        free_flow_speed_kph_max=60.0,
        city="Chennai",
    ),
)


def make_real_settings(**overrides: str) -> Settings:
    """Settings for a fully configured TomTom deployment.

    ``TRAFFIC_API_BASE_URL`` defaults to the real TomTom host, so a test that only
    omits the key exercises the genuine default rather than a placeholder URL.
    """

    defaults = {
        "TRAFFIC_PROVIDER": "real",
        "TRAFFIC_API_VENDOR": "tomtom",
        "TRAFFIC_API_KEY": "test-key-not-a-real-credential",
        "TRAFFIC_API_BASE_URL": "https://api.tomtom.com",
        "TRAFFIC_MIN_REQUEST_INTERVAL": "0",  # never sleep in a test
        "DATABASE_URL": "",
    }
    return Settings(**{**defaults, **overrides})


def test_simulated_provider_implements_the_interface() -> None:
    """The provider must satisfy the abstract contract it is selected through."""

    provider = SimulatedTrafficProvider()

    assert isinstance(provider, TrafficDataProvider)
    assert provider.name == "simulation"
    assert provider.is_simulation is True
    assert provider.is_configured is True


def test_every_simulated_record_is_labelled_simulation() -> None:
    """No generated record may escape unlabelled.

    This is the guarantee that keeps synthetic traffic from being mistaken for
    live data once it is stored.
    """

    provider = SimulatedTrafficProvider()
    records = provider.fetch_current_traffic()

    assert records
    assert all(record.source == SIMULATION_SOURCE for record in records)
    assert all(record.metadata.get("simulated") is True for record in records)


def test_generation_is_deterministic() -> None:
    """The same timestamp yields the same reading.

    Determinism is what lets a developer see a repeatable history and lets tests
    assert on values instead of ranges.
    """

    provider = SimulatedTrafficProvider()
    moment = datetime(2026, 1, 18, 8, 0, tzinfo=timezone.utc)

    first = provider._generate("LOC-001", "SEG-01", "Anna Salai", 60.0, moment)
    second = provider._generate("LOC-001", "SEG-01", "Anna Salai", 60.0, moment)

    assert first.vehicle_count == second.vehicle_count
    assert first.avg_speed_kph == second.avg_speed_kph
    assert first.weather_condition == second.weather_condition
    assert first.rainfall_mm == second.rainfall_mm


def test_different_providers_need_not_agree() -> None:
    """A different seed must actually change the output.

    Otherwise the seed is not wired in and the determinism test above would pass
    for the wrong reason.
    """

    moment = datetime(2026, 1, 18, 8, 0, tzinfo=timezone.utc)
    default = SimulatedTrafficProvider()
    other = SimulatedTrafficProvider(seed=99)

    a = default._generate("LOC-001", "SEG-01", "Anna Salai", 60.0, moment)
    b = other._generate("LOC-001", "SEG-01", "Anna Salai", 60.0, moment)

    assert (a.avg_speed_kph, a.vehicle_count) != (b.avg_speed_kph, b.vehicle_count)


def test_simulated_speeds_stay_below_free_flow() -> None:
    """Generated speed must respect the free-flow reference.

    The database now enforces ``free_flow_speed_kph > avg_speed_kph`` as a CHECK
    constraint, so a provider that violated it would fail at insert time.
    """

    provider = SimulatedTrafficProvider()
    records = provider.fetch_historical_traffic(location_id="LOC-001", limit=48)

    assert len(records) == 48
    for record in records:
        assert record.avg_speed_kph < record.free_flow_speed_kph
        assert record.avg_speed_kph > 0
        assert record.vehicle_count >= 0
        assert record.rainfall_mm >= 0


def test_simulated_segments_match_the_models_known_segments() -> None:
    """Simulated locations must map onto road segments the model can score.

    A segment outside the trained category set would produce a one-hot row of all
    zeros and a silently wrong prediction.
    """

    from ml.train import load_preprocessor

    known = set(load_preprocessor().categorical_levels["road_segment_id"])

    assert {segment for segment, _, _ in SIMULATED_LOCATIONS.values()} <= known


def test_unknown_location_is_rejected() -> None:
    """Asking for a location the simulation does not model must fail loudly."""

    provider = SimulatedTrafficProvider()

    with pytest.raises(ProviderNotConfiguredError, match="LOC-999"):
        provider.fetch_current_traffic(location_id="LOC-999")


def test_history_is_oldest_first_and_hourly() -> None:
    """Lag features need chronological order with a regular interval."""

    provider = SimulatedTrafficProvider()
    records = provider.fetch_historical_traffic(location_id="LOC-001", limit=12)

    assert len(records) == 12
    timestamps = [record.timestamp for record in records]
    assert timestamps == sorted(timestamps)
    gaps = {right - left for left, right in zip(timestamps, timestamps[1:])}
    assert gaps == {timedelta(hours=1)}


def test_history_respects_start_time() -> None:
    """A start bound must truncate the generated range."""

    provider = SimulatedTrafficProvider()
    end = datetime(2026, 1, 18, 12, 0, tzinfo=timezone.utc)
    start = end - timedelta(hours=4)

    records = provider.fetch_historical_traffic(
        location_id="LOC-001", start_time=start, end_time=end, limit=24
    )

    assert records
    assert all(record.timestamp >= start for record in records)
    assert len(records) == 5


def test_history_rejects_inverted_range() -> None:
    """``start_time`` after ``end_time`` is a caller mistake."""

    provider = SimulatedTrafficProvider()
    moment = datetime(2026, 1, 18, 12, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="start_time"):
        provider.fetch_historical_traffic(
            location_id="LOC-001", start_time=moment + timedelta(hours=1), end_time=moment
        )


def test_current_fetch_without_location_covers_every_location() -> None:
    """An unqualified fetch must be the whole set, not an arbitrary location.

    Returning "some" location would make a collect run's output depend on dict
    ordering.
    """

    provider = SimulatedTrafficProvider()

    records = provider.fetch_current_traffic()

    assert {record.location_id for record in records} == set(SIMULATED_LOCATIONS)


def test_model_frame_row_is_target_free() -> None:
    """The provider must not emit derived target columns.

    ``speed_ratio`` is computed by the ML pipeline from raw inputs. A provider
    supplying it would bypass that calculation, which is the leakage Stage 3
    removed.
    """

    provider = SimulatedTrafficProvider()
    record = provider.fetch_current_traffic(location_id="LOC-001")[0]

    row = record.to_model_frame_row()

    assert "speed_ratio" not in row
    assert "congestion_level" not in row
    assert row["precipitation_mm"] == record.rainfall_mm
    assert row["road_segment_id"] == record.road_segment_id
    # Model version 4.0.0 dropped flow as a feature. It may still be present in the
    # raw data, but emitting it into the model's input frame would imply it is used.
    assert "flow_veh_per_hr" not in row


def test_real_provider_without_credentials_raises() -> None:
    """An unconfigured real provider must fail rather than fabricate data."""

    provider = RealTrafficProvider(make_settings(TRAFFIC_PROVIDER="real"))

    assert provider.is_configured is False

    with pytest.raises(ProviderNotConfiguredError, match="TRAFFIC_API_BASE_URL"):
        provider.fetch_current_traffic(location_id="LOC-001")


def test_real_provider_missing_key_names_the_key() -> None:
    """With a base URL set, the missing credential is named explicitly."""

    settings = make_settings(
        TRAFFIC_PROVIDER="real",
        TRAFFIC_API_BASE_URL="https://api.example.invalid/traffic",
    )
    provider = RealTrafficProvider(settings)

    assert provider.is_configured is False

    with pytest.raises(ProviderNotConfiguredError, match="TRAFFIC_API_KEY"):
        provider.fetch_current_traffic()


def test_real_provider_never_falls_back_to_simulation() -> None:
    """The critical safety property: no silent substitution.

    Even with full credentials present, an upstream failure must raise rather than
    return synthetic readings. Fabricated traffic under a real provider's name is
    worse than an outage, and a future implementer must not be able to reintroduce
    it by accident.
    """

    provider = RealTrafficProvider(
        make_real_settings(),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    httpx.ConnectError("upstream unreachable")
                )
            )
        ),
        locations=_LOCATIONS,
    )

    assert provider.is_configured is True
    assert provider.is_simulation is False

    with pytest.raises(UpstreamProviderError):
        provider.fetch_current_traffic()

    # And a history request fails explicitly rather than being faked.
    with pytest.raises(UpstreamProviderError, match="historical"):
        provider.fetch_historical_traffic(location_id="LOC-001")


def test_real_provider_description_never_reveals_the_key() -> None:
    """``describe()`` reaches a client, so it must not echo the credential."""

    secret = "super-secret-traffic-key"
    provider = RealTrafficProvider(
        make_settings(
            TRAFFIC_PROVIDER="real",
            TRAFFIC_API_BASE_URL="https://api.example.invalid/traffic",
            TRAFFIC_API_KEY=secret,
        )
    )

    description = provider.describe()

    assert secret not in description
    # Readiness is reported in words; the credential itself never appears.
    assert "Not configured" not in description
    assert "Connected" in description


def test_real_provider_description_reports_missing_parts() -> None:
    """The description must say what is missing, not just that something is."""

    provider = RealTrafficProvider(make_settings(TRAFFIC_PROVIDER="real"))

    description = provider.describe()

    # The base URL now defaults to TomTom, so only the key can be missing, and the
    # description must name it specifically rather than saying "not configured".
    assert "TRAFFIC_API_KEY" in description


# ---------------------------------------------------------------------------
# Real provider: TomTom request shape and Open-Meteo enrichment
# ---------------------------------------------------------------------------


def _tomtom_body(
    *,
    current_speed: float = 31.4,
    free_flow_speed: float = 60.0,
    road_closure: str = "N",
) -> dict:
    """A documented Flow Segment Data response."""

    return {
        "flowSegmentData": {
            "coordinates": {
                "point": {"latitude": 13.0569, "longitude": 80.2425}
            },
            "frc": "FRC_F2",
            "openlr": "C1C2C3|ABCDEF",
            "currentSpeed": current_speed,
            "freeFlowSpeed": free_flow_speed,
            "confidence": 0.91,
            "roadClosure": road_closure,
        }
    }


def _weather_body(
    *, code: int = 0, temperature: float = 31.2, precipitation: float = 0.0
) -> dict:
    """A documented Open-Meteo current-weather response."""

    return {
        "current": {
            "time": "2026-10-05T12:00",
            "temperature_2m": temperature,
            "precipitation": precipitation,
            "weather_code": code,
        }
    }


def _provider(
    handler,
    *,
    settings: Settings | None = None,
    locations: Sequence[MonitoredLocation] | None = None,
) -> RealTrafficProvider:
    """Build a provider whose every HTTP call is served by ``handler``.

    The weather client is constructed here with the same transport rather than
    left to the provider's default. Without this the provider opens its own real
    ``httpx.Client``, and a test asserting on a mocked weather body would instead
    reach Open-Meteo over the network: slow, and dependent on live conditions.
    """

    weather = OpenMeteoClient(
        settings or make_real_settings(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return RealTrafficProvider(
        settings or make_real_settings(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        weather_client=weather,
        locations=locations if locations is not None else _LOCATIONS,
    )


def test_real_provider_maps_a_documented_response() -> None:
    """A well-formed response becomes one honest record."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=_tomtom_body())
        return httpx.Response(200, json=_weather_body())

    record = _provider(handler).fetch_current_traffic(location_id="LOC-001")[0]

    assert record.location_id == "LOC-001"
    assert record.road_segment_id == "SEG-01"
    assert record.avg_speed_kph == 31.4
    assert record.free_flow_speed_kph == 60.0
    assert record.weather_condition == "clear"
    assert record.temperature_c == 31.2
    assert record.rainfall_mm == 0.0
    assert record.is_incident is False
    assert record.source == "tomtom"


def test_real_provider_leaves_vehicle_count_null() -> None:
    """The honest core of Stage 5: throughput is unavailable, so it stays null.

    No self-serve traffic API publishes a vehicle count. Storing a fabricated one
    would be indistinguishable from a measurement once written, which is why
    ``vehicle_count`` is nullable and this assertion exists.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=_tomtom_body())
        return httpx.Response(200, json=_weather_body())

    record = _provider(handler).fetch_current_traffic(location_id="LOC-001")[0]

    assert record.vehicle_count is None
    assert "does not publish" in record.metadata["vehicle_count_unavailable_reason"]


def test_real_provider_records_vendor_provenance() -> None:
    """Each observation must say which vendor and weather source produced it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=_tomtom_body())
        return httpx.Response(200, json=_weather_body())

    record = _provider(handler).fetch_current_traffic(location_id="LOC-001")[0]

    assert record.metadata["traffic_vendor"] == "tomtom"
    assert record.metadata["weather_source"] == "open-meteo"
    # Vendor road identifiers are preserved for traceability, but they are not
    # substituted for road_segment_id, which is an ML feature the model learned.
    assert record.metadata["tomtom_frc"] == "FRC_F2"
    assert record.metadata["tomtom_openlr"] == "C1C2C3|ABCDEF"
    assert record.road_segment_id == "SEG-01"


def test_real_provider_sends_the_documented_query() -> None:
    """The request must match TomTom's documented parameter names."""

    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=_tomtom_body())
        return httpx.Response(200, json=_weather_body())

    _provider(handler).fetch_current_traffic(location_id="LOC-001")

    traffic_url = next(url for url in seen if "flowSegmentData" in str(url))
    assert traffic_url.params["point"] == "13.05690,80.24250"
    assert traffic_url.params["zoom"] == "10"
    # Explicit rather than relying on a vendor default that could change upstream.
    assert traffic_url.params["unit"] == "KphMPH"
    assert "currentSpeed" in traffic_url.params["fields"]


def test_real_provider_maps_road_closure_to_an_incident() -> None:
    """A closed road is a real incident signal, and is recorded as one."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=_tomtom_body(road_closure="Y"))
        return httpx.Response(200, json=_weather_body())

    record = _provider(handler).fetch_current_traffic(location_id="LOC-001")[0]

    assert record.is_incident is True


def test_real_provider_unrecognised_closure_is_not_an_incident() -> None:
    """An unknown closure code must not manufacture an incident.

    TomTom documents ``Y``/``N``. Treating any other value as blocked would invent
    an event the vendor never reported.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=_tomtom_body(road_closure="maybe"))
        return httpx.Response(200, json=_weather_body())

    record = _provider(handler).fetch_current_traffic(location_id="LOC-001")[0]

    assert record.is_incident is False


@pytest.mark.parametrize(
    "body, reason",
    [
        ({}, "flowSegmentData"),
        ({"flowSegmentData": {}}, "currentSpeed"),
        ({"flowSegmentData": {"currentSpeed": 30.0}}, "freeFlowSpeed"),
        (
            {"flowSegmentData": {"currentSpeed": "fast", "freeFlowSpeed": 60.0}},
            "non-numeric",
        ),
        (
            {"flowSegmentData": {"currentSpeed": 400.0, "freeFlowSpeed": 500.0}},
            "outside",
        ),
        (
            {"flowSegmentData": {"currentSpeed": 61.0, "freeFlowSpeed": 60.0}},
            "above",
        ),
    ],
)
def test_real_provider_rejects_an_unusable_traffic_body(body, reason) -> None:
    """A 2xx body missing a usable speed is a failure, not a defaulted reading.

    Current speed is the model's primary input. Substituting anything for it would
    store a real-looking observation that says nothing about the road.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=body)
        return httpx.Response(200, json=_weather_body())

    with pytest.raises(ProviderResponseError, match=reason):
        _provider(handler).fetch_current_traffic(location_id="LOC-001")


def test_real_provider_maps_rate_limit_to_503() -> None:
    """Throttling is expected and self-limiting, so it gets its own error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "quota exceeded"})

    provider = _provider(handler)

    with pytest.raises(ProviderRateLimitedError) as caught:
        provider.fetch_current_traffic(location_id="LOC-001")

    assert caught.value.status_code == 503
    assert caught.value.vendor == "tomtom"


def test_real_provider_maps_rejected_key_to_502() -> None:
    """A rejected key is an upstream failure, and must not echo the key back."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden: key test-key-not-a-real-credential")

    with pytest.raises(UpstreamProviderError) as caught:
        _provider(handler).fetch_current_traffic(location_id="LOC-001")

    assert "test-key-not-a-real-credential" not in caught.value.message


def test_real_provider_maps_timeout_to_504() -> None:
    """A timeout is distinct from a rejection so the scheduler can retry it."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(UpstreamTimeoutError) as caught:
        _provider(handler).fetch_current_traffic(location_id="LOC-001")

    assert caught.value.status_code == 504


def test_real_provider_history_raises_with_an_actionable_message() -> None:
    """No historical feed exists, and the error must say what to do instead.

    An empty sequence would read as "no history is available", which would leave a
    real deployment permanently unable to score a prediction without explaining why.
    """

    with pytest.raises(UpstreamProviderError) as caught:
        _provider(lambda request: httpx.Response(200, json={})).fetch_historical_traffic()

    message = caught.value.message
    assert "current" in message
    assert "COLLECTION_ENABLED" in message


def test_real_provider_fails_a_whole_reading_when_weather_fails() -> None:
    """Weather is a required model input, so a partial reading is not stored.

    Persisting traffic without weather would create an observation the model cannot
    score, which is worse than collecting again on the next cycle.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=_tomtom_body())
        return httpx.Response(500, json={"error": "upstream"})

    with pytest.raises(UpstreamProviderError):
        _provider(handler).fetch_current_traffic(location_id="LOC-001")


def test_real_provider_rejects_an_unconfigured_location() -> None:
    """An unknown location id is a configuration error naming the known set."""

    provider = _provider(lambda request: httpx.Response(200, json=_tomtom_body()))

    with pytest.raises(ConfigurationError, match="LOC-999"):
        provider.fetch_current_traffic(location_id="LOC-999")


def test_real_provider_uses_configured_metadata_not_vendor_ids() -> None:
    """Segment id and road name come from config; the vendor supplies neither."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "flowSegmentData" in str(request.url):
            return httpx.Response(200, json=_tomtom_body())
        return httpx.Response(200, json=_weather_body())

    record = _provider(handler).fetch_current_traffic(location_id="LOC-001")[0]

    assert record.road_name == "Anna Salai"
    assert record.road_segment_id == "SEG-01"


def test_get_provider_selects_simulation_by_default() -> None:
    """The default provider is the labelled simulation."""

    provider = get_provider(make_settings())

    assert isinstance(provider, SimulatedTrafficProvider)


def test_get_provider_selects_real_when_configured() -> None:
    """Selecting the real provider yields the real provider, not the simulation."""

    provider = get_provider(make_settings(TRAFFIC_PROVIDER="real"))

    assert isinstance(provider, RealTrafficProvider)
    assert provider.is_simulation is False


def test_simulation_provider_overrides_a_lying_record() -> None:
    """A simulating provider's records are always stamped ``simulation``.

    The trust boundary runs provider to database. If the record's own ``source``
    were trusted, a provider could mislabel itself and defeat the guarantee.
    """

    provider = SimulatedTrafficProvider()
    record = TrafficRecord(
        timestamp=datetime(2026, 1, 18, 8, 0, tzinfo=timezone.utc),
        location_id="LOC-001",
        road_name="Anna Salai",
        road_segment_id="SEG-01",
        vehicle_count=500,
        avg_speed_kph=30.0,
        free_flow_speed_kph=60.0,
        weather_condition="clear",
        temperature_c=27.0,
        rainfall_mm=0.0,
        is_incident=False,
        source="tomcat-production-feed",
    )

    assert effective_source(provider, record) == SIMULATION_SOURCE
    assert record_to_observation(record, SIMULATION_SOURCE).source == SIMULATION_SOURCE


def test_non_simulated_record_keeps_its_provenance() -> None:
    """A real provider's records must keep their own origin label."""

    class FakeReal(TrafficDataProvider):
        name = "fake"
        is_simulation = False

        def fetch_current_traffic(self, *, location_id=None):
            return ()

        def fetch_historical_traffic(
            self, *, location_id=None, start_time=None, end_time=None, limit=100
        ):
            return ()

    record = TrafficRecord(
        timestamp=datetime(2026, 1, 18, 8, 0, tzinfo=timezone.utc),
        location_id="LOC-001",
        road_name="Anna Salai",
        road_segment_id="SEG-01",
        vehicle_count=500,
        avg_speed_kph=30.0,
        free_flow_speed_kph=60.0,
        weather_condition="clear",
        temperature_c=27.0,
        rainfall_mm=0.0,
        source="vendor-x",
    )

    assert effective_source(FakeReal(), record) == "vendor-x"


def test_store_records_skips_duplicates_rather_than_failing(db_session) -> None:
    """A collector re-reading an hour must not fail the whole run.

    Only the single explicit POST treats a duplicate as a client mistake.
    """

    provider = SimulatedTrafficProvider()
    moment = datetime(2026, 1, 18, 8, 0, tzinfo=timezone.utc)
    records = (
        provider._generate("LOC-001", "SEG-01", "Anna Salai", 60.0, moment),
        provider._generate("LOC-002", "SEG-02", "Raj Bhavan Road", 60.0, moment),
    )

    first = store_records(db_session, records, provider)
    second = store_records(db_session, records, provider)

    assert len(first) == 2
    assert len(second) == 0  # both already present


def test_store_records_raises_on_duplicate_when_asked(db_session) -> None:
    """With ``skip_duplicates=False`` the caller wants to hear about collisions."""

    from app.core.exceptions import DuplicateObservationError

    provider = SimulatedTrafficProvider()
    record = provider.fetch_current_traffic(location_id="LOC-001")[0]

    store_records(db_session, [record], provider)

    with pytest.raises(DuplicateObservationError):
        store_records(db_session, [record], provider, skip_duplicates=False)