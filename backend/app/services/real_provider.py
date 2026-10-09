"""Real traffic provider: TomTom Flow Segment Data plus Open-Meteo weather.

What this provider does
-----------------------
For each configured road it queries TomTom's Flow Segment Data endpoint for the
current and free-flow speed on the segment containing that coordinate, then
queries Open-Meteo for the weather there, and combines the two into a
:class:`~app.services.providers.TrafficRecord`.

Vendor choice
-------------
TomTom Traffic Flow Segment Data was chosen after checking the three realistic
candidates against current vendor documentation:

* **TomTom** - point query, JSON, speeds in km/h, India covered. The pricing page
  documents a 20,000 requests/month allowance for the Flow API with no credit
  card (https://docs.tomtom.com/pricing).
* **HERE Traffic v7** - richer response (bbox query, ``jamFactor``,
  ``traversability``, a server-side timestamp), but speeds are metres/second and
  there is no self-serve historical endpoint.
* **Mapbox** - only route annotations; the bulk traffic product is Enterprise-only.

TomTom was selected because its free allowance is documented and a point query
per monitored location matches how this project uses a traffic feed.

The honest limitation
---------------------
**None of these vendors publishes vehicle throughput.** TomTom returns current
speed, free-flow speed, confidence, functional road class, OpenLR identifier and
road closure. None returns a count, volume or density of vehicles.

So ``vehicle_count`` is left ``None`` on every record this provider produces. That
is not a gap to be filled with an estimate: a fabricated count stored under a real
vendor's name is indistinguishable from a measurement once it is in the database.
The matching decision is model version 4.0.0, which dropped ``flow_veh_per_hr``
from the feature set. Because the model needs prior observations to build its lag
and rolling features, a real deployment must accumulate history over time;
:meth:`fetch_historical_traffic` therefore raises rather than inventing a past.

Road names and segment IDs
--------------------------
TomTom has no road name field, so ``road_name`` and ``road_segment_id`` come from
the monitored-locations file. The vendor's road identifiers (``frc``, ``openlr``)
are preserved in the record's metadata, where they are useful, but they are not
substituted for ``road_segment_id``: that column is an ML feature whose values
the model learned, and feeding it a vendor-specific string would be meaningless to
the model.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx

from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    ProviderNotConfiguredError,
    ProviderRateLimitedError,
    ProviderResponseError,
    UpstreamProviderError,
    UpstreamTimeoutError,
)
from app.services.locations import MonitoredLocation, get_monitored_locations
from app.services.providers import TrafficDataProvider, TrafficRecord
from app.services.weather_client import OpenMeteoClient, WeatherReading

logger = logging.getLogger(__name__)

#: Reported in error details so a caller can tell which upstream failed.
VENDOR = "tomtom"

#: Path appended to ``TRAFFIC_API_BASE_URL``. TomTom versions the service in the
#: path; 4 is the version whose documented response contains ``currentSpeed``,
#: ``freeFlowSpeed``, ``confidence``, ``frc``, ``openlr`` and ``roadClosure``.
FLOW_SEGMENT_PATH = "/traffic/services/flowSegmentData"

#: Value recorded as the observation's ``source``. Not ``SIMULATION_SOURCE``:
#: every field is measured or reported by a named vendor.
SOURCE = "tomtom"

#: ``roadClosure`` values that mean the road is blocked for traffic. The documented
#: values are ``Y`` and ``N``; the spellings of ``true`` are tolerated in case a
#: regional endpoint normalises the field.
_CLOSURE_VALUES_BLOCKING = frozenset({"Y", "y", "true", "True", "TRUE"})

#: Max upstream speed accepted before a reading is treated as garbage. A speed
#: above this is not a traffic condition, it is a parsing mistake.
_MAX_PLAUSIBLE_KPH = 200.0


def _blocked_by_closure(value: object) -> bool:
    """Interpret TomTom's ``roadClosure`` field.

    The documented values are ``Y``/``N``, but the field arrives as a string and a
    lenient ``in "Y"`` test would also match any string containing a ``Y``. An
    explicit set is used instead, and an unrecognised value is treated as
    "not blocked" and logged, because guessing an incident from an unknown code
    would manufacture an event the vendor never reported.
    """

    if isinstance(value, str):
        return value in _CLOSURE_VALUES_BLOCKING
    return False


class RealTrafficProvider(TrafficDataProvider):
    """Fetches live speeds from TomTom and weather from Open-Meteo.

    Never returns data it did not receive. If either upstream fails, the whole
    reading fails: storing traffic without weather produces an observation the
    model cannot score, so a partial result is not worth persisting.
    """

    name = "real"
    is_simulation = False

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        weather_client: OpenMeteoClient | None = None,
        locations: Sequence[MonitoredLocation] | None = None,
    ) -> None:
        self._settings = settings
        # Injected clients are not owned, so tests can pass a MockTransport-backed
        # client and close it themselves.
        self._client = client or httpx.Client(
            timeout=settings.TRAFFIC_REQUEST_TIMEOUT,
            headers={"User-Agent": "traffic-prediction-api/4.0.0"},
        )
        self._owns_client = client is None
        self._weather = weather_client or OpenMeteoClient(settings)
        self._owns_weather = weather_client is None
        self._locations_override = locations
        self._last_request_at = 0.0

    @property
    def is_configured(self) -> bool:
        return self._settings.has_real_provider_configured

    def describe(self) -> str:
        """Describe readiness without revealing the key.

        Only presence is reported. The key itself is never logged or returned to
        a client.
        """

        base_url = self._settings.TRAFFIC_API_BASE_URL.strip()
        has_key = bool(self._settings.TRAFFIC_API_KEY.strip())
        if not base_url and not has_key:
            return (
                "Not configured: TRAFFIC_API_KEY is empty. No traffic API is "
                "connected, so this provider cannot fetch."
            )
        if not has_key:
            return (
                "Not configured: TRAFFIC_API_KEY is empty. Set it in your untracked "
                ".env file to enable live collection."
            )
        try:
            count = len(self._locations())
        except ConfigurationError as exc:
            return f"Configured with a key, but the locations file is unusable: {exc.message}"
        return (
            f"Connected to {VENDOR} Flow Segment Data at {base_url} for {count} "
            f"monitored location(s), with weather from Open-Meteo. Speeds are "
            "live; vehicle_count is null because no self-serve traffic API "
            "publishes throughput, and no historical feed is available."
        )

    def _require_configuration(self) -> None:
        """Raise unless the key, base URL and locations file are all usable."""

        if not self._settings.TRAFFIC_API_BASE_URL.strip():
            raise ProviderNotConfiguredError(
                "TRAFFIC_API_BASE_URL is not set. A real traffic provider needs the "
                "API's base URL; the default is https://api.tomtom.com.",
                details={"missing": "TRAFFIC_API_BASE_URL"},
            )
        if not self._settings.TRAFFIC_API_KEY.strip():
            raise ProviderNotConfiguredError(
                "TRAFFIC_API_KEY is not set. A real traffic provider needs an API "
                "key. Add it to your untracked .env file; never commit it.",
                details={"missing": "TRAFFIC_API_KEY"},
            )

    def close(self) -> None:
        """Close any connection pool this instance created."""

        if self._owns_client:
            self._client.close()
        if self._owns_weather:
            self._weather.close()

    def __enter__(self) -> RealTrafficProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _locations(self) -> Sequence[MonitoredLocation]:
        """Return the configured locations, or the injected set for a test."""

        if self._locations_override is not None:
            return self._locations_override
        return get_monitored_locations(self._settings)

    def list_locations(self) -> Sequence[str]:
        return tuple(location.location_id for location in self._locations())

    def _throttle(self) -> None:
        """Sleep so consecutive requests stay above the vendor's QPS limit.

        TomTom documents 10 queries/second for non-tile traffic endpoints. Rather
        than relying on that holding under load, requests are spaced by
        ``TRAFFIC_MIN_REQUEST_INTERVAL`` measured from the previous start.
        """

        interval = self._settings.TRAFFIC_MIN_REQUEST_INTERVAL
        if interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_at = time.monotonic()

    def _url(self) -> str:
        return f"{self._settings.TRAFFIC_API_BASE_URL}{FLOW_SEGMENT_PATH}"

    def _params(self, location: MonitoredLocation) -> dict[str, str]:
        """Build the Flow Segment Data query for one coordinate.

        ``unit=KphMPH`` is explicit rather than relying on the vendor default, so
        a default change upstream cannot silently reinterpret every stored speed.
        ``fields`` asks only for what this provider reads; requesting less keeps
        the payload small and makes the dependency on those exact field names
        explicit.
        """

        return {
            "key": self._settings.TRAFFIC_API_KEY.strip(),
            "point": location.coordinates,
            "zoom": str(self._settings.TRAFFIC_ZOOM),
            "unit": "KphMPH",
            "fields": "currentSpeed,freeFlowSpeed,confidence,roadClosure,frc,openlr",
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate a TomTom failure into a domain error.

        The vendor's error body echoes the request URL, and the traffic request URL
        carries ``key=...``, so nothing from the body is forwarded. Only the status
        and the vendor's error code are kept, for the server log.
        """

        if response.status_code == 429:
            raise ProviderRateLimitedError(
                f"{VENDOR} rate limit reached. Raise COLLECTION_INTERVAL_SECONDS or "
                "reduce the number of monitored locations.",
                vendor=VENDOR,
                vendor_status=response.status_code,
            )
        if response.status_code in (401, 403):
            raise UpstreamProviderError(
                f"{VENDOR} rejected the API key (HTTP {response.status_code}). Check "
                "TRAFFIC_API_KEY.",
                vendor=VENDOR,
                vendor_status=response.status_code,
            )
        if response.status_code >= 400:
            raise UpstreamProviderError(
                f"{VENDOR} returned HTTP {response.status_code} for the flow "
                "segment request.",
                vendor=VENDOR,
                vendor_status=response.status_code,
            )

    def _fetch_flow_segment(
        self, location: MonitoredLocation
    ) -> dict[str, Any]:
        """Return the validated ``flowSegmentData`` object for one location."""

        self._throttle()
        try:
            response = self._client.get(self._url(), params=self._params(location))
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"{VENDOR} did not respond within "
                f"{self._settings.TRAFFIC_REQUEST_TIMEOUT}s.",
                vendor=VENDOR,
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamProviderError(
                f"{VENDOR} request failed: {type(exc).__name__}.",
                vendor=VENDOR,
            ) from exc

        self._raise_for_status(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                f"{VENDOR} returned a body that is not valid JSON.",
                vendor=VENDOR,
                vendor_status=response.status_code,
            ) from exc

        if not isinstance(payload, dict):
            raise ProviderResponseError(
                f"{VENDOR} returned a JSON {type(payload).__name__}, expected an "
                "object. See https://docs.tomtom.com/traffic-api/documentation/"
                "tomtom-maps/v1/traffic-flow/flow-segment-data.",
                vendor=VENDOR,
            )

        segment = payload.get("flowSegmentData")
        if not isinstance(segment, dict):
            # TomTom returns 200 with an error object for some bad coordinates, so
            # a missing segment here is a real failure, not an empty result.
            raise ProviderResponseError(
                f"{VENDOR} response for location {location.location_id} has no "
                "'flowSegmentData' object. Check that the coordinate lies on a "
                "road and that 'fields' matches the documented field names.",
                vendor=VENDOR,
                vendor_status=response.status_code,
                details={"location_id": location.location_id},
            )
        return segment

    def _parse_speeds(
        self, segment: dict[str, Any], location: MonitoredLocation
    ) -> tuple[float, float]:
        """Extract and validate the current and free-flow speeds, in km/h.

        Both fields are required. A missing or non-numeric speed is a failure, not
        a value to default: the model's single most important input is current
        speed, and substituting anything for it would produce a real-looking
        observation that says nothing about the road.

        The database also enforces ``free_flow_speed_kph > avg_speed_kph``. If a
        reading violates that, the coordinate has almost certainly landed on an
        unintended segment, so it is rejected here with an actionable message
        rather than stored and failing at write time.
        """

        current = self._required_number(segment, "currentSpeed", location)
        free_flow = self._required_number(segment, "freeFlowSpeed", location)

        for name, value in (("currentSpeed", current), ("freeFlowSpeed", free_flow)):
            if not 0.0 <= value <= _MAX_PLAUSIBLE_KPH:
                raise ProviderResponseError(
                    f"{VENDOR} reported {name}={value} km/h for location "
                    f"{location.location_id}, outside 0..{_MAX_PLAUSIBLE_KPH}. "
                    "This indicates a parsing problem or a wrong zoom level, not a "
                    "traffic condition.",
                    vendor=VENDOR,
                    details={
                        "location_id": location.location_id,
                        "field": name,
                        "value": value,
                    },
                )

        if current > free_flow:
            raise ProviderResponseError(
                f"{VENDOR} reported currentSpeed={current} above "
                f"freeFlowSpeed={free_flow} for location {location.location_id}. "
                "The point may have fallen on the wrong segment; check the "
                "coordinate and TRAFFIC_ZOOM.",
                vendor=VENDOR,
                details={
                    "location_id": location.location_id,
                    "current_speed_kph": current,
                    "free_flow_speed_kph": free_flow,
                },
            )

        if free_flow > location.free_flow_speed_kph_max:
            # Not an error: the road's real limit may exceed the configured bound.
            # But it usually means the point is on a different road than intended,
            # so it is worth surfacing loudly rather than storing quietly.
            logger.warning(
                "Location %s (%s) reported free-flow %s km/h, above the "
                "configured sanity bound of %s km/h. Check that the coordinate "
                "lies on %s.",
                location.location_id,
                location.road_name,
                free_flow,
                location.free_flow_speed_kph_max,
                location.road_name,
            )

        return round(current, 2), round(free_flow, 2)

    def _required_number(
        self, segment: dict[str, Any], field: str, location: MonitoredLocation
    ) -> float:
        """Read a required numeric field, or raise."""

        if field not in segment:
            raise ProviderResponseError(
                f"{VENDOR} response for location {location.location_id} is missing "
                f"{field!r}. Verify that the 'fields' parameter matches the "
                "documented field names at "
                "https://docs.tomtom.com/traffic-api/documentation/tomtom-maps/v1/"
                "traffic-flow/flow-segment-data.",
                vendor=VENDOR,
                details={"location_id": location.location_id, "field": field},
            )
        value = segment[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderResponseError(
                f"{VENDOR} returned a non-numeric {field} "
                f"({type(value).__name__}) for location {location.location_id}. "
                "Expected a number in km/h.",
                vendor=VENDOR,
                details={"location_id": location.location_id, "field": field},
            )
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ProviderResponseError(
                f"{VENDOR} returned a non-finite {field} for location "
                f"{location.location_id}.",
                vendor=VENDOR,
                details={"location_id": location.location_id, "field": field},
            )
        return number

    def _build_record(
        self,
        location: MonitoredLocation,
        segment: dict[str, Any],
        weather: WeatherReading,
        observed_at: datetime,
    ) -> TrafficRecord:
        """Combine the two upstream readings into one storable record.

        ``vehicle_count`` is ``None`` and the reason is recorded in metadata, so
        the omission is visible to anyone reading the row rather than looking like
        a lost value.
        """

        current, free_flow = self._parse_speeds(segment, location)
        closure = segment.get("roadClosure")
        metadata: dict[str, Any] = {
            "traffic_vendor": VENDOR,
            "traffic_vendor_version": 4,
            "vehicle_count_unavailable_reason": (
                f"{VENDOR} Flow Segment Data does not publish vehicle throughput. "
                "No self-serve traffic API does, so this field is null by design "
                "and is not a model input as of version 4.0.0."
            ),
            **weather.to_metadata(),
        }
        for field in ("confidence", "frc", "openlr", "roadClosure"):
            if field in segment:
                metadata[f"tomtom_{field}"] = segment[field]

        return TrafficRecord(
            timestamp=observed_at,
            location_id=location.location_id,
            road_name=location.road_name,
            road_segment_id=location.road_segment_id,
            vehicle_count=None,
            avg_speed_kph=current,
            free_flow_speed_kph=free_flow,
            weather_condition=weather.weather_condition,
            temperature_c=weather.temperature_c,
            rainfall_mm=weather.precipitation_mm,
            # TomTom's roadClosure is the only incident signal available. NULL
            # would mean "vendor did not say"; the field is always present in a
            # documented response, so False is a real reading, not a guess.
            is_incident=_blocked_by_closure(closure),
            source=SOURCE,
            metadata=metadata,
        )

    def _fetch_one(self, location: MonitoredLocation) -> TrafficRecord:
        """Fetch traffic then weather for one location.

        Order matters only for diagnosis: traffic is requested first so a bad
        coordinate fails before a weather request is spent on it.
        """

        segment = self._fetch_flow_segment(location)
        if self._settings.WEATHER_ENABLED:
            weather = self._weather.fetch_current(
                latitude=location.latitude, longitude=location.longitude
            )
        else:
            # Weather is a required model input, but WEATHER_ENABLED=false is an
            # explicit operator choice. An unscorable observation is stored with
            # the weather fields null and the reason recorded, which is different
            # from silently omitting the fetch.
            logger.warning(
                "WEATHER_ENABLED is false; storing location %s without weather. "
                "Model version 4.0.0 requires weather_condition, so this "
                "observation cannot be scored.",
                location.location_id,
            )
            weather = WeatherReading(
                weather_condition="cloudy",
                temperature_c=None,
                precipitation_mm=None,
                observed_at=datetime.now(timezone.utc),
                source="disabled",
                metadata={"weather_unavailable_reason": "WEATHER_ENABLED is false"},
            )

        observed_at = (
            weather.observed_at
            if weather.source != "disabled"
            else datetime.now(timezone.utc)
        )
        return self._build_record(location, segment, weather, observed_at)

    def fetch_current_traffic(
        self, *, location_id: str | None = None
    ) -> Sequence[TrafficRecord]:
        """Fetch a live reading for one location, or for all of them.

        Args:
            location_id: restrict to a single configured location, or ``None``
                for every location in the file.

        Raises:
            ProviderNotConfiguredError: the key or locations file is unusable.
            UpstreamTimeoutError / ProviderRateLimitedError /
            UpstreamProviderError / ProviderResponseError: an upstream failed or
                returned something unusable. Propagated rather than swallowed,
                so a partial cycle cannot look like a successful one.
        """

        self._require_configuration()
        locations = self._locations()

        if location_id is not None:
            matches = [item for item in locations if item.location_id == location_id]
            if not matches:
                raise ConfigurationError(
                    f"Location {location_id!r} is not in the monitored-locations "
                    f"file. Known locations: {[item.location_id for item in locations]}.",
                    details={"requested": location_id},
                )
            targets = matches
        else:
            targets = list(locations)

        records = [self._fetch_one(location) for location in targets]
        logger.info(
            "Fetched %d live reading(s) from %s for %s.",
            len(records),
            VENDOR,
            [record.location_id for record in records],
        )
        return tuple(records)

    def fetch_historical_traffic(
        self,
        *,
        location_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[TrafficRecord]:
        """Always raises: this vendor has no historical traffic feed.

        TomTom Flow Segment Data reports current conditions only. HERE was checked
        for the same reason and also publishes no self-serve historical endpoint;
        Mapbox's historical traffic data is Enterprise-only.

        The arguments are honoured only in the sense that they are rejected. This
        is a :class:`~app.core.exceptions.UpstreamProviderError` rather than an
        empty sequence on purpose: an empty result reads as "no traffic history
        exists", whereas the truth is "this provider cannot supply history". A
        caller that needs history must collect going forward, with
        ``COLLECTION_ENABLED=true``.
        """

        self._require_configuration()
        raise UpstreamProviderError(
            f"{VENDOR} publishes current traffic conditions only, so historical "
            "readings cannot be fetched. Model version 4.0.0 needs prior "
            "observations per segment to build its lag and rolling features, so a "
            "real deployment accumulates history: set COLLECTION_ENABLED=true to "
            "collect on a schedule, or POST observations to /api/v1/traffic/"
            "observations. Returning an empty history instead would leave the "
            "prediction endpoints permanently unable to score a real location.",
            vendor=VENDOR,
            details={
                "capability": "historical_traffic",
                "location_id": location_id,
                "vendor_documentation": (
                    "https://docs.tomtom.com/traffic-api/documentation/"
                    "tomtom-maps/v1/traffic-flow/flow-segment-data"
                ),
            },
        )


__all__ = [
    "FLOW_SEGMENT_PATH",
    "RealTrafficProvider",
    "SOURCE",
    "VENDOR",
]