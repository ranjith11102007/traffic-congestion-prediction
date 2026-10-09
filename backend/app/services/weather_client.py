"""Weather readings from Open-Meteo.

Separated from :mod:`app.services.real_provider` for two reasons.

First, weather is not traffic data. TomTom's Flow Segment Data carries no weather
fields at all, so the observation's ``weather_condition``/``temperature_c``/
``rainfall_mm`` columns have to come from somewhere else, and keeping that in its
own module means a future HERE or Mapbox provider does not have to touch the
weather parsing.

Second, the model requires weather. A traffic observation stored without it is
stored unscorable, so the two upstream calls are part of one logical reading and
must fail together rather than half-succeed.

Open-Meteo
----------
Free for non-commercial use with no API key, and documented at 10,000 calls/day
and 300,000 calls/month (https://open-meteo.com/en/pricing). That is the reason
for choosing it: it is the only weather source that needs no signup.

The request deliberately asks only for three variables. Open-Meteo offers many
more, but every extra field is another thing that can be missing, and the model
consumes exactly three.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from app.core.config import Settings
from app.core.exceptions import (
    ProviderRateLimitedError,
    ProviderResponseError,
    UpstreamProviderError,
    UpstreamTimeoutError,
)

logger = logging.getLogger(__name__)

#: Reported in error details so a caller can tell which upstream failed.
VENDOR = "open-meteo"

#: Path appended to ``WEATHER_API_BASE_URL``.
FORECAST_PATH = "/forecast"

#: The three model inputs weather can supply. Nothing else is requested.
CURRENT_FIELDS = "temperature_2m,precipitation,weather_code"

#: WMO weather interpretation codes, from the Open-Meteo documentation table
#: (https://open-meteo.com/en/docs). Only the model's three categories are needed,
#: so codes are grouped rather than mapped one to one.
#:
#: The grouping is intentionally coarse. The model was trained on a synthetic
#: fixture using exactly ``clear``/``cloudy``/``rain``, so inventing finer labels
#: here would produce a category the model has never seen and would silently score
#: it as unknown.
_WMO_CLEAR = frozenset({0, 1})
_WMO_CLOUDY = frozenset({2, 3, 45, 48})

#: WMO codes 51-67 (drizzle and rain), 71-77 (snowfall), 80-82 (rain showers),
#: 85-86 (snow showers) and 95-99 (thunderstorm). Any precipitation-bearing or
#: storm code collapses to ``rain``, which is the label that means "wet road
#: conditions" in the training fixture.
WMO_CODE_TO_CONDITION: dict[int, str] = {
    **{code: "clear" for code in _WMO_CLEAR},
    **{code: "cloudy" for code in _WMO_CLOUDY},
    **{code: "rain" for code in range(51, 68)},    # drizzle and rain
    **{code: "rain" for code in range(71, 78)},    # snowfall
    **{code: "rain" for code in range(80, 83)},    # rain showers
    **{code: "rain" for code in range(85, 87)},    # snow showers
    **{code: "rain" for code in range(95, 100)},   # thunderstorm
}

#: Used only for a code the vendor documents as unassigned. Not a guess about
#: conditions: the model has no "unknown" label, so cloudy is the least
#: load-bearing of the three and is recorded in metadata either way.
FALLBACK_CONDITION = "cloudy"


@dataclass(frozen=True, slots=True)
class WeatherReading:
    """One point-in-time weather observation.

    ``temperature_c`` and ``precipitation_mm`` stay ``None`` when the vendor
    omits them, rather than defaulting to 0. A missing temperature and a genuine
    zero degrees are different facts, and the model should not be told they are
    the same.
    """

    weather_condition: str
    temperature_c: float | None
    precipitation_mm: float | None
    observed_at: datetime
    source: str = VENDOR
    metadata: dict[str, object] | None = None

    def to_metadata(self) -> dict[str, object]:
        """Return the provenance recorded on a stored observation."""

        return {
            "weather_source": self.source,
            "weather_observed_at": self.observed_at.isoformat(),
            **({"weather_code": self.metadata["weather_code"]} if self.metadata else {}),
        }


def classify_weather_code(code: object) -> str:
    """Map a WMO code to one of the model's three condition labels.

    Args:
        code: the ``weather_code`` value from the Open-Meteo response.

    Returns:
        ``"clear"``, ``"cloudy"`` or ``"rain"``.

    Raises:
        ProviderResponseError: if ``code`` is not an integer. A string or null
            here means the response shape changed, and silently returning
            ``cloudy`` would let a shape change look like normal weather.
    """

    if isinstance(code, bool) or not isinstance(code, int):
        raise ProviderResponseError(
            f"{VENDOR} returned a non-integer weather_code "
            f"({type(code).__name__}). Expected a WMO code as documented at "
            "https://open-meteo.com/en/docs.",
            vendor=VENDOR,
        )
    condition = WMO_CODE_TO_CONDITION.get(code, FALLBACK_CONDITION)
    if code not in WMO_CODE_TO_CONDITION:
        logger.warning(
            "Unrecognised WMO weather code %s from %s; recorded as %r.",
            code,
            VENDOR,
            condition,
        )
    return condition


class OpenMeteoClient:
    """Fetches current weather for a coordinate pair.

    One instance is reused across a collection cycle so the underlying
    connection is pooled rather than re-established per location.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        # A supplied client is not owned: the caller closes it. This is what lets
        # tests inject a MockTransport without the client trying to close it.
        self._client = client or httpx.Client(
            timeout=settings.WEATHER_REQUEST_TIMEOUT,
            headers={"User-Agent": "traffic-prediction-api/4.0.0"},
        )
        self._owns_client = client is None

    def close(self) -> None:
        """Close the connection pool if this instance created it."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenMeteoClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _url(self) -> str:
        return f"{self._settings.WEATHER_API_BASE_URL}{FORECAST_PATH}"

    def _params(self, latitude: float, longitude: float) -> dict[str, str]:
        """Build the query string.

        ``timezone=GMT`` makes Open-Meteo return local time in UTC, so the
        ``current.time`` field is directly comparable with the UTC timestamps
        stored on observations. Without it the value is local to the coordinate,
        which for Chennai is +05:30 and would shift every reading by half a day
        once written down.
        """

        return {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "current": CURRENT_FIELDS,
            "timezone": "GMT",
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate a vendor failure into a domain error.

        Open-Meteo's error body echoes the requested URL. The traffic request URL
        carries the API key, so nothing from the response body is passed on; only
        the status and the vendor's error code are kept, for the server log.
        """

        if response.status_code == 429:
            raise ProviderRateLimitedError(
                f"{VENDOR} rate limit reached. Reduce COLLECTION_INTERVAL_SECONDS "
                "or the number of monitored locations.",
                vendor=VENDOR,
                vendor_status=response.status_code,
            )
        if response.status_code >= 400:
            raise UpstreamProviderError(
                f"{VENDOR} returned HTTP {response.status_code} for the current "
                "weather request.",
                vendor=VENDOR,
                vendor_status=response.status_code,
            )

    def fetch_current(
        self, *, latitude: float, longitude: float
    ) -> WeatherReading:
        """Return the current weather at a coordinate.

        Raises:
            UpstreamTimeoutError: the request exceeded ``WEATHER_REQUEST_TIMEOUT``.
            ProviderRateLimitedError: the vendor reported throttling.
            UpstreamProviderError: any other non-2xx response.
            ProviderResponseError: the 2xx body was missing a required field.
        """

        url = self._url()
        params = self._params(latitude, longitude)

        try:
            response = self._client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                f"{VENDOR} did not respond within "
                f"{self._settings.WEATHER_REQUEST_TIMEOUT}s.",
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

        return self._parse(payload)

    def _parse(self, payload: object) -> WeatherReading:
        """Map a documented Open-Meteo response onto a reading.

        Every field is validated rather than defaulted. A 200 response without
        ``current`` means the request was wrong or the vendor changed shape, and
        inventing a temperature would store a fabricated observation.
        """

        if not isinstance(payload, dict):
            raise ProviderResponseError(
                f"{VENDOR} returned a JSON {type(payload).__name__}, expected an object.",
                vendor=VENDOR,
            )

        current = payload.get("current")
        if not isinstance(current, dict):
            raise ProviderResponseError(
                f"{VENDOR} response has no 'current' object. The request used "
                "current=temperature_2m,precipitation,weather_code as documented "
                "at https://open-meteo.com/en/docs.",
                vendor=VENDOR,
            )

        if "weather_code" not in current:
            raise ProviderResponseError(
                f"{VENDOR} response is missing 'current.weather_code'.",
                vendor=VENDOR,
            )
        condition = classify_weather_code(current["weather_code"])

        # temperature_2m and precipitation are genuinely optional here: the
        # columns are nullable and a missing value is more honest than a zero.
        temperature = self._optional_float(current.get("temperature_2m"), "temperature_2m")
        precipitation = self._optional_float(current.get("precipitation"), "precipitation")

        observed_at = self._parse_time(current.get("time"))
        metadata: dict[str, object] = {
            "weather_code": current["weather_code"],
            "weather_code_fallback_used": (
                current["weather_code"] not in WMO_CODE_TO_CONDITION
            ),
        }
        return WeatherReading(
            weather_condition=condition,
            temperature_c=temperature,
            precipitation_mm=precipitation,
            observed_at=observed_at,
            source=VENDOR,
            metadata=metadata,
        )

    def _optional_float(self, value: object, field: str) -> float | None:
        """Return a finite float, or ``None`` when the vendor omitted the field."""

        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderResponseError(
                f"{VENDOR} returned a non-numeric {field} "
                f"({type(value).__name__}).",
                vendor=VENDOR,
            )
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ProviderResponseError(
                f"{VENDOR} returned a non-finite {field}.", vendor=VENDOR
            )
        return number

    def _parse_time(self, value: object) -> datetime:
        """Read ``current.time``, which Open-Meteo returns as ``YYYY-MM-DDTHH:MM``.

        ``timezone=GMT`` was requested, so the value is UTC and is made
        timezone-aware explicitly rather than assumed. A naive datetime compared
        with an aware one raises in Python; assuming UTC keeps the stored value
        consistent with every other timestamp in the system.
        """

        if not isinstance(value, str):
            raise ProviderResponseError(
                f"{VENDOR} response is missing 'current.time'.", vendor=VENDOR
            )
        try:
            moment = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ProviderResponseError(
                f"{VENDOR} returned an unparseable current.time {value!r}.",
                vendor=VENDOR,
            ) from exc
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)


__all__ = [
    "CURRENT_FIELDS",
    "FALLBACK_CONDITION",
    "FORECAST_PATH",
    "OpenMeteoClient",
    "VENDOR",
    "WMO_CODE_TO_CONDITION",
    "WeatherReading",
    "classify_weather_code",
]