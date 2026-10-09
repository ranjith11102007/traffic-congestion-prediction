"""The roads this deployment monitors, loaded from configuration.

Why this is configuration and not code
--------------------------------------
A traffic vendor is asked "what is happening at 13.0569, 80.2425?". It has no
concept of ``SEG-01``. That mapping is a property of *this* deployment, not of the
vendor or the model, so it belongs in a file a user can edit without a code change
or a retrain.

Two constraints from the model make the file stricter than it looks:

* ``road_segment_id`` must be one of the categories the model was trained on,
  ``SEG-01``..``SEG-06``. An unknown segment is rejected on load, because the
  model would otherwise score a location under a label it never learned and
  produce a confident, meaningless answer.
* ``free_flow_speed_kph_max`` is a sanity bound per location. TomTom returns the
  free-flow speed of whatever segment contains the coordinate, and a coordinate
  that drifts onto the wrong road returns a plausible but wrong number. Bounding
  it means a mis-placed point produces a rejected reading rather than a stored
  wild value.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

#: The road-segment categories the Stage 3 fixture and model version 4.0.0 know.
#: Loaded from the model metadata rather than hard-coded, so retraining on a
#: different segment set cannot leave this list quietly stale.
KNOWN_ROAD_SEGMENTS: frozenset[str] = frozenset(
    f"SEG-{index:02d}" for index in range(1, 7)
)

#: Tolerance when checking a location's road segment against the known set.
_MAX_FREE_FLOW_KPH = 200.0


@dataclass(frozen=True, slots=True)
class MonitoredLocation:
    """One road this deployment queries, with its ML feature mapping."""

    location_id: str
    road_segment_id: str
    road_name: str
    latitude: float
    longitude: float
    free_flow_speed_kph_max: float
    city: str | None = None

    @property
    def coordinates(self) -> str:
        """Return ``"lat,lon"`` in the order TomTom's ``point`` parameter wants."""

        return f"{self.latitude:.5f},{self.longitude:.5f}"


def _require(entry: dict, field: str, location_id: str, path: Path) -> object:
    """Fetch a required key, or explain exactly what is missing."""

    if field not in entry:
        raise ConfigurationError(
            f"Monitored location {location_id} in {path} is missing {field!r}.",
            details={"location_id": location_id, "field": field},
        )
    return entry[field]


def _as_float(value: object, field: str, location_id: str, path: Path) -> float:
    """Coerce a JSON number, rejecting booleans and non-numeric strings."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"Monitored location {location_id} in {path} has a non-numeric "
            f"{field}: {value!r}.",
            details={"location_id": location_id, "field": field},
        )
    number = float(value)
    if not -180.0 <= number <= 180.0 and field == "longitude":
        raise ConfigurationError(
            f"Monitored location {location_id} in {path} has longitude "
            f"{number}, outside -180..180.",
            details={"location_id": location_id},
        )
    if not -90.0 <= number <= 90.0 and field == "latitude":
        raise ConfigurationError(
            f"Monitored location {location_id} in {path} has latitude {number}, "
            "outside -90..90.",
            details={"location_id": location_id},
        )
    return number


def _parse_location(entry: object, index: int, path: Path) -> MonitoredLocation:
    """Validate one location entry and return it.

    Validation happens at load time rather than at fetch time so a typo surfaces
    as a configuration error the operator sees immediately, not as an upstream
    error twenty minutes into a collection cycle.
    """

    if not isinstance(entry, dict):
        raise ConfigurationError(
            f"Entry {index} of the locations list in {path} is a "
            f"{type(entry).__name__}, expected an object.",
            details={"index": index},
        )

    location_id = str(_require(entry, "location_id", f"index {index}", path))
    segment = str(_require(entry, "road_segment_id", location_id, path))
    if segment not in KNOWN_ROAD_SEGMENTS:
        raise ConfigurationError(
            f"Monitored location {location_id} in {path} uses road segment "
            f"{segment!r}, which the model does not know. Valid segments: "
            f"{sorted(KNOWN_ROAD_SEGMENTS)}. A location outside this set cannot "
            "be scored, so it is rejected rather than stored unscorable.",
            details={
                "location_id": location_id,
                "road_segment_id": segment,
                "known_segments": sorted(KNOWN_ROAD_SEGMENTS),
            },
        )

    road_name = str(_require(entry, "road_name", location_id, path))
    latitude = _as_float(
        _require(entry, "latitude", location_id, path), "latitude", location_id, path
    )
    longitude = _as_float(
        _require(entry, "longitude", location_id, path), "longitude", location_id, path
    )
    free_flow_max = _as_float(
        entry.get("free_flow_speed_kph_max", _MAX_FREE_FLOW_KPH),
        "free_flow_speed_kph_max",
        location_id,
        path,
    )
    if not 0.0 < free_flow_max <= _MAX_FREE_FLOW_KPH:
        raise ConfigurationError(
            f"Monitored location {location_id} in {path} has "
            f"free_flow_speed_kph_max={free_flow_max}; it must be above 0 and at "
            f"most {_MAX_FREE_FLOW_KPH}.",
            details={"location_id": location_id},
        )

    city = entry.get("city")
    return MonitoredLocation(
        location_id=location_id,
        road_segment_id=segment,
        road_name=road_name,
        latitude=latitude,
        longitude=longitude,
        free_flow_speed_kph_max=free_flow_max,
        city=str(city) if isinstance(city, str) else None,
    )


def load_monitored_locations(
    path: str | Path, *, max_locations: int | None = None
) -> tuple[MonitoredLocation, ...]:
    """Read and validate the monitored-locations file.

    Args:
        path: location of the JSON file.
        max_locations: reject a file with more entries than this. Present so a
            configuration mistake cannot spend a month's free vendor quota in a
            single collection cycle.

    Raises:
        ConfigurationError: if the file is missing, is not JSON, has no usable
            locations, repeats a ``location_id``, or contains an invalid entry.
    """

    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigurationError(
            f"Monitored locations file {resolved} does not exist. Create it, or "
            "point TRAFFIC_LOCATIONS_FILE at an existing file. See "
            "config/monitored_locations.json for the expected format.",
            details={"path": str(resolved)},
        )

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Monitored locations file {resolved} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno}).",
            details={"path": str(resolved)},
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Monitored locations file {resolved} could not be read: {exc.strerror}.",
            details={"path": str(resolved)},
        ) from exc

    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Monitored locations file {resolved} must contain a JSON object with "
            f"a 'locations' list, found {type(payload).__name__}.",
            details={"path": str(resolved)},
        )

    entries = payload.get("locations")
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError(
            f"Monitored locations file {resolved} has no 'locations' list, or it "
            "is empty. A real provider has no built-in locations, so at least one "
            "must be configured.",
            details={"path": str(resolved)},
        )

    if max_locations is not None and len(entries) > max_locations:
        raise ConfigurationError(
            f"Monitored locations file {resolved} lists {len(entries)} locations "
            f"but TRAFFIC_MAX_LOCATIONS is {max_locations}. Reduce the list, or "
            "raise the limit if the vendor quota supports it.",
            details={
                "path": str(resolved),
                "locations": len(entries),
                "max_locations": max_locations,
            },
        )

    locations: list[MonitoredLocation] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        location = _parse_location(entry, index, resolved)
        if location.location_id in seen:
            raise ConfigurationError(
                f"Monitored locations file {resolved} repeats location_id "
                f"{location.location_id!r}. Identifiers must be unique, because "
                "they key stored observations.",
                details={
                    "path": str(resolved),
                    "location_id": location.location_id,
                },
            )
        seen.add(location.location_id)
        locations.append(location)

    logger.info(
        "Loaded %d monitored location(s) from %s: %s",
        len(locations),
        resolved,
        [location.location_id for location in locations],
    )
    return tuple(locations)


@lru_cache(maxsize=1)
def _cached_locations(resolved: str, max_locations: int | None) -> tuple[MonitoredLocation, ...]:
    return load_monitored_locations(Path(resolved), max_locations=max_locations)


def get_monitored_locations(settings: Settings | None = None) -> tuple[MonitoredLocation, ...]:
    """Return the configured locations, loading the file at most once.

    Cached because a collection cycle asks for the list once per location and the
    file cannot change without a restart anyway. ``cache_clear`` on this function
    is how a test forces a reload after editing the file.
    """

    active = settings or get_settings()
    path = active.traffic_locations_path
    return _cached_locations(str(path), active.TRAFFIC_MAX_LOCATIONS)


__all__ = [
    "KNOWN_ROAD_SEGMENTS",
    "MonitoredLocation",
    "get_monitored_locations",
    "load_monitored_locations",
]