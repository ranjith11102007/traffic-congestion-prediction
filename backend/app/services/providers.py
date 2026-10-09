"""Abstraction over where traffic observations come from.

The collector depends on :class:`TrafficDataProvider`, never on a concrete vendor.
Adding a real provider in Stage 5 means implementing this interface and selecting
it with ``TRAFFIC_PROVIDER=real``; no route, service or repository changes.

The contract is deliberately narrow: return fully-formed
:class:`TrafficRecord` values, or raise. A provider is not allowed to invent a
record it could not fetch, which is why the base class has no default
implementation of either method.

Provenance is part of the contract rather than an afterthought. Every record
carries the ``source`` it came from, and the simulated provider sets it to
``"simulation"`` without exception, so a stored row can never misrepresent
synthetic data as real traffic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from app.models.traffic import API_SOURCE, SIMULATION_SOURCE


@dataclass(frozen=True, slots=True)
class TrafficRecord:
    """One observation, normalised to the field names the ML model expects.

    Field names match the Stage 3 feature frame (``avg_speed_kph``,
    ``free_flow_speed_kph``, ``weather_condition``) so a record maps onto the
    model's input frame without a translation layer that could go out of step.

    Attributes:
        timestamp: when the state was observed, timezone-aware.
        location_id: stable sensor or junction identifier.
        road_name: human-readable road name.
        road_segment_id: ML road segment, e.g. ``SEG-01``.
        vehicle_count: vehicles in the interval, or ``None`` when the provider
            does not report throughput. Since model version 4.0.0 this is not a
            model input, because no self-serve traffic API publishes it.
        avg_speed_kph: mean observed speed.
        free_flow_speed_kph: uncongested reference speed.
        weather_condition: weather label known to the model.
        temperature_c: air temperature, if reported.
        rainfall_mm: rainfall, if reported.
        is_incident: incident flag, if reported.
        latitude: latitude, if reported.
        longitude: longitude, if reported.
        source: provenance label. ``"simulation"`` means synthetic data.
        metadata: provider-specific extras, kept out of the database columns.
    """

    timestamp: datetime
    location_id: str
    road_name: str
    road_segment_id: str | None
    vehicle_count: int | None
    avg_speed_kph: float
    free_flow_speed_kph: float
    weather_condition: str
    temperature_c: float | None = None
    rainfall_mm: float | None = None
    is_incident: bool | None = None
    latitude: float | None = None
    longitude: float | None = None
    source: str = API_SOURCE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model_frame_row(self) -> dict[str, Any]:
        """Render this record as a raw ML input row.

        Only the columns the Stage 3 schema actually requires are emitted.
        ``speed_ratio`` and ``congestion_level`` are absent on purpose: they are
        derived by the pipeline from ``avg_speed_kph`` and ``free_flow_speed_kph``,
        and accepting them from a caller would reintroduce the target leakage that
        Stage 3 removed.
        """

        return {
            "timestamp": self.timestamp,
            "road_segment_id": self.road_segment_id,
            "avg_speed_kph": self.avg_speed_kph,
            "free_flow_speed_kph": self.free_flow_speed_kph,
            "weather_condition": self.weather_condition,
            "temperature_c": self.temperature_c,
            "precipitation_mm": self.rainfall_mm,
            "is_incident": int(self.is_incident) if self.is_incident is not None else 0,
        }


class TrafficDataProvider(abc.ABC):
    """Base class for every traffic data source.

    Subclasses must implement both fetch methods. A subclass that cannot serve
    one of them must raise rather than return synthetic data, so an unimplemented
    capability is visible instead of being papered over.
    """

    #: Short provider name recorded on every record it produces.
    name: str = "base"

    #: True only for providers that generate synthetic data.
    is_simulation: bool = False

    @property
    def is_configured(self) -> bool:
        """Whether this provider has everything it needs to fetch.

        Used by the status endpoint so a client can discover an unconfigured
        provider without triggering a failed collection.
        """

        return True

    def describe(self) -> str:
        """One-line human-readable description for status responses."""

        return f"{self.name} provider"

    @abc.abstractmethod
    def fetch_current_traffic(
        self, *, location_id: str | None = None
    ) -> Sequence[TrafficRecord]:
        """Fetch the most recent reading(s).

        Args:
            location_id: restrict to one location, or ``None`` for all known
                locations.

        Returns:
            One or more records. May be empty if the provider has no current data;
            must never fabricate a record to avoid returning an empty sequence.

        Raises:
            ProviderNotConfiguredError: if the provider is not usable.
        """

    @abc.abstractmethod
    def fetch_historical_traffic(
        self,
        *,
        location_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[TrafficRecord]:
        """Fetch a historical range.

        Args:
            location_id: restrict to one location.
            start_time: inclusive lower bound.
            end_time: inclusive upper bound.
            limit: maximum rows to return.

        Returns:
            Records ordered oldest first.

        Raises:
            ProviderNotConfiguredError: if the provider is not usable.
        """

    def list_locations(self) -> Sequence[str]:
        """Return the location identifiers this provider can report on."""

        return ()


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)


__all__ = [
    "SIMULATION_SOURCE",
    "TrafficDataProvider",
    "TrafficRecord",
    "utc_now",
]