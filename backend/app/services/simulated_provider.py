"""Simulated traffic provider for local development and testing.

**This provider does not observe real traffic.** Every record it produces is
generated from a fixed-seed parametric model and is stamped
``source="simulation"`` without exception, so nothing it returns can be mistaken
for a live reading once stored.

It exists so the ingestion path, the database layer and the prediction service can
be exercised end to end on a developer machine with no external API and no
credentials. It is selected with ``TRAFFIC_PROVIDER=simulation``, which is the
default.

Determinism is deliberate: the same ``timestamp`` always yields the same reading,
so tests can assert on values and a developer sees a repeatable history rather
than noise that changes on every request.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Sequence

from app.core.exceptions import ProviderNotConfiguredError
from app.models.traffic import SIMULATION_SOURCE
from app.services.providers import TrafficDataProvider, TrafficRecord

#: Fixed seed. Every generated value derives from this, so runs are repeatable.
SIMULATION_SEED = 20260101

#: Simulated locations, mirroring the road segments the Stage 3 model knows.
#: Keeping them aligned is what lets a simulated history be scored for real.
SIMULATED_LOCATIONS: dict[str, tuple[str, str, float]] = {
    "LOC-001": ("SEG-01", "Anna Salai", 60.0),
    "LOC-002": ("SEG-02", "Raj Bhavan Road", 60.0),
    "LOC-003": ("SEG-03", "Koramangala Main Road", 50.0),
    "LOC-004": ("SEG-04", "MG Road", 55.0),
    "LOC-005": ("SEG-05", "NH-44 Bypass", 80.0),
    "LOC-006": ("SEG-06", "Airport Expressway", 80.0),
}

#: Weather cycle over the day, so consecutive readings vary.
_WEATHER_CYCLE = ("clear", "clear", "cloudy", "cloudy", "clear", "rain")

#: Congestion peaks at these hours, matching the Stage 3 fixture's shape.
_PEAK_HOURS = (8.0, 18.0)
_PEAK_WIDTH_HOURS = 2.0


class SimulatedTrafficProvider(TrafficDataProvider):
    """Generates clearly-labelled synthetic traffic readings.

    Not a substitute for a real feed. It exists so the pipeline can be run and
    tested without one.
    """

    name = "simulation"
    is_simulation = True

    def __init__(self, *, seed: int = SIMULATION_SEED) -> None:
        self._seed = seed

    def describe(self) -> str:
        return (
            "Synthetic development data generated locally. Not real traffic "
            "observations and not connected to any external feed."
        )

    def list_locations(self) -> Sequence[str]:
        return tuple(SIMULATED_LOCATIONS)

    def _resolve(self, location_id: str) -> tuple[str, str, float]:
        """Look up a simulated location's segment, road name and free-flow speed.

        Raises:
            ProviderNotConfiguredError: if the location is not modelled. Checked
                here, before any dict access, so an unknown location produces an
                actionable 503 instead of a ``KeyError`` surfacing as a 500.
        """

        try:
            return SIMULATED_LOCATIONS[location_id]
        except KeyError:
            raise ProviderNotConfiguredError(
                f"The simulation provider does not know location {location_id!r}. "
                f"Known locations: {sorted(SIMULATED_LOCATIONS)}.",
                details={"requested": location_id},
            ) from None

    def _generate(
        self,
        location_id: str,
        road_segment_id: str,
        road_name: str,
        free_flow_speed_kph: float,
        timestamp: datetime,
    ) -> TrafficRecord:
        """Produce one reading deterministically from ``(seed, location, time)``.

        A hash of those inputs gives a value in ``[0, 1)``. Using a hash rather
        than a random generator is what makes the same timestamp always produce
        the same reading.
        """

        self._resolve(location_id)

        aware = timestamp if timestamp.tzinfo is not None else timestamp.replace(
            tzinfo=timezone.utc
        )
        hour = aware.hour + aware.minute / 60.0

        # Two Gaussian peaks over a daytime base, flattened at weekends.
        daily = 0.18 + 0.10 * math.sin(2 * math.pi * (hour - 7) / 24.0)
        for peak in _PEAK_HOURS:
            daily += 0.38 * math.exp(-(((hour - peak) ** 2) / (2 * _PEAK_WIDTH_HOURS**2)))
        if aware.weekday() >= 5:
            daily *= 0.55

        jitter = self._unit_random(f"flow:{location_id}", aware) - 0.5
        # A fraction of free-flow speed, clamped so speed stays plausible and
        # strictly below the free-flow reference the label depends on.
        ratio = min(0.95, max(0.05, daily + 0.05 * jitter))

        avg_speed = round(free_flow_speed_kph * ratio, 2)
        weather = _WEATHER_CYCLE[int(aware.hour) % len(_WEATHER_CYCLE)]
        rainfall = 4.5 if weather == "rain" else 0.0
        incident = 1 if self._unit_random(f"incident:{location_id}", aware) > 0.97 else 0
        vehicle_count = int(
            300 + 1400 * ratio + 120 * (self._unit_random(f"count:{location_id}", aware) - 0.5)
        )

        return TrafficRecord(
            timestamp=aware,
            location_id=location_id,
            road_name=road_name,
            road_segment_id=road_segment_id,
            vehicle_count=max(0, vehicle_count),
            avg_speed_kph=avg_speed,
            free_flow_speed_kph=free_flow_speed_kph,
            weather_condition=weather,
            temperature_c=round(24.0 + 6.0 * math.sin(2 * math.pi * (hour - 15) / 24.0), 2),
            rainfall_mm=rainfall,
            is_incident=bool(incident),
            source=SIMULATION_SOURCE,
            metadata={"seed": self._seed, "simulated": True},
        )

    def _unit_random(self, salt: str, moment: datetime) -> float:
        """Return a deterministic value in ``[0, 1)`` for ``(salt, moment)``.

        ``hash()`` is deliberately avoided: Python randomises string hashing per
        process, which would make values differ between runs and across workers.
        """

        key = f"{self._seed}|{salt}|{moment.isoformat()}"
        digest = 0
        for char in key:
            digest = (digest * 131 + ord(char)) % 2**61
        return digest / float(2**61)

    def fetch_current_traffic(
        self, *, location_id: str | None = None
    ) -> Sequence[TrafficRecord]:
        """Return the current reading for one or all simulated locations.

        A missing ``location_id`` yields every known location rather than an
        arbitrary one, so the result is reproducible.
        """

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        targets = (
            [location_id] if location_id is not None else sorted(SIMULATED_LOCATIONS)
        )
        records = []
        for target in targets:
            segment, road_name, free_flow = self._resolve(target)
            records.append(
                self._generate(target, segment, road_name, free_flow, now)
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
        """Return generated readings on an hourly grid, oldest first.

        The grid runs backwards from ``end_time`` (default: now) so ``limit`` rows
        always end at the most recent hour rather than starting in the past.
        """

        if limit < 1:
            raise ValueError("limit must be at least 1")

        end = end_time or datetime.now(timezone.utc).replace(second=0, microsecond=0)
        if start_time is not None and start_time > end:
            raise ValueError("start_time must not be after end_time")

        targets = (
            [location_id] if location_id is not None else sorted(SIMULATED_LOCATIONS)
        )
        records: list[TrafficRecord] = []
        for target in targets:
            segment, road_name, free_flow = self._resolve(target)
            for offset in range(limit):
                moment = end - timedelta(hours=offset)
                if start_time is not None and moment < start_time:
                    break
                records.append(
                    self._generate(target, segment, road_name, free_flow, moment)
                )
        records.sort(key=lambda record: record.timestamp)
        return tuple(records)


__all__ = [
    "SIMULATED_LOCATIONS",
    "SIMULATION_SEED",
    "SimulatedTrafficProvider",
]