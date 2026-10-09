"""Generator for the **SIMULATED** development dataset.

.. warning::

   Everything this module produces is **synthetic**. It is generated from a
   parametric traffic model, not measured from any road. It exists for exactly one
   purpose: to exercise and test the preprocessing pipeline so that Stage 3 has a
   working end-to-end path to debug against. **No result derived from this file
   is a statement about real-world traffic behaviour, and no model trained on it
   means anything.**

The real dataset is expected in ``ml/data/raw/``. See ``ml/data/raw/README.md``.

Why generate it at all
----------------------
The pipeline needs a file to run against. Hand-writing a fixture CSV would drift
from the code that reads it, so the fixture is generated from documented
parameters with a fixed seed: the schema, the column types and the failure modes
are all explicit, and the output is byte-for-byte reproducible.

What the generator models
-------------------------
* A bimodal daily demand curve (morning and evening peaks) with a weekday/weekend
  difference.
* A speed that falls as demand rises, with a hard floor at a congested crawl.
* A volume-flow relationship, so vehicle count and speed are not independent.
* Diurnal temperature, occasional rain, and rain reducing speed.
* Rare incidents with a sustained local speed penalty.

Usage
-----
::

    python -m ml.data.sample.generate_simulated_dataset            # clean fixture
    python -m ml.data.sample.generate_simulated_dataset --defects   # inject defects
    python -m ml.data.sample.generate_simulated_dataset --output other.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

SAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SAMPLE_DIR / "SIMULATED_traffic_observations.csv"
PROVENANCE_SUFFIX = ".provenance.json"

DEFAULT_SEED = 20260101
DEFAULT_SEGMENTS = 6
DEFAULT_DAYS = 14
INTERVAL_MINUTES = 60

#: Plausible urban free-flow speeds, in km/h.
SEGMENT_FREE_FLOW_KPH = (60.0, 70.0, 50.0, 80.0, 60.0, 55.0)

#: Approximate capacity per lane, vehicles/hour. Used for the volume-flow curve.
SEGMENT_CAPACITY = (1900.0, 2200.0, 1650.0, 2600.0, 1800.0, 1700.0)

#: Relative peak demand per segment, so the segments are not identical.
SEGMENT_DEMAND_SCALE = (1.0, 0.92, 1.08, 0.85, 1.15, 0.78)

#: Morning and evening peak centres, in hours, with their widths.
PEAK_MORNING = (8.0, 2.0)
PEAK_EVENING = (17.5, 2.5)

BASE_TEMPERATURE_C = 14.0
TEMPERATURE_AMPLITUDE_C = 7.0

WEATHER_CATEGORIES = ("clear", "cloudy", "rain")

SCHEMA_COLUMNS = (
    "timestamp",
    "road_segment_id",
    "avg_speed_kph",
    "free_flow_speed_kph",
    "flow_veh_per_hr",
    "occupancy_pct",
    "temperature_c",
    "precipitation_mm",
    "weather_condition",
    "is_incident",
)


def _demand_profile(hours: np.ndarray, weekend: np.ndarray, scale: float) -> np.ndarray:
    """Return a normalised demand value in ``[0, 1]`` for each timestamp.

    Two Gaussian peaks over a daytime baseline, flattened at weekends. This is a
    schematic shape, not a fitted curve to any real road.
    """

    baseline = 0.18 + 0.10 * np.sin(2 * np.pi * (hours - 7) / 24.0)

    morning = 0.72 * np.exp(-0.5 * ((hours - PEAK_MORNING[0]) / PEAK_MORNING[1]) ** 2)
    evening = 0.88 * np.exp(-0.5 * ((hours - PEAK_EVENING[0]) / PEAK_EVENING[1]) ** 2)

    weekday = baseline + morning + evening
    weekend_profile = 0.34 + 0.22 * np.exp(
        -0.5 * ((hours - 13.0) / 4.0) ** 2
    )

    demand = np.where(weekend, weekend_profile, weekday)
    return np.clip(demand * scale, 0.0, 1.0)


def _generate_frame(
    *,
    seed: int,
    segments: int,
    days: int,
    start: str,
) -> pd.DataFrame:
    """Build the simulated observation table."""

    random = np.random.RandomState(seed)

    if segments > len(SEGMENT_FREE_FLOW_KPH):
        raise ValueError(
            f"At most {len(SEGMENT_FREE_FLOW_KPH)} segments are parameterised, "
            f"got {segments}."
        )

    timestamps = pd.date_range(
        start=start, periods=days * 24, freq=f"{INTERVAL_MINUTES}min"
    )
    hours = timestamps.hour.to_numpy().astype(float)
    weekend = (timestamps.dayofweek.to_numpy() >= 5)

    temperature = (
        BASE_TEMPERATURE_C
        + TEMPERATURE_AMPLITUDE_C
        * np.sin(2 * np.pi * (hours - 9.0) / 24.0)
        + random.normal(0.0, 1.2, size=len(timestamps))
    )

    # A few wet hours spread across the fortnight.
    rainfall = np.zeros(len(timestamps), dtype=float)
    wet_hours = random.choice(len(timestamps), size=len(timestamps) // 40, replace=False)
    rainfall[wet_hours] = np.round(random.uniform(0.4, 9.0, size=len(wet_hours)), 1)

    frames: list[pd.DataFrame] = []

    for index in range(segments):
        free_flow = SEGMENT_FREE_FLOW_KPH[index]
        capacity = SEGMENT_CAPACITY[index]
        scale = SEGMENT_DEMAND_SCALE[index]

        demand = _demand_profile(hours, weekend, scale)

        # Speed falls as demand rises; rain and incidents push it lower.
        speed_ratio = 1.02 - 0.92 * demand + random.normal(0.0, 0.045, size=len(timestamps))
        speed_ratio -= 0.06 * (rainfall > 0)

        incidents = (random.random_sample(len(timestamps)) < 0.004).astype(int)
        speed_ratio -= 0.25 * incidents

        speed_ratio = np.clip(speed_ratio, 0.04, 1.0)
        speed = np.round(free_flow * speed_ratio, 2)

        # Volume-flow relationship: flow peaks slightly before speed collapses.
        flow = capacity * demand * (1.15 - 0.45 * speed_ratio)
        flow = np.clip(flow, 0.0, None)
        flow = np.round(flow + random.normal(0.0, 18.0, size=len(timestamps)), 0)
        flow = np.clip(flow, 0.0, None)

        occupancy = np.clip(100.0 * flow / (capacity * 1.35), 0.0, 100.0)

        weather = np.full(len(timestamps), "clear", dtype=object)
        weather[rainfall > 6.0] = "rain"
        weather[(rainfall > 0) & (rainfall <= 6.0)] = "cloudy"

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "road_segment_id": f"SEG-{index + 1:02d}",
                    "avg_speed_kph": speed,
                    "free_flow_speed_kph": free_flow,
                    "flow_veh_per_hr": flow,
                    "occupancy_pct": np.round(occupancy, 2),
                    "temperature_c": np.round(temperature, 2),
                    "precipitation_mm": rainfall,
                    "weather_condition": weather,
                    "is_incident": incidents,
                }
            )
        )

    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return frame.sort_values(["road_segment_id", "timestamp"]).reset_index(drop=True)


def _inject_defects(
    frame: pd.DataFrame, *, seed: int, duplicate_rows: int, null_cells: int
) -> pd.DataFrame:
    """Insert a small, deliberate number of realistic data-quality defects.

    Used only by the test suite, never by the committed fixture. Injects
    missing optional values, impossible numeric readings, and both exact and
    conflicting duplicates, so validation and cleaning can be proven to detect
    each one.
    """

    random = np.random.RandomState(seed)
    defective = frame.copy()

    if null_cells:
        for column in ("flow_veh_per_hr", "occupancy_pct", "temperature_c"):
            positions = random.choice(
                len(defective), size=min(null_cells, len(defective)), replace=False
            )
            defective.loc[positions, column] = np.nan

    if duplicate_rows:
        impossible_positions = random.choice(
            len(defective), size=max(duplicate_rows // 2, 1), replace=False
        )
        defective.loc[impossible_positions, "avg_speed_kph"] = -3.0

        out_of_range = random.choice(
            len(defective), size=max(duplicate_rows // 2, 1), replace=False
        )
        defective.loc[out_of_range, "occupancy_pct"] = 140.0

        exact = defective.sample(n=min(duplicate_rows, len(defective)), replace=False)
        conflicting = defective.sample(n=min(duplicate_rows, len(defective)), replace=False)
        conflicting.loc[conflicting.index[: max(len(conflicting) // 2, 1)], "flow_veh_per_hr"] *= 1.9

        defective = pd.concat([defective, exact, conflicting], ignore_index=True)

    return defective


def write_provenance(
    output_path: Path,
    *,
    seed: int,
    segments: int,
    days: int,
    start: str,
    rows: int,
    defects: bool,
) -> Path:
    """Write the sidecar that marks this file as simulated.

    :func:`ml.preprocessing.detect_provenance` reads it, so the pipeline reports
    ``kind="simulated"`` from a machine-readable source rather than from a guess
    about the filename.
    """

    sidecar = output_path.with_name(output_path.name + PROVENANCE_SUFFIX)
    payload: dict[str, Any] = {
        "kind": "simulated",
        "generator": "ml/data/sample/generate_simulated_dataset.py",
        "description": (
            "Synthetic traffic observations generated from a parametric demand "
            "model with a fixed seed. NOT real-world data and NOT a live data "
            "feed. Provided only to test the preprocessing pipeline."
        ),
        "seed": seed,
        "segments": segments,
        "days": days,
        "start": start,
        "interval_minutes": INTERVAL_MINUTES,
        "rows": rows,
        "defects_injected": defects,
        "schema": list(SCHEMA_COLUMNS),
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar


def generate(
    *,
    seed: int = DEFAULT_SEED,
    segments: int = DEFAULT_SEGMENTS,
    days: int = DEFAULT_DAYS,
    start: str = "2026-01-05",
    defects: bool = False,
) -> pd.DataFrame:
    """Return the simulated dataset as a DataFrame."""

    frame = _generate_frame(seed=seed, segments=segments, days=days, start=start)
    if defects:
        frame = _inject_defects(frame, seed=seed + 1, duplicate_rows=6, null_cells=12)
    return frame


def write_dataset(
    output_path: Path | None = None,
    *,
    seed: int = DEFAULT_SEED,
    segments: int = DEFAULT_SEGMENTS,
    days: int = DEFAULT_DAYS,
    start: str = "2026-01-05",
    defects: bool = False,
) -> tuple[Path, Path, pd.DataFrame]:
    """Generate the fixture and write it with its provenance sidecar.

    Returns:
        ``(csv_path, provenance_path, frame)``.
    """

    target = Path(output_path) if output_path is not None else DEFAULT_OUTPUT
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    frame = generate(
        seed=seed, segments=segments, days=days, start=start, defects=defects
    )
    frame.to_csv(target, index=False)

    sidecar = write_provenance(
        target,
        seed=seed,
        segments=segments,
        days=days,
        start=start,
        rows=int(len(frame)),
        defects=defects,
    )
    return target, sidecar, frame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the SIMULATED development dataset. The output is not "
            "real-world data."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--start", type=str, default="2026-01-05")
    parser.add_argument(
        "--defects",
        action="store_true",
        help="Inject missing values, impossible readings and duplicates.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    output_path, sidecar, frame = write_dataset(
        args.output,
        seed=args.seed,
        segments=args.segments,
        days=args.days,
        start=args.start,
        defects=args.defects,
    )

    print("SIMULATED dataset written (not real-world data)")
    print(f"  file       : {output_path}")
    print(f"  provenance : {sidecar}")
    print(f"  rows       : {len(frame)}")
    print(f"  columns    : {len(frame.columns)}")
    print(f"  segments   : {frame['road_segment_id'].nunique()}")
    print(
        f"  period     : {frame['timestamp'].min()} -> {frame['timestamp'].max()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())