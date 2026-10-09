"""Traffic dataset validation, cleaning and feature engineering.

This module is the entire data pipeline and the single source of truth for the
feature contract. ``train``, ``evaluate`` and ``predict`` all import their column
names and feature construction from here, so a schema change applies everywhere at
once.

Pipeline stages
---------------
Each stage is a standalone function with an explicit signature, so any one of them
can be tested or reused in isolation:

===============  ==========================================================
Stage            Function
===============  ==========================================================
Load             :func:`load_data`
Validate         :func:`validate_data`
Clean            :func:`clean_data`
Engineer         :func:`create_features`
Target + split   :func:`prepare_features_and_target`, :func:`split_chronologically`
Persist          :func:`save_processed_data`
Orchestrate      :func:`run_pipeline`
===============  ==========================================================

Dataset provenance
------------------
No real traffic dataset ships with this repository, and none is claimed to.
``ml/data/raw/`` is reserved for a real dataset the operator supplies;
``ml/data/sample/`` holds a clearly labelled **simulated** fixture used only to
exercise and test the pipeline. Every function that touches data carries a
:class:`DatasetProvenance` record so a simulated dataset can never be silently
mistaken for real-world measurements.

Expected input schema
---------------------
Required
    ``timestamp``          observation time, any pandas-parsable format
    ``road_segment_id``    identifier of the road link the reading belongs to
    ``avg_speed_kph``      mean traversal speed for the interval
Either
    ``free_flow_speed_kph``  free-flow reference speed for the segment, or
    ``speed_ratio``          pre-computed speed / free-flow ratio
Optional, used only when present
    ``occupancy_pct``, ``temperature_c``, ``precipitation_mm``, ``is_incident``
Accepted but not a model input since version 4.0.0
    ``flow_veh_per_hr`` — still loaded and validated, but no longer a feature,
    because no self-serve traffic API publishes vehicle throughput
Nominal categoricals, encoded automatically when present
    ``weather_condition`` and any other non-numeric column

Target construction
-------------------
Priority order, resolved by :func:`prepare_features_and_target`:

1. A dataset-supplied label column is **preserved as-is**. Candidates are
   matched against :data:`CANDIDATE_LABEL_COLUMNS`.
2. Otherwise the four-level label is derived from ``speed_ratio`` using
   :data:`CONGESTION_RATIO_BINS`.

The derived label is an **engineering convention, not ground truth**. Congestion
in traffic engineering is routinely expressed as a speed ratio against the
segment's free-flow speed, and banding that ratio is the standard way to turn a
continuous detector measurement into a level. The specific cut points are
configurable defaults that must be validated against the real dataset's own
free-flow speed distribution before the resulting scores mean anything. Which
path was taken is recorded in ``target_source`` and written into the processed
file's metadata sidecar.

Target leakage
--------------
The label is derived from the same measurement that a naive feature set would
include, so contemporaneous sensor readings are **excluded from the feature
matrix**: :data:`CONTEMPORANEOUS_MEASUREMENT_COLUMNS` lists them and
:func:`select_feature_columns` refuses to emit them. Feeding ``speed_ratio`` to
the model would let it read its own answer and score near-perfect accuracy with
no real predictive skill. Only information genuinely available *before* the
prediction instant is allowed: calendar position, segment identity and capacity,
weather, incident flags, and strictly backward-looking lags and rolling means of
past observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings as _warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# --------------------------------------------------------------------- paths

ML_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = ML_ROOT / "data" / "raw"
DATA_SAMPLE_DIR = ML_ROOT / "data" / "sample"
DATA_PROCESSED_DIR = ML_ROOT / "data" / "processed"
MODELS_DIR = ML_ROOT / "models"
SUMMARY_FILENAME = "data_summary.md"

DEFAULT_DATASET_NAME = "traffic_observations.csv"
DEFAULT_SAMPLE_NAME = "SIMULATED_traffic_observations.csv"
DEFAULT_SAMPLE_PATH = DATA_SAMPLE_DIR / DEFAULT_SAMPLE_NAME
DEFAULT_PROCESSED_NAME = "traffic_processed.csv"
DEFAULT_METADATA_SUFFIX = ".meta.json"
PROVENANCE_SUFFIX = ".provenance.json"

# ------------------------------------------------------------ column contract

TIMESTAMP_COLUMN = "timestamp"
SEGMENT_COLUMN = "road_segment_id"
SPEED_COLUMN = "avg_speed_kph"
FLOW_COLUMN = "flow_veh_per_hr"
FREE_FLOW_COLUMN = "free_flow_speed_kph"
SPEED_RATIO_COLUMN = "speed_ratio"
OCCUPANCY_COLUMN = "occupancy_pct"
TEMPERATURE_COLUMN = "temperature_c"
PRECIPITATION_COLUMN = "precipitation_mm"
INCIDENT_COLUMN = "is_incident"
WEATHER_COLUMN = "weather_condition"

TARGET_COLUMN = "congestion_level"

REQUIRED_COLUMNS: tuple[str, ...] = (TIMESTAMP_COLUMN, SEGMENT_COLUMN, SPEED_COLUMN)
RATIO_SOURCE_COLUMNS: tuple[str, ...] = (FREE_FLOW_COLUMN, SPEED_RATIO_COLUMN)

OPTIONAL_FEATURE_COLUMNS: tuple[str, ...] = (
    FLOW_COLUMN,
    OCCUPANCY_COLUMN,
    TEMPERATURE_COLUMN,
    PRECIPITATION_COLUMN,
    INCIDENT_COLUMN,
    WEATHER_COLUMN,
)

#: Columns that describe a segment rather than an instant. Missing values here are
#: filled with the segment's own median, never a global one, because a global
#: median would silently corrupt the identity of a specific road.
SEGMENT_CONSTANT_COLUMNS: tuple[str, ...] = (FREE_FLOW_COLUMN,)

#: Never model inputs, because the label is derived from them.
CONTEMPORANEOUS_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    SPEED_COLUMN,
    FLOW_COLUMN,
    OCCUPANCY_COLUMN,
    SPEED_RATIO_COLUMN,
)

#: Column names that may hold a dataset-supplied congestion label. An explicit
#: candidate list is used rather than fuzzy matching so the resolution is
#: predictable and reviewable.
CANDIDATE_LABEL_COLUMNS: tuple[str, ...] = (
    TARGET_COLUMN,
    "congestion_class",
    "congestion_state",
    "traffic_level",
    "traffic_state",
    "level_of_service",
    "los",
)

LAG_STEPS: tuple[int, ...] = (1, 2, 3)

#: Columns that get backward-looking lag features.
#:
#: ``flow_veh_per_hr`` was a lag source through model version 3.0.0 and was
#: **removed** for version 4.0.0. Stage 5 verified at field level that no
#: self-serve traffic REST API publishes vehicle throughput: TomTom Flow Segment
#: Data, HERE Traffic v7 and Mapbox Directions all return speeds, free-flow
#: speeds and congestion indicators only, with no count, volume or density field.
#: Keeping flow lags would therefore have forced one of two dishonest choices
#: either model inputs that no real feed can ever populate, or an imputed value
#: presented as a measurement. Dropping the feature removes the requirement
#: without inventing anything, and keeps the remaining inputs genuinely obtainable.
#: A dataset may still carry the column; it is simply not a model input.
LAG_SOURCE_COLUMNS: tuple[str, ...] = (SPEED_COLUMN, SPEED_RATIO_COLUMN)

#: Rolling means of *past* observations, in observation intervals.
ROLLING_WINDOWS: tuple[int, ...] = (3, 6)

# Feature columns derived from the timestamp itself. Ordinal by nature, so they
# are kept numeric rather than one-hot encoded.
CALENDAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "month",
)

# ---------------------------------------------------------------- target bands

# Left-closed bins on speed_ratio: each boundary belongs to the less congested
# class. Labels are 0 (free flow) through 3 (severe), so a higher ratio means
# less congestion. See derive_congestion_level() for the full table.
CONGESTION_RATIO_BINS: tuple[float, ...] = (0.15, 0.40, 0.75)
CONGESTION_LABELS: tuple[float, ...] = (3.0, 2.0, 1.0, 0.0)
CONGESTION_NAMES: dict[int, str] = {
    0: "free_flow",
    1: "moderate",
    2: "heavy",
    3: "severe",
}

# Word orders recognised when a dataset supplies the label as text. Listed from
# most to least congested so that mapping a label to a higher integer never means
# "less congested" — the same direction as CONGESTION_NAMES.
ORDINAL_LABEL_VOCABULARIES: tuple[tuple[str, ...], ...] = (
    ("free_flow", "moderate", "heavy", "severe"),
    ("low", "moderate", "high", "severe"),
    ("low", "medium", "high"),
    ("level_0", "level_1", "level_2", "level_3"),
    ("a", "b", "c", "d", "e", "f"),
)

# ------------------------------------------------------------------- behaviour

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_TEST_RATIO = 0.2

#: Nominal categoricals with more distinct values than this are frequency-encoded
#: instead of one-hot encoded, to stop a high-cardinality segment id from
#: exploding the feature count.
ONE_HOT_MAX_CARDINALITY = 20

#: Physically plausible inclusive ranges. Values outside are treated as sensor
#: errors and nulled rather than clipped, so the cleaning report stays honest.
VALID_VALUE_RULES: dict[str, tuple[float, float]] = {
    SPEED_COLUMN: (0.0, 200.0),
    FLOW_COLUMN: (0.0, 50_000.0),
    FREE_FLOW_COLUMN: (1.0, 200.0),
    OCCUPANCY_COLUMN: (0.0, 100.0),
    TEMPERATURE_COLUMN: (-70.0, 65.0),
    PRECIPITATION_COLUMN: (0.0, 300.0),
}

#: Cleaning refuses to drop more than this fraction of rows.
MAX_DROP_RATIO = 0.20

#: Complete feature contract: engineered columns present regardless of dataset
#: shape. Derived and lag columns are added by :func:`create_features`.
BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    *CALENDAR_FEATURE_COLUMNS,
    FREE_FLOW_COLUMN,
    INCIDENT_COLUMN,
)

FEATURE_COLUMNS: tuple[str, ...] = BASE_FEATURE_COLUMNS

ProvenanceKind = Literal["real", "simulated", "unknown"]


class DatasetError(RuntimeError):
    """Base class for data-contract violations."""


class DatasetNotFoundError(DatasetError):
    """Raised when the requested dataset file does not exist."""


class MissingColumnsError(DatasetError):
    """Raised when a dataset does not satisfy the documented schema."""


class ValidationFailedError(DatasetError):
    """Raised when a dataset has schema errors that make it unusable."""


class ExcessiveDataLossError(DatasetError):
    """Raised when cleaning would discard an implausible share of the rows."""


# ------------------------------------------------------------- column naming


def lag_feature_columns(steps: Iterable[int] = LAG_STEPS) -> tuple[str, ...]:
    """Return the generated lag column names for the given step sizes."""

    return tuple(
        f"{source}_lag_{step}"
        for step in steps
        for source in LAG_SOURCE_COLUMNS
    )


def rolling_feature_columns(
    source: str = SPEED_RATIO_COLUMN,
    windows: Iterable[int] = ROLLING_WINDOWS,
) -> tuple[str, ...]:
    """Return the generated rolling-mean column names."""

    return tuple(f"{source}_rolling_mean_{window}" for window in windows)


def one_hot_column(source: str, level: Any) -> str:
    """Return the one-hot column name for a categorical level."""

    return f"{source}__{_slugify(str(level))}"


def frequency_column(source: str) -> str:
    """Return the frequency-encoding column name for a categorical column."""

    return f"{source}_freq"


def _slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


# ------------------------------------------------------------------ provenance


@dataclass(frozen=True)
class DatasetProvenance:
    """Where a dataset came from and how trustworthy it is.

    ``kind`` is never inferred optimistically. A file under
    ``ml/data/sample/`` is ``simulated`` by definition, and an unrecognised
    location is ``unknown`` rather than being assumed real.
    """

    kind: ProvenanceKind
    source_path: str
    description: str
    sha256: str
    detected_by: str

    @property
    def is_simulated(self) -> bool:
        return self.kind == "simulated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_path": self.source_path,
            "description": self.description,
            "sha256": self.sha256,
            "detected_by": self.detected_by,
        }


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file, for immutability checks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_provenance(path: str | Path) -> DatasetProvenance:
    """Classify a dataset file as real, simulated or unknown.

    Resolution order:

    1. An adjacent ``<file>.provenance.json`` sidecar, if present and readable.
    2. Location: anything inside ``ml/data/sample/`` is simulated.
    3. Location: anything inside ``ml/data/raw/`` is treated as operator-supplied.
    4. A ``SIMULATED`` / ``sample`` marker in the filename, as defence in depth.
    """

    dataset_path = Path(path).expanduser().resolve()
    sha256 = file_sha256(dataset_path)

    sidecar = dataset_path.with_name(dataset_path.name + PROVENANCE_SUFFIX)
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        if payload.get("kind") in {"real", "simulated", "unknown"}:
            return DatasetProvenance(
                kind=payload["kind"],
                source_path=str(dataset_path),
                description=str(payload.get("description", "")).strip() or "(no description)",
                sha256=sha256,
                detected_by=f"sidecar:{sidecar.name}",
            )

    try:
        relative_parts = dataset_path.relative_to(ML_ROOT).parts
    except ValueError:
        relative_parts = ()

    if "sample" in relative_parts:
        return DatasetProvenance(
            kind="simulated",
            source_path=str(dataset_path),
            description=(
                "Simulated development fixture generated by "
                "ml/data/sample/generate_simulated_dataset.py. Not real-world data."
            ),
            sha256=sha256,
            detected_by="directory:ml/data/sample",
        )

    if "raw" in relative_parts:
        return DatasetProvenance(
            kind="real",
            source_path=str(dataset_path),
            description="Operator-supplied dataset in ml/data/raw/.",
            sha256=sha256,
            detected_by="directory:ml/data/raw",
        )

    marker = dataset_path.stem.lower()
    if "simulat" in marker or "sample" in marker or "dummy" in marker:
        return DatasetProvenance(
            kind="simulated",
            source_path=str(dataset_path),
            description="Filename marks this file as simulated or sample data.",
            sha256=sha256,
            detected_by="filename_marker",
        )

    return DatasetProvenance(
        kind="unknown",
        source_path=str(dataset_path),
        description=(
            "Dataset outside the recognised data directories. Provenance could "
            "not be established; treat it as unverified."
        ),
        sha256=sha256,
        detected_by="default",
    )


# --------------------------------------------------------------- stage 1: load


def resolve_dataset_path(path: str | Path | None = None) -> Path:
    """Resolve the dataset location.

    Precedence: an explicit ``path``, then the default name inside
    ``ml/data/raw/``, then the labelled simulated fixture in ``ml/data/sample/``.
    The fallback exists so the pipeline is runnable out of the box, and the
    resulting :class:`DatasetProvenance` records that it was simulated.
    """

    if path is not None:
        return Path(path).expanduser().resolve()

    raw_default = DATA_RAW_DIR / DEFAULT_DATASET_NAME
    if raw_default.is_file():
        return raw_default

    sample_default = DATA_SAMPLE_DIR / DEFAULT_SAMPLE_NAME
    if sample_default.is_file():
        return sample_default

    return raw_default


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Stage 1: load a CSV dataset with Pandas.

    The file is read verbatim. Type coercion, validation and cleaning belong to
    later stages so each can be tested on its own.

    Raises:
        DatasetNotFoundError: if the file does not exist. No dataset is ever
            synthesised to satisfy this call.
    """

    dataset_path = resolve_dataset_path(path)
    if not dataset_path.is_file():
        raise DatasetNotFoundError(
            f"No dataset found at {dataset_path}.\n"
            "This project does not ship real traffic data. Either:\n"
            "  * place a real CSV dataset at ml/data/raw/"
            f"{DEFAULT_DATASET_NAME} (see ml/data/raw/README.md), or\n"
            "  * use the labelled simulated fixture: "
            f"python -m ml.data.sample.generate_simulated_dataset\n"
            "  * or pass --data <path> explicitly."
        )

    frame = pd.read_csv(dataset_path)
    return frame


# ----------------------------------------------------------- stage 2: validate


@dataclass(frozen=True)
class ValidationReport:
    """Measured quality of a raw dataset. Never mutates the data."""

    rows: int
    columns: tuple[str, ...]
    shape: tuple[int, int]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    missing_total: int
    missing_ratio: float
    exact_duplicate_rows: int
    duplicate_keys: int
    conflicting_duplicate_keys: int
    unparsable_timestamps: int
    non_numeric_cells: dict[str, int]
    out_of_range_cells: dict[str, int]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        """True when no blocking problem was found."""

        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "shape": list(self.shape),
            "columns": list(self.columns),
            "dtypes": self.dtypes,
            "missing_counts": self.missing_counts,
            "missing_total": self.missing_total,
            "missing_ratio": round(self.missing_ratio, 6),
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "duplicate_keys": self.duplicate_keys,
            "conflicting_duplicate_keys": self.conflicting_duplicate_keys,
            "unparsable_timestamps": self.unparsable_timestamps,
            "non_numeric_cells": self.non_numeric_cells,
            "out_of_range_cells": self.out_of_range_cells,
            "categorical_columns": list(self.categorical_columns),
            "numeric_columns": list(self.numeric_columns),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "is_usable": self.is_usable,
        }

    def render(self) -> str:
        """Return a human-readable report of the measured dataset quality."""

        def show(values: Mapping[str, int]) -> str:
            if not values:
                return "none"
            return ", ".join(f"{name}={count}" for name, count in sorted(values.items()))

        lines = [
            f"shape                 : {self.rows} rows x {len(self.columns)} columns",
            f"columns               : {', '.join(self.columns) or 'none'}",
            f"numeric columns       : {', '.join(self.numeric_columns) or 'none'}",
            f"categorical columns   : {', '.join(self.categorical_columns) or 'none'}",
            f"missing values        : {self.missing_total} cell(s) "
            f"({self.missing_ratio * 100:.2f}%) across "
            f"{len(self.missing_counts)} column(s)",
            f"  by column           : {show(self.missing_counts)}",
            f"duplicate rows (exact)         : {self.exact_duplicate_rows}",
            f"duplicate (segment, timestamp) : {self.duplicate_keys}",
            f"  of which conflicting values  : {self.conflicting_duplicate_keys}",
            f"unparsable timestamps : {self.unparsable_timestamps}",
            f"non-numeric cells     : {show(self.non_numeric_cells)}",
            f"out-of-range cells    : {show(self.out_of_range_cells)}",
            f"errors                : {'none' if not self.errors else '; '.join(self.errors)}",
            f"warnings              : {'none' if not self.warnings else '; '.join(self.warnings)}",
        ]
        return "\n".join(lines)


def _count_exact_duplicates(frame: pd.DataFrame) -> int:
    return int(frame.duplicated().sum())


def _count_duplicate_keys(
    frame: pd.DataFrame, key_columns: Sequence[str]
) -> tuple[int, int]:
    """Return ``(duplicate_keys, conflicting_keys)`` for the given key.

    ``duplicate_keys`` counts extra rows beyond the first occurrence of each key.
    ``conflicting_keys`` counts keys where those extra rows disagree on at least
    one value, which is a real data-quality problem rather than a harmless
    repeated export.
    """

    present = [column for column in key_columns if column in frame.columns]
    if len(present) < 2 or frame.duplicated(subset=present).sum() == 0:
        return 0, 0

    duplicates = frame[frame.duplicated(subset=present, keep=False)]
    conflicting = (
        duplicates.groupby(present, dropna=False)
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
        .sum()
    )
    return int(frame.duplicated(subset=present).sum()), int(conflicting)


def validate_data(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = (SEGMENT_COLUMN, TIMESTAMP_COLUMN),
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
    value_rules: Mapping[str, tuple[float, float]] | None = VALID_VALUE_RULES,
) -> ValidationReport:
    """Stage 2: measure a dataset's quality without modifying it.

    Checks performed: shape, column names, dtypes, missing values, exact and
    key-based duplicates, timestamp parsability, numeric coercion failures and
    physically implausible values, plus categorical-column detection.
    """

    columns = tuple(str(column) for column in frame.columns)
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [name for name in required_columns if name not in frame.columns]
    if missing_columns:
        errors.append(
            "missing required column(s): " + ", ".join(missing_columns)
        )
    if not any(name in frame.columns for name in RATIO_SOURCE_COLUMNS):
        errors.append(
            f"need one of {FREE_FLOW_COLUMN!r} or {SPEED_RATIO_COLUMN!r} to measure congestion"
        )
    if len(frame) == 0:
        errors.append("dataset contains no rows")

    # --- timestamps ------------------------------------------------------
    unparsable = 0
    if TIMESTAMP_COLUMN in frame.columns:
        # The count of unparsable values is reported below, so pandas' own
        # "could not infer format" notice would only add noise.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
        unparsable = int(parsed.isna().sum() - frame[TIMESTAMP_COLUMN].isna().sum())
        if unparsable:
            warnings.append(
                f"{unparsable} value(s) in {TIMESTAMP_COLUMN!r} are not parsable datetimes"
            )

    # --- duplicates ------------------------------------------------------
    exact_duplicates = _count_exact_duplicates(frame)
    duplicate_keys, conflicting_keys = _count_duplicate_keys(frame, key_columns)
    if exact_duplicates:
        warnings.append(f"{exact_duplicates} fully duplicated row(s)")
    if conflicting_keys:
        warnings.append(
            f"{conflicting_keys} (segment, timestamp) key(s) carry conflicting values"
        )

    # --- numeric health --------------------------------------------------
    rules = VALID_VALUE_RULES if value_rules is None else value_rules
    non_numeric_cells: dict[str, int] = {}
    out_of_range_cells: dict[str, int] = {}

    for column in columns:
        series = frame[column]
        if column in rules or (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
            coerced = pd.to_numeric(series, errors="coerce")
            failures = int((series.notna() & coerced.isna()).sum())
            if failures:
                non_numeric_cells[column] = failures
            if column in rules:
                low, high = rules[column]
                numeric_series = coerced.dropna()
                breaches = int(((numeric_series < low) | (numeric_series > high)).sum())
                if breaches:
                    out_of_range_cells[column] = breaches

    # --- missing values --------------------------------------------------
    missing_counts = {
        str(column): int(count)
        for column, count in frame.isna().sum().items()
        if count
    }
    missing_total = int(sum(missing_counts.values()))
    missing_ratio = missing_total / float(len(frame) * len(frame.columns)) if len(frame) and len(frame.columns) else 0.0

    for column, count in missing_counts.items():
        if column in required_columns:
            errors.append(
                f"required column {column!r} has {count} missing value(s); it cannot be imputed"
            )
        elif count / max(len(frame), 1) > MAX_DROP_RATIO:
            warnings.append(
                f"column {column!r} is {count / max(len(frame), 1) * 100:.1f}% missing"
            )

    categorical_columns = tuple(
        str(column)
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    )
    numeric_columns = tuple(
        str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
    )

    return ValidationReport(
        rows=int(len(frame)),
        columns=columns,
        shape=(int(len(frame)), int(len(frame.columns))),
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        missing_counts=missing_counts,
        missing_total=missing_total,
        missing_ratio=missing_ratio,
        exact_duplicate_rows=exact_duplicates,
        duplicate_keys=duplicate_keys,
        conflicting_duplicate_keys=conflicting_keys,
        unparsable_timestamps=unparsable,
        non_numeric_cells=non_numeric_cells,
        out_of_range_cells=out_of_range_cells,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


# ------------------------------------------------------------- stage 3: clean


@dataclass(frozen=True)
class CleaningReport:
    """Every action cleaning took, so no change is silent."""

    rows_in: int
    rows_out: int
    rows_dropped_duplicates: int
    nulled_invalid_values: dict[str, int] = field(default_factory=dict)
    imputed_by_carried_forward: dict[str, int] = field(default_factory=dict)
    imputed_by_backward_fill: dict[str, int] = field(default_factory=dict)
    imputed_by_segment_median: dict[str, int] = field(default_factory=dict)
    imputed_by_global_median: dict[str, int] = field(default_factory=dict)
    imputed_categorical_by_mode: dict[str, int] = field(default_factory=dict)
    rows_dropped_invalid_timestamps: int = 0

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def drop_ratio(self) -> float:
        return self.rows_dropped / self.rows_in if self.rows_in else 0.0

    @property
    def imputed_total(self) -> int:
        buckets = (
            self.imputed_by_carried_forward,
            self.imputed_by_backward_fill,
            self.imputed_by_segment_median,
            self.imputed_by_global_median,
            self.imputed_categorical_by_mode,
        )
        return sum(sum(bucket.values()) for bucket in buckets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "drop_ratio": round(self.drop_ratio, 6),
            "rows_dropped_duplicates": self.rows_dropped_duplicates,
            "rows_dropped_invalid_timestamps": self.rows_dropped_invalid_timestamps,
            "nulled_invalid_values": self.nulled_invalid_values,
            "imputed_by_carried_forward": self.imputed_by_carried_forward,
            "imputed_by_backward_fill": self.imputed_by_backward_fill,
            "imputed_by_segment_median": self.imputed_by_segment_median,
            "imputed_by_global_median": self.imputed_by_global_median,
            "imputed_categorical_by_mode": self.imputed_categorical_by_mode,
        }

    def render(self) -> str:
        def show(values: Mapping[str, int]) -> str:
            if not values:
                return "none"
            return ", ".join(f"{name}={count}" for name, count in sorted(values.items()))

        return "\n".join(
            [
                f"rows in / out        : {self.rows_in} -> {self.rows_out} "
                f"({self.rows_dropped} dropped, {self.drop_ratio * 100:.2f}%)",
                f"duplicates removed   : {self.rows_dropped_duplicates}",
                f"rows dropped (bad ts): {self.rows_dropped_invalid_timestamps}",
                f"values nulled        : {show(self.nulled_invalid_values)}",
                f"imputed (forward)    : {show(self.imputed_by_carried_forward)}",
                f"imputed (backward)   : {show(self.imputed_by_backward_fill)}",
                f"imputed (seg median) : {show(self.imputed_by_segment_median)}",
                f"imputed (glob median): {show(self.imputed_by_global_median)}",
                f"imputed (mode)       : {show(self.imputed_categorical_by_mode)}",
            ]
        )


def clean_data(
    frame: pd.DataFrame,
    *,
    validation: ValidationReport | None = None,
    group_column: str = SEGMENT_COLUMN,
    value_rules: Mapping[str, tuple[float, float]] = VALID_VALUE_RULES,
    max_drop_ratio: float = MAX_DROP_RATIO,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Stage 3: repair a dataset and report every change.

    Actions, in order:

    1. Drop fully duplicated rows.
    2. Resolve conflicting ``(segment, timestamp)`` keys by keeping the first
       occurrence. Averaging disagreeing readings would invent a measurement.
    3. Drop rows whose timestamp cannot be parsed.
    4. Null out non-numeric and physically implausible numeric values, so the
       imputation step below handles them consistently.
    5. Impute remaining gaps. Segment constant columns use the segment median;
       other numeric columns are carried forward then backward within the segment
       (traffic state is autocorrelated, so a neighbouring observation is a better
       estimate than a global average); anything still missing falls back to the
       global median. Nominal columns use the global mode.

    Raises:
        ExcessiveDataLossError: if more than ``max_drop_ratio`` of rows would be
            discarded. Guessing at data that is mostly missing is not cleaning.
        ValidationFailedError: if required columns are missing or a required
            column still has gaps after imputation.
    """

    report = validation or validate_data(frame)
    if not report.is_usable:
        raise ValidationFailedError(
            "Dataset failed validation:\n  - " + "\n  - ".join(report.errors)
        )

    rows_in = int(len(frame))
    nulled: dict[str, int] = {}

    # --- 1/2. duplicates --------------------------------------------------
    before = len(frame)
    frame = frame.drop_duplicates()
    duplicates_removed = before - len(frame)

    key_columns = [column for column in (group_column, TIMESTAMP_COLUMN) if column in frame.columns]
    if len(key_columns) == 2:
        before = len(frame)
        frame = frame.drop_duplicates(subset=key_columns, keep="first")
        duplicates_removed += before - len(frame)

    # --- 3. timestamps ---------------------------------------------------
    dropped_bad_timestamps = 0
    frame = frame.copy()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    mask_bad_timestamp = frame[TIMESTAMP_COLUMN].isna()
    dropped_bad_timestamps = int(mask_bad_timestamp.sum())
    if dropped_bad_timestamps:
        frame = frame.loc[~mask_bad_timestamp]

    # --- 4. implausible values -------------------------------------------
    for column, (low, high) in value_rules.items():
        if column not in frame.columns:
            continue
        coerced = pd.to_numeric(frame[column], errors="coerce")
        breaches = coerced.notna() & ((coerced < low) | (coerced > high))
        non_numeric = frame[column].notna() & coerced.isna()
        null_count = int((breaches | non_numeric).sum())
        if null_count:
            nulled[column] = null_count
        frame[column] = coerced.where(~(breaches | non_numeric))

    # --- 5. imputation ---------------------------------------------------
    forward: dict[str, int] = {}
    backward: dict[str, int] = {}
    segment_median: dict[str, int] = {}
    global_median: dict[str, int] = {}
    categorical_mode: dict[str, int] = {}

    group_series = frame[group_column] if group_column in frame.columns else None

    for column in frame.columns:
        if column in (TIMESTAMP_COLUMN,) or column == group_column:
            continue

        if not int(frame[column].isna().sum()):
            continue

        if pd.api.types.is_numeric_dtype(frame[column]):
            # A segment property is best filled from the segment's own typical
            # value, so this runs *before* the neighbour fill: copying the adjacent
            # reading would be meaningless for a constant like free-flow speed.
            if column in SEGMENT_CONSTANT_COLUMNS and group_series is not None:
                per_segment = frame[column].groupby(group_series).transform(
                    lambda values: values.fillna(values.median())
                )
                filled = int(frame[column].isna().sum() - per_segment.isna().sum())
                if filled:
                    segment_median[column] = filled
                frame[column] = per_segment

            if group_series is not None:
                carried = frame[column].groupby(group_series).ffill()
                filled = int(frame[column].isna().sum() - carried.isna().sum())
                if filled:
                    forward[column] = filled
                frame[column] = carried

                carried_back = frame[column].groupby(group_series).bfill()
                filled = int(frame[column].isna().sum() - carried_back.isna().sum())
                if filled:
                    backward[column] = filled
                frame[column] = carried_back

            fallback = frame[column].median()
            if pd.notna(fallback):
                remaining = frame[column].isna()
                filled = int(remaining.sum())
                if filled:
                    global_median[column] = filled
                frame.loc[remaining, column] = fallback
        else:
            mode = frame[column].mode(dropna=True)
            if not len(mode):
                continue
            remaining = frame[column].isna()
            filled = int(remaining.sum())
            if filled:
                categorical_mode[column] = filled
            frame.loc[remaining, column] = mode.iloc[0]

    # --- guards ----------------------------------------------------------
    rows_out = int(len(frame))
    cleaning_report = CleaningReport(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped_duplicates=int(duplicates_removed),
        nulled_invalid_values=nulled,
        imputed_by_carried_forward=forward,
        imputed_by_backward_fill=backward,
        imputed_by_segment_median=segment_median,
        imputed_by_global_median=global_median,
        imputed_categorical_by_mode=categorical_mode,
        rows_dropped_invalid_timestamps=dropped_bad_timestamps,
    )

    if cleaning_report.drop_ratio > max_drop_ratio:
        raise ExcessiveDataLossError(
            f"Cleaning would drop {cleaning_report.drop_ratio * 100:.1f}% of rows, "
            f"above the {max_drop_ratio * 100:.0f}% limit. Inspect the dataset "
            "rather than accepting this loss."
        )

    for column in REQUIRED_COLUMNS:
        if column in frame.columns and frame[column].isna().any():
            remaining = int(frame[column].isna().sum())
            raise ValidationFailedError(
                f"Required column {column!r} still has {remaining} missing value(s) "
                "after cleaning. It cannot be imputed without inventing data."
            )

    frame = frame.sort_values(
        [c for c in (group_column, TIMESTAMP_COLUMN) if c in frame.columns]
    ).reset_index(drop=True)
    return frame, cleaning_report


# --------------------------------------------------- stage 4: feature creation


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive discrete time features from the timestamp.

    These are ordinal, not nominal: Monday genuinely follows Sunday, so a tree
    model benefits from an ordered integer far more than from one-hot columns.
    """

    result = frame.copy()
    timestamps = result[TIMESTAMP_COLUMN]
    result["hour"] = timestamps.dt.hour.astype("float64")
    result["day_of_week"] = timestamps.dt.dayofweek.astype("float64")
    result["day_of_month"] = timestamps.dt.day.astype("float64")
    result["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype("float64")
    result["month"] = timestamps.dt.month.astype("float64")
    return result


def add_speed_ratio(
    frame: pd.DataFrame,
    free_flow_column: str = FREE_FLOW_COLUMN,
    ratio_column: str = SPEED_RATIO_COLUMN,
) -> pd.DataFrame:
    """Ensure a ``speed_ratio`` column exists.

    ``speed_ratio`` is the basis of the derived label, so it is a *target*
    quantity, never a model input. It is retained in the processed frame for
    transparency and is excluded by :func:`select_feature_columns`.

    Raises:
        MissingColumnsError: if neither a pre-computed ratio nor a free-flow
            reference speed is available.
    """

    result = frame.copy()
    if ratio_column not in result.columns:
        if free_flow_column not in result.columns:
            raise MissingColumnsError(
                f"Provide either {ratio_column!r} or {free_flow_column!r}; "
                "congestion cannot be derived without a free-flow reference."
            )
        reference = pd.to_numeric(result[free_flow_column], errors="coerce")
        if (reference <= 0).any():
            raise MissingColumnsError(
                f"Column {free_flow_column!r} contains non-positive values."
            )
        result[ratio_column] = pd.to_numeric(result[SPEED_COLUMN], errors="coerce") / reference
    result[ratio_column] = pd.to_numeric(result[ratio_column], errors="coerce").clip(upper=1.0)
    return result


def add_lag_features(
    frame: pd.DataFrame,
    steps: Iterable[int] = LAG_STEPS,
    group_column: str = SEGMENT_COLUMN,
) -> pd.DataFrame:
    """Add strictly backward-looking value lags per road segment.

    ``groupby(segment).shift(step)`` guarantees a row can only ever see its own
    past. This is the central leakage control of the whole pipeline: a forecast
    of congestion at time *t* must not depend on observations after *t*.
    """

    result = frame.copy()
    ordered = result.sort_values([group_column, TIMESTAMP_COLUMN])
    for step in steps:
        for source in LAG_SOURCE_COLUMNS:
            if source not in result.columns:
                continue
            shifted = ordered.groupby(group_column)[source].shift(step)
            result[f"{source}_lag_{step}"] = pd.to_numeric(shifted, errors="coerce").astype(
                "float64"
            )
    return result


def add_rolling_features(
    frame: pd.DataFrame,
    windows: Iterable[int] = ROLLING_WINDOWS,
    source: str = SPEED_RATIO_COLUMN,
    group_column: str = SEGMENT_COLUMN,
) -> pd.DataFrame:
    """Add rolling means over *past* observations only.

    The series is shifted by one before rolling, so the current observation is
    excluded from its own window. A centred window here would be textbook label
    leakage.
    """

    result = frame.copy()
    if source not in result.columns:
        return result

    ordered = result.sort_values([group_column, TIMESTAMP_COLUMN])
    past = ordered.groupby(group_column)[source].shift(1)
    grouped_past = past.groupby(ordered[group_column])

    for window in windows:
        column = f"{source}_rolling_mean_{window}"
        rolled = (
            grouped_past.rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        result[column] = rolled.astype("float64")
    return result


def detect_categorical_columns(
    frame: pd.DataFrame, *, exclude: Sequence[str] = ()
) -> tuple[str, ...]:
    """Return the non-numeric columns eligible for categorical encoding."""

    excluded = set(exclude) | {TIMESTAMP_COLUMN, TARGET_COLUMN}
    return tuple(
        str(column)
        for column in frame.columns
        if column not in excluded
        and not pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    )


def encode_categorical_columns(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    max_one_hot_cardinality: int = ONE_HOT_MAX_CARDINALITY,
    exclude: Sequence[str] = (),
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Encode nominal categoricals as numeric columns.

    * ``cardinality <= max_one_hot_cardinality`` → one-hot indicators. Road and
      weather categories have no intrinsic order, so one-hot avoids inventing one.
    * above the cap → a single frequency column. One-hotting hundreds of segment
      ids would bloat the matrix and fragment the splits.

    Returns ``(frame, encoded_column_names)``.
    """

    result = frame.copy()
    names = list(columns) if columns is not None else list(
        detect_categorical_columns(result, exclude=exclude)
    )
    encoded: list[str] = []

    for column in names:
        if column not in result.columns:
            continue
        series = result[column]
        cardinality = int(series.nunique(dropna=True))

        if cardinality == 0:
            continue

        if cardinality <= max_one_hot_cardinality:
            dummies = pd.get_dummies(series.astype("object"), dtype="int8")
            dummies.columns = [one_hot_column(column, level) for level in dummies.columns]
            for dummy in dummies.columns:
                result[dummy] = dummies[dummy]
                encoded.append(dummy)
        else:
            counts = series.value_counts(normalize=True, dropna=True)
            result[frequency_column(column)] = series.map(counts).astype("float64")
            encoded.append(frequency_column(column))

    return result, tuple(encoded)


def create_features(
    frame: pd.DataFrame,
    *,
    lags: Iterable[int] = LAG_STEPS,
    rolling_windows: Iterable[int] = ROLLING_WINDOWS,
    group_column: str = SEGMENT_COLUMN,
    encode: bool = True,
    max_one_hot_cardinality: int = ONE_HOT_MAX_CARDINALITY,
    exclude_from_encoding: Sequence[str] = (),
) -> pd.DataFrame:
    """Stage 4: add every model-ready feature to a cleaned frame.

    Adds calendar features, the speed ratio, strictly backward-looking lags and
    rolling means, an incident flag when the feed is absent, and numeric encodings
    of any nominal categoricals.
    """

    result = add_calendar_features(frame)
    result = add_speed_ratio(result)
    result = add_lag_features(result, steps=lags, group_column=group_column)
    result = add_rolling_features(
        result, windows=rolling_windows, group_column=group_column
    )

    if INCIDENT_COLUMN not in result.columns:
        # Absent incident feed: a constant zero honestly encodes "unknown", while
        # imputing a nonzero rate would fabricate a signal.
        result[INCIDENT_COLUMN] = 0.0

    if encode:
        categoricals = detect_categorical_columns(
            result, exclude=exclude_from_encoding
        )
        if categoricals:
            result, _ = encode_categorical_columns(
                result,
                categoricals,
                max_one_hot_cardinality=max_one_hot_cardinality,
                exclude=exclude_from_encoding,
            )

    return result


def select_feature_columns(
    frame: pd.DataFrame,
    *,
    excluded: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the model-input columns of a prepared frame.

    Excluded by construction:

    * identifiers and time keys, which are not measurements;
    * the target;
    * :data:`CONTEMPORANEOUS_MEASUREMENT_COLUMNS`, whose values determine the
      label. Including them would let the model read its own answer.
    """

    if excluded is None:
        excluded = (TIMESTAMP_COLUMN, SEGMENT_COLUMN, TARGET_COLUMN, *CONTEMPORANEOUS_MEASUREMENT_COLUMNS)
    excluded_set = set(excluded)
    return tuple(
        str(column)
        for column in frame.columns
        if column not in excluded_set and is_numeric_dtype(frame[column])
    )


# ------------------------------------------- stage 5: target and train/test prep


def detect_label_column(frame: pd.DataFrame) -> str | None:
    """Return a dataset-supplied label column, if one is recognisable.

    Only :data:`CANDIDATE_LABEL_COLUMNS` is considered. No fuzzy or
    substring matching is used, so the outcome is predictable.
    """

    for candidate in CANDIDATE_LABEL_COLUMNS:
        if candidate in frame.columns:
            return candidate
    return None


def derive_congestion_level(
    frame: pd.DataFrame, ratio_column: str = SPEED_RATIO_COLUMN
) -> pd.DataFrame:
    """Derive the four-level congestion target from ``speed_ratio``.

    Bands are **left-closed** so the boundaries belong to the less congested
    class, matching the table in ``docs/ml-pipeline.md``:

    ============  ======================  ===================
    Class         Band                    ``speed_ratio``
    ============  ======================  ===================
    0 free_flow   ``[0.75, inf)``        uncongested
    1 moderate    ``[0.40, 0.75)``       above capacity
    2 heavy       ``[0.15, 0.40)``       breakdown forming
    3 severe      ``(-inf, 0.15)``       near standstill
    ============  ======================  ===================

    This is a documented convention, not measured ground truth. See the module
    docstring.
    """

    result = frame.copy()
    result[TARGET_COLUMN] = pd.cut(
        result[ratio_column],
        bins=[-np.inf, *CONGESTION_RATIO_BINS, np.inf],
        labels=CONGESTION_LABELS,
        include_lowest=True,
        right=False,
    ).astype("float64")
    return result


def encode_target_labels(series: pd.Series) -> tuple[pd.Series, dict[str, str]]:
    """Return the target as integers plus the label dictionary that explains it.

    A dataset's own labels are *preserved*, never replaced by the derived bands:

    * numeric non-negative integers are used unchanged, so a source class ``3``
      stays class ``3``;
    * recognised ordinal words are mapped by their natural order, so
      ``low < medium < high`` becomes ``0 < 1 < 2``;
    * any other text is mapped in sorted order and the mapping is recorded, so an
      arbitrary vocabulary can still be decoded later.

    Returns:
        ``(codes, mapping)`` where ``mapping`` is ``{"0": "low", …}``.
    """

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        mapping = {str(int(value)): str(int(value)) for value in sorted(numeric.dropna().unique())}
        return numeric, mapping

    cleaned = series.astype("string").str.strip()
    if bool(cleaned.isna().all()):
        return pd.Series(np.nan, index=series.index, dtype="float64"), {}

    observed = {str(value).strip().lower() for value in cleaned.dropna().unique()}
    for vocabulary in ORDINAL_LABEL_VOCABULARIES:
        if observed <= set(vocabulary):
            order = {label: position for position, label in enumerate(vocabulary)}
            break
    else:
        order = {label: position for position, label in enumerate(sorted(observed))}

    codes = pd.to_numeric(cleaned.str.lower().map(order), errors="coerce")
    mapping = {str(position): label for label, position in order.items()}
    return codes.astype("float64"), mapping


def build_prepared_frame(
    frame: pd.DataFrame,
    *,
    label_column: str | None = None,
    lags: Iterable[int] = LAG_STEPS,
    rolling_windows: Iterable[int] = ROLLING_WINDOWS,
    max_one_hot_cardinality: int = ONE_HOT_MAX_CARDINALITY,
    drop_incomplete: bool = True,
) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    """Stage 5a: produce one frame holding identifiers, features and the target.

    Columns are ordered ``road_segment_id``, ``timestamp``, features…, target so
    the processed CSV reads sensibly and the target is always last.

    The resolved label column is also recorded in ``frame.attrs`` as
    ``target_source`` and ``target_label_map``, so the mapping survives without
    changing this function's signature.

    Returns:
        ``(frame, feature_columns, target_source)``.

    Raises:
        MissingColumnsError: if no label can be resolved or produced.
    """

    # The label must be resolved *before* features are built: a text label such
    # as traffic_level would otherwise be one-hot encoded into the inputs, which
    # is target leakage rather than a feature.
    if label_column is not None and label_column not in frame.columns:
        raise MissingColumnsError(
            f"Requested label column {label_column!r} is not in the dataset. "
            f"Available columns: {', '.join(frame.columns)}"
        )
    resolved_label = label_column if label_column is not None else detect_label_column(frame)

    result = create_features(
        frame,
        lags=lags,
        rolling_windows=rolling_windows,
        max_one_hot_cardinality=max_one_hot_cardinality,
        exclude_from_encoding=(resolved_label,) if resolved_label else (),
    )

    if resolved_label is not None:
        source = "dataset_provided"
        result[TARGET_COLUMN] = result[resolved_label]
    else:
        source = "derived_speed_ratio"
        result = derive_congestion_level(result)

    codes, mapping = encode_target_labels(result[TARGET_COLUMN])
    if source == "derived_speed_ratio":
        # The bands have canonical names; report those rather than "0", "1", …
        mapping = {str(key): name for key, name in CONGESTION_NAMES.items()}
    if codes.isna().all():
        raise MissingColumnsError(
            f"Target column {TARGET_COLUMN!r} could not be resolved from the dataset. "
            "Supply it with --label-column."
        )
    result[TARGET_COLUMN] = codes
    result.attrs["target_source"] = source
    result.attrs["target_label_map"] = mapping

    feature_columns = select_feature_columns(result)
    if not feature_columns:
        raise MissingColumnsError(
            "No numeric feature columns remain after preprocessing. Check the "
            "dataset schema against ml/data/raw/README.md."
        )

    if drop_incomplete:
        result = result.dropna(subset=[*feature_columns, TARGET_COLUMN])

    if result.empty:
        raise MissingColumnsError(
            "No rows survived feature preparation. The dataset is probably too "
            "short for the requested lags and rolling windows."
        )

    identifiers = [
        column
        for column in (SEGMENT_COLUMN, TIMESTAMP_COLUMN)
        if column in result.columns
    ]
    ordered_columns = [
        *identifiers,
        *[column for column in feature_columns if column not in identifiers],
        TARGET_COLUMN,
    ]
    result = result.loc[:, ordered_columns]
    result = result.sort_values(identifiers).reset_index(drop=True)
    # Re-assert after the reorder, because selecting and sorting do not carry
    # DataFrame.attrs across in every pandas version.
    result.attrs["target_source"] = source
    result.attrs["target_label_map"] = mapping
    return result, feature_columns, source


def prepare_features_and_target(
    frame: pd.DataFrame,
    *,
    label_column: str | None = None,
    lags: Iterable[int] = LAG_STEPS,
    rolling_windows: Iterable[int] = ROLLING_WINDOWS,
    max_one_hot_cardinality: int = ONE_HOT_MAX_CARDINALITY,
    drop_incomplete: bool = True,
) -> tuple[pd.DataFrame, pd.Series, tuple[str, ...], str]:
    """Stage 5b: split a prepared frame into a feature matrix and target series.

    The target is never part of ``X``.

    Args:
        frame: an already cleaned frame (see :func:`clean_data`).
        label_column: dataset-supplied label to preserve. Auto-detected from
            :data:`CANDIDATE_LABEL_COLUMNS` when omitted.
        drop_incomplete: drop rows that cannot supply every feature. Warm-up rows
            are dropped rather than zero-filled, because a zero speed is not a
            plausible real measurement.

    Returns:
        ``(X, y, feature_columns, target_source)`` where ``target_source`` is
        ``"dataset_provided"`` or ``"derived_speed_ratio"``.
    """

    prepared_frame, feature_columns, source = build_prepared_frame(
        frame,
        label_column=label_column,
        lags=lags,
        rolling_windows=rolling_windows,
        max_one_hot_cardinality=max_one_hot_cardinality,
        drop_incomplete=drop_incomplete,
    )

    X = prepared_frame.loc[:, list(feature_columns)].reset_index(drop=True)
    y = prepared_frame.loc[:, TARGET_COLUMN].astype("int64").reset_index(drop=True)
    return X, y, feature_columns, source


def ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise :class:`MissingColumnsError` naming every absent column.

    All missing names are listed in one message, so a caller fixes its dataset in
    one pass rather than one error at a time.
    """

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MissingColumnsError(
            f"Required columns are missing: {', '.join(missing)}. "
            f"Present columns: {', '.join(map(str, frame.columns))}"
        )


def build_feature_matrix(
    frame: pd.DataFrame, feature_columns: Sequence[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a prepared frame into ``X`` and ``y``."""

    columns = (
        tuple(feature_columns) if feature_columns else select_feature_columns(frame)
    )
    ensure_columns(frame, columns)
    X = frame.loc[:, list(columns)]
    y = frame.loc[:, TARGET_COLUMN].astype("int64")
    return X, y


def build_supervised_frame(
    path: str | Path | None = None,
    *,
    label_column: str | None = None,
    lags: Iterable[int] = LAG_STEPS,
    rolling_windows: Iterable[int] = ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load a dataset and return a prepared supervised frame.

    Convenience wrapper over the individual stages — ``load_data``,
    ``validate_data``, ``clean_data``, ``build_prepared_frame`` — kept for the
    Stage 3 training and evaluation entry points, which need a single frame plus
    its feature list.

    Returns:
        ``(frame, feature_columns)``.
    """

    frame = load_data(path)
    cleaned, _ = clean_data(frame)
    prepared, feature_columns, _ = build_prepared_frame(
        cleaned, label_column=label_column, lags=lags, rolling_windows=rolling_windows
    )
    return prepared, feature_columns


def ensure_datetime_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame with ``timestamp`` parsed into real datetimes.

    A frame that has been through a CSV round-trip carries the timestamp as text,
    so every time-based operation (splitting, grouping, range filters) would
    otherwise compare strings. Parsing here keeps the rest of the pipeline
    working on either representation.

    Raises:
        MissingColumnsError: if the timestamp column is absent.
        DatasetError: if any value cannot be parsed.
    """

    ensure_columns(frame, (TIMESTAMP_COLUMN,))
    if pd.api.types.is_datetime64_any_dtype(frame[TIMESTAMP_COLUMN]):
        return frame

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")

    unparsable = int(parsed.isna().sum() - frame[TIMESTAMP_COLUMN].isna().sum())
    if unparsable:
        raise DatasetError(
            f"{unparsable} value(s) in {TIMESTAMP_COLUMN!r} could not be parsed as "
            "a datetime. Clean the dataset before splitting it."
        )

    result = frame.copy()
    result[TIMESTAMP_COLUMN] = parsed
    return result


def split_chronologically(
    frame: pd.DataFrame, *, test_ratio: float = DEFAULT_TEST_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on time, never randomly, for Stage 3 training.

    Traffic observations are strongly autocorrelated: adjacent rows describe the
    same evolving traffic state. A random row split would place neighbouring
    instants on both sides of the boundary and leak the future into training,
    inflating scores. The split point is a single global timestamp, so every
    segment contributes its past to training and its future to testing.

    Args:
        frame: a prepared frame. The timestamp may be text or datetime.
        test_ratio: fraction of rows reserved for testing.

    Returns:
        ``(train, test)``, both sorted by timestamp.

    Raises:
        ValueError: if ``test_ratio`` is not strictly between 0 and 1.
        DatasetError: if either side of the split would be empty.
    """

    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    timed = ensure_datetime_timestamps(frame)
    cutoff = timed[TIMESTAMP_COLUMN].quantile(1 - test_ratio)
    train = timed.loc[timed[TIMESTAMP_COLUMN] < cutoff]
    test = timed.loc[timed[TIMESTAMP_COLUMN] >= cutoff]

    if train.empty or test.empty:
        raise DatasetError(
            "Chronological split produced an empty side. Increase the dataset "
            "time span or lower test_ratio."
        )

    return (
        train.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True),
        test.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True),
    )


def class_distribution(y: pd.Series) -> dict[str, int]:
    """Return the per-class row counts, for imbalance reporting."""

    counts = y.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def chronological_split_info(frame: pd.DataFrame, test_ratio: float) -> dict[str, Any]:
    """Describe the chronological split Stage 3 should use."""

    train, test = split_chronologically(frame, test_ratio=test_ratio)
    return {
        "strategy": "chronological",
        "test_ratio": test_ratio,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_time_range": [
            str(train[TIMESTAMP_COLUMN].min()),
            str(train[TIMESTAMP_COLUMN].max()),
        ],
        "test_time_range": [
            str(test[TIMESTAMP_COLUMN].min()),
            str(test[TIMESTAMP_COLUMN].max()),
        ],
        "boundary": str(train[TIMESTAMP_COLUMN].max()),
        "note": (
            "Used in Stage 3. A random split is invalid here because adjacent "
            "traffic observations are strongly autocorrelated."
        ),
    }


# ------------------------------------------------------------ stage 6: persist


def resolve_processed_path(output_path: str | Path | None = None) -> Path:
    """Resolve the processed-dataset destination."""

    if output_path is not None:
        return Path(output_path).expanduser().resolve()
    return DATA_PROCESSED_DIR / DEFAULT_PROCESSED_NAME


def write_processed_frame(
    frame: pd.DataFrame, output_path: str | Path | None = None
) -> Path:
    """Write a frame to the processed-data directory."""

    target = resolve_processed_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def save_processed_data(
    frame: pd.DataFrame,
    output_path: str | Path | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    write_metadata: bool = True,
) -> Path:
    """Stage 6: persist the processed dataset plus a metadata sidecar.

    The raw dataset is never modified. The sidecar records provenance, the
    SHA-256 of the source, the feature list, the target and how it was
    constructed, so the processed file can be traced back and reproduced.
    """

    target = write_processed_frame(frame, output_path)

    if write_metadata and metadata is not None:
        sidecar = target.with_name(target.name + DEFAULT_METADATA_SUFFIX)
        payload = dict(metadata)
        payload["output_path"] = str(target)
        payload["output_rows"] = int(len(frame))
        payload["output_columns"] = [str(column) for column in frame.columns]
        sidecar.write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )

    return target


# ------------------------------------------------------------ orchestration


@dataclass(frozen=True)
class PreparedDataset:
    """Everything one pipeline run produced."""

    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    feature_columns: tuple[str, ...]
    target_source: str
    provenance: DatasetProvenance
    validation: ValidationReport
    cleaning: CleaningReport
    output_path: Path | None = None
    metadata_path: Path | None = None

    @property
    def class_distribution(self) -> dict[str, int]:
        return class_distribution(self.target)

    def metadata(self, *, test_ratio: float = DEFAULT_TEST_RATIO) -> dict[str, Any]:
        """Return the JSON-serialisable record of this run."""

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "ml.preprocessing.run_pipeline",
            "source": self.provenance.to_dict(),
            "validation": self.validation.to_dict(),
            "cleaning": self.cleaning.to_dict(),
            "feature_columns": list(self.feature_columns),
            "feature_count": len(self.feature_columns),
            "target_column": TARGET_COLUMN,
            "target_source": self.target_source,
            "target_label_map": {
                str(key): value for key, value in CONGESTION_NAMES.items()
            },
            "excluded_from_features": {
                "identifiers": [TIMESTAMP_COLUMN, SEGMENT_COLUMN],
                "target_derived_quantities": list(CONTEMPORANEOUS_MEASUREMENT_COLUMNS),
                "reason": (
                    "Contemporaneous sensor readings determine the label; using "
                    "them as inputs would be target leakage."
                ),
            },
            "class_distribution": self.class_distribution,
            "recommended_split": chronological_split_info(self.frame, test_ratio),
        }


def run_pipeline(
    path: str | Path | None = None,
    *,
    label_column: str | None = None,
    lags: Iterable[int] = LAG_STEPS,
    rolling_windows: Iterable[int] = ROLLING_WINDOWS,
    max_drop_ratio: float = MAX_DROP_RATIO,
    max_one_hot_cardinality: int = ONE_HOT_MAX_CARDINALITY,
    output_path: str | Path | None = None,
    test_ratio: float = DEFAULT_TEST_RATIO,
    save: bool = True,
) -> PreparedDataset:
    """Run load → validate → clean → engineer → target → persist.

    Returns a :class:`PreparedDataset` carrying every report, so the caller can
    print them, assert on them, or discard them.

    Raises:
        DatasetError: on any data-contract violation. Nothing is fabricated.
    """

    dataset_path = resolve_dataset_path(path)
    provenance = detect_provenance(dataset_path)

    raw = load_data(dataset_path)
    validation = validate_data(raw)
    if not validation.is_usable:
        raise ValidationFailedError(
            "Dataset failed validation:\n  - " + "\n  - ".join(validation.errors)
        )

    cleaned, cleaning = clean_data(
        raw, validation=validation, max_drop_ratio=max_drop_ratio
    )
    frame, feature_columns, target_source = build_prepared_frame(
        cleaned,
        label_column=label_column,
        lags=lags,
        rolling_windows=rolling_windows,
        max_one_hot_cardinality=max_one_hot_cardinality,
    )

    X = frame.loc[:, list(feature_columns)].reset_index(drop=True)
    y = frame.loc[:, TARGET_COLUMN].astype("int64").reset_index(drop=True)

    prepared = PreparedDataset(
        frame=frame,
        features=X,
        target=y,
        feature_columns=feature_columns,
        target_source=target_source,
        provenance=provenance,
        validation=validation,
        cleaning=cleaning,
    )

    if save:
        written = save_processed_data(
            frame,
            output_path,
            metadata=prepared.metadata(test_ratio=test_ratio),
        )
        prepared = PreparedDataset(
            frame=prepared.frame,
            features=prepared.features,
            target=prepared.target,
            feature_columns=prepared.feature_columns,
            target_source=prepared.target_source,
            provenance=prepared.provenance,
            validation=prepared.validation,
            cleaning=prepared.cleaning,
            output_path=written,
            metadata_path=written.with_name(written.name + DEFAULT_METADATA_SUFFIX),
        )

    return prepared


def render_report(prepared: PreparedDataset) -> str:
    """Render the full pipeline report for a terminal or a markdown block."""

    provenance = prepared.provenance
    banner = (
        "!! SIMULATED DATA !! This dataset is a synthetic fixture for testing "
        "the pipeline. It is not real-world traffic data and no result derived "
        "from it is meaningful."
        if provenance.is_simulated
        else f"Dataset provenance: {provenance.kind} ({provenance.detected_by})"
    )

    sections = [
        banner,
        "",
        "== source ==",
        f"path            : {provenance.source_path}",
        f"kind            : {provenance.kind}",
        f"description     : {provenance.description}",
        f"sha256          : {provenance.sha256}",
        "",
        "== validation ==",
        prepared.validation.render(),
        "",
        "== cleaning ==",
        prepared.cleaning.render(),
        "",
        "== features ==",
        f"feature count   : {len(prepared.feature_columns)}",
        f"features        : {', '.join(prepared.feature_columns)}",
        f"target          : {TARGET_COLUMN} (source: {prepared.target_source})",
        f"class counts    : {json.dumps(prepared.class_distribution)}",
        f"rows prepared   : {len(prepared.frame)}",
    ]

    if prepared.output_path is not None:
        sections.extend(
            [
                f"written to      : {prepared.output_path}",
                f"metadata        : {prepared.metadata_path}",
            ]
        )

    return "\n".join(sections)


def render_summary_markdown(prepared: PreparedDataset) -> str:
    """Render ``ml/data/data_summary.md`` from measured values only.

    Every number in the output comes from the reports produced by this run, so
    the document cannot drift from the data it describes.
    """

    provenance = prepared.provenance
    validation = prepared.validation
    cleaning = prepared.cleaning
    split = chronological_split_info(prepared.frame, DEFAULT_TEST_RATIO)

    if provenance.is_simulated:
        nature = (
            "**SIMULATED — not real-world data.** Generated by "
            "`ml/data/sample/generate_simulated_dataset.py` for the sole purpose "
            "of exercising and testing this pipeline. No metric derived from it is "
            "meaningful."
        )
    elif provenance.kind == "real":
        nature = (
            "**Real-world data**, supplied by the operator into `ml/data/raw/`. "
            f"Source description: {provenance.description}"
        )
    else:
        nature = (
            "**Provenance unestablished.** "
            f"Source description: {provenance.description}"
        )

    missing_rows = "\n".join(
        f"| `{column}` | {count} | {count / validation.rows * 100:.2f}% |"
        for column, count in sorted(validation.missing_counts.items())
    ) or "| _none_ | 0 | 0.00% |"

    feature_rows = "\n".join(
        f"| `{column}` | {_describe_feature(column)} |"
        for column in prepared.feature_columns
    )

    imputation_strategies = {
        "imputed_by_carried_forward": "carried forward",
        "imputed_by_backward_fill": "carried backward",
        "imputed_by_segment_median": "segment median",
        "imputed_by_global_median": "global median",
        "imputed_categorical_by_mode": "column mode",
    }
    imputation_parts = [
        f"{column}={count} ({imputation_strategies[key]})"
        for key, label in imputation_strategies.items()
        for column, count in sorted(getattr(cleaning, key).items())
    ]
    imputation_detail = (
        ", ".join(imputation_parts) if imputation_parts else "no gaps required filling"
    )

    return f"""# Dataset summary

_Generated by `ml.preprocessing.run_pipeline` on \
{datetime.now(timezone.utc).isoformat()}. Every figure below is measured from \
the dataset by that run; nothing here is hand-written or estimated._

## Dataset name and source

| Field | Value |
| --- | --- |
| Path | `{provenance.source_path}` |
| Nature | {nature} |
| Provenance detected by | `{provenance.detected_by}` |
| SHA-256 | `{provenance.sha256}` |

## Size

| Metric | Value |
| --- | --- |
| Raw records | {validation.rows} |
| Raw columns | {len(validation.columns)} |
| Raw shape | {validation.shape[0]} × {validation.shape[1]} |
| Processed records | {len(prepared.frame)} |
| Model features | {len(prepared.feature_columns)} |
| Road segments | {int(prepared.frame[SEGMENT_COLUMN].nunique())} |
| Time span | {prepared.frame[TIMESTAMP_COLUMN].min()} → \
{prepared.frame[TIMESTAMP_COLUMN].max()} |

## Raw columns

| Column | Role |
| --- | --- |
{chr(10).join(f"| `{column}` | input ({validation.dtypes.get(column, '?')}) |" for column in validation.columns)}
| `{TARGET_COLUMN}` | derived target, not present in the raw file |

## Feature descriptions

| Feature | Description |
| --- | --- |
{feature_rows}

Columns deliberately excluded from the feature matrix:

| Excluded | Reason |
| --- | --- |
| `{TIMESTAMP_COLUMN}` | Ordering key, already decomposed into calendar features. |
| `{SEGMENT_COLUMN}` | Identity key, encoded as indicator columns instead. |
| `{TARGET_COLUMN}` | The target. |
| {", ".join(f"`{name}`" for name in CONTEMPORANEOUS_MEASUREMENT_COLUMNS)} | Contemporaneous readings. The label is derived from them, so using them as inputs would be target leakage. |

## Target variable

| Field | Value |
| --- | --- |
| Name | `{TARGET_COLUMN}` |
| Source | `{prepared.target_source}` |
| Type | four-level ordinal class, 0 = least congested |
| Classes | {", ".join(f"{key} = {value}" for key, value in sorted(CONGESTION_NAMES.items()))} |

{"The dataset supplied no congestion label, so the target was derived from " + SPEED_RATIO_COLUMN + " using left-closed bands at " + ", ".join(str(bound) for bound in CONGESTION_RATIO_BINS) + ". This is a documented engineering convention, not measured ground truth; the cut points must be validated against the real dataset's free-flow speed distribution before any score is meaningful." if prepared.target_source == "derived_speed_ratio" else "A dataset-supplied label was found and preserved unchanged."}

Class balance in the processed dataset:

| Class | Rows |
| --- | --- |
{chr(10).join(f"| {key} | {value} |" for key, value in sorted(prepared.class_distribution.items()))}

## Missing-value summary (measured on the raw file)

| Column | Missing | Share |
| --- | --- | --- |
{missing_rows}

Total missing cells: {validation.missing_total} \
({validation.missing_ratio * 100:.2f}% of all cells).

## Duplicate summary (measured on the raw file)

| Metric | Count |
| --- | --- |
| Fully duplicated rows | {validation.exact_duplicate_rows} |
| Duplicate `{SEGMENT_COLUMN}` + `{TIMESTAMP_COLUMN}` keys | {validation.duplicate_keys} |
| …of which carried conflicting values | {validation.conflicting_duplicate_keys} |
| Rows removed by cleaning | {cleaning.rows_dropped_duplicates} |

## Preprocessing performed

1. **Load** — CSV read verbatim with Pandas, no coercion at read time.
2. **Validate** — shape, columns, dtypes, missing values, duplicates, timestamp
   parsability, numeric coercion failures and physically implausible values
   measured and reported.
3. **Clean** — {cleaning.rows_dropped_duplicates} duplicate row(s) removed;
   {sum(cleaning.nulled_invalid_values.values())} out-of-range value(s) nulled;
   {cleaning.imputed_total} gap(s) imputed ({imputation_detail}); rows sorted by
   segment then timestamp.
4. **Feature engineering** — calendar decomposition, speed ratio, strictly
   backward lags at steps {list(LAG_STEPS)}, rolling means over the previous
   {list(ROLLING_WINDOWS)} observations, incident flag, one-hot encoding of
   nominal categoricals.
5. **Target** — resolved as `{prepared.target_source}`.
6. **Train/test preparation** — chronological split reserved for Stage 3:
   {split["train_rows"]} training rows up to `{split["boundary"]}`, then
   {split["test_rows"]} test rows.
7. **Persist** — written to `{prepared.output_path}` with a JSON metadata sidecar.

### Leakage controls

* Lags use `groupby(segment).shift(step)`, so a row only sees its own past.
* Rolling means shift by one before rolling, so the current observation is never
  in its own window.
* Contemporaneous sensor readings are excluded from the feature matrix.
* The train/test split is chronological, not random, because adjacent traffic
  observations are strongly autocorrelated.

## Limitations

* {"**This dataset is simulated.** It demonstrates that the pipeline runs; it does not support any claim about real traffic behaviour or model accuracy." if provenance.is_simulated else "The dataset's real-world coverage, sensor quality and missing-data mechanisms have not been independently verified."}
* The four-level congestion target is derived from speed ratio using fixed,
  configurable cut points. It is an engineering convention, not ground truth.
* Class balance is reported above and is expected to be imbalanced; severe
  congestion is the rarest class. Stage 3 must not report accuracy alone.
* Rows without enough history for the requested lags and rolling windows are
  dropped rather than imputed, which costs up to `max(lags, rolling_windows)`
  rows per segment.
* No ground-truth incident or road-closure labels are present; `is_incident` is a
  synthetic indicator in the fixture.
* Missing optional values are imputed, not measured. Their imputed values must not
  be reported as observations.
"""


def _describe_feature(column: str) -> str:
    """Return a one-line description for a generated feature column."""

    known: dict[str, str] = {
        "hour": "Hour of day, 0–23 (ordinal).",
        "day_of_week": "Day of week, 0 = Monday (ordinal).",
        "day_of_month": "Day of month, 1–31.",
        "is_weekend": "1 for Saturday or Sunday, else 0.",
        "month": "Calendar month, 1–12.",
        FREE_FLOW_COLUMN: "Segment free-flow reference speed, a static capacity property.",
        INCIDENT_COLUMN: "Incident indicator; constant 0 when the feed is absent.",
    }
    if column in known:
        return known[column]
    if column.startswith(SPEED_RATIO_COLUMN + "_lag_"):
        step = column.rsplit("_", 1)[-1]
        return f"{SPEED_RATIO_COLUMN} {step} observation interval(s) earlier for this segment."
    if column.startswith(SPEED_RATIO_COLUMN + "_rolling_mean_"):
        window = column.rsplit("_", 1)[-1]
        return f"Mean {SPEED_RATIO_COLUMN} over the previous {window} observations, excluding the current one."
    if column.startswith(f"{SPEED_COLUMN}_lag_"):
        step = column.rsplit("_", 1)[-1]
        return f"Average speed {step} observation interval(s) earlier for this segment."
    if "__" in column:
        source, level = column.split("__", 1)
        return f"One-hot indicator: {source} equals `{level}`."
    if column.endswith("_freq"):
        source = column[: -len("_freq")]
        return f"Share of observations with this {source} value (frequency encoding)."
    return "Engineered numeric feature."


# ----------------------------------------------------------------- entry point


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, clean and engineer features for the traffic congestion "
            "model. Reads a CSV dataset and writes a processed CSV plus a "
            "metadata sidecar. Never modifies the source dataset."
        )
    )
    parser.add_argument("--data", type=Path, default=None, help="Input CSV path.")
    parser.add_argument("--out", type=Path, default=None, help="Processed CSV path.")
    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Dataset-supplied congestion label to preserve. Auto-detected when "
            "omitted; otherwise the target is derived from speed ratio."
        ),
    )
    parser.add_argument(
        "--lags", type=int, nargs="*", default=list(LAG_STEPS), help="Lag steps."
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="*",
        default=list(ROLLING_WINDOWS),
        help="Rolling mean windows over past observations.",
    )
    parser.add_argument(
        "--max-drop-ratio",
        type=float,
        default=MAX_DROP_RATIO,
        help="Refuse to clean if more than this fraction of rows would be dropped.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Test share recorded for the Stage 3 chronological split.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help=f"Also write ml/data/{SUMMARY_FILENAME} from the measured results.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Run the pipeline without writing output."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        prepared = run_pipeline(
            path=args.data,
            label_column=args.label_column,
            lags=tuple(args.lags),
            rolling_windows=tuple(args.rolling_windows),
            max_drop_ratio=args.max_drop_ratio,
            output_path=args.out,
            test_ratio=args.test_ratio,
            save=not args.no_save,
        )
    except DatasetError as error:
        print(f"error: {error}")
        return 1

    print(render_report(prepared))

    if args.write_summary and not args.no_save:
        summary_path = DATA_PROCESSED_DIR.parent / SUMMARY_FILENAME
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_summary_markdown(prepared), encoding="utf-8")
        print(f"\nsummary written to: {summary_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())