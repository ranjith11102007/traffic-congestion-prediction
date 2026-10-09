"""Inference for the traffic congestion model.

Loads the artifacts written by :mod:`ml.train` and exposes
:func:`predict_traffic`, which returns a structured, JSON-safe result.

Confidence is only reported when the model genuinely produces probabilities. The
estimator's ``predict_proba`` output is used directly and
``estimator.classes_`` supplies the class order — no value is invented, rounded to
a fabricated figure, or defaulted when it cannot be computed.

Feature preparation is delegated to :class:`ml.train.FeatureSchema`, the object
saved at training time, so inference cannot drift from the feature set the model
was fitted on.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ml.preprocessing import (
    CONGESTION_NAMES,
    SEGMENT_COLUMN,
    TIMESTAMP_COLUMN,
    DatasetError,
    MissingColumnsError,
    create_features,
    ensure_columns,
    ensure_datetime_timestamps,
)
from ml.train import ModelArtifactNotFoundError, load_model, load_preprocessor

#: Result keys, named once so callers and tests agree on the contract.
PREDICTED_CLASS_KEY = "predicted_congestion_level"
PREDICTED_LABEL_KEY = "predicted_congestion"
CONFIDENCE_KEY = "confidence"
PROBABILITIES_KEY = "class_probabilities"
RECORD_COUNT_KEY = "records_predicted"
RECORDS_SKIPPED_KEY = "records_skipped_insufficient_history"


class PredictionInputError(DatasetError):
    """Raised when supplied records cannot be scored."""


def _coerce_frame(
    input_data: pd.DataFrame | Sequence[dict[str, Any]] | dict[str, Any],
) -> pd.DataFrame:
    """Accept a DataFrame, a list of records or a single record mapping.

    Raises:
        PredictionInputError: if the input is empty or the wrong shape.
    """

    if isinstance(input_data, pd.DataFrame):
        frame = input_data.copy()
    elif isinstance(input_data, dict):
        frame = pd.DataFrame([input_data])
    else:
        try:
            frame = pd.DataFrame(list(input_data))
        except TypeError as error:
            raise PredictionInputError(
                "input_data must be a DataFrame, a list of records, or a single "
                f"record mapping. Got {type(input_data).__name__}."
            ) from error

    if frame.empty:
        raise PredictionInputError(
            "No records supplied. Provide at least one observation."
        )
    return frame


def predict_traffic(
    input_data: pd.DataFrame | Sequence[dict[str, Any]] | dict[str, Any],
    *,
    models_dir: str | Path | None = None,
    include_probabilities: bool = True,
    drop_incomplete_history: bool = True,
) -> dict[str, Any]:
    """Predict congestion for one or more observation records.

    Args:
        input_data: a DataFrame, a list of record mappings, or a single mapping
            with at least ``timestamp`` and ``road_segment_id``. Lag and rolling
            features are built from the supplied rows, so include enough history
            per road segment — see the schema's ``min_history_per_segment``.
        models_dir: artifact directory; defaults to ``ml/models``.
        include_probabilities: include the full per-class distribution.
        drop_incomplete_history: drop leading rows that lack the lag and rolling
            history the model needs, and report how many were dropped. Set to
            ``False`` to treat them as an error instead.

    Returns:
        A JSON-safe dictionary. For a single record::

            {
              "predicted_congestion_level": 1,
              "predicted_congestion": "moderate",
              "confidence": 0.74,
              "class_probabilities": {"free_flow": 0.02, ...},
              "records_predicted": 1,
              ...
            }

        For several records, ``predictions`` holds one entry per input row and
        the top-level fields describe the first row.

    Raises:
        ModelArtifactNotFoundError: if no model has been trained.
        MissingColumnsError: if a required input column is absent.
        PredictionInputError: if the input is empty or cannot supply history.
    """

    estimator = load_model(models_dir)
    schema = load_preprocessor(models_dir)

    frame = _coerce_frame(input_data)

    missing = [
        column
        for column in schema.required_input_columns
        if column not in frame.columns
    ]
    if missing:
        raise MissingColumnsError(
            "Prediction input is missing required column(s): "
            + ", ".join(missing)
            + ". Supply "
            + str(schema.min_history_per_segment)
            + " prior observations per road segment so lag and rolling features "
            "can be built."
        )

    prepared = schema.transform(frame)
    scoreable = prepared.notna().all(axis=1)

    if not bool(scoreable.any()):
        incomplete = [column for column in prepared.columns if bool(prepared[column].isna().any())]
        raise PredictionInputError(
            "Input does not contain enough history to build feature(s): "
            + ", ".join(incomplete)
            + f". Provide at least {schema.min_history_per_segment} observations "
            "per road segment."
        )
    if not drop_incomplete_history and not bool(scoreable.all()):
        raise PredictionInputError(
            f"{int((~scoreable).sum())} record(s) lack the "
            f"{schema.min_history_per_segment}-observation history the model "
            "needs. Supply more history, or pass drop_incomplete_history=True to "
            "score only the records that have it."
        )

    skipped = int((~scoreable).sum())
    scorable_index = prepared.index[scoreable]
    prepared = prepared.loc[scoreable]

    predictions = np.asarray(estimator.predict(prepared), dtype=int)

    probabilities: np.ndarray | None = None
    if hasattr(estimator, "predict_proba"):
        try:
            probabilities = np.asarray(estimator.predict_proba(prepared), dtype=float)
        except (AttributeError, NotImplementedError):  # pragma: no cover
            probabilities = None

    classes = (
        np.asarray(getattr(estimator, "classes_", []), dtype=int)
        if probabilities is not None
        else np.asarray([], dtype=int)
    )

    label_mapping = schema.label_mapping or {
        str(key): name for key, name in CONGESTION_NAMES.items()
    }

    entries: list[dict[str, Any]] = []
    for position, value in enumerate(predictions):
        entry: dict[str, Any] = {
            PREDICTED_CLASS_KEY: int(value),
            PREDICTED_LABEL_KEY: label_mapping.get(str(int(value)), "unknown"),
        }
        if probabilities is not None and probabilities.size:
            row = probabilities[position]
            entry[CONFIDENCE_KEY] = round(float(row.max()), 6)
            if include_probabilities:
                entry[PROBABILITIES_KEY] = {
                    label_mapping.get(str(int(label)), str(int(label))): round(
                        float(probability), 6
                    )
                    for label, probability in zip(classes, row, strict=True)
                }
        entry["road_segment_id"] = _first_value(frame, SEGMENT_COLUMN, scorable_index[position])
        entry["timestamp"] = _first_value(frame, TIMESTAMP_COLUMN, scorable_index[position])
        entries.append(entry)

    result: dict[str, Any] = {
        **entries[0],
        "predictions": entries,
        RECORD_COUNT_KEY: len(entries),
        "records_received": len(frame),
        RECORDS_SKIPPED_KEY: skipped,
        "model_available": True,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }
    if probabilities is None:
        # Stated explicitly rather than omitted, so a consumer never assumes a
        # confidence value is missing by accident.
        result["confidence_available"] = False

    return result


def prepare_records(records: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 compatibility wrapper returning the full feature frame.

    Stage 3 scores through :class:`ml.train.FeatureSchema`, which returns only the
    model's columns. Existing callers expect every engineered column, so this
    delegates to the same shared builder training uses and applies the recorded
    input requirements, keeping inference and training on one code path.

    Raises:
        MissingColumnsError: if ``records`` is empty, lacks a required column, or
            carries a timestamp that cannot be parsed.
    """

    schema = load_preprocessor()

    if records is None or len(records) == 0:
        raise MissingColumnsError(
            "No records supplied. Provide at least one observation."
        )

    ensure_columns(records, schema.required_input_columns)

    return create_features(
        ensure_datetime_timestamps(records),
        lags=schema.lag_steps,
        rolling_windows=schema.rolling_windows,
        max_one_hot_cardinality=schema.one_hot_max_cardinality,
    )


def _first_value(frame: pd.DataFrame, column: str, position: int) -> Any:
    if column not in frame.columns or position >= len(frame):
        return None
    value = frame[column].iloc[position]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):  # pragma: no cover
            return str(value)
    return value


def predict(
    records: pd.DataFrame,
    artifact_path: str | Path | None = None,
    require_lag_features: bool = True,
) -> pd.DataFrame:
    """Backwards-compatible frame-returning entry point.

    Kept because the Stage 1 tests and any existing caller use it. Returns one
    row per input record with the predicted class, its label and the confidence.

    Args:
        records: raw observation records.
        artifact_path: Stage 1 compatibility argument; only its parent directory
            is read.
        require_lag_features: unused, retained for signature compatibility. Lag
            features are always required by the schema.
    """

    models_dir = Path(artifact_path).expanduser().resolve().parent if artifact_path else None
    result = predict_traffic(records, models_dir=models_dir)
    rows = result["predictions"]

    frame = pd.DataFrame(
        {
            PREDICTED_CLASS_KEY: [row[PREDICTED_CLASS_KEY] for row in rows],
            PREDICTED_LABEL_KEY: [row[PREDICTED_LABEL_KEY] for row in rows],
        }
    )
    if CONFIDENCE_KEY in rows[0]:
        frame[CONFIDENCE_KEY] = [row[CONFIDENCE_KEY] for row in rows]
    for column in (SEGMENT_COLUMN, TIMESTAMP_COLUMN):
        if column in rows[0] and rows[0][column] is not None:
            frame[column] = [row[column] for row in rows]
    return frame


def load_records(path: str | Path) -> pd.DataFrame:
    """Load prediction input from a CSV or JSON file.

    A JSON file may be a bare list of records or an object with a ``records`` key.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DatasetError(f"No input file at {source}.")

    if source.suffix.lower() == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DatasetError(
                f"{source} is not valid JSON: {error}. The file must contain a "
                'list of records or an object of the form {"records": [...]}.'
            ) from error
        if isinstance(payload, dict):
            payload = payload.get("records", [])
        if not isinstance(payload, list):
            raise DatasetError("JSON input must be a list of records or {'records': [...]}.")
        return pd.DataFrame(payload)
    return pd.read_csv(source)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict congestion level for a set of observation records."
    )
    parser.add_argument("input", type=Path, help="Input CSV or JSON file of records.")
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Artifact directory."
    )
    parser.add_argument(
        "--no-probabilities",
        action="store_true",
        help="Omit the full per-class distribution, keeping only confidence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        records = load_records(args.input)
        result = predict_traffic(
            records,
            models_dir=args.models_dir,
            include_probabilities=not args.no_probabilities,
        )
    except (DatasetError, ModelArtifactNotFoundError) as error:
        print(f"error: {error}")
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())