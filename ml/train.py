"""Model training for the traffic congestion classifier.

Phase 3 trains a **Random Forest classifier** on the dataset produced by Stage 2
(``ml/data/processed/traffic_processed.csv``).

What this module owns
---------------------
* loading the processed dataset,
* the **chronological** train/test split,
* estimator construction behind a single seam (so other models can be compared
  later),
* persisting the fitted model and everything inference needs.

Split methodology
-----------------
Traffic observations are strongly autocorrelated: consecutive rows describe the
same evolving congestion state. A random row split would put neighbouring
instants on both sides of the boundary and leak the future into training, which
inflates scores. This module therefore cuts the dataset at **one global
timestamp** — every segment contributes its past to training and its future to
testing. The boundary is stored in the model metadata so evaluation and any
future run reproduce the identical split.

Note the honest consequence: a chronological split cannot be stratified, so the
test window may not contain the same class proportions as the whole dataset.
:func:`ml.evaluate.evaluate` reports per-class metrics precisely because of this.

Artifacts written
-----------------
``ml/models/traffic_model.pkl``
    The fitted estimator.
``ml/models/preprocessor.pkl``
    The :class:`FeatureSchema` — feature order, label map and the feature-build
    parameters. Required, because inference must rebuild features identically.
    No scaler is saved: a Random Forest is invariant to feature scale, so a
    fitted `StandardScaler` would be dead weight. This is stated rather than
    silently omitted.
``ml/models/model_metadata.json``
    Provenance and split record.
``ml/models/feature_importance.json``
    Feature importances measured from the fitted estimator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from ml import __version__ as ml_package_version
from ml.preprocessing import (
    CONGESTION_NAMES,
    CONTEMPORANEOUS_MEASUREMENT_COLUMNS,
    DEFAULT_PROCESSED_NAME,
    DEFAULT_SAMPLE_NAME,
    DATA_PROCESSED_DIR,
    LAG_STEPS,
    MODELS_DIR,
    ONE_HOT_MAX_CARDINALITY,
    ROLLING_WINDOWS,
    SEGMENT_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    DatasetError,
    DatasetNotFoundError,
    ensure_datetime_timestamps,
    split_chronologically,
)
from ml.schema import FeatureSchema, derive_required_inputs

MODEL_NAME = "random_forest"
MODEL_VERSION = "4.0.0"
MODEL_FILENAME = "traffic_model.pkl"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
METADATA_FILENAME = "model_metadata.json"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.json"

DEFAULT_TEST_RATIO = 0.2
RANDOM_STATE = 42

#: Raw columns a caller must supply so every recorded feature can be rebuilt.
#: Derived from the feature names rather than hard-coded, so a schema change
#: cannot leave this list stale.
_ALWAYS_REQUIRED = (TIMESTAMP_COLUMN, SEGMENT_COLUMN)


class ModelArtifactNotFoundError(DatasetError):
    """Raised when inference is attempted before a model has been trained."""


# --------------------------------------------------------------- model registry


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build and describe one candidate estimator.

    Comparing models in Phase 4 means adding entries here and re-running the
    same split, not editing training logic.
    """

    key: str
    display_name: str
    builder: Any
    parameters: dict[str, Any] = field(default_factory=dict)


def _random_forest() -> RandomForestClassifier:
    """The Stage 3 baseline estimator.

    ``class_weight='balanced'`` compensates for the imbalance typical of
    congestion data, where severe congestion is far rarer than free flow.
    """

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        key="random_forest",
        display_name="Random Forest Classifier",
        builder=_random_forest,
    ),
    "gradient_boosting": ModelSpec(
        key="gradient_boosting",
        display_name="HistGradientBoosting Classifier",
        builder=lambda: __import__(
            "sklearn.ensemble", fromlist=["HistGradientBoostingClassifier"]
        ).HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    ),
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        builder=lambda: __import__(
            "sklearn.linear_model", fromlist=["LogisticRegression"]
        ).LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    ),
}

DEFAULT_MODEL_KEY = "random_forest"


def build_estimator(
    model_key: str = DEFAULT_MODEL_KEY, params: dict[str, Any] | None = None
) -> Any:
    """Instantiate a registered estimator.

    Args:
        model_key: a key from :data:`MODEL_REGISTRY`.
        params: overrides applied after the defaults.

    Raises:
        DatasetError: if ``model_key`` is not registered.
    """

    if model_key not in MODEL_REGISTRY:
        raise DatasetError(
            f"Unknown model {model_key!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    estimator = MODEL_REGISTRY[model_key].builder()
    if params:
        estimator.set_params(**params)
    return estimator


# The feature contract lives in ml/schema.py so its pickled name is stable;
# see that module's docstring for why.


# ------------------------------------------------------------ data loading


def resolve_dataset_path(data_path: str | Path | None = None) -> Path:
    """Locate the processed dataset produced by Stage 2.

    Raises:
        DatasetNotFoundError: if neither the given path nor the default exists.
    """

    if data_path is not None:
        path = Path(data_path).expanduser().resolve()
        if not path.is_file():
            raise DatasetNotFoundError(
                f"No dataset at {path}. Run 'python -m ml.preprocessing' first to "
                "create the processed dataset."
            )
        return path

    default = DATA_PROCESSED_DIR / DEFAULT_PROCESSED_NAME
    if default.is_file():
        return default

    raise DatasetNotFoundError(
        f"No processed dataset at {default}. Generate it with:\n"
        "  python -m ml.preprocessing --write-summary"
    )


def load_processed_dataset(
    data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load the Stage 2 dataset and verify it is fit for classification.

    Returns:
        ``(frame, source_path)``.

    Raises:
        DatasetError: if the target is missing, continuous, or the frame holds
            values that cannot be used as model inputs.
    """

    source = resolve_dataset_path(data_path)
    frame = pd.read_csv(source)

    if frame.empty:
        raise DatasetError(f"{source} is empty. Nothing to train on.")

    if TARGET_COLUMN not in frame.columns:
        raise DatasetError(
            f"{source} has no {TARGET_COLUMN!r} column, so there is nothing to "
            f"predict. Columns present: {', '.join(map(str, frame.columns))}"
        )

    target = frame[TARGET_COLUMN]
    if bool(target.isna().any()):
        raise DatasetError(
            f"{TARGET_COLUMN} contains {int(target.isna().sum())} missing value(s). "
            "Re-run preprocessing; warm-up rows should have been dropped."
        )

    distinct = np.sort(target.unique())
    if not np.all(np.equal(np.mod(distinct, 1), 0)):
        raise DatasetError(
            f"{TARGET_COLUMN} holds non-integer values {distinct[:5].tolist()}, so "
            "it is continuous and needs a regressor, not a classifier."
        )
    if target.nunique() < 2:
        raise DatasetError(
            f"{TARGET_COLUMN} has a single value, so no classifier can be trained."
        )
    if target.nunique() > 20:
        raise DatasetError(
            f"{TARGET_COLUMN} has {target.nunique()} distinct values. That is too "
            "many for the congestion classes defined in ml.preprocessing."
        )

    frame = ensure_datetime_timestamps(frame)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int64")

    return frame, source


def select_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the numeric model inputs, excluding keys and the target.

    Delegates to the Stage 2 selection so training cannot reintroduce a column
    the pipeline excluded for leakage reasons. The comparison in
    :func:`ml.train.train` also asserts that none of
    :data:`~ml.preprocessing.CONTEMPORANEOUS_MEASUREMENT_COLUMNS` is present.
    """

    from ml.preprocessing import select_feature_columns as _stage2_select

    return _stage2_select(frame)


def detect_simulated_dataset(source: Path) -> bool:
    """Report whether a processed dataset descends from simulated data.

    A filename check alone is not enough: Stage 3 trains on the *processed* file,
    whose name carries no ``SIMULATED`` marker even when its source was synthetic.
    The Stage 2 sidecar records the true provenance, so it is consulted first and
    the filename is only a fallback. Getting this wrong would let a report present
    synthetic numbers as real traffic performance.
    """

    sidecar = source.with_suffix(source.suffix + ".meta.json")
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        provenance = payload.get("provenance", payload)
        kind = provenance.get("source", {}).get("kind") if isinstance(provenance, dict) else None
        if kind in {"real", "simulated"}:
            return kind == "simulated"

    if source.parent.name == "sample":
        return True
    return DEFAULT_SAMPLE_NAME in source.name


def collect_categorical_levels(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Record every category level the dataset contains.

    Written into the schema so inference can rebuild the same one-hot columns from
    a single row. Deriving levels from a per-batch frame instead would silently
    drop the dummies for any level that batch omits.

    The levels are read from the raw categorical column when it is present, and
    otherwise recovered from the encoded ``<source>__<level>`` columns, which is
    the case for ``weather_condition`` in the Stage 2 output.
    """

    from ml.preprocessing import SEGMENT_COLUMN, WEATHER_COLUMN

    levels: dict[str, tuple[str, ...]] = {}
    for column in (SEGMENT_COLUMN, WEATHER_COLUMN):
        if column in frame.columns:
            observed = {str(value) for value in frame[column].dropna().unique()}
        else:
            prefix = f"{column}__"
            observed = {
                name[len(prefix) :]
                for name in frame.columns
                if name.startswith(prefix)
            }
        if observed:
            levels[column] = tuple(sorted(observed))
    return levels


def assert_no_leakage_columns(feature_columns: Sequence[str]) -> None:
    """Fail loudly if a contemporaneous measurement became a model input.

    These columns determine the label, so including any of them would let the
    model read its own answer. Checked at training time as well as in the test
    suite, because this is the property most easily broken by a later change.

    Raises:
        DatasetError: if a leakage column is present.
    """

    offenders = sorted(set(feature_columns) & set(CONTEMPORANEOUS_MEASUREMENT_COLUMNS))
    if offenders:
        raise DatasetError(
            "Refusing to train: leakage column(s) present in the feature set: "
            + ", ".join(offenders)
            + ". These determine the target, so the model would score on its own "
            "answer. Remove them in ml.preprocessing."
        )


# ----------------------------------------------------------------- training


@dataclass
class TrainingResult:
    """Outcome of a training run, with only measured values."""

    model_name: str
    model_key: str
    artifact_path: Path
    preprocessor_path: Path
    metadata_path: Path
    feature_importance_path: Path
    training_rows: int
    test_rows: int
    feature_columns: tuple[str, ...]
    split_cutoff: str
    train_class_distribution: dict[str, int]
    test_class_distribution: dict[str, int]
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"model            : {self.model_name}",
            f"artifact         : {self.artifact_path}",
            f"preprocessor     : {self.preprocessor_path}",
            f"metadata         : {self.metadata_path}",
            f"features         : {len(self.feature_columns)}",
            f"training rows    : {self.training_rows}",
            f"test rows        : {self.test_rows}",
            f"split            : chronological at {self.split_cutoff}",
            f"train classes    : {self.train_class_distribution}",
            f"test classes     : {self.test_class_distribution}",
        ]
        return "\n".join(lines)


def train(
    data_path: str | Path | None = None,
    model_key: str = DEFAULT_MODEL_KEY,
    params: dict[str, Any] | None = None,
    test_ratio: float = DEFAULT_TEST_RATIO,
    models_dir: str | Path | None = None,
) -> TrainingResult:
    """Train the congestion classifier and persist every required artifact.

    Args:
        data_path: processed CSV; defaults to the Stage 2 output.
        model_key: key from :data:`MODEL_REGISTRY`.
        params: estimator overrides.
        test_ratio: fraction of rows held out, split chronologically.
        models_dir: output directory; defaults to ``ml/models``.

    Returns:
        A :class:`TrainingResult`. Metrics describe the *training* run only;
        authoritative evaluation is :func:`ml.evaluate.evaluate`.
    """

    frame, source = load_processed_dataset(data_path)
    simulated = detect_simulated_dataset(source)
    feature_columns = select_feature_columns(frame)
    if not feature_columns:
        raise DatasetError(
            f"{source} yielded no numeric feature columns. Check the Stage 2 "
            "output against ml/data/raw/README.md."
        )
    assert_no_leakage_columns(feature_columns)

    train_frame, test_frame = split_chronologically(frame, test_ratio=test_ratio)
    if train_frame.empty or test_frame.empty:
        raise DatasetError("Chronological split produced an empty side.")

    X_train = train_frame.loc[:, list(feature_columns)]
    y_train = train_frame[TARGET_COLUMN]
    X_test = test_frame.loc[:, list(feature_columns)]
    y_test = test_frame[TARGET_COLUMN]

    classes_in_train = sorted(int(value) for value in y_train.unique())
    missing_classes = sorted(set(int(v) for v in frame[TARGET_COLUMN].unique()) - set(classes_in_train))
    if missing_classes:
        raise DatasetError(
            f"The chronological training window contains no examples of class(es) "
            f"{missing_classes}. The dataset's time span does not cover all "
            "congestion levels early enough. Widen the date range or lower "
            "test_ratio."
        )

    estimator = build_estimator(model_key, params)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    # Scores on the training run are a sanity signal, not the reported result.
    from ml.evaluate import compute_classification_metrics

    metrics = compute_classification_metrics(y_test, predictions)

    schema = FeatureSchema(
        feature_columns=feature_columns,
        label_mapping={str(key): name for key, name in CONGESTION_NAMES.items()},
        target_column=TARGET_COLUMN,
        lag_steps=tuple(LAG_STEPS),
        rolling_windows=tuple(ROLLING_WINDOWS),
        one_hot_max_cardinality=ONE_HOT_MAX_CARDINALITY,
        required_input_columns=derive_required_inputs(feature_columns, frame.columns),
        min_history_per_segment=int(max(LAG_STEPS + ROLLING_WINDOWS)),
        categorical_levels=collect_categorical_levels(frame),
    )

    target_dir = Path(models_dir).expanduser().resolve() if models_dir else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / MODEL_FILENAME
    preprocessor_path = target_dir / PREPROCESSOR_FILENAME
    metadata_path = target_dir / METADATA_FILENAME
    importance_path = target_dir / FEATURE_IMPORTANCE_FILENAME

    joblib.dump(estimator, artifact_path)
    joblib.dump(schema, preprocessor_path)

    importances = _feature_importance(estimator, feature_columns)
    importance_payload = {
        "model_name": model_key,
        "trained_at": _utc_now(),
        "method": "mean decrease in impurity across all trees; "
        "sums to 1.0. Permutation importance on held-out data is the more "
        "trustworthy measure and is not computed here.",
        "features": importances,
    }
    importance_path.write_text(
        json.dumps(importance_payload, indent=2), encoding="utf-8"
    )

    metadata = {
        "model_name": model_key,
        "model_display_name": MODEL_REGISTRY[model_key].display_name,
        "model_version": MODEL_VERSION,
        "model_type": type(estimator).__name__,
        "trained_at": _utc_now(),
        "dataset_name": source.name,
        "dataset_path": str(source),
        "dataset_is_simulated": simulated,
        "dataset_rows_total": int(len(frame)),
        "training_records": int(len(train_frame)),
        "test_records": int(len(test_frame)),
        "features": list(feature_columns),
        "feature_count": len(feature_columns),
        "target_name": TARGET_COLUMN,
        "target_classes": len(classes_in_train),
        "label_mapping": schema.label_mapping,
        "split_strategy": "chronological",
        "split_ratio": test_ratio,
        "split_cutoff": str(train_frame[TIMESTAMP_COLUMN].max()),
        "split_cutoff_epoch": pd.Timestamp(train_frame[TIMESTAMP_COLUMN].max()).timestamp(),
        "train_time_range": [
            str(train_frame[TIMESTAMP_COLUMN].min()),
            str(train_frame[TIMESTAMP_COLUMN].max()),
        ],
        "test_time_range": [
            str(test_frame[TIMESTAMP_COLUMN].min()),
            str(test_frame[TIMESTAMP_COLUMN].max()),
        ],
        "train_class_distribution": _distribution(y_train),
        "test_class_distribution": _distribution(y_test),
        # Native JSON types are kept where possible; only values JSON cannot
        # represent are stringified, so a reader can compare n_estimators to 300
        # as a number instead of parsing "300".
        "hyperparameters": {
            key: value if isinstance(value, (int, float, bool, type(None), str)) else str(value)
            for key, value in estimator.get_params().items()
            if key in {"n_estimators", "max_depth", "min_samples_leaf", "max_features",
                       "class_weight", "n_jobs", "random_state", "criterion",
                       "max_iter", "solver"}
        },
        "required_input_columns": list(schema.required_input_columns),
        "min_history_per_segment": schema.min_history_per_segment,
        "sklearn_version": sklearn.__version__,
        "ml_package_version": ml_package_version,
        "training_run_metrics": metrics,
        "artifacts": {
            "model": artifact_path.name,
            "preprocessor": preprocessor_path.name,
            "metadata": metadata_path.name,
            "feature_importance": importance_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingResult(
        model_name=model_key,
        model_key=model_key,
        artifact_path=artifact_path,
        preprocessor_path=preprocessor_path,
        metadata_path=metadata_path,
        feature_importance_path=importance_path,
        training_rows=int(len(train_frame)),
        test_rows=int(len(test_frame)),
        feature_columns=feature_columns,
        split_cutoff=str(train_frame[TIMESTAMP_COLUMN].max()),
        train_class_distribution=_distribution(y_train),
        test_class_distribution=_distribution(y_test),
        metrics=metrics,
    )


# ------------------------------------------------------------------ loading


def load_model(models_dir: str | Path | None = None) -> Any:
    """Load the fitted estimator.

    Raises:
        ModelArtifactNotFoundError: if no model has been trained yet.
    """

    path = _models_dir(models_dir) / MODEL_FILENAME
    if not path.is_file():
        raise ModelArtifactNotFoundError(
            f"No trained model found at {path}. Train one with 'python -m ml.train'."
        )
    return joblib.load(path)


def load_preprocessor(models_dir: str | Path | None = None) -> FeatureSchema:
    """Load the :class:`FeatureSchema` saved alongside the model.

    Raises:
        ModelArtifactNotFoundError: if the schema has not been written.
    """

    path = _models_dir(models_dir) / PREPROCESSOR_FILENAME
    if not path.is_file():
        raise ModelArtifactNotFoundError(
            f"No preprocessor at {path}. Retrain with 'python -m ml.train'."
        )
    schema = joblib.load(path)
    if not isinstance(schema, FeatureSchema):
        raise ModelArtifactNotFoundError(
            f"{path} does not contain a FeatureSchema (got {type(schema).__name__}). "
            "Retrain with 'python -m ml.train'."
        )
    return schema


def load_metadata(models_dir: str | Path | None = None) -> dict[str, Any]:
    """Load ``model_metadata.json``.

    Raises:
        ModelArtifactNotFoundError: if the metadata has not been written.
    """

    path = _models_dir(models_dir) / METADATA_FILENAME
    if not path.is_file():
        raise ModelArtifactNotFoundError(
            f"No model metadata at {path}. Train with 'python -m ml.train'."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def model_info(models_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a JSON-safe description of the trained model, without the estimator."""

    metadata = load_metadata(models_dir)
    return {key: value for key, value in metadata.items() if key != "estimator"}


def load_artifact(artifact_path: str | Path | None = None) -> dict[str, Any]:
    """Stage 1 compatibility wrapper returning the model bundle as a mapping.

    Stage 3 split the bundle into ``traffic_model.pkl`` (estimator),
    ``preprocessor.pkl`` (:class:`FeatureSchema`) and ``model_metadata.json``, and
    :func:`load_model` / :func:`load_preprocessor` / :func:`load_metadata` are the
    real entry points. Existing callers still expect a single dictionary, so this
    reassembles it rather than leaving them broken.

    Args:
        artifact_path: a Stage 1-style path such as ``traffic_model.joblib``. Its
            parent directory is used as the artifact directory; ``None`` uses the
            default.

    Raises:
        ModelArtifactNotFoundError: if any part of the bundle is missing.
    """

    models_dir = Path(artifact_path).expanduser().resolve().parent if artifact_path else None
    return {
        "estimator": load_model(models_dir),
        "preprocessor": load_preprocessor(models_dir),
        "metadata": load_metadata(models_dir),
    }


def _models_dir(models_dir: str | Path | None) -> Path:
    return Path(models_dir).expanduser().resolve() if models_dir else MODELS_DIR


# ------------------------------------------------------------------ helpers


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _distribution(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _feature_importance(estimator: Any, feature_columns: Sequence[str]) -> list[dict[str, Any]]:
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return []
    ranked = sorted(
        zip(feature_columns, (float(value) for value in values), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [
        {"rank": index, "feature": name, "importance": round(value, 6)}
        for index, (name, value) in enumerate(ranked, start=1)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the congestion classifier on the Stage 2 dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Processed dataset CSV (default: ml/data/processed/traffic_processed.csv).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        choices=sorted(MODEL_REGISTRY),
        help="Estimator to train.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Fraction of the latest data held out for testing.",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Output directory for artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        result = train(
            data_path=args.data,
            model_key=args.model,
            test_ratio=args.test_ratio,
            models_dir=args.models_dir,
        )
    except DatasetError as error:
        print(f"error: {error}")
        return 1

    print(result.summary())
    print(
        "\nRun 'python -m ml.evaluate' for the authoritative held-out metrics."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())