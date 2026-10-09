"""Stage 3: the trained-model contract.

These tests train a real model once per session on the simulated fixture and then
check the things that would silently make the reported numbers wrong:

* the chronological split is reproduced exactly, so evaluation cannot score rows
  the model already saw;
* no contemporaneous measurement leaks into the feature set;
* inference rebuilds the feature set from raw records alone, and produces
  probabilities that actually come from the estimator;
* the artifact set on disk is complete and self-consistent.

Nothing here asserts a specific accuracy value. The fixture is synthetic, so a
hard-coded score would only assert that the fixture did not change; it would not
test anything about the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml import evaluate as evaluate_module
from ml import predict as predict_module
from ml import preprocessing, train
from ml.data.sample.generate_simulated_dataset import generate
from ml.preprocessing import (
    CONTEMPORANEOUS_MEASUREMENT_COLUMNS,
    SEGMENT_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    DatasetError,
    MissingColumnsError,
)

REQUIRED_ARTIFACTS = (
    train.MODEL_FILENAME,
    train.PREPROCESSOR_FILENAME,
    train.METADATA_FILENAME,
    train.FEATURE_IMPORTANCE_FILENAME,
    evaluate_module.RESULTS_FILENAME,
    evaluate_module.CONFUSION_MATRIX_FILENAME,
    evaluate_module.REPORT_FILENAME,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train and evaluate once into a temporary artifact directory."""

    models_dir = tmp_path_factory.mktemp("stage3_models")
    train.train(models_dir=models_dir)
    evaluate_module.evaluate(models_dir=models_dir)
    return models_dir


@pytest.fixture(scope="module")
def metadata(trained: Path) -> dict:
    return train.load_metadata(trained)


@pytest.fixture(scope="module")
def processed_path() -> Path:
    return preprocessing.DATA_PROCESSED_DIR / preprocessing.DEFAULT_PROCESSED_NAME


@pytest.fixture(scope="module")
def processed(processed_path: Path) -> pd.DataFrame:
    return pd.read_csv(processed_path)


# --------------------------------------------------------------------- training


def test_training_writes_every_required_artifact(trained: Path) -> None:
    for filename in REQUIRED_ARTIFACTS:
        path = trained / filename
        assert path.is_file(), f"missing artifact: {filename}"
        assert path.stat().st_size > 0, f"empty artifact: {filename}"


def test_split_is_chronological_and_disjoint(metadata: dict) -> None:
    cutoff = pd.Timestamp(metadata["split_cutoff"])
    assert metadata["split_strategy"] == "chronological"
    assert pd.Timestamp(metadata["train_time_range"][1]) <= cutoff
    assert pd.Timestamp(metadata["test_time_range"][0]) > cutoff


def test_evaluation_scores_exactly_the_held_out_rows(
    trained: Path, metadata: dict, processed_path: Path
) -> None:
    """The evaluated window must match training's, row for row.

    Reproducing this wrongly is easy and quiet: an inclusive/exclusive slip at the
    boundary moves rows between the two sets and changes every reported number
    without raising anything.
    """

    results = json.loads(
        (trained / evaluate_module.RESULTS_FILENAME).read_text(encoding="utf-8")
    )
    frame = pd.read_csv(processed_path)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN])
    cutoff = pd.Timestamp(metadata["split_cutoff"])

    train_rows = int((frame[TIMESTAMP_COLUMN] <= cutoff).sum())
    test_rows = int((frame[TIMESTAMP_COLUMN] > cutoff).sum())

    assert (train_rows, test_rows) == (
        metadata["training_records"],
        metadata["test_records"],
    )
    assert results["dataset"]["training_records"] == train_rows
    assert results["dataset"]["test_records"] == test_rows


def test_metrics_are_measured_not_placeholder(metadata: dict) -> None:
    """Metrics must come from a real score() call, not hard-coded defaults."""

    results = json.loads(
        (Path(train.MODELS_DIR) / evaluate_module.RESULTS_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    metrics = results["metrics"]

    for key in ("accuracy", "balanced_accuracy", "f1_macro", "precision_macro"):
        value = metrics[key]
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    # Per-class support must add up to the test set, which only happens when the
    # scores were computed on real predictions.
    per_class = metrics["per_class"]
    assert sum(int(entry["support"]) for entry in per_class.values()) == results["dataset"][
        "test_records"
    ]
    assert metrics["classification_report"].strip()


def test_feature_set_excludes_contemporaneous_measurements(metadata: dict) -> None:
    features = set(metadata["features"])
    assert not features & set(CONTEMPORANEOUS_MEASUREMENT_COLUMNS)
    assert train.assert_no_leakage_columns(metadata["features"]) is None


def test_training_refuses_a_leakage_feature_set() -> None:
    with pytest.raises(DatasetError, match="leakage"):
        train.assert_no_leakage_columns(
            ["hour", "avg_speed_kph", "speed_ratio_lag_1"]
        )


def test_target_is_excluded_from_the_feature_set(metadata: dict) -> None:
    assert TARGET_COLUMN not in metadata["features"]
    assert SEGMENT_COLUMN not in metadata["features"]
    assert TIMESTAMP_COLUMN not in metadata["features"]


def test_baseline_majority_class_comes_from_training_not_test() -> None:
    """Choosing the baseline from test labels would leak test information."""

    y_train = [0, 0, 0, 0, 1, 1, 2, 3]
    y_test = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    baseline = evaluate_module.compute_majority_baseline(y_test, y_train)
    assert baseline["majority_class"] == 0, "must follow the training majority"

    # The test set's own majority is 2. Using it would score a different constant.
    assert evaluate_module.compute_majority_baseline(y_test, y_test)[
        "majority_class"
    ] == 2
    assert baseline["accuracy"] == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------------- artifacts


def test_metadata_records_provenance_and_hyperparameters(metadata: dict) -> None:
    assert metadata["dataset_is_simulated"] is True, (
        "the fixture is synthetic and the metadata must say so"
    )
    assert metadata["feature_count"] == len(metadata["features"]) == 26
    assert not any(name.startswith("flow_veh_per_hr") for name in metadata["features"]), (
        "vehicle throughput was removed as a feature in 4.0.0 because no "
        "self-serve traffic API publishes it"
    )
    assert "flow_veh_per_hr" not in metadata["required_input_columns"]
    assert metadata["model_type"] == "RandomForestClassifier"
    assert metadata["hyperparameters"]["random_state"] == preprocessing.RANDOM_STATE
    assert metadata["label_mapping"]["1"] == "moderate"


def test_preprocessor_round_trips_through_a_plain_import(trained: Path) -> None:
    """The schema must unpickle without importing ml.train as __main__.

    Training runs as ``python -m ml.train``, so a schema class defined in that
    module is recorded as ``__main__.FeatureSchema`` and cannot be loaded back.
    """

    schema = train.load_preprocessor(trained)
    assert schema.feature_columns
    assert schema.min_history_per_segment >= max(schema.rolling_windows)
    assert schema.categorical_levels


def test_feature_importance_is_ranked_and_normalised(trained: Path) -> None:
    payload = json.loads(
        (trained / train.FEATURE_IMPORTANCE_FILENAME).read_text(encoding="utf-8")
    )
    entries = payload["features"]
    assert [entry["rank"] for entry in entries] == list(range(1, len(entries) + 1))
    assert all(entry["importance"] >= 0 for entry in entries)
    assert pytest.approx(sum(e["importance"] for e in entries), abs=1e-3) == 1.0


def test_report_discloses_simulated_data_and_matches_json(trained: Path) -> None:
    report = (trained / evaluate_module.REPORT_FILENAME).read_text(encoding="utf-8")
    results = json.loads(
        (trained / evaluate_module.RESULTS_FILENAME).read_text(encoding="utf-8")
    )

    assert "simulated" in report.lower()
    assert f"{results['metrics']['f1_macro']:.4f}" in report
    assert f"{results['metrics']['accuracy']:.4f}" in report
    assert results["results_path"].endswith(evaluate_module.RESULTS_FILENAME)


def test_confusion_matrix_image_is_a_real_png(trained: Path) -> None:
    image = (trained / evaluate_module.CONFUSION_MATRIX_FILENAME).read_bytes()
    assert image[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"


# ------------------------------------------------------------------ prediction


def _history(rows: int = 8) -> pd.DataFrame:
    """Raw observations for one road segment, enough to build every lag."""

    frame = generate()
    segment = str(frame[SEGMENT_COLUMN].iloc[0])
    return (
        frame[frame[SEGMENT_COLUMN].astype(str) == segment]
        .sort_values(TIMESTAMP_COLUMN)
        .head(rows)
        .reset_index(drop=True)
    )


#: Enough rows that some are scoreable once the warm-up rows are removed.
SCORABLE_ROWS = 12


def test_prediction_returns_a_structured_result(trained: Path) -> None:
    schema = train.load_preprocessor(trained)
    result = predict_module.predict_traffic(
        _history(SCORABLE_ROWS), models_dir=trained
    )

    expected = SCORABLE_ROWS - schema.min_history_per_segment
    assert result["records_predicted"] == expected
    assert result["predicted_congestion_level"] in {0, 1, 2, 3}
    assert result["predicted_congestion"] in {
        "free_flow", "moderate", "heavy", "severe"
    }
    assert isinstance(result["predicted_congestion_level"], int)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_available"] is True
    assert len(result["predictions"]) == expected


def test_confidence_comes_from_the_estimator_probabilities(trained: Path) -> None:
    """Confidence must equal the largest predicted probability, not a constant."""

    estimator = train.load_model(trained)
    schema = train.load_preprocessor(trained)
    records = _history()

    prepared = schema.transform(records)
    scorable = prepared.notna().all(axis=1)
    expected = estimator.predict_proba(prepared.loc[scorable]).max(axis=1)

    result = predict_module.predict_traffic(
        records, models_dir=trained, include_probabilities=False
    )
    reported = np.array([row["confidence"] for row in result["predictions"]])

    np.testing.assert_allclose(reported, np.round(expected, 6))


def test_class_probabilities_form_a_distribution(trained: Path) -> None:
    result = predict_module.predict_traffic(_history(SCORABLE_ROWS), models_dir=trained)
    for row in result["predictions"]:
        probabilities = row["class_probabilities"]
        assert len(probabilities) == 4
        assert all(0.0 <= value <= 1.0 for value in probabilities.values())
        assert pytest.approx(sum(probabilities.values()), abs=1e-4) == 1.0


def test_probabilities_can_be_omitted(trained: Path) -> None:
    result = predict_module.predict_traffic(
        _history(SCORABLE_ROWS), models_dir=trained, include_probabilities=False
    )
    assert "class_probabilities" not in result["predictions"][0]
    assert "confidence" in result["predictions"][0]


def test_records_may_be_supplied_as_plain_dictionaries(trained: Path) -> None:
    records = _history(SCORABLE_ROWS).to_dict(orient="records")
    result = predict_module.predict_traffic(records, models_dir=trained)
    assert result["records_received"] == SCORABLE_ROWS
    assert result["records_predicted"] > 0


def test_inference_rebuilds_features_without_precomputed_inputs(
    trained: Path, metadata: dict
) -> None:
    """A single raw record must still produce the model's full column set.

    This is the case a naive implementation gets wrong: one-hot encoding a
    one-row frame emits only the dummies that row happens to contain, leaving the
    fitted model's other expected columns missing.
    """

    schema = train.load_preprocessor(trained)
    prepared = schema.transform(_history().head(1))

    assert list(prepared.columns) == list(metadata["features"])
    # The one-hot columns for absent levels must exist and be zero, not missing.
    assert prepared[list(metadata["features"])].shape[1] == 26


def test_features_are_fully_populated_once_history_exists(trained: Path) -> None:
    schema = train.load_preprocessor(trained)
    prepared = schema.transform(_history(SCORABLE_ROWS)).tail(1)
    assert not prepared.isna().any().any()


def test_inference_rejects_a_missing_required_column(trained: Path) -> None:
    records = _history().drop(columns=[SEGMENT_COLUMN])
    with pytest.raises(MissingColumnsError, match=SEGMENT_COLUMN):
        predict_module.predict_traffic(records, models_dir=trained)


def test_inference_rejects_an_unparsable_timestamp(trained: Path) -> None:
    records = _history()
    records[SEGMENT_COLUMN] = records[SEGMENT_COLUMN].astype(str)
    records.loc[0, TIMESTAMP_COLUMN] = "not-a-time"
    with pytest.raises(DatasetError, match=TIMESTAMP_COLUMN):
        predict_module.predict_traffic(records, models_dir=trained)


def test_inference_rejects_an_empty_input(trained: Path) -> None:
    with pytest.raises(predict_module.PredictionInputError):
        predict_module.predict_traffic(pd.DataFrame(), models_dir=trained)


def test_insufficient_history_is_reported_not_silently_scored(trained: Path) -> None:
    schema = train.load_preprocessor(trained)
    rows = 8
    result = predict_module.predict_traffic(
        _history(rows), models_dir=trained, include_probabilities=False
    )
    # A row needs one prior observation per lag step and a full rolling window, so
    # the first min_history_per_segment rows are unscoreable. They must be counted
    # as skipped, not filled in with imputed values and scored anyway.
    expected_skipped = min(schema.min_history_per_segment, rows - 1)
    assert result["records_skipped_insufficient_history"] == expected_skipped
    assert result["records_predicted"] == rows - expected_skipped
    assert result["records_received"] == rows


def test_insufficient_history_can_be_made_fatal(trained: Path) -> None:
    with pytest.raises(predict_module.PredictionInputError, match="history"):
        predict_module.predict_traffic(
            _history(8), models_dir=trained, drop_incomplete_history=False
        )


def test_prediction_without_an_artifact_raises() -> None:
    with pytest.raises(train.ModelArtifactNotFoundError):
        predict_module.predict_traffic(_history(), models_dir=Path("no/such/dir"))


def test_schema_required_inputs_exclude_the_target_quantity(trained: Path) -> None:
    """speed_ratio determines the label, so a caller must not supply it."""

    schema = train.load_preprocessor(trained)
    assert "speed_ratio" not in schema.required_input_columns
    assert "avg_speed_kph" in schema.required_input_columns
    assert "free_flow_speed_kph" in schema.required_input_columns


def test_legacy_frame_api_still_returns_a_frame(trained: Path) -> None:
    frame = predict_module.predict(_history(), artifact_path=trained / train.MODEL_FILENAME)
    assert len(frame) == 2
    assert predict_module.PREDICTED_CLASS_KEY in frame.columns
    assert predict_module.CONFIDENCE_KEY in frame.columns