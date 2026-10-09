"""Tests for the ML module's data contract and its refusal to fabricate results.

Phase 1 ships no dataset and no model artifact, so these tests assert two things:
the documented schema helpers behave correctly, and every entry point fails
loudly rather than silently producing invented output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml import evaluate, predict, preprocessing, train


# --------------------------------------------------------------- data contract


def test_required_columns_are_declared() -> None:
    assert preprocessing.REQUIRED_COLUMNS == (
        "timestamp",
        "road_segment_id",
        "avg_speed_kph",
    )


def test_target_column_name() -> None:
    assert preprocessing.TARGET_COLUMN == "congestion_level"


def test_feature_columns_include_calendar_features() -> None:
    assert "hour" in preprocessing.FEATURE_COLUMNS
    assert "day_of_week" in preprocessing.FEATURE_COLUMNS
    assert "is_weekend" in preprocessing.FEATURE_COLUMNS


def test_contemporaneous_measurements_are_not_features() -> None:
    """The label is derived from these columns, so they must never be inputs."""

    for column in (
        "avg_speed_kph",
        "speed_ratio",
        "flow_veh_per_hr",
        "occupancy_pct",
    ):
        assert column in preprocessing.CONTEMPORANEOUS_MEASUREMENT_COLUMNS
        assert column not in preprocessing.FEATURE_COLUMNS

    assert preprocessing.SPEED_COLUMN in preprocessing.CONTEMPORANEOUS_MEASUREMENT_COLUMNS
    assert preprocessing.SPEED_RATIO_COLUMN in preprocessing.CONTEMPORANEOUS_MEASUREMENT_COLUMNS
    assert preprocessing.FLOW_COLUMN in preprocessing.CONTEMPORANEOUS_MEASUREMENT_COLUMNS
    assert preprocessing.OCCUPANCY_COLUMN in preprocessing.CONTEMPORANEOUS_MEASUREMENT_COLUMNS


def test_every_congestion_class_has_a_label() -> None:
    assert sorted(preprocessing.CONGESTION_NAMES) == [0, 1, 2, 3]
    assert all(preprocessing.CONGESTION_NAMES.values())


def test_congestion_thresholds_are_strictly_increasing() -> None:
    bins = preprocessing.CONGESTION_RATIO_BINS

    assert len(bins) == len(set(bins))
    assert list(bins) == sorted(bins)


@pytest.mark.parametrize(
    ("speed_ratio", "expected_level"),
    [
        (0.90, 0),  # free flow
        (0.75, 0),  # boundary is inclusive at the top band
        (0.74, 1),
        (0.40, 1),
        (0.39, 2),
        (0.15, 2),
        (0.14, 3),
    ],
)
def test_congestion_level_derivation_boundaries(
    speed_ratio: float, expected_level: int
) -> None:
    """Unit test of the band mapping itself, using boundary ratios only."""

    frame = pd.DataFrame({preprocessing.SPEED_RATIO_COLUMN: [speed_ratio]})

    result = preprocessing.derive_congestion_level(frame)

    assert int(result[preprocessing.TARGET_COLUMN].iloc[0]) == expected_level


def test_missing_ratio_and_free_flow_columns_are_rejected() -> None:
    frame = pd.DataFrame(
        {preprocessing.SPEED_COLUMN: [40.0], preprocessing.TARGET_COLUMN: [1]}
    )

    with pytest.raises(preprocessing.MissingColumnsError):
        preprocessing.add_speed_ratio(frame)


def test_validate_columns_reports_every_missing_name() -> None:
    frame = pd.DataFrame({"timestamp": ["2026-01-01 08:00:00"]})

    with pytest.raises(preprocessing.MissingColumnsError) as error:
        preprocessing.ensure_columns(frame, preprocessing.REQUIRED_COLUMNS)

    assert "road_segment_id" in str(error.value)
    assert "avg_speed_kph" in str(error.value)


# ------------------------------------------------- refusal to invent results


def test_loading_a_missing_dataset_raises_with_guidance() -> None:
    with pytest.raises(preprocessing.DatasetNotFoundError) as error:
        preprocessing.load_data("definitely_not_a_real_file.csv")

    assert "does not ship real traffic data" in str(error.value)


def test_training_without_a_dataset_raises() -> None:
    with pytest.raises(preprocessing.DatasetError):
        train.train(data_path="definitely_not_a_real_file.csv")


def test_ml_entry_points_remain_importable_after_stage_two() -> None:
    """Stage 2 added functions; it must not remove names the other modules use."""

    for name in (
        "build_supervised_frame",
        "build_feature_matrix",
        "class_distribution",
        "lag_feature_columns",
        "prepare_features_and_target",
        "split_chronologically",
    ):
        assert hasattr(preprocessing, name), name


def test_loading_a_missing_artifact_raises() -> None:
    with pytest.raises(train.ModelArtifactNotFoundError) as error:
        train.load_artifact("definitely_not_a_real_model.joblib")

    assert "No trained model found" in str(error.value)


def test_evaluation_without_an_artifact_raises() -> None:
    with pytest.raises(train.ModelArtifactNotFoundError):
        evaluate.evaluate(artifact_path="definitely_not_a_real_model.joblib")


def test_prediction_without_an_artifact_raises() -> None:
    records = pd.DataFrame(
        {
            "timestamp": ["2026-01-05 08:00:00"],
            "road_segment_id": ["SEG-1"],
            "avg_speed_kph": [42.0],
            "free_flow_speed_kph": [60.0],
        }
    )

    with pytest.raises(train.ModelArtifactNotFoundError):
        predict.predict(records, artifact_path="definitely_not_a_real_model.joblib")


def test_predict_rejects_records_without_required_columns() -> None:
    records = pd.DataFrame({"avg_speed_kph": [42.0]})

    with pytest.raises(preprocessing.MissingColumnsError):
        predict.prepare_records(records)


def test_predict_rejects_unparsable_timestamps() -> None:
    records = pd.DataFrame(
        {
            "timestamp": ["not-a-time"],
            "road_segment_id": ["SEG-1"],
            "avg_speed_kph": [42.0],
            "free_flow_speed_kph": [60.0],
        }
    )

    with pytest.raises(preprocessing.MissingColumnsError) as error:
        predict.prepare_records(records)

    assert "timestamp" in str(error.value)


def test_predict_rejects_an_empty_frame() -> None:
    records = pd.DataFrame(columns=["timestamp", "road_segment_id"])

    with pytest.raises(preprocessing.MissingColumnsError):
        predict.prepare_records(records)


def test_predict_builds_the_same_features_as_training() -> None:
    """Inference must reuse create_features, not a drifting copy of it."""

    records = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05", periods=8, freq="60min"),
            "road_segment_id": ["SEG-1"] * 8,
            "avg_speed_kph": [40.0] * 8,
            "free_flow_speed_kph": [60.0] * 8,
            "flow_veh_per_hr": [900.0] * 8,
            "weather_condition": ["clear"] * 8,
            # Pass-through model inputs. These are required by the feature schema:
            # the trained model uses them directly, so a raw record without them
            # cannot be scored.
            "temperature_c": [18.0] * 8,
            "precipitation_mm": [0.0] * 8,
        }
    )

    prepared = predict.prepare_records(records)
    _, _, feature_columns, _ = preprocessing.prepare_features_and_target(records)

    # Every training feature must be producible from raw records.
    assert set(feature_columns) <= set(prepared.columns)
    # And the rolling means that only exist in the shared builder must be present.
    assert "speed_ratio_rolling_mean_6" in prepared.columns


def test_ml_package_does_not_import_the_web_layer() -> None:
    """The ML module must stay independent of the FastAPI application."""

    for module in (preprocessing, train, evaluate, predict):
        source_names = {
            name
            for name in dir(module)
            if not name.startswith("_")
        }
        assert "app" not in source_names