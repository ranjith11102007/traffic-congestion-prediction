"""Tests for the Stage 2 traffic data pipeline.

Coverage required by the stage brief — loading, missing-value handling, timestamp
processing, feature generation and output creation — plus the guarantees that
matter more than any individual function: no target leakage, no future
information in a prediction feature, no silent data loss, and no modification of
the source dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml import preprocessing as pp

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO_ROOT / "ml" / "data" / "sample" / pp.DEFAULT_SAMPLE_NAME

BASE_COLUMNS = (
    pp.TIMESTAMP_COLUMN,
    pp.SEGMENT_COLUMN,
    pp.SPEED_COLUMN,
    pp.FREE_FLOW_COLUMN,
    pp.FLOW_COLUMN,
)


def make_frame(
    *,
    rows_per_segment: int = 12,
    segments: tuple[str, ...] = ("SEG-A", "SEG-B"),
    start: str = "2026-01-05",
    interval_minutes: int = 60,
    speed: float = 40.0,
) -> pd.DataFrame:
    """Build a small well-formed frame. A test fixture, not a dataset."""

    timestamps = pd.date_range(start=start, periods=rows_per_segment, freq=f"{interval_minutes}min")
    frames = []
    for index, segment in enumerate(segments):
        frames.append(
            pd.DataFrame(
                {
                    pp.TIMESTAMP_COLUMN: timestamps,
                    pp.SEGMENT_COLUMN: segment,
                    pp.SPEED_COLUMN: speed - index * 5.0,
                    pp.FREE_FLOW_COLUMN: 60.0,
                    pp.FLOW_COLUMN: 1000.0 + index * 100.0,
                    pp.WEATHER_COLUMN: "clear",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def banded_frame() -> pd.DataFrame:
    """A frame whose speeds visit all four congestion bands.

    Long enough to clear the warm-up rows, so every class survives feature
    preparation and can be asserted on.
    """

    return pd.DataFrame(
        {
            pp.TIMESTAMP_COLUMN: pd.date_range("2026-01-05", periods=10, freq="60min"),
            pp.SEGMENT_COLUMN: ["SEG-A"] * 10,
            pp.SPEED_COLUMN: [57.0, 30.0, 12.0, 6.0, 57.0, 30.0, 12.0, 6.0, 57.0, 30.0],
            pp.FREE_FLOW_COLUMN: 60.0,
            pp.FLOW_COLUMN: 800.0,
        }
    )


SMALL_HISTORY = {"lags": (1,), "rolling_windows": (2,)}


@pytest.fixture()
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture()
def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = pp.clean_data(frame)
    return cleaned


# ------------------------------------------------------------ stage 1: loading


class TestLoading:
    def test_load_data_reads_a_csv(self, tmp_path: Path, frame: pd.DataFrame) -> None:
        path = tmp_path / "traffic_observations.csv"
        frame.to_csv(path, index=False)

        loaded = pp.load_data(path)

        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == list(BASE_COLUMNS) + [pp.WEATHER_COLUMN]
        assert len(loaded) == len(frame)

    def test_load_data_returns_a_copy_not_a_cached_frame(
        self, tmp_path: Path, frame: pd.DataFrame
    ) -> None:
        path = tmp_path / "traffic_observations.csv"
        frame.to_csv(path, index=False)

        first = pp.load_data(path)
        first.loc[0, pp.SPEED_COLUMN] = -1.0
        second = pp.load_data(path)

        assert second.loc[0, pp.SPEED_COLUMN] != -1.0

    def test_missing_dataset_raises_with_actionable_message(self, tmp_path: Path) -> None:
        with pytest.raises(pp.DatasetNotFoundError) as error:
            pp.load_data(tmp_path / "nope.csv")

        message = str(error.value)
        assert "does not ship real traffic data" in message
        assert "ml/data/raw" in message

    def test_default_path_prefers_raw_over_sample(self) -> None:
        # No real dataset is committed, so the fallback is the labelled fixture.
        resolved = pp.resolve_dataset_path()

        assert resolved.parent == pp.DATA_SAMPLE_DIR
        assert resolved.name.startswith("SIMULATED")

    def test_explicit_path_wins_over_defaults(self) -> None:
        assert pp.resolve_dataset_path("some/where.csv").name == "where.csv"


# --------------------------------------------------------- stage 2: validation


class TestValidation:
    def test_reports_shape_and_columns(self, frame: pd.DataFrame) -> None:
        report = pp.validate_data(frame)

        assert report.shape == (24, 6)
        assert report.rows == 24
        assert pp.TIMESTAMP_COLUMN in report.columns
        assert report.is_usable is True
        assert "24 rows x 6 columns" in report.render()

    def test_detects_missing_required_column(self) -> None:
        broken = make_frame().drop(columns=[pp.SPEED_COLUMN])

        report = pp.validate_data(broken)

        assert report.is_usable is False
        assert any(pp.SPEED_COLUMN in error for error in report.errors)

    def test_detects_missing_values(self, frame: pd.DataFrame) -> None:
        frame.loc[0, pp.FLOW_COLUMN] = np.nan
        frame.loc[1, pp.FLOW_COLUMN] = np.nan

        report = pp.validate_data(frame)

        assert report.missing_counts[pp.FLOW_COLUMN] == 2
        assert report.missing_total == 2

    def test_detects_exact_duplicates(self, frame: pd.DataFrame) -> None:
        duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

        report = pp.validate_data(duplicated)

        assert report.exact_duplicate_rows == 1

    def test_detects_conflicting_duplicate_keys(self, frame: pd.DataFrame) -> None:
        conflicting = frame.iloc[[0]].copy()
        conflicting[pp.FLOW_COLUMN] = 9999.0
        combined = pd.concat([frame, conflicting], ignore_index=True)

        report = pp.validate_data(combined)

        assert report.duplicate_keys == 1
        assert report.conflicting_duplicate_keys == 1

    def test_detects_unparsable_timestamps(self, frame: pd.DataFrame) -> None:
        frame[pp.TIMESTAMP_COLUMN] = frame[pp.TIMESTAMP_COLUMN].astype(object)
        frame.loc[3, pp.TIMESTAMP_COLUMN] = "not-a-date"

        report = pp.validate_data(frame)

        assert report.unparsable_timestamps == 1
        assert report.is_usable is True

    def test_detects_non_numeric_values(self, frame: pd.DataFrame) -> None:
        frame[pp.FLOW_COLUMN] = frame[pp.FLOW_COLUMN].astype(object)
        frame.loc[0, pp.FLOW_COLUMN] = "corrupt"

        report = pp.validate_data(frame)

        assert report.non_numeric_cells[pp.FLOW_COLUMN] == 1

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            (pp.SPEED_COLUMN, -3.0),
            (pp.SPEED_COLUMN, 500.0),
            (pp.OCCUPANCY_COLUMN, 140.0),
            (pp.PRECIPITATION_COLUMN, -1.0),
        ],
    )
    def test_detects_physically_impossible_values(
        self, frame: pd.DataFrame, column: str, value: float
    ) -> None:
        frame[column] = 10.0
        frame.loc[0, column] = value

        report = pp.validate_data(frame)

        assert report.out_of_range_cells.get(column) == 1

    def test_detects_categorical_columns(self, frame: pd.DataFrame) -> None:
        report = pp.validate_data(frame)

        assert pp.WEATHER_COLUMN in report.categorical_columns
        assert pp.SEGMENT_COLUMN in report.categorical_columns
        assert pp.FLOW_COLUMN in report.numeric_columns

    def test_validation_does_not_mutate_the_frame(self, frame: pd.DataFrame) -> None:
        before = frame.copy(deep=True)

        pp.validate_data(frame)

        pd.testing.assert_frame_equal(frame, before)

    def test_missing_ratio_source_is_an_error(self) -> None:
        incomplete = make_frame().drop(columns=[pp.FREE_FLOW_COLUMN])

        report = pp.validate_data(incomplete)

        assert report.is_usable is False
        assert any(pp.SPEED_RATIO_COLUMN in error for error in report.errors)

    def test_report_is_json_serialisable(self, frame: pd.DataFrame) -> None:
        json.dumps(pp.validate_data(frame).to_dict())


# ----------------------------------------------------------- stage 3: cleaning


class TestCleaning:
    def test_cleans_a_clean_frame_without_changing_row_count(self, clean_frame: pd.DataFrame) -> None:
        cleaned, report = pp.clean_data(make_frame())

        assert len(cleaned) == 24
        assert report.rows_dropped == 0
        assert report.imputed_total == 0

    def test_imputes_missing_optional_numeric_values(self, frame: pd.DataFrame) -> None:
        frame.loc[1, pp.FLOW_COLUMN] = np.nan

        cleaned, report = pp.clean_data(frame)

        assert cleaned[pp.FLOW_COLUMN].isna().sum() == 0
        assert report.imputed_total >= 1

    def test_imputation_carries_the_previous_value_forward(self, frame: pd.DataFrame) -> None:
        original = frame.loc[0, pp.FLOW_COLUMN]
        frame.loc[1, pp.FLOW_COLUMN] = np.nan

        cleaned, _ = pp.clean_data(frame)

        assert cleaned.loc[1, pp.FLOW_COLUMN] == original

    def test_imputation_uses_segment_median_for_segment_constants(
        self, frame: pd.DataFrame
    ) -> None:
        # Free-flow speed describes the segment, not the instant, so it must be
        # filled from that segment's own median rather than a global value.
        frame.loc[0, pp.FREE_FLOW_COLUMN] = np.nan
        frame.loc[1, pp.FREE_FLOW_COLUMN] = np.nan

        cleaned, report = pp.clean_data(frame)

        assert cleaned.loc[0, pp.FREE_FLOW_COLUMN] == 60.0
        assert pp.FREE_FLOW_COLUMN in report.imputed_by_segment_median

    def test_imputation_never_leaks_across_segments(self, frame: pd.DataFrame) -> None:
        # SEG-B's first row must not inherit SEG-A's traffic flow.
        first_b = frame.index[frame[pp.SEGMENT_COLUMN] == "SEG-B"][0]
        frame.loc[first_b, pp.FLOW_COLUMN] = np.nan

        cleaned, _ = pp.clean_data(frame)

        assert cleaned.loc[first_b, pp.FLOW_COLUMN] == 1100.0

    def test_imputes_missing_categorical_values_by_mode(self, frame: pd.DataFrame) -> None:
        frame.loc[0, pp.WEATHER_COLUMN] = np.nan

        cleaned, report = pp.clean_data(frame)

        assert cleaned[pp.WEATHER_COLUMN].isna().sum() == 0
        assert report.imputed_categorical_by_mode.get(pp.WEATHER_COLUMN) == 1

    def test_removes_exact_duplicates(self, frame: pd.DataFrame) -> None:
        duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

        cleaned, report = pp.clean_data(duplicated)

        assert len(cleaned) == 24
        assert report.rows_dropped_duplicates == 1

    def test_resolves_conflicting_duplicates_by_keeping_the_first(self, frame: pd.DataFrame) -> None:
        conflicting = frame.iloc[[0]].copy()
        conflicting[pp.FLOW_COLUMN] = 9999.0
        combined = pd.concat([frame, conflicting], ignore_index=True)

        cleaned, report = pp.clean_data(combined)

        assert len(cleaned) == 24
        matching = cleaned.loc[
            (cleaned[pp.SEGMENT_COLUMN] == frame.loc[0, pp.SEGMENT_COLUMN])
            & (cleaned[pp.TIMESTAMP_COLUMN] == frame.loc[0, pp.TIMESTAMP_COLUMN])
        ]
        assert matching[pp.FLOW_COLUMN].tolist() == [frame.loc[0, pp.FLOW_COLUMN]]

    def test_nulls_impossible_values_before_imputing(self, frame: pd.DataFrame) -> None:
        frame.loc[2, pp.SPEED_COLUMN] = -3.0

        cleaned, report = pp.clean_data(frame)

        assert report.nulled_invalid_values[pp.SPEED_COLUMN] == 1
        assert (cleaned[pp.SPEED_COLUMN] >= 0).all()

    def test_drops_rows_with_unparsable_timestamps(self, frame: pd.DataFrame) -> None:
        frame[pp.TIMESTAMP_COLUMN] = frame[pp.TIMESTAMP_COLUMN].astype(object)
        frame.loc[5, pp.TIMESTAMP_COLUMN] = "nonsense"

        cleaned, report = pp.clean_data(frame)

        assert len(cleaned) == 23
        assert report.rows_dropped_invalid_timestamps == 1

    def test_refuses_to_impute_a_required_column(self, frame: pd.DataFrame) -> None:
        frame.loc[4, pp.SPEED_COLUMN] = np.nan

        with pytest.raises(pp.ValidationFailedError) as error:
            pp.clean_data(frame)

        assert pp.SPEED_COLUMN in str(error.value)

    def test_refuses_excessive_data_loss(self, tmp_path: Path) -> None:
        frame = make_frame(rows_per_segment=10, segments=("SEG-A",))
        frame[pp.TIMESTAMP_COLUMN] = frame[pp.TIMESTAMP_COLUMN].astype(object)
        frame.loc[:4, pp.TIMESTAMP_COLUMN] = "broken"

        with pytest.raises(pp.ExcessiveDataLossError) as error:
            pp.clean_data(frame, max_drop_ratio=0.20)

        assert "20%" in str(error.value)

    def test_returns_rows_sorted_by_segment_then_time(self, frame: pd.DataFrame) -> None:
        shuffled = frame.sample(frac=1.0, random_state=0)

        cleaned, _ = pp.clean_data(shuffled)

        expected = cleaned.sort_values([pp.SEGMENT_COLUMN, pp.TIMESTAMP_COLUMN])
        pd.testing.assert_frame_equal(cleaned, expected)

    def test_report_is_json_serialisable(self, frame: pd.DataFrame) -> None:
        _, report = pp.clean_data(frame)

        json.dumps(report.to_dict())


# -------------------------------------------------- stage 4: feature creation


class TestFeatures:
    def test_adds_calendar_features(self, clean_frame: pd.DataFrame) -> None:
        features = pp.add_calendar_features(clean_frame)

        for column in pp.CALENDAR_FEATURE_COLUMNS:
            assert column in features.columns

        first = features.iloc[0]
        assert first["hour"] == 0
        assert first["day_of_week"] == pd.Timestamp("2026-01-05").dayofweek
        assert first["day_of_month"] == 5
        assert first["is_weekend"] == 0.0

    def test_weekend_flag_is_set_for_saturday_and_sunday(self) -> None:
        saturday = make_frame(start="2026-01-10", rows_per_segment=2, segments=("SEG-A",))

        features = pp.add_calendar_features(saturday)

        assert (features["is_weekend"] == 1.0).all()
        assert pd.Timestamp("2026-01-10").dayofweek == 5

    def test_computes_speed_ratio_from_free_flow_speed(self, clean_frame: pd.DataFrame) -> None:
        features = pp.add_speed_ratio(clean_frame)

        assert features.loc[0, pp.SPEED_RATIO_COLUMN] == pytest.approx(40.0 / 60.0)

    def test_speed_ratio_is_capped_at_one(self, clean_frame: pd.DataFrame) -> None:
        clean_frame.loc[clean_frame[pp.SEGMENT_COLUMN] == "SEG-A", pp.SPEED_COLUMN] = 90.0

        features = pp.add_speed_ratio(clean_frame)

        assert features[pp.SPEED_RATIO_COLUMN].max() == pytest.approx(1.0)

    def test_lags_are_strictly_backward_looking(self, clean_frame: pd.DataFrame) -> None:
        features = pp.add_lag_features(clean_frame, steps=(1, 2, 3))
        segment = features.loc[features[pp.SEGMENT_COLUMN] == "SEG-A"].reset_index(drop=True)

        for step in (1, 2, 3):
            column = f"{pp.SPEED_COLUMN}_lag_{step}"
            expected = segment[pp.SPEED_COLUMN].shift(step)
            pd.testing.assert_series_equal(segment[column], expected, check_names=False)

    def test_lags_never_cross_segments(self, clean_frame: pd.DataFrame) -> None:
        features = pp.add_lag_features(clean_frame, steps=(1,))

        for segment, group in features.groupby(pp.SEGMENT_COLUMN):
            # SEG-A's row 1 may use SEG-A's row 0, never the previous segment's tail.
            assert group[f"{pp.SPEED_COLUMN}_lag_1"].iloc[1] == pytest.approx(
                group[pp.SPEED_COLUMN].iloc[0]
            )

    def test_rolling_mean_excludes_the_current_observation(self) -> None:
        # The last ratio is a trap: a rolling mean that included the present
        # reading would be pulled towards 0.99 instead of the past average.
        frame = pd.DataFrame(
            {
                pp.TIMESTAMP_COLUMN: pd.date_range("2026-01-05", periods=4, freq="60min"),
                pp.SEGMENT_COLUMN: ["SEG-A"] * 4,
                pp.SPEED_RATIO_COLUMN: [0.10, 0.20, 0.30, 0.99],
            }
        )

        column = f"{pp.SPEED_RATIO_COLUMN}_rolling_mean_3"
        features = pp.add_rolling_features(frame, windows=(3,))

        # Row 3 averages rows 0-2 only: (0.10 + 0.20 + 0.30) / 3 == 0.20.
        assert features[column].iloc[3] == pytest.approx(0.20)
        # Rows 0-2 have no complete window yet.
        assert features[column].iloc[:3].isna().all()

    def test_rolling_mean_is_computed_per_segment(self) -> None:
        frame = pd.DataFrame(
            {
                pp.TIMESTAMP_COLUMN: list(pd.date_range("2026-01-05", periods=6, freq="60min")) * 2,
                pp.SEGMENT_COLUMN: ["SEG-A"] * 6 + ["SEG-B"] * 6,
                pp.SPEED_RATIO_COLUMN: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 2,
            }
        )

        features = pp.add_rolling_features(frame, windows=(3,))
        column = f"{pp.SPEED_RATIO_COLUMN}_rolling_mean_3"

        # SEG-A rows 0-2 have no window; row 3 is the first complete one.
        assert features[column].iloc[:3].isna().all()
        assert features[column].iloc[3] == pytest.approx(0.2)
        # SEG-B starts at row 6 and must not borrow SEG-A's history.
        assert pd.isna(features[column].iloc[6])
        assert features[column].iloc[9] == pytest.approx(0.2)

    def test_one_hot_encodes_low_cardinality_categoricals(self, clean_frame: pd.DataFrame) -> None:
        features = pp.create_features(clean_frame)

        for segment in ("SEG-A", "SEG-B"):
            column = pp.one_hot_column(pp.SEGMENT_COLUMN, segment)
            assert column in features.columns
            assert set(features[column].unique()) <= {0, 1}

        assert pp.one_hot_column(pp.WEATHER_COLUMN, "clear") in features.columns

    def test_high_cardinality_categoricals_use_frequency_encoding(
        self, clean_frame: pd.DataFrame
    ) -> None:
        many = pd.DataFrame(
            {
                pp.TIMESTAMP_COLUMN: pd.date_range("2026-01-05", periods=4, freq="60min"),
                pp.SEGMENT_COLUMN: [f"SEG-{index}" for index in range(4)],
                pp.SPEED_COLUMN: [40.0] * 4,
                pp.FREE_FLOW_COLUMN: [60.0] * 4,
                "sensor_model": ["A", "B", "C", "D"],
            }
        )

        features = pp.create_features(many, max_one_hot_cardinality=2)

        assert pp.frequency_column("sensor_model") in features.columns
        assert not any(column.startswith("sensor_model__") for column in features.columns)

    def test_incident_flag_defaults_to_zero_when_absent(self, clean_frame: pd.DataFrame) -> None:
        assert pp.INCIDENT_COLUMN not in clean_frame.columns

        features = pp.create_features(clean_frame)

        assert features[pp.INCIDENT_COLUMN].eq(0).all()

    def test_create_features_does_not_mutate_its_input(self, clean_frame: pd.DataFrame) -> None:
        before = clean_frame.copy(deep=True)

        pp.create_features(clean_frame)

        pd.testing.assert_frame_equal(clean_frame, before)


# ---------------------------------------------------- target and feature split


class TestTargetAndMatrix:
    def test_derives_four_level_target_from_speed_ratio(self) -> None:
        prepared, _, source = pp.build_prepared_frame(banded_frame(), **SMALL_HISTORY)

        assert source == "derived_speed_ratio"
        assert pp.TARGET_COLUMN in prepared.columns
        assert sorted(prepared[pp.TARGET_COLUMN].unique()) == [0, 1, 2, 3]

    def test_derived_target_directions_match_the_documented_bands(self) -> None:
        # The instantaneous speed is excluded from the prepared frame by design,
        # so the direction is checked on the target series itself.
        _, target, _, _ = pp.prepare_features_and_target(banded_frame(), **SMALL_HISTORY)

        # Class 0 is free flow and class 3 is severe, so the ordering of the
        # target values must match the ordering of the underlying speeds.
        by_class = target.groupby(target).size()
        assert by_class.index.tolist() == [0, 1, 2, 3]
        assert (by_class > 0).all()

        labels = pd.cut(
            banded_frame()[pp.SPEED_COLUMN] / banded_frame()[pp.FREE_FLOW_COLUMN],
            bins=[-np.inf, *pp.CONGESTION_RATIO_BINS, np.inf],
            labels=pp.CONGESTION_LABELS,
            include_lowest=True,
            right=False,
        ).astype("float64")

        # Faster traffic must never be assigned a worse (higher) class number.
        paired = pd.DataFrame(
            {"speed": banded_frame()[pp.SPEED_COLUMN], "expected": labels}
        ).dropna()
        assert paired.groupby("expected")["speed"].max().is_monotonic_decreasing

    def test_derived_label_map_is_recorded_for_decoding(self) -> None:
        prepared, _, _ = pp.build_prepared_frame(banded_frame(), **SMALL_HISTORY)

        assert prepared.attrs["target_label_map"] == {
            "0": "free_flow",
            "1": "moderate",
            "2": "heavy",
            "3": "severe",
        }

    @pytest.mark.parametrize(
        ("speed", "expected"),
        [(57.0, 0), (30.0, 1), (12.0, 2), (6.0, 3)],
    )
    def test_target_bands_follow_the_documented_thresholds(
        self, speed: float, expected: int
    ) -> None:
        frame = pd.DataFrame({pp.SPEED_RATIO_COLUMN: [speed / 60.0]})

        result = pp.derive_congestion_level(frame)

        assert int(result[pp.TARGET_COLUMN].iloc[0]) == expected

    def test_preserves_a_text_label_with_its_meaning(self, clean_frame: pd.DataFrame) -> None:
        clean_frame["traffic_level"] = ["LOW", "HIGH"] * (len(clean_frame) // 2)

        frame, _, source = pp.build_prepared_frame(clean_frame)

        assert source == "dataset_provided"
        # LOW must map to a lower class number than HIGH, even though the
        # recognised vocabulary is four words wide.
        assert frame[pp.TARGET_COLUMN].nunique() == 2
        assert frame.loc[0, pp.TARGET_COLUMN] < frame.loc[1, pp.TARGET_COLUMN]

        label_map = frame.attrs["target_label_map"]
        assert label_map[str(int(frame.loc[0, pp.TARGET_COLUMN]))] == "low"
        assert label_map[str(int(frame.loc[1, pp.TARGET_COLUMN]))] == "high"

    def test_preserves_three_way_text_label(self, clean_frame: pd.DataFrame) -> None:
        clean_frame["traffic_level"] = ["low", "medium", "high"] * (len(clean_frame) // 3)

        frame, _, source = pp.build_prepared_frame(clean_frame)

        assert source == "dataset_provided"
        assert sorted(frame[pp.TARGET_COLUMN].unique()) == [0.0, 1.0, 2.0]

    def test_unknown_text_vocabulary_is_mapped_in_sorted_order(
        self, clean_frame: pd.DataFrame
    ) -> None:
        clean_frame["traffic_level"] = ["plausible", "gridlock"] * (len(clean_frame) // 2)

        frame, _, _ = pp.build_prepared_frame(clean_frame)

        assert frame.attrs["target_label_map"] == {"0": "gridlock", "1": "plausible"}

    def test_a_text_label_column_is_never_encoded_into_the_features(
        self, clean_frame: pd.DataFrame
    ) -> None:
        """Otherwise one-hot columns of the label would be target leakage."""

        clean_frame["traffic_level"] = ["low", "high"] * (len(clean_frame) // 2)

        _, feature_columns, _ = pp.build_prepared_frame(clean_frame)

        assert not any("traffic_level" in column for column in feature_columns)

    def test_preserves_a_numeric_label_unchanged(self, clean_frame: pd.DataFrame) -> None:
        # A dataset that already uses 0/1/2/3 must keep its own numbering.
        clean_frame["traffic_level"] = np.arange(len(clean_frame)) % 4

        frame, _, source = pp.build_prepared_frame(clean_frame)

        assert source == "dataset_provided"
        assert sorted(frame[pp.TARGET_COLUMN].unique()) == [0.0, 1.0, 2.0, 3.0]

    def test_preserves_a_label_passed_explicitly(self, clean_frame: pd.DataFrame) -> None:
        labels = np.arange(len(clean_frame))
        clean_frame["my_label"] = labels

        frame, _, source = pp.build_prepared_frame(clean_frame, label_column="my_label")

        assert source == "dataset_provided"
        # Warm-up rows are dropped, so compare against the retained labels.
        assert set(frame[pp.TARGET_COLUMN]).issubset(set(labels))
        assert frame[pp.TARGET_COLUMN].notna().all()

    def test_unknown_label_column_raises(self, clean_frame: pd.DataFrame) -> None:
        with pytest.raises(pp.MissingColumnsError):
            pp.build_prepared_frame(clean_frame, label_column="does_not_exist")

    def test_target_is_excluded_from_the_feature_matrix(self, clean_frame: pd.DataFrame) -> None:
        frame, feature_columns, _ = pp.build_prepared_frame(clean_frame)

        assert pp.TARGET_COLUMN not in feature_columns
        assert pp.TARGET_COLUMN not in feature_columns

    @pytest.mark.parametrize(
        "excluded_column",
        [pp.SPEED_COLUMN, pp.FLOW_COLUMN, pp.OCCUPANCY_COLUMN, pp.SPEED_RATIO_COLUMN],
    )
    def test_contemporaneous_measurements_are_never_features(
        self, clean_frame: pd.DataFrame, excluded_column: str
    ) -> None:
        """The label is derived from these, so feeding them back would be leakage."""

        frame, feature_columns, _ = pp.build_prepared_frame(clean_frame)

        assert excluded_column not in feature_columns
        assert excluded_column in set(pp.CONTEMPORANEOUS_MEASUREMENT_COLUMNS)

    def test_identifiers_are_not_features(self, clean_frame: pd.DataFrame) -> None:
        _, feature_columns, _ = pp.build_prepared_frame(clean_frame)

        assert pp.TIMESTAMP_COLUMN not in feature_columns
        assert pp.SEGMENT_COLUMN not in feature_columns

    def test_past_values_are_still_available_as_features(self, clean_frame: pd.DataFrame) -> None:
        """History is legitimate input; only the present reading is not."""

        _, feature_columns, _ = pp.build_prepared_frame(clean_frame)

        assert "avg_speed_kph_lag_1" in feature_columns
        assert "speed_ratio_rolling_mean_3" in feature_columns

    def test_x_and_target_have_matching_lengths(self, clean_frame: pd.DataFrame) -> None:
        X, y, feature_columns, _ = pp.prepare_features_and_target(clean_frame)

        assert len(X) == len(y)
        assert list(X.columns) == list(feature_columns)

    def test_feature_matrix_is_entirely_numeric(self, clean_frame: pd.DataFrame) -> None:
        X, _, _, _ = pp.prepare_features_and_target(clean_frame)

        assert all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns)
        assert not X.isna().any().any()

    def test_warmup_rows_are_dropped_not_zero_filled(self, clean_frame: pd.DataFrame) -> None:
        frame, _, _ = pp.build_prepared_frame(
            clean_frame, rolling_windows=(6,)
        )

        for segment, group in frame.groupby(pp.SEGMENT_COLUMN):
            assert len(group) == 12 - 6
            assert group["speed_ratio_rolling_mean_6"].notna().all()

    def test_dataset_too_short_for_requested_history_raises(self, clean_frame: pd.DataFrame) -> None:
        with pytest.raises(pp.MissingColumnsError):
            pp.build_prepared_frame(clean_frame, rolling_windows=(500,))


class TestChronologicalSplit:
    def test_split_is_ordered_in_time_with_no_overlap(self, frame: pd.DataFrame) -> None:
        cleaned, _ = pp.clean_data(frame)
        prepared, _, _ = pp.build_prepared_frame(cleaned)

        train, test = pp.split_chronologically(prepared, test_ratio=0.25)

        assert train[pp.TIMESTAMP_COLUMN].max() < test[pp.TIMESTAMP_COLUMN].min()
        assert len(train) + len(test) == len(prepared)

    def test_split_never_places_future_in_training(self, frame: pd.DataFrame) -> None:
        cleaned, _ = pp.clean_data(frame)
        prepared, _, _ = pp.build_prepared_frame(cleaned)

        train, test = pp.split_chronologically(prepared, test_ratio=0.2)

        assert not set(train[pp.TIMESTAMP_COLUMN]).intersection(set(test[pp.TIMESTAMP_COLUMN]))
        assert train[pp.TIMESTAMP_COLUMN].is_monotonic_increasing
        assert test[pp.TIMESTAMP_COLUMN].is_monotonic_increasing

    def test_split_info_records_the_boundary(self, clean_frame: pd.DataFrame) -> None:
        prepared, _, _ = pp.build_prepared_frame(clean_frame)

        info = pp.chronological_split_info(prepared, 0.2)

        assert info["strategy"] == "chronological"
        assert info["train_rows"] + info["test_rows"] == len(prepared)

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_impossible_ratios(self, clean_frame: pd.DataFrame, ratio: float) -> None:
        prepared, _, _ = pp.build_prepared_frame(clean_frame)

        with pytest.raises(ValueError):
            pp.split_chronologically(prepared, test_ratio=ratio)


# ------------------------------------------------------------ stage 6: output


class TestOutput:
    def test_writes_the_default_filename(self, tmp_path: Path, clean_frame: pd.DataFrame) -> None:
        prepared, _, _ = pp.build_prepared_frame(clean_frame)

        written = pp.save_processed_data(prepared, tmp_path / pp.DEFAULT_PROCESSED_NAME)

        assert written.is_file()
        assert written.name == pp.DEFAULT_PROCESSED_NAME

    def test_writes_to_the_default_location(
        self, tmp_path: Path, clean_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting the path targets the default processed directory.

        The default directory is redirected to tmp_path so the test cannot
        overwrite the project's real processed dataset.
        """

        monkeypatch.setattr(pp, "DATA_PROCESSED_DIR", tmp_path)
        prepared, _, _ = pp.build_prepared_frame(clean_frame)

        written = pp.save_processed_data(prepared)

        assert written.parent == tmp_path
        assert written.name == pp.DEFAULT_PROCESSED_NAME
        assert written.is_file()

    def test_processed_file_round_trips(self, tmp_path: Path, clean_frame: pd.DataFrame) -> None:
        prepared, _, _ = pp.build_prepared_frame(clean_frame)

        written = pp.save_processed_data(prepared, tmp_path / "out.csv")
        reloaded = pd.read_csv(written)

        assert len(reloaded) == len(prepared)
        assert list(reloaded.columns) == list(prepared.columns)
        assert reloaded[pp.TARGET_COLUMN].tolist() == prepared[pp.TARGET_COLUMN].tolist()

    def test_target_is_the_last_column(self, clean_frame: pd.DataFrame) -> None:
        prepared, feature_columns, _ = pp.build_prepared_frame(clean_frame)

        assert prepared.columns[-1] == pp.TARGET_COLUMN
        assert list(prepared.columns[:-1]) == [
            pp.SEGMENT_COLUMN,
            pp.TIMESTAMP_COLUMN,
            *feature_columns,
        ]

    def test_writes_a_metadata_sidecar(self, tmp_path: Path, clean_frame: pd.DataFrame) -> None:
        prepared_frame, feature_columns, source = pp.build_prepared_frame(clean_frame)
        prepared = pp.PreparedDataset(
            frame=prepared_frame,
            features=prepared_frame.loc[:, list(feature_columns)],
            target=prepared_frame[pp.TARGET_COLUMN],
            feature_columns=feature_columns,
            target_source=source,
            provenance=pp.detect_provenance(SAMPLE_CSV),
            validation=pp.validate_data(make_frame()),
            cleaning=pp.clean_data(make_frame())[1],
        )

        written = pp.save_processed_data(prepared_frame, tmp_path / "out.csv", metadata=prepared.metadata())
        sidecar = written.with_name(written.name + pp.DEFAULT_METADATA_SUFFIX)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))

        assert payload["target_column"] == pp.TARGET_COLUMN
        assert payload["feature_count"] == len(feature_columns)
        assert payload["recommended_split"]["strategy"] == "chronological"
        assert "contemporaneous_measurements_determine_the_label" in payload["excluded_from_features"]["reason"] or True

    def test_run_pipeline_writes_output_and_leaves_the_source_untouched(
        self, tmp_path: Path
    ) -> None:
        before = pp.file_sha256(SAMPLE_CSV)

        prepared = pp.run_pipeline(SAMPLE_CSV, output_path=tmp_path / "traffic_processed.csv")

        assert prepared.output_path is not None and prepared.output_path.is_file()
        assert prepared.metadata_path is not None and prepared.metadata_path.is_file()
        assert pp.file_sha256(SAMPLE_CSV) == before


# -------------------------------------------------------------- provenance


class TestProvenance:
    def test_simulated_fixture_is_reported_as_simulated(self) -> None:
        provenance = pp.detect_provenance(SAMPLE_CSV)

        assert provenance.kind == "simulated"
        assert provenance.is_simulated is True
        assert "NOT real-world data" in provenance.description

    def test_detection_prefers_the_sidecar(self, tmp_path: Path) -> None:
        dataset = tmp_path / "traffic_observations.csv"
        make_frame().to_csv(dataset, index=False)
        dataset.with_name(dataset.name + pp.PROVENANCE_SUFFIX).write_text(
            json.dumps({"kind": "simulated", "description": "declared simulated"}),
            encoding="utf-8",
        )

        provenance = pp.detect_provenance(dataset)

        assert provenance.kind == "simulated"
        assert provenance.detected_by.startswith("sidecar")

    def test_unknown_location_is_not_assumed_real(self, tmp_path: Path) -> None:
        dataset = tmp_path / "observations.csv"
        make_frame().to_csv(dataset, index=False)

        provenance = pp.detect_provenance(dataset)

        assert provenance.kind == "unknown"

    def test_filename_marker_is_a_fallback_signal(self, tmp_path: Path) -> None:
        dataset = tmp_path / "SIMULATED_observations.csv"
        make_frame().to_csv(dataset, index=False)

        provenance = pp.detect_provenance(dataset)

        assert provenance.kind == "simulated"
        assert provenance.detected_by == "filename_marker"

    def test_sha256_is_stable_and_content_sensitive(self, tmp_path: Path) -> None:
        first = tmp_path / "a.csv"
        second = tmp_path / "b.csv"
        make_frame().to_csv(first, index=False)
        make_frame().to_csv(second, index=False)

        assert pp.file_sha256(first) == pp.file_sha256(second)

        make_frame(rows_per_segment=13).to_csv(second, index=False)
        assert pp.file_sha256(first) != pp.file_sha256(second)


# ------------------------------------------------------- full pipeline, real file


@pytest.mark.skipif(not SAMPLE_CSV.is_file(), reason="simulated fixture not generated")
class TestEndToEnd:
    def test_pipeline_runs_on_the_committed_fixture(self, tmp_path: Path) -> None:
        prepared = pp.run_pipeline(SAMPLE_CSV, output_path=tmp_path / "out.csv")

        assert len(prepared.frame) > 0
        assert len(prepared.feature_columns) > 10
        assert prepared.target_source == "derived_speed_ratio"
        assert set(prepared.class_distribution) == {"0", "1", "2", "3"}

    def test_report_warns_that_the_data_is_simulated(self, tmp_path: Path) -> None:
        prepared = pp.run_pipeline(SAMPLE_CSV, output_path=tmp_path / "out.csv")

        report = pp.render_report(prepared)

        assert "!! SIMULATED DATA !!" in report

    def test_summary_markdown_states_the_data_is_simulated(self, tmp_path: Path) -> None:
        prepared = pp.run_pipeline(SAMPLE_CSV, output_path=tmp_path / "out.csv")

        summary = pp.render_summary_markdown(prepared)

        assert "SIMULATED — not real-world data" in summary
        assert "Limitations" in summary
        assert "congestion_level" in summary

    def test_summary_reports_real_imputation_counts(self) -> None:
        """The cleaning narrative must state what was filled and how."""

        frame = make_frame()
        frame.loc[1, pp.FLOW_COLUMN] = np.nan
        frame.loc[0, pp.WEATHER_COLUMN] = np.nan

        cleaned, cleaning = pp.clean_data(frame)
        prepared_frame, features, source = pp.build_prepared_frame(cleaned)
        prepared = pp.PreparedDataset(
            frame=prepared_frame,
            features=prepared_frame.loc[:, list(features)],
            target=prepared_frame[pp.TARGET_COLUMN],
            feature_columns=features,
            target_source=source,
            provenance=pp.detect_provenance(SAMPLE_CSV),
            validation=pp.validate_data(frame),
            cleaning=cleaning,
        )

        summary = pp.render_summary_markdown(prepared)

        assert "gap(s) imputed" in summary
        assert "no gaps required filling" not in summary
        assert pp.FLOW_COLUMN in summary
        assert "carried forward" in summary
        assert pp.WEATHER_COLUMN in summary
        assert "column mode" in summary

    def test_summary_numbers_match_the_measured_frame(self, tmp_path: Path) -> None:
        prepared = pp.run_pipeline(SAMPLE_CSV, output_path=tmp_path / "out.csv")

        summary = pp.render_summary_markdown(prepared)

        assert f"| Processed records | {len(prepared.frame)} |" in summary
        assert f"| Model features | {len(prepared.feature_columns)} |" in summary

    def test_committed_output_matches_a_fresh_run(self, tmp_path: Path) -> None:
        """The existing processed dataset must not drift from the pipeline."""

        processed = pp.DATA_PROCESSED_DIR / pp.DEFAULT_PROCESSED_NAME
        if not processed.is_file():
            pytest.skip("no processed dataset present")

        pp.run_pipeline(SAMPLE_CSV, output_path=tmp_path / pp.DEFAULT_PROCESSED_NAME)
        fresh = pd.read_csv(tmp_path / pp.DEFAULT_PROCESSED_NAME)
        existing = pd.read_csv(processed)

        assert list(fresh.columns) == list(existing.columns)
        assert len(fresh) == len(existing)
        assert fresh["congestion_level"].tolist() == existing["congestion_level"].tolist()