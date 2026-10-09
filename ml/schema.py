"""The feature contract shared by training and inference.

:class:`FeatureSchema` lives in its own module for one concrete reason: it is
pickled to ``ml/models/preprocessor.pkl``. A class defined in a module that also
has an ``if __name__ == "__main__"`` block gets recorded in the pickle as
``__main__.FeatureSchema`` when training runs as ``python -m ml.train``, and
loading it again later fails with *"Can't get attribute 'FeatureSchema' on
__main__"*. Defining it in a plain module means the pickle always names
``ml.schema.FeatureSchema``, which resolves on load.

Keeping it in a module also puts the schema under review in ordinary source form
rather than hiding it inside a binary blob.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from ml.preprocessing import (
    INCIDENT_COLUMN,
    LAG_SOURCE_COLUMNS,
    SEGMENT_COLUMN,
    SEGMENT_CONSTANT_COLUMNS,
    SPEED_COLUMN,
    SPEED_RATIO_COLUMN,
    WEATHER_COLUMN,
    MissingColumnsError,
    create_features,
    ensure_columns,
    ensure_datetime_timestamps,
    one_hot_column,
)

#: Columns ``add_calendar_features`` derives from the timestamp, so a caller
#: never has to send them. Kept explicit and adjacent to
#: :func:`_built_feature_bases`, which depends on it.
CALENDAR_FEATURES = frozenset(
    {"hour", "day_of_week", "day_of_month", "is_weekend", "month"}
)

#: Raw columns a caller must always supply, whatever the schema contains.
ALWAYS_REQUIRED = ("timestamp", "road_segment_id")

#: Feature-name markers, resolved from the schema rather than hard-coded per model.
_ROLLING_MARKER = "_rolling_"
_LAG_MARKER = "_lag_"


@dataclass(frozen=True)
class FeatureSchema:
    """Everything inference needs to rebuild the model's inputs.

    Attributes:
        feature_columns: exact model-input order. Inference must follow this
            order, since a fitted estimator's columns are positional.
        label_mapping: class id (as a string) to human-readable name.
        target_column: the label being predicted.
        lag_steps: lag steps used at training time.
        rolling_windows: rolling window sizes used at training time.
        one_hot_max_cardinality: above this many distinct values a categorical is
            frequency-encoded instead of one-hot encoded.
        required_input_columns: raw columns a caller must supply.
        min_history_per_segment: prior observations per segment needed so the
            largest lag and rolling window are fully populated.
        categorical_levels: category -> levels seen at training time. Recorded so
            one-hot columns are rebuilt identically from a single row. Encoding
            each batch independently would create only the dummies that batch
            happens to contain, leaving the model's other expected columns absent.
    """

    feature_columns: tuple[str, ...]
    label_mapping: dict[str, str]
    target_column: str
    lag_steps: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    one_hot_max_cardinality: int
    required_input_columns: tuple[str, ...]
    min_history_per_segment: int
    categorical_levels: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        """Rebuild the model's input columns from raw observation records.

        Args:
            records: raw observations, one row per interval per road segment.

        Returns:
            A frame whose columns are exactly :attr:`feature_columns`, in order.

        Raises:
            MissingColumnsError: if a required input column is absent, or a
                recorded feature cannot be rebuilt.
        """

        ensure_columns(records, self.required_input_columns)

        # Callers often read input from CSV, where pandas parses a timestamp as
        # text. Normalising here keeps .dt accessors working on inference input
        # exactly as they do on the training path.
        prepared = create_features(
            ensure_datetime_timestamps(records),
            lags=self.lag_steps,
            rolling_windows=self.rolling_windows,
            max_one_hot_cardinality=self.one_hot_max_cardinality,
            exclude_from_encoding=tuple(self.categorical_levels),
        )

        # Reindex one-hot columns to the training level set. Without this, a batch
        # containing only one segment would lack the other segments' dummies and
        # the model's positional columns would no longer line up.
        prepared = self._align_encoded_levels(prepared)

        missing = [name for name in self.feature_columns if name not in prepared.columns]
        if missing:
            raise MissingColumnsError(
                "Input cannot supply feature(s) the model was trained on: "
                + ", ".join(missing)
                + f". Supply at least {self.min_history_per_segment} observations "
                "per road segment."
            )

        return prepared.loc[:, list(self.feature_columns)]

    def _align_encoded_levels(self, prepared: pd.DataFrame) -> pd.DataFrame:
        """Restore every recorded one-hot column, zero-filling absent levels.

        ``encode_categorical_columns`` one-hots whatever levels a frame happens
        to contain. That is correct for training on the whole dataset, but wrong
        for a single inference row: it would emit one dummy where the fitted model
        expects six. Rebuilding the full level set here keeps inference independent
        of which rows the caller happened to pass.
        """

        for source, levels in self.categorical_levels.items():
            expected = [one_hot_column(source, level) for level in levels]
            if all(name in prepared.columns for name in expected):
                continue
            for name in expected:
                if name not in prepared.columns:
                    prepared[name] = 0

        return prepared

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""

        return {
            "categorical_levels": {
                key: list(levels) for key, levels in self.categorical_levels.items()
            },
            "feature_columns": list(self.feature_columns),
            "label_mapping": dict(self.label_mapping),
            "target_column": self.target_column,
            "lag_steps": list(self.lag_steps),
            "rolling_windows": list(self.rolling_windows),
            "one_hot_max_cardinality": self.one_hot_max_cardinality,
            "required_input_columns": list(self.required_input_columns),
            "min_history_per_segment": self.min_history_per_segment,
        }


def derive_required_inputs(
    feature_columns: Sequence[str], available: Sequence[str]
) -> tuple[str, ...]:
    """Work out which raw columns are needed to rebuild ``feature_columns``.

    The feature names are walked rather than listing columns by hand, so a schema
    change cannot leave this list stale:

    * ``x_lag_3`` and ``x_rolling_mean_6`` both require ``x``;
    * a one-hot or frequency-encoded column requires its source categorical;
    * calendar features are derived from the timestamp, so they add nothing.

    Args:
        feature_columns: the model's input columns.
        available: columns present in the training dataset, used to distinguish a
            genuine source column from a derived one.

    Returns:
        The sorted raw columns a caller must provide.
    """

    # A feature is either built by create_features() (calendar, lags, rolling
    # means, one-hot indicators) or passed straight through from the raw record.
    # Only the built ones can be produced without the caller sending data, so any
    # feature that is *also* a real column in the dataset must be requested.
    feature_set = set(feature_columns)
    built_by_helper = _built_feature_bases(feature_columns)

    passthrough = {
        column
        for column in feature_columns
        if column in set(available) and column not in built_by_helper
    }

    required: set[str] = set(ALWAYS_REQUIRED) | passthrough

    for column in feature_columns:
        base = column
        if _ROLLING_MARKER in base:
            base = base.split(_ROLLING_MARKER)[0]
        if _LAG_MARKER in base:
            base = base.split(_LAG_MARKER)[0]

        # speed_ratio determines the label, so it is recomputed from its
        # components rather than accepted from the caller: that keeps a
        # contemporaneous target-derived value out of the input contract.
        if base == SPEED_RATIO_COLUMN:
            required.add(SPEED_COLUMN)
            required.update(SEGMENT_CONSTANT_COLUMNS)
            continue

        # Lag sources are always required: they cannot be reconstructed from
        # anything else the caller sends.
        if base in LAG_SOURCE_COLUMNS:
            required.add(base)

    if any(column.startswith(f"{WEATHER_COLUMN}__") for column in feature_columns):
        required.add(WEATHER_COLUMN)
    if any(column.startswith(f"{SEGMENT_COLUMN}__") for column in feature_columns):
        required.add(SEGMENT_COLUMN)

    return tuple(sorted(required))


def _built_feature_bases(feature_columns: Sequence[str]) -> set[str]:
    """Return the feature names that ``create_features`` generates itself.

    Everything else is a pass-through column the caller has to supply.
    """

    built: set[str] = set()
    for column in feature_columns:
        if _ROLLING_MARKER in column or _LAG_MARKER in column or "__" in column:
            built.add(column)
            continue
        if column in CALENDAR_FEATURES or column == INCIDENT_COLUMN:
            built.add(column)
    return built