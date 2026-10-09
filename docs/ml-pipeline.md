# ML pipeline design

Stage 2 delivers the **preprocessing pipeline**, which is implemented, tested and
runnable. Stage 1 delivered the structure and data contract; no model has been
trained, so this document specifies what the pipeline consumes and produces
rather than reporting model results.

> **No real dataset has been acquired.** The pipeline has been exercised only
> against a deterministic *simulated* fixture, prefixed `SIMULATED_`. Numbers
> derived from it describe the pipeline's behaviour, not real traffic.

## Module boundary

`ml/` is a standalone package. It must never import `backend/app` and must not
depend on FastAPI, SQLAlchemy or any web concept. The only coupling between the ML
layer and the API is the serialized artifact file.

Consequences:

- A training machine needs only `ml/requirements.txt`.
- The API container does not need NumPy, pandas or scikit-learn in production —
  Stage 3 loads the artifact with `joblib` and converts it to plain arrays.
- Alternative models can be benchmarked without touching the API.

## Data contract

### Input

A single wide CSV, one row per observation interval per road segment.

| Column | Required | Role |
| --- | --- | --- |
| `timestamp` | yes | Observation time, any pandas-parsable format |
| `road_segment_id` | yes | Groups observations per road link; lag features never cross segments |
| `avg_speed_kph` | yes | Mean speed for the interval |
| `free_flow_speed_kph` | one of these two | Reference free-flow speed for the segment |
| `speed_ratio` | one of these two | Pre-computed `avg_speed_kph / free_flow_speed_kph` |
| `flow_veh_per_hr` | accepted, **not a feature** | Vehicle throughput; see the note below |
| `occupancy_pct` | optional | Detector occupancy |
| `temperature_c` | optional | Weather feature |
| `precipitation_mm` | optional | Weather feature |
| `is_incident` | optional | Incident flag; absent means a constant `0.0` |

Validation is strict: missing required columns, unparsable timestamps or missing
speeds raise `MissingColumnsError` rather than being silently imputed.

#### Why `flow_veh_per_hr` is no longer a feature (model 4.0.0)

The column is still accepted, so an older dataset trains without edits, but it is
excluded from the feature set.

Stage 5 connected a real traffic provider and that changed what the system can
actually observe. **No self-serve traffic API publishes vehicle throughput** — not
TomTom Flow Segment Data, not HERE Traffic v7, not Mapbox. TomTom returns current
speed, free-flow speed, confidence, functional road class, OpenLR id and road
closure; nothing returns a count, volume or density of vehicles.

A feature the system can never populate at inference time is a feature that is
always missing for real observations. The alternatives were both worse:

- store an estimated count, which is indistinguishable from a measurement once it
  is in the database, and would make a fabricated number look real;
- refuse to record genuine traffic readings that lack a count.

So `vehicle_count` became nullable (migration `b1f2c3d4e5a6`), live rows record
`NULL` with the reason in their metadata, and the feature was dropped so the model
and the data agree. 29 features became 26.

### Target derivation

A dataset-supplied label is always **preferred and preserved**.
`detect_label_column` only considers the explicit list
`CANDIDATE_LABEL_COLUMNS`, so resolution is predictable rather than fuzzy. Pass
`--label-column` to force a specific column.

`encode_target_labels` then converts that label to integers without discarding
its meaning:

| Input | Treatment |
| --- | --- |
| Non-negative integers | Used unchanged — a source class `3` stays class `3` |
| Recognised ordinal words | Mapped by natural order, so `low < medium < high` becomes `0 < 1 < 2` |
| Any other text | Mapped in sorted order, with the mapping recorded |

The chosen vocabulary is written to `frame.attrs["target_label_map"]` and into the
processed metadata sidecar, so any prediction can be decoded back to its
original label. The label column is excluded from encoding, so a text label is
never one-hot encoded into the model's own inputs.

Only when no label is supplied is `congestion_level` derived from `speed_ratio`
in four bands. Bands are **left-closed**, so each boundary belongs to the less
congested class:

| Class | Name | `speed_ratio` range | Interpretation |
| --- | --- | --- | --- |
| 0 | `free_flow` | `[0.75, inf)` | Unconstrained flow |
| 1 | `moderate` | `[0.40, 0.75)` | Slowing, above capacity |
| 2 | `heavy` | `[0.15, 0.40)` | Breakdown forming |
| 3 | `severe` | `(-inf, 0.15)` | Near standstill |

These thresholds are engineering defaults held in
`preprocessing.CONGESTION_RATIO_BINS`, not a published standard.

> A derived label is a **convention, not ground truth**. A model trained this way
> learns these four thresholds. It cannot validate them, and its accuracy says
> nothing about whether the thresholds match any authority's definition of
> congestion.

### Features

Calendar features from `timestamp`: `hour`, `day_of_week`, `day_of_month`,
`is_weekend`, `month`.

Static segment context: `free_flow_speed_kph` (capacity, not an instant reading).

Categorical encoding: one-hot for low cardinality (segment, weather), frequency
encoding above `ONE_HOT_MAX_CARDINALITY`.

Lagged history, computed per `road_segment_id` and shifted strictly backwards at
steps 1, 2 and 3 intervals:

```
avg_speed_kph_lag_1   avg_speed_kph_lag_2   avg_speed_kph_lag_3
speed_ratio_lag_1     speed_ratio_lag_2     speed_ratio_lag_3
```

The source columns are `LAG_SOURCE_COLUMNS = ("avg_speed_kph", "speed_ratio")`.
`flow_veh_per_hr` was removed from that tuple in model 4.0.0 for the reason
described in the data contract above.

Rolling means of `speed_ratio` over windows of 3 and 6 intervals.

Two properties are enforced by construction:

- **No future leakage.** Lags are `groupby(segment).shift(step)` and rolling
  windows are `.shift(1).rolling(window)`, so a row only ever sees its own past.
  Both are computed per segment and never cross a segment boundary.
- **No fabricated warm-up rows.** Rows without a complete history are dropped
  rather than zero-filled, because a zero speed is a real-world impossible value.

### Leakage control

This is the most important property of the stage, and it is enforced in three
places.

**1. Contemporaneous measurements are not features.** `avg_speed_kph`,
`flow_veh_per_hr`, `occupancy_pct` and `speed_ratio` *determine* the label.
Feeding them back as inputs would let the model read its own answer and report a
meaningless score. They are listed in
`CONTEMPORANEOUS_MEASUREMENT_COLUMNS` and excluded by
`select_feature_columns()`. Their history remains available as lags.

**2. History is strictly backward.** See above.

**3. Splitting is chronological.** Adjacent traffic observations are strongly
autocorrelated, so a random row split would place neighbouring instants on both
sides of the boundary and leak the future into training. `split_chronologically()`
cuts at a single global timestamp, so every segment contributes its past to
training and its future to testing. `recommended_split` in the processed metadata
records the intended boundary before Stage 3 runs.

Identifiers (`timestamp`, `road_segment_id`) are also excluded — they are keys,
not measurements.

### Imputation policy

Missing values are filled only when a defensible prior value exists, and never
across a segment boundary:

| Column kind | Strategy |
| --- | --- |
| Segment constant (e.g. `free_flow_speed_kph`) | That segment's median |
| Other numeric | Previous value within segment, then next value, then global median |
| Categorical | Column mode |

The segment-median step runs *before* the neighbour fill, because copying an
adjacent reading is meaningless for a constant like free-flow speed.

Impossible values (negative speed, occupancy above 100%, precipitation below
zero) are nulled and then imputed rather than deleted, and every count is
reported in `CleaningReport`. Cleaning **refuses to proceed** if it would drop
more than `MAX_DROP_RATIO` (20%) of rows.

### Provenance

- Every source file is hashed with SHA-256.
- `ml/data/raw/` holds a real operator-supplied dataset and is never committed.
- `ml/data/sample/` holds the deterministic simulated fixture; its CSV is
  regenerated by `python -m ml.data.sample.generate_simulated_dataset` rather
  than tracked.
- Provenance is detected from a `.provenance.json` sidecar first, then the
  filename marker, then the directory. Anything unrecognised is reported as
  `unknown` — never assumed real.
- Every report and summary prints a `!! SIMULATED DATA !!` banner when the source
  is simulated.

## Model

**Random Forest classifier** (`sklearn.ensemble.RandomForestClassifier`) as the
first working model, built by `ml.train.build_estimator`:

| Setting | Value | Reason |
| --- | --- | --- |
| `n_estimators` | 300 | Variance reduction without tuning |
| `class_weight` | `"balanced"` | Severe congestion is far rarer than free flow |
| `max_features` | `"sqrt"` | Standard default for classification |
| `min_samples_leaf` | 2 | Mild smoothing |
| `n_jobs` | `-1` | Use all cores |

### Artifact bundle

`ml/models/random_forest.joblib` holds one dictionary:

| Key | Purpose |
| --- | --- |
| `schema_version` | Bundle format version, for forward compatibility |
| `model_name` | `"random_forest"` |
| `estimator` | The fitted sklearn estimator |
| `feature_columns` | Ordered feature names — inference reorders input to match |
| `label_mapping` | Class id → human-readable name |
| `metrics` | Held-out accuracy, macro F1/precision/recall, confusion matrix, report |
| `class_distribution` | Training class counts, for imbalance reporting |
| `trained_at` | ISO-8601 UTC timestamp |
| `sklearn_version`, `ml_package_version` | Provenance and compatibility checks |

Storing the feature order inside the artifact is what prevents the classic
train/serve feature-skew bug: inference never has to assume a column order.

## Metrics

Accuracy alone is misleading on imbalanced congestion data, so the primary
reported figure is **macro-averaged F1** across the four classes, supported by:

- Confusion matrix, to see which transitions are confused
- Per-class precision and recall, to expose a model that ignores severe congestion
- Class distribution of the training split, so a reader can judge the imbalance

The baseline that must be beaten: always predicting the majority class. A model
that cannot beat it is not reported as working.

## Model comparison (Stage 3)

`build_estimator` is the only place model identity is decided, so comparison is a
loop over candidate builders scored by the same split:

1. Random Forest (baseline)
2. Gradient Boosting (`HistGradientBoostingClassifier`)
3. Logistic Regression (interpretable baseline)
4. Sequence model on lag features (optional stretch goal)

Each candidate is scored on identical splits, with a random-state sweep to report
mean ± spread rather than a single lucky number.

## Execution order

```bash
# Generate the SIMULATED fixture (not real-world data).
python -m ml.data.sample.generate_simulated_dataset

# Stage 2 - implemented and working.
python -m ml.preprocessing --data <dataset>.csv --write-summary

# Stage 3 - implemented; exits non-zero with an actionable message.
python -m ml.train         --data <dataset>.csv
python -m ml.evaluate      --data <dataset>.csv
python -m ml.predict       <records>.csv
```

`ml.preprocessing` writes `ml/data/processed/traffic_processed.csv`, a
`.meta.json` sidecar carrying provenance, validation, cleaning, feature and
target metadata plus the recommended split, and optionally
`ml/data/data_summary.md`.

`ml.train`, `ml.evaluate` and `ml.predict` exit non-zero while no artifact
exists, and create nothing. That is the intended behaviour.

## Pipeline API

Each stage is an independently testable function. The stages compose through
`run_pipeline()`, which is what the CLI calls:

| Stage | Function |
| --- | --- |
| Load | `load_data(path=None)` |
| Validate | `validate_data(frame)` → `ValidationReport` |
| Clean | `clean_data(frame, ...)` → `(frame, CleaningReport)` |
| Featurise | `create_features(frame, ...)` |
| Target | `build_prepared_frame(frame, ...)` → `(frame, feature_columns, source)` |
| Split | `split_chronologically(frame, test_ratio=...)` |
| Output | `save_processed_data(frame, ...)` / `run_pipeline(...)` |
| Convenience | `build_supervised_frame(path=None)` for the Stage 3 entry points |

`predict.prepare_records()` calls the same `create_features()` used for
training, so rolling means and encodings cannot drift between training and
inference.

## Not yet decided

- Source and licence of the real dataset (candidates are listed in
  `ml/data/raw/README.md`).
- Whether `flow`/`occupancy`/`weather` are available in the source dataset or
  must be joined from a separate provider.
- Forecast horizon and prediction interval reporting.
- Schema migrations for persisting stored predictions.
- Whether a real ground-truth congestion label can be obtained. Without one, the
  derived bands remain a convention and model scores must be reported with that
  caveat attached.