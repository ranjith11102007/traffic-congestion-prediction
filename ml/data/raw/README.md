# Real traffic dataset goes here

This directory is **reserved for a real dataset you supply**. Nothing in this
repository is real-world traffic data, and the pipeline will not invent any.

If a file named `traffic_observations.csv` exists here, `python -m ml.preprocessing`
uses it automatically. Otherwise the pipeline falls back to the clearly labelled
simulated fixture in `../sample/`, and its report states that the data is
simulated.

## Accepted sources

Use a public urban traffic dataset with a documented licence, for example:

| Source | What it provides | Notes |
| --- | --- | --- |
| **PEMS (Caltrans Performance Measurement System)** | 5-minute speed, flow and occupancy from ~39 000 loop detectors on California freeways, several years | Free, but freeway rather than urban arterial; large downloads, so aggregate before committing |
| **METR-LA / PeMSD7 / PeMSD8 (DAGNN benchmark suites)** | Pre-aggregated hourly sensor readings, widely used for congestion-forecasting research | Small CSV downloads, ideal for a student project; already aggregated to 5–15 minute intervals |
| **Urban Traffic Data from an Open Data Portal** | Speeds, counts and occupancy from city loop or probe detectors | Check each city's licence and citation requirements |

Record the source name, URL, licence and download date in `data_summary.md` when
you add a real dataset. Do not commit a dataset whose licence forbids it.

## Required schema

The pipeline needs these columns. Names are configurable only by renaming in the
file; the defaults below are what the code looks for.

| Column | Type | Required | Meaning |
| --- | --- | --- | --- |
| `timestamp` | datetime | **yes** | Observation time. Any pandas-parsable format. |
| `road_segment_id` | string / int | **yes** | Road link identifier. Groups observations so lag and rolling features never cross segments. |
| `avg_speed_kph` | float | **yes** | Mean traversal speed over the interval. |
| `free_flow_speed_kph` | float | one of these two | Uncongested reference speed for the segment. |
| `speed_ratio` | float | one of these two | Pre-computed `avg_speed_kph / free_flow_speed_kph`. |

### Optional columns

Used automatically when present. Nothing is forced — a minimal dataset still works.

| Column | Type | Meaning |
| --- | --- | --- |
| `flow_veh_per_hr` | float | Vehicle throughput, your "vehicle count". |
| `occupancy_pct` | float | Detector occupancy, your "density". |
| `temperature_c` | float | Air temperature at the nearest station. |
| `precipitation_mm` | float | Rainfall over the interval. |
| `weather_condition` | string | Nominal weather label; one-hot encoded. |
| `is_incident` | 0/1 | Incident or roadworks flag on the segment. |

### Optional label

If the dataset already ships congestion categories, the pipeline **preserves
them unchanged**. It looks for a column named one of:

```
congestion_level, congestion_class, congestion_state,
traffic_level, traffic_state, level_of_service, los
```

The values are taken as the target and encoded to consecutive integers. If the
labels are ordinal strings (`LOW`, `MEDIUM`, `HIGH`), map them in ascending order
of severity before saving. To force a specific column, pass
`--label-column <name>`.

If no label exists, the target is derived from `speed_ratio` using fixed,
configurable bands. That is an engineering convention, not ground truth — see
`docs/ml-pipeline.md`.

### Unit conversions

If your source uses other units, convert before running the pipeline:

- miles per hour → km/h: multiply by `1.609344`
- vehicles per interval → vehicles per hour: divide by the interval length in hours
  (divide by 0.25 for 15-minute data)
- mph → km/h in the free-flow column as well

## Advice before running the pipeline

- **Aggregate to a fixed interval.** PEMS publishes 5-minute data; hourly data is
  usually plenty for a project of this size and keeps the file small.
- **One row per segment per timestamp.** The pipeline detects and reports
  conflicting duplicates rather than silently averaging them.
- **Avoid gaps larger than a few intervals.** Lag features are computed on
  consecutive rows, so a long gap produces a lag that silently means "an hour and
  a half ago" rather than "one interval ago". A time-based reindex per segment is
  worth doing before this step.
- **Check the licence.** Do not commit data you are not permitted to redistribute.