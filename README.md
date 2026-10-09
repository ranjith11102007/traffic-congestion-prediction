# AI-Based Urban Traffic Congestion Prediction System

A modular system that ingests real urban traffic observations, trains machine-learning
models to predict congestion levels per road segment, and serves those forecasts
through an API and an interactive map dashboard.

> **Current state: Stages 1-8 complete** - data pipeline, model training, prediction API, live traffic and weather providers, the real-time prediction engine with its dashboard API, the interactive frontend dashboard (Leaflet map, Chart.js analytics, auto-refresh), and Stage 8 hardening (production configuration guard, security and XSS review, health-vs-readiness, deployment guide, project report and presentation deck).
> The model is trained (version 4.0.0, Random Forest on 26 features) and served by
> the API; live TomTom and Open-Meteo readings flow through the collector and are
> scored automatically after collection. **The training data is a clearly labelled
> simulated fixture**, so predictions are demonstrative rather than field-validated -
> see [Data pipeline](#data-pipeline) and
> [Implementation status](#implementation-status).

---

## Table of contents

- [Problem statement](#problem-statement)
- [Project objective](#project-objective)
- [Planned features](#planned-features)
- [Technology stack](#technology-stack)
- [Architecture overview](#architecture-overview)
- [Local setup](#local-setup)
- [Starting the FastAPI server](#starting-the-fastapi-server)
- [Opening the frontend](#opening-the-frontend)
- [Deployment](#deployment)
- [Data pipeline](#data-pipeline)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Traffic API](#traffic-api)
- [Running the tests](#running-the-tests)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Implementation status](#implementation-status)
- [Development stages](#development-stages)

---

## Problem statement

Urban road networks experience congestion that people only discover once they are
already inside it. Reactive traffic control responds after congestion has formed,
and static timetables cannot account for day-to-day variation caused by incidents,
weather, construction and shifting peak hours. The consequences are measurable:
lost travel time, wasted fuel, avoidable emissions, and unreliability in urban
travel planning.

Existing dashboards mostly describe *current* conditions. They answer "where is
it congested right now?" but not "will this road be congested in 30 minutes?" —
which is the question that would actually let someone change their behaviour.

## Project objective

Build a system that predicts congestion for urban road segments ahead of time:

1. Collect **real** urban traffic observations into a persistent store.
2. Engineer features from time, road geometry, traffic flow and weather.
3. Train and compare machine-learning models against a majority-class baseline.
4. Expose predictions through a documented REST API.
5. Visualise current and predicted congestion on an interactive map.

Every metric and every prediction must be traceable to real data. The project does
not ship synthetic traffic data, placeholder model scores, or mocked API responses.

## Planned features

| Area | Feature | Stage |
| --- | --- | --- |
| Data | Real-time observation ingestion and batching | 4 |
| Data | PostgreSQL schema, migrations, time-series indexes | 3 |
| Data | Weather and incident feature joins | 5 |
| ML | Feature engineering pipeline with leakage control | 2 |
| ML | Random Forest baseline classifier | 3 |
| ML | Side-by-side model comparison on identical splits | 3 |
| ML | Stored, reproducible model artifacts with provenance | 2 |
| API | Road segment and congestion endpoints | 3 |
| API | Prediction and model-metadata endpoints | 3, 6 |
| Frontend | Leaflet map coloured by predicted congestion | 7 |
| Frontend | Chart.js observed-vs-predicted analytics | 7 |
| Ops | Containerisation, CI, deployment documentation | 8 |

Explicitly **out of scope**: authentication and user accounts, payments, and
mobile applications. None of them serve the project's research objective.

## Technology stack

| Layer | Technology | Notes |
| --- | --- | --- |
| Frontend | HTML5, CSS3, JavaScript (ES2020) | No framework, no build step |
| Maps | Leaflet.js | Vendored locally, OpenStreetMap tiles, colour by predicted congestion |
| Charts | Chart.js | Vendored locally, observed-vs-predicted trend chart |
| Backend | Python 3.11+, FastAPI, Pydantic, Uvicorn | Application-factory pattern |
| Database | PostgreSQL, SQLAlchemy 2.x, psycopg2 | Env-driven, lazy engine |
| ML | NumPy, pandas, scikit-learn, joblib | Independent of the API |
| Tooling | VS Code, Git/GitHub, `.env`, `requirements.txt` | `pytest` for tests |

Runtime dependencies are split so the API can be deployed without the ML stack and
the ML pipeline can run without FastAPI: see `backend/requirements.txt` and
`ml/requirements.txt`.

## Architecture overview

```
Browser (frontend/)
   │  HTTP/JSON, CORS enabled
   ▼
FastAPI (backend/app)
   api/         thin routers: parse, delegate, serialise
   schemas/     Pydantic contracts (validation lives here)
      │
      ▼
   services/    business logic
      │              │
      │              └── ml/predict.py ──► ML pipeline (ml/)
      ▼
   repositories/  every SQL statement
      │
      ▼
   database/   declarative base · lazy engine · session_scope
      │
      ▼
PostgreSQL  (DSN from DATABASE_URL, managed by Alembic)

Data providers
   services/providers.py  ← interface
      ├── SimulatedTrafficProvider   deterministic, always source="simulation"
      └── RealTrafficProvider        live: TomTom speeds + Open-Meteo weather
              ├── services/weather_client.py   Open-Meteo, no API key needed
              └── services/locations.py        which roads, mapped to SEG-01..06

Background collection
   services/collection_scheduler.py   opt-in, asyncio task, never blocks the API
```

Key architectural decisions:

- **The ML layer is independent of the web layer.** `ml/` never imports
  `backend/app`. The only coupling is the serialized artifact bundle, so models
  can be trained and compared with no API running. The dependency runs one way:
  the *service* layer calls the ML package.
- **Route handlers stay thin.** Business logic lives in `app/services/`, all SQL
  in `app/repositories/`, all validation in `app/schemas/`. No route imports
  SQLAlchemy query helpers.
- **Provenance is a first-class column.** Every observation records where it came
  from, and a simulating provider's label overrides the record's own claim. A
  forecast inherits the provenance of the reading it describes.
- **One error envelope.** Every non-2xx response uses `{"error": {...}}`.
- **Configuration comes from the environment.** No secret is committed; the DSN
  is resolved from settings so `alembic.ini` stays credential-free.
- **Missing capability is an explicit error.** Prediction fails with 503
  `model_not_available` without an artifact, 422 `insufficient_history` without
  enough readings, and 503 `provider_not_configured` without a real provider's
  credentials. Nothing is invented to fill a gap.
- **An unavailable measurement is stored as absent, not estimated.** No self-serve
  traffic API publishes vehicle throughput, so `vehicle_count` is nullable and live
  rows record `NULL` with the reason in their metadata. Model 4.0.0 dropped
  `flow_veh_per_hr` so the features and the data agree.

Full detail: [`docs/architecture.md`](docs/architecture.md) · [`docs/api.md`](docs/api.md)

## Local setup

**Prerequisites:** Python 3.11 or newer, Git, and PostgreSQL 14+ only once you
reach Stage 2.

```powershell
# Windows PowerShell
git clone <repository-url> traffic-prediction-system
cd traffic-prediction-system

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
```

```bash
# macOS / Linux
git clone <repository-url> traffic-prediction-system
cd traffic-prediction-system

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Edit `.env` and set `DATABASE_URL` when you need the database. The API starts
without it in local development; database-backed features return a `configuration_error`
that tells you exactly what to set.

Optional VS Code setup is in [`.vscode/settings.json`](.vscode/settings.json).

## Starting the FastAPI server

```powershell
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Or from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload
```

Verify:

```bash
curl http://127.0.0.1:8000/api/health
```

```json
{"status":"healthy","service":"traffic-prediction-api"}
```

Interactive documentation: <http://127.0.0.1:8000/docs>

## Opening the frontend

The frontend is static — no server or build step required. The simplest path in
development is to let the API serve it (single origin, no CORS):

<http://127.0.0.1:8000/> (landing page) or <http://127.0.0.1:8000/dashboard.html>.

For a pure-frontend preview on the `http` origin without the API serving files:

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --directory frontend
```

Then open <http://127.0.0.1:5500/index.html>. Because the preview runs on a
different origin than the API, set an absolute backend location in
`frontend/js/config.js` (see the inline comments there):
`API_BASE_URL: 'http://127.0.0.1:8000'`. The dashboard served by the API itself
uses the default empty `API_BASE_URL`, which means "the same origin".

If the API is running, the landing page shows a live `GET /api/health` result.
Stop the API and press **Re-check** to see the frontend report the failure
accurately.

### Dashboard (`frontend/dashboard.html`, Stage 7)

The dashboard is a single responsive page with a sidebar (Overview, Live Traffic,
Predictions, Analytics, Locations, System Status) and a location selector:

- **Overview** - six KPI cards (reporting locations, forecasts, mean speed,
  highest predicted level, freshness, model provenance), the Leaflet map with one
  marker per monitored location coloured by the model's predicted congestion
  class, and a per-location detail card.
- **Live Traffic** - the current reading, forecast and freshness for every
  monitored location, straight from `GET /api/dashboard/current`.
- **Predictions** - the latest stored forecast and the forecast history from
  `GET /api/traffic/predictions` and `/predictions/latest`.
- **Analytics** - a Chart.js line chart (average speed, free-flow speed and the
  predicted congestion class per point) for the selected 1/6/12/24-hour window
  from `GET /api/dashboard/trends`, plus the model card from
  `GET /api/dashboard/model`.
- **Locations** - readiness cards ("X of N readings", stale/fresh, provider).
- **System Status** - database, providers, scheduler, model and freshness from
  `GET /api/dashboard/status`.

Behaviour:

- Auto-refresh every 30 seconds by default, selectable 15/30/60 seconds or
  manual; refreshes pause while the tab is hidden. The header pill reports the
  live connection and backend health (`degraded` when the database is stale).
- Every panel has loading, error and honest empty states; retrying re-queries the
  API. The model's own congestion labels are displayed (Free flow / Moderate /
  Heavy / Severe) - nothing is invented, and `N/A` is used for missing values.
- Leaflet and Chart.js are vendored under `frontend/vendor/` so no CDN is
  required; if a tile server fails or a library is missing, that panel shows a
  notice and every other panel keeps working.

Configuration lives in `frontend/js/config.js`
(`window.TRAFFIC_CONFIG`): `API_BASE_URL` (default `''` — same origin, i.e. the
API serves the dashboard), `REFRESH_INTERVAL_MS`, `TREND_HOUR_OPTIONS`,
`PREDICTION_HISTORY_LIMIT`, the map tile URL, request timeout, and no
credentials of any kind.

## Data pipeline

The preprocessing pipeline turns raw traffic observations into a model-ready
dataset. It is the Stage 2 deliverable and runs entirely offline, with no API or
database required.

```powershell
# 1. Generate the SIMULATED fixture (only needed to exercise the pipeline).
#    This is synthetic data, never real-world traffic.
python -m ml.data.sample.generate_simulated_dataset

# 2. Run the pipeline against a real dataset and write the summary.
python -m ml.preprocessing --data ml\data\raw\<dataset>.csv --write-summary
```

### Stages

| Stage | Function | What it does |
| --- | --- | --- |
| Load | `load_data` | Reads CSV or Parquet, records SHA-256, returns a copy |
| Validate | `validate_data` | Schema, dtypes, missing values, duplicates, timestamp parseability, physical ranges |
| Clean | `clean_data` | Resolves duplicate keys, nulls impossible values, imputes per-segment; refuses to drop more than 20% of rows |
| Features | `create_features` | Calendar fields, `speed_ratio`, strictly backward lags and rolling means, one-hot / frequency encoding |
| Target | `build_prepared_frame` | Preserves a supplied label, otherwise derives four congestion bands from `speed_ratio` |
| Output | `save_processed_data` | Writes `traffic_processed.csv` plus a `.meta.json` sidecar |

### Required dataset columns

`timestamp`, `road_segment_id`, `avg_speed_kph`, plus either
`free_flow_speed_kph` or `speed_ratio`.

Optional, used when present: `flow_veh_per_hr`, `occupancy_pct`, `temperature_c`,
`precipitation_mm`, `weather_condition`, `is_incident`.

### The congestion target

If the dataset already contains a congestion category, it is **preserved**.
Otherwise the target is derived from `speed_ratio = avg_speed_kph / free_flow_speed_kph`
using left-closed bands:

| Class | Label | `speed_ratio` |
| --- | --- | --- |
| 0 | `free_flow` | `[0.75, ∞)` |
| 1 | `moderate` | `[0.40, 0.75)` |
| 2 | `heavy` | `[0.15, 0.40)` |
| 3 | `severe` | `(-∞, 0.15)` |

> **This is a documented convention, not measured ground truth.** Any model
> trained against a derived label learns *these thresholds*, not a field-verified
> definition of congestion.

### Leakage controls

The single most important property of this stage. `avg_speed_kph`, `flow_veh_per_hr`,
`occupancy_pct` and `speed_ratio` **determine the label**, so they are excluded
from the model's inputs — otherwise the model would read its own answer and report
a meaningless score.

Legitimate inputs remain: calendar features, static segment capacity, encoded
segment and weather identity, incident flags, and strictly *past* values via lags
and rolling means. Rolling windows are shifted so they never include the current
reading, and all imputation is per-segment so no segment inherits another's values.

`split_chronologically()` is provided for Stage 2. Adjacent traffic observations
are strongly autocorrelated, so a random split would leak the future into training;
the split point is one global timestamp.

### Data locations

| Path | Contents | Tracked in git |
| --- | --- | --- |
| `ml/data/raw/` | Real dataset supplied by an operator | No — schema README only |
| `ml/data/sample/` | Deterministic **simulated** fixture | No — regenerate with the generator |
| `ml/data/processed/` | `traffic_processed.csv` and its metadata | No — reproducible |
| `ml/data/data_summary.md` | Generated summary | No |

The fixture is prefixed `SIMULATED_`, carries a `.provenance.json` sidecar, and
every report states prominently that it is not real-world data.

## Machine Learning Pipeline

The pipeline is fully working end to end: preprocess → train → evaluate → predict.
Every command below runs against the Stage 2 output, and every one of them fails
loudly rather than inventing a result if its preconditions are not met.

```powershell
# 0. Generate the SIMULATED fixture (synthetic; not real traffic) and preprocess it.
python -m ml.data.sample.generate_simulated_dataset
python -m ml.preprocessing

# 1. Train. Writes the model, preprocessor, metadata and feature importances.
python -m ml.train

# 2. Evaluate on the held-out window. Writes metrics, a report and the matrix image.
python -m ml.evaluate

# 3. Predict. Takes a CSV or JSON file of observation records.
python -m ml.predict ml\data\sample\SIMULATED_traffic_observations.csv
```

Point any of them at your own data instead of the fixture:

```powershell
python -m ml.train    --data ml\data\processed\traffic_processed.csv --test-ratio 0.2
python -m ml.evaluate --data ml\data\processed\traffic_processed.csv
python -m ml.predict  records.csv --models-dir ml\models
```

### Artifacts

Training and evaluation write into `ml/models/`:

| File | Contents |
| --- | --- |
| `traffic_model.pkl` | The fitted estimator |
| `preprocessor.pkl` | `FeatureSchema`: feature order, required inputs, label map |
| `model_metadata.json` | Hyperparameters, split boundary, class balance, provenance |
| `feature_importance.json` | Impurity-based importances, ranked |
| `evaluation_results.json` | Every measured metric, the baseline and the split record |
| `model_report.md` | The same run as a readable report |
| `confusion_matrix.png` | Confusion matrix with congestion-level colours |

These are git-ignored because they are reproducible from the dataset. Re-run
`ml.train` and `ml.evaluate` to regenerate them.

### Prediction input

`ml.predict` accepts raw observation records and rebuilds the features itself:

```python
from ml.predict import predict_traffic

result = predict_traffic(records)          # DataFrame, list of dicts, or one dict
result["predicted_congestion"]              # 'free_flow' | 'moderate' | 'heavy' | 'severe'
result["confidence"]                        # max predicted probability, never hard-coded
result["class_probabilities"]               # full distribution, sums to 1.0
```

Required columns are `timestamp`, `road_segment_id`, `avg_speed_kph`,
`free_flow_speed_kph`, `flow_veh_per_hr`, `weather_condition`, `temperature_c` and
`precipitation_mm`. Because the lag and rolling features read backwards, supply at
least six prior observations per road segment; the leading rows that cannot be
scored are counted in `records_skipped_insufficient_history` rather than being
imputed and scored anyway.

Derived quantities are **never** inputs. `speed_ratio` and `congestion_level` are
computed from the raw columns above; the API rejects them if a caller supplies
them, which is what keeps the Stage 3 leakage fix from being undone at the
service boundary.

### Reported performance

On the current simulated dataset, the chronological 80/20 split gives **macro F1
0.7370** and accuracy **0.8561**, against a majority-class baseline of 0.2143 and
0.7500. These numbers describe a synthetic fixture only. They demonstrate that the
pipeline runs; they are not a statement about real traffic. `ml/models/model_report.md`
is regenerated on every evaluation and repeats that caveat.

The data contract, feature design, leakage controls and model-comparison plan are
specified in [`docs/ml-pipeline.md`](docs/ml-pipeline.md).

## Traffic API

### Creating the database schema

Stage 4 stores observations and forecasts in PostgreSQL. Alembic applies the
schema; the DSN comes from `DATABASE_URL`, so no credential is ever committed to
`alembic.ini`.

```powershell
# Set DATABASE_URL in .env first, then:
alembic upgrade head
```

Review the SQL without touching a server:

```powershell
alembic upgrade head --sql
```

Alembic refuses to run with no `DATABASE_URL` rather than guessing a target
database.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness. Unchanged Stage 1 contract. |
| `POST` | `/api/traffic/observations` | Record a reading. Duplicate location+timestamp → `409`. |
| `GET` | `/api/traffic/observations` | List readings, newest first, with filters echoed back |
| `GET` | `/api/traffic/observations/{id}` | One reading, `404` if absent |
| `GET` | `/api/traffic/latest` | Most recent reading, `404` when the table is empty |
| `POST` | `/api/traffic/predict` | Store a reading, predict it, store the forecast |
| `POST` | `/api/traffic/observations/{id}/predict` | Predict from an already stored reading |
| `GET` | `/api/traffic/predictions/status` | Model version, required columns, history depth |
| `GET` | `/api/traffic/collector/status` | Active provider and whether it is simulated |
| `POST` | `/api/traffic/collect` | Collect, then predict automatically from each new reading |
| `POST` | `/api/traffic/collect/history` | Backfill history so a location becomes predictable |
| `GET` | `/api/traffic/predictions` | Prediction history, newest first, joined to its observation |
| `GET` | `/api/traffic/predictions/latest` | Most recent stored forecast, `404` when none exists |
| `GET` | `/api/dashboard/current` | Current speed, forecast and freshness for every location |
| `GET` | `/api/dashboard/trends` | Speed and forecast series for a chart |
| `GET` | `/api/dashboard/locations` | Per-location readiness, e.g. "5 of 6 readings" |
| `GET` | `/api/dashboard/status` | Database, model, provider and scheduler readiness |
| `GET` | `/api/dashboard/model` | Features, provenance, metrics and known limitations |

Interactive docs are at `/docs`. Full request/response reference in
[`docs/api.md`](docs/api.md).

### Collection predicts automatically

`POST /api/traffic/collect` stores each reading **first**, then scores it. The
order matters: a prediction failure can never cost an observation, because the
reading is already committed.

The response reports one outcome per reading:

| `status` | Meaning |
| --- | --- |
| `predicted` | Scored and stored. |
| `insufficient_history` | Too few prior readings — normal for a location's first six. |
| `model_unavailable` | No artifact. Training one is required; collecting more will not help. |
| `unscoreable_location` | The reading has no ML segment the model knows. |
| `failed` | Threw. Logged server-side; the response carries no internals. |

`insufficient_history` and `model_unavailable` are deliberately distinct: they
call for different fixes, and merging them would send an operator to collect
readings that will never be scored.

The scheduler runs the same path, so `COLLECTION_ENABLED=true` accumulates
history *and* forecasts without any further calls.

```powershell
# One reading now, six more, then a forecast on the seventh — no manual predict.
curl -X POST http://127.0.0.1:8000/api/traffic/collect `
  -H "Content-Type: application/json" -d '{"location_id":"LOC-001"}'
```

### Data provenance

The ML model was trained on **simulated data**. Connecting a live provider does not
change that, so three mechanisms keep the distinction from being ambiguous:

- every stored row carries a `source` label — `simulation`, `tomtom`, or whatever
  an API caller supplied — and a forecast over a `simulation` observation carries
  the note *"Predicted from simulated development data."*;
- `TRAFFIC_PROVIDER=real` fails with `provider_not_configured` when credentials are
  absent. It **never** falls back to simulated data — fabricated readings stored
  under a real provider's name are worse than an outage;
- `vehicle_count` is `NULL` on every live row, with the reason recorded in the
  observation's metadata, because no traffic API publishes throughput.

`GET /api/traffic/collector/status` makes the active provider explicit.

### Live traffic and weather

Set these in your untracked `.env`:

```powershell
TRAFFIC_PROVIDER=real
TRAFFIC_API_KEY=<your TomTom key>          # https://developer.tomtom.com/
TRAFFIC_API_BASE_URL=https://api.tomtom.com
COLLECTION_ENABLED=true                    # see "No history" below
```

Then collect once:

```powershell
curl http://127.0.0.1:8000/api/traffic/collector/status
curl -X POST "http://127.0.0.1:8000/api/traffic/collect?location_id=LOC-001"
```

Roads come from `config/monitored_locations.json`, which maps each coordinate to
one of the six ML segments. Edit it to monitor your own roads; an unknown
`road_segment_id` is rejected on load, because the model never learned that
category.

| Setting | Default | Purpose |
| --- | --- | --- |
| `TRAFFIC_API_VENDOR` | `tomtom` | Response mapping. An unknown vendor is rejected at start-up. |
| `TRAFFIC_API_KEY` | *(empty)* | TomTom key. Empty means unconfigured; the API still starts. |
| `TRAFFIC_ZOOM` | `10` | Point-query zoom. Raising it changes which segment a coordinate resolves to. |
| `TRAFFIC_MIN_REQUEST_INTERVAL` | `0.15` | Throttle. TomTom documents 10 QPS for non-tile traffic endpoints. |
| `TRAFFIC_MAX_LOCATIONS` | `25` | Guard against a config mistake spending the monthly quota in one cycle. |
| `WEATHER_ENABLED` | `true` | Open-Meteo needs no key. Turning it off stores readings the model cannot score. |
| `COLLECTION_ENABLED` | `false` | Background collection. Off by default: it spends quota and writes rows. |
| `COLLECTION_INTERVAL_SECONDS` | `300` | Minimum 30. |

### No history, so the scheduler matters

TomTom publishes **current conditions only**, and the model needs six prior
readings per segment before it will score. So a live deployment has to accumulate
its own history: either `COLLECTION_ENABLED=true`, or POST observations yourself.

`fetch_historical_traffic` raises `upstream_provider_error` explaining this rather
than returning an empty list, because an empty history reads as "no data exists"
and would leave a live location permanently unscoreable with no indication why.

At the default 300s interval with the 6 shipped locations, that is 288 traffic
reads and 288 weather calls per day — within TomTom's documented 20,000/month
traffic allowance and Open-Meteo's 300,000/month.

### Trying it

```powershell
# 1. Confirm the model and provider are wired up.
curl http://127.0.0.1:8000/api/traffic/predictions/status
curl http://127.0.0.1:8000/api/traffic/collector/status

# 2. Seed ~12 hours of history for LOC-001 (synthetic, labelled).
curl -X POST http://127.0.0.1:8000/api/traffic/collect/history `
  -H "Content-Type: application/json" -d '{"location_id":"LOC-001"}'

# 3. Collect a fresh reading: it is stored first, then scored automatically.
curl -X POST http://127.0.0.1:8000/api/traffic/collect `
  -H "Content-Type: application/json" -d '{"location_id":"LOC-001"}'

# 4. Read the forecast back from the history and latest endpoints.
curl http://127.0.0.1:8000/api/traffic/predictions
curl http://127.0.0.1:8000/api/traffic/predictions/latest

# 5. One poll per dashboard view - no per-location requests from the client.
curl http://127.0.0.1:8000/api/dashboard/current
curl "http://127.0.0.1:8000/api/dashboard/trends?location_id=LOC-001"
curl http://127.0.0.1:8000/api/dashboard/locations
```

Collect reports the outcome rather than failing: without six prior readings the
reading is stored and returned as `insufficient_history`, never as a fabricated number.
The same readings are scored by manual predict calls too, which return a `422`.

### Why prediction needs history

The model's lag and rolling features read backwards over the previous six
observations on the same road segment. A fresh location has nothing to read, so
`POST /api/traffic/predict` stores the reading, finds too little history, and
returns a `422` that names how many readings exist and how many are required.

History is scoped to `road_segment_id`, not just `location_id`: the features are
built per segment, so mixing another segment's readings in would corrupt them.
This also means `speed_ratio` is always recomputed by `ml.predict` rather than
supplied by the caller.

## Deployment

Production is a **hard guard, not a soft fallback**. When `ENVIRONMENT=production`
the application refuses to start unless `TRAFFIC_PROVIDER=real`, a
`TRAFFIC_API_KEY` and a PostgreSQL `DATABASE_URL` are all set — a missing key (or
a `sqlite://` DSN) fails at boot instead of silently serving simulated traffic.
`DB_FAIL_FAST=true` makes an unreachable database fail the same way.

The verified server startup command runs from the **project root** (running from
`backend/` breaks the `ml` import):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

- `GET /api/health` is the liveness probe.
- `GET /api/dashboard/status` is readiness: database, provider, scheduler, model
  and data freshness.
- Apply the schema with `alembic upgrade head` (DSN comes from `.env`).

The full deployment guide — prerequisites, production `.env`, reverse-proxy and
static-frontend serving, quotas, verification and troubleshooting — is in
[`docs/deployment.md`](docs/deployment.md).

## Running the tests

```powershell
pytest -q
```

450 tests covering the health endpoint contract, the error envelope,
configuration validation (including the Stage 8 production guard), database
engine wiring, the traffic schemas, the data providers, the TomTom and Open-Meteo
response mappings, the locations file, the background scheduler, the full
observation → prediction → persistence flow against
the real trained model, automatic prediction after collection (including its
failure modes), the prediction history and latest endpoints, every dashboard
endpoint with its freshness and query-count behaviour, the Alembic migrations
against both SQLite and the PostgreSQL dialect, the whole ML pipeline and
trained-model contract, Stage-7 frontend structure, element-contract and
no-secrets validation, and the Stage 8 frontend XSS/escaping contract.
No external services are required. The database tests use an isolated
in-memory SQLite database, provider tests drive `httpx.MockTransport`, and the
migrations are verified by rendering their SQL for PostgreSQL offline. The
simulated fixture is generated automatically. The Open-Meteo provider has also
been verified with a live one-off call; live TomTom requires a real API key.

### What the tests pin down

- simulated readings are **always** labelled `source="simulation"`, even if a
  provider record claims otherwise;
- the real provider raises rather than falling back to synthetic data;
- insufficient history is a `422`, never an invented prediction;
- a `speed_ratio` or `congestion_level` field in a request is rejected;
- the database rejects a `free_flow_speed_kph` below `avg_speed_kph` even when
  the write bypasses the API;
- no credential appears in any response, error envelope or committed file.

## Project structure

```
traffic-prediction-system/
├── backend/
│   ├── app/
│   │   ├── main.py              application factory, lifespan, CORS
│   │   ├── api/                 routers: health · traffic · prediction · collector
│   │   ├── core/                config · logging · exceptions
│   │   ├── database/            declarative base · lazy engine · sessions
│   │   ├── models/              ORM models (observations · predictions)
│   │   ├── repositories/        all SQL lives here
│   │   ├── schemas/             Pydantic contracts
│   │   └── services/            traffic · collector · providers · prediction
│   │                            weather_client · locations · collection_scheduler
│   └── requirements.txt
├── config/
│   └── monitored_locations.json  monitored roads, mapped to SEG-01..06
├── ml/
│   ├── data/
│   │   ├── raw/                 real dataset + schema README (untracked data)
│   │   ├── sample/              SIMULATED fixture generator (untracked CSV)
│   │   ├── processed/           traffic_processed.csv + .meta.json
│   │   └── data_summary.md      generated pipeline summary
│   ├── models/                  joblib artifacts (empty by design)
│   ├── notebooks/
│   ├── preprocessing.py         load · validate · clean · features · target
│   ├── train.py                 estimator, training, artifact IO
│   ├── evaluate.py              held-out scoring
│   ├── predict.py               inference
│   └── requirements.txt
├── migrations/                  Alembic environment + versions
├── frontend/
│   ├── index.html               landing page + navigation
│   ├── dashboard.html           stage-7 dashboard page
│   ├── css/{style.css,dashboard.css}
│   ├── js/{config.js,api.js,ui.js,app.js,map.js,charts.js,dashboard.js}
│   └── vendor/                   Leaflet + Chart.js (no CDN)
├── tests/                       pytest suite
├── docs/                        architecture · api · ml-pipeline · roadmap · deployment · project-report · presentation
├── .env.example
├── .gitignore
├── alembic.ini                  no DSN committed; resolved from settings
├── pyproject.toml
├── requirements.txt             aggregates backend + ml + test deps
└── README.md
```

## Configuration

All settings live in `.env` (copy `.env.example`; `.env` is git-ignored).

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | *(empty)* | PostgreSQL DSN; empty allows a database-free start |
| `DB_ECHO` | `false` | Log all SQL |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` / `DB_POOL_TIMEOUT` | `5` / `10` / `30` | Pool tuning |
| `DB_CONNECT_TIMEOUT` | `5` | Seconds to wait for a new connection |
| `DB_FAIL_FAST` | `false` | Refuse to start if PostgreSQL is unreachable |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Bind address |
| `API_RELOAD` | `true` | Auto-reload in development |
| `API_LOG_LEVEL` | `INFO` | Log verbosity |
| `SERVICE_NAME` | `traffic-prediction-api` | Reported by `/api/health` |
| `ENVIRONMENT` | `development` | Environment label |
| `CORS_ORIGINS` | local origins + `"null"` | Allowed browser origins |
| `TRAFFIC_PROVIDER` | `simulation` | `simulation` (synthetic, labelled) or `real` (TomTom + Open-Meteo) |
| `TRAFFIC_API_VENDOR` | `tomtom` | Response mapping; unknown vendors are rejected at start-up |
| `TRAFFIC_API_BASE_URL` | `https://api.tomtom.com` | Traffic API base URL; must be https |
| `TRAFFIC_API_KEY` | *(empty)* | TomTom key; empty means unconfigured |
| `TRAFFIC_ZOOM` | `10` | Point-query zoom level (0–22) |
| `TRAFFIC_REQUEST_TIMEOUT` | `10` | Seconds to wait for one traffic request |
| `TRAFFIC_MIN_REQUEST_INTERVAL` | `0.15` | Throttle between upstream requests |
| `TRAFFIC_MAX_LOCATIONS` | `25` | Cap on roads per collection cycle |
| `TRAFFIC_LOCATIONS_FILE` | `config/monitored_locations.json` | Monitored roads and their ML segments |
| `WEATHER_ENABLED` | `true` | Fetch Open-Meteo weather alongside traffic |
| `WEATHER_API_BASE_URL` | `https://api.open-meteo.com/v1` | Open-Meteo endpoint; must be https |
| `WEATHER_REQUEST_TIMEOUT` | `10` | Seconds to wait for one weather request |
| `WEATHER_API_KEY` | *(empty)* | Unused; Open-Meteo's free tier needs no key |
| `COLLECTION_ENABLED` | `false` | Run background collection |
| `COLLECTION_INTERVAL_SECONDS` | `300` | Seconds between cycles; minimum 30 |
| `AUTO_PREDICT_AFTER_COLLECTION` | `true` | Score each reading as it is collected |
| `DATA_FRESHNESS_THRESHOLD_SECONDS` | `900` | Age beyond which a reading is reported `stale` |
| `DASHBOARD_TREND_MAX_HOURS` | `168` | Upper bound on a trend window |
| `ML_MODELS_DIR` | *(empty)* | Artifact directory; empty means `ml/models` |
| `ML_PREDICTION_HISTORY` | `12` | Prior observations loaded per segment; minimum 6 |

## Implementation status

### Working and verified

| Capability | Detail |
| --- | --- |
| FastAPI application | App factory, lifespan hooks, CORS, structured logging |
| Health endpoint | `GET /api/health` returns exactly `{"status":"healthy","service":"traffic-prediction-api"}` |
| Configuration | Validated settings from `.env`; no secrets in source |
| Database layer | Declarative base with a naming convention, lazy cached engine, `session_scope`, `get_db` dependency, pool tuning, PostgreSQL-only options |
| Error handling | One envelope for domain, validation, HTTP and unexpected errors; no internals or DSNs leaked to clients |
| Frontend | Landing page, responsive navigation shell for Dashboard / Traffic Map / Predictions / Analytics / About, live API health indicator, honest empty states |
| ML module | Independent package: data contract, training, evaluation and inference entry points, artifact bundles |
| Data pipeline | Six-stage pipeline: load, validate, clean, featurise, target, write. Emits `traffic_processed.csv` + metadata sidecar + `data_summary.md` |
| Model training | Random Forest on 26 features, chronological 80/20 split, macro F1 as the primary metric; refuses to train if a leakage column appears |
| Model evaluation | Accuracy, balanced accuracy, macro/weighted F1, per-class detail, confusion matrix PNG, majority-class baseline, and a split-integrity check |
| Inference | `predict_traffic` rebuilds features from raw records, returns the level, its name, the true max probability and the full distribution |
| Leakage control | Contemporaneous measurements excluded from inputs and refused at training time **and at the API boundary**; backward-only lags and rolling means; chronological split, never random |
| Provenance | SHA-256 per source, machine-readable sidecars, simulated data labelled as such in every report, stored row and API response |
| ORM | `traffic_observations` and `traffic_predictions`, FK with `ON DELETE CASCADE`, unique `(location_id, timestamp)`, indexes on timestamp/location/road name, CHECK constraints mirroring the API validation. `vehicle_count` is nullable |
| Migrations | Alembic; DSN resolved from settings so no credential is committed; refuses to run without a target; downgrade is a table drop, never a schema wipe. Three revisions: table creation, the nullable-`vehicle_count` change, and the Stage 6 prediction-history index |
| Traffic API | Store, list, fetch and retrieve observations; duplicate detection; bounded pages with echoed filters |
| Prediction API | Stores the reading, scores it against that location's stored history, persists the forecast, returns both; 422 when history is insufficient |
| Data providers | `TrafficDataProvider` interface; deterministic seeded `SimulatedTrafficProvider` always stamped `simulation`; `RealTrafficProvider` fails loudly rather than substituting synthetic data |
| Live traffic | TomTom Flow Segment Data per monitored road: current and free-flow speed, confidence, road closure. Every field validated; a 2xx body missing a usable speed is an error, not a default. `vehicle_count` stays `NULL` with the reason recorded |
| Live weather | Open-Meteo, no API key; WMO codes mapped to the model's three labels, `timezone=GMT` so stored times are UTC. A missing temperature stays `NULL` rather than becoming 0 °C |
| Upstream errors | 429 → 503 `provider_rate_limited`, 401/403 and other 4xx/5xx → 502, timeout → 504, unusable body → 502 `provider_invalid_response`. Vendor bodies are never echoed, because they contain the request URL and the API key |
| Monitored locations | `config/monitored_locations.json` maps coordinates to `SEG-01..06`, validates ranges and uniqueness, and rejects unknown segments at load |
| Background collection | Opt-in `CollectionScheduler` on an asyncio task; synchronous work runs in a worker thread, a failed cycle is counted not fatal, and shutdown cancels a pending sleep |
| Automatic prediction | Collection stores readings first, then scores each one and reports an outcome per reading. Insufficient history, a missing artifact, an unknown segment and a scoring failure are four distinct statuses, and none of them costs an observation |
| Prediction history | `GET /api/traffic/predictions` and `/latest`, joined to the observation each forecast describes, with a bounded limit and an explicit `404` when nothing exists yet |
| Dashboard API | `current`, `trends`, `locations`, `status` and `model`. Fixed query count per request (window functions, never N+1), freshness measured from stored timestamps against a configurable threshold, and a `require_fresh` opt-out that returns `409` instead of stale data |
| Missing data stays missing | An uncollected road reports `null` speed and `freshness: "unavailable"`; an observation with no forecast contributes a chart gap. Nothing is zero-filled or interpolated, because both would render as claims about traffic |
| Model disclosure | `/api/dashboard/model` reports `dataset_is_simulated`, `validated_on_real_traffic`, `majority_class_baseline_accuracy` and an explicit `known_limitations` list |
| Security | Credentials only from `.env`; empty in `.env.example`; no DSN in `alembic.ini` or any migration; generic database error messages so a driver exception cannot echo the DSN; vendor bodies never echoed (they contain the request URL and the API key) |
| Production config guard | `ENVIRONMENT=production` requires `TRAFFIC_PROVIDER=real`, a `TRAFFIC_API_KEY` and a PostgreSQL `DATABASE_URL` (no `sqlite://`); start-up fails naming the missing setting, never echoing its value — no silent fallback to simulated traffic |
| Frontend XSS hardening | `UI.esc` escapes `& < > " '`; every `innerHTML`-producing template routes API data through it, pinned by contract tests |
| Health vs readiness | `/api/health` is liveness; `/api/dashboard/status` is readiness (database, provider, scheduler, model, freshness); `DB_FAIL_FAST=true` recommended for boot-time failures |
| Deployment documentation | `docs/deployment.md`: production `.env`, the verified server startup command (project root + `--app-dir backend`), Alembic apply, reverse-proxy and static-frontend serving, vendor quotas, a verification checklist and a troubleshooting table |
| Tests | 450 passing tests spanning API contract, errors, config (incl. production guard), DB wiring, schemas, providers, mocked TomTom/Open-Meteo, locations, scheduler, end-to-end prediction against the real artifact, migrations, dashboard and freshness behaviour, the ML pipeline, Stage-7 frontend structure/contract, and Stage-8 XSS/escaping |
| Docs | Architecture, API reference, ML pipeline design, roadmap, deployment guide, 24-section project report, 12-slide presentation, talking points |

### Not implemented — and deliberately so

| Item | Why |
| --- | --- |
| Vehicle throughput | **No self-serve traffic API publishes it.** Not TomTom, not HERE v7, not Mapbox. `vehicle_count` is `NULL` on live rows and `flow_veh_per_hr` was dropped from the model in 4.0.0, rather than storing an estimate that would look like a measurement |
| Historical traffic | TomTom publishes current conditions only; HERE v7 has no self-serve history; Mapbox's historical product is Enterprise-only. History is accumulated by the scheduler or POSTed |
| Real traffic dataset | Not acquired; only a simulated fixture exists, so reported metrics describe synthetic data only |
| Container images, CI pipeline | Documented in `docs/deployment.md` and the roadmap as follow-up: no registry or build runner is owned here, so they are not pretended to be shipped |
| Authentication, payments | Out of scope by design |

A trained model exists and the full ingest → store → predict → persist path works
against it, but the model is trained on synthetic data with a label derived from
`speed_ratio`, not on measured ground truth. Feeding it live traffic does not
validate it. Every artifact, report and API response states this. Treat it as a
working pipeline awaiting a real dataset and model retraining, not as a validated
traffic predictor.

## Development stages

| Stage | Focus | Status |
| --- | --- | --- |
| **1 - Foundation** | API, config, DB wiring, ML layout, frontend shell, tests, docs | **Complete** |
| **2 - Data pipeline** | Data contract, leakage-safe features, six-stage preprocessing, provenance | **Complete** |
| **3 - Model** | Chronological split, Random Forest, evaluation with an honest baseline, inference API | **Complete** |
| **4 - Data layer** | PostgreSQL schema, Alembic, providers with a labelled simulation, observation endpoints | **Complete** |
| **5 - Live traffic** | TomTom + Open-Meteo, monitored locations, opt-in scheduler, provider error mapping | **Complete** |
| **6 - Prediction engine** | Auto-prediction after collection, history/latest endpoints, dashboard API, freshness, query-count discipline | **Complete** |
| **7 - Frontend dashboard** | Leaflet map, Chart.js trends, auto-refresh, honest empty states | **Complete** |
| **8 - Hardening** | Production config guard, security/XSS review, health-vs-readiness, deployment guide, project report, presentation deck | **Complete** |

See `docs/roadmap.md` for the full breakdown.

## License

Not yet specified. Add a `LICENSE` file before publishing the repository.
