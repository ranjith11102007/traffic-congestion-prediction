# Architecture

## Design goals

1. **Honest capability.** No endpoint, metric or chart may present fabricated
   traffic data or a prediction that no trained model produced. Missing
   capability is reported as an explicit error.
2. **Modularity.** Every component is independently testable and replaceable:
   the ML layer never imports the API, and the API never imports training code.
3. **Separation of concerns.** Requests in, schemas out. Business logic lives in
   services, persistence in a repository-style session layer, transport in routers.
4. **Config over hard-coding.** All environment-specific values arrive through
   environment variables; nothing secret is stored in the repository.

## Component diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│  Browser                                                              │
│  frontend/  (static HTML5 + CSS3 + JavaScript)                       │
│    index.html      landing page + status, calls GET /api/health       │
│    dashboard.html  placeholder for Stage 7 (Leaflet + Chart.js)       │
│    js/config.js    API base URL (no secrets)                          │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ HTTP / JSON  (CORS enabled)
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FastAPI application  —  backend/app                                  │
│                                                                       │
│  api/router.py ──► health, traffic, prediction, collector, dashboard   │
│        │                                                              │
│        ▼                                                              │
│  services/    health_service, traffic_collector,                       │
│               prediction_service, dashboard_service,                  │
│               collection_scheduler   business logic, no HTTP types    │
│        │                                                              │
│        ▼                                                              │
│  schemas/     health, traffic, prediction, dashboard                  │
│                                                                       │
│  core/config.py        settings from environment / .env               │
│  core/exceptions.py    domain errors + uniform error envelope         │
│  core/logging_config.py                                                    │
│                                                                       │
│  database/base.py      SQLAlchemy DeclarativeBase                     │
│  database/session.py   lazy engine, session_scope, get_db dependency  │
│  models/traffic.py     traffic_observations, traffic_predictions      │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │ SQLAlchemy Core / ORM
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  PostgreSQL  (connection details from DATABASE_URL, Alembic-managed)  │
└───────────────────────────────┘

              ▲ provider reads, collector writes
              │
┌───────────────────────────────────────────────────────────────────────┐
│  Ingestion  —  backend/app/services/                                  │
│    providers.py          TrafficDataProvider interface + TrafficRecord│
│    simulated_provider.py deterministic synthetic data, always labelled │
│    real_provider.py      TomTom Flow Segment Data + weather enrichment│
│    weather_client.py     Open-Meteo (no API key), WMO code mapping     │
│    locations.py          monitored roads -> SEG-01..06, validated      │
│    traffic_collector.py  fetch -> store -> score, duplicate policy     │
│    collection_scheduler.py  opt-in loop, failure counts, status         │
│    dashboard_service.py     aggregation reads, freshness, no N+1       │
└───────────────────────────────┘
                 │ HTTPS, opt-in
                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  External APIs                                                        │
│    TomTom   api.tomtom.com/traffic/services/flowSegmentData          │
│    Open-Meteo  api.open-meteo.com/v1/forecast                         │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│  ML module  —  ml/   (independent; never imports backend/app)         │
│    preprocessing.py   schema, features, congestion-label derivation   │
│    train.py           estimator build, fit, artifact IO               │
│    evaluate.py        held-out scoring                                │
│    predict.py         inference                                        │
│    models/            joblib artifact bundle                          │
└───────────────────────────────────────────────────────────────────────┘
```

The ML module is connected to the API only through a **model artifact file**
(`ml/models/*.joblib`), not through Python imports. That boundary is what lets the
API run without scikit-learn installed.

## Ingestion flow

```
POST /api/traffic/collect          CollectionScheduler (if COLLECTION_ENABLED)
  -> services/traffic_collector.py       resolves the provider from settings
  -> services/real_provider.py           TomTom point query per monitored road
       -> services/weather_client.py     Open-Meteo for the same coordinate
  -> services/traffic_collector.py       stamps provenance, skips duplicates
  -> repositories/traffic_repository.py  store FIRST, one transaction, commits
  -> services/prediction_service.py      then score the stored readings
       -> ml/predict.py                  outcome per reading: predicted,
                                         insufficient_history,
                                         model_unavailable,
                                         unscoreable_location, failed
  -> repositories/traffic_repository.py  persists any forecasts produced

POST /api/traffic/predict
  -> services/traffic_collector.py       stores the reading
  -> repositories/traffic_repository.py  loads prior readings for the segment
  -> services/prediction_service.py      422 if fewer than 6 readings exist
  -> ml/predict.py                       rebuilds features, scores
  -> repositories/traffic_repository.py  persists the forecast
```

Three properties of this path are deliberate and tested:

- **No fallback.** `TRAFFIC_PROVIDER=real` never substitutes synthetic data. An
  upstream failure surfaces as 502/503/504.
- **No fabrication.** A 2xx vendor response missing a usable speed is an error, not
  a default. `vehicle_count` stays `NULL` because no traffic API reports it.
- **Provenance is not optional.** The stored `source` comes from the provider's own
  `is_simulation` flag, so a provider cannot mislabel itself.
- **Store before score.** A prediction failure can never cost an observation: the
  reading is committed before the model is asked for anything, and each reading
  reports its own outcome instead of failing the whole cycle.

## Read path (dashboard)

```
GET /api/dashboard/current   (and trends, locations, status, model)
  -> api/dashboard.py            thin route, query validation
  -> services/dashboard_service.py
       - fixed number of aggregate queries per request (window functions)
       - latest observation and latest forecast per location
       - freshness from the newest stored timestamp vs the threshold
       - no per-location queries from the caller: N+1 is forbidden
  -> schemas/dashboard.py        Pydantic response contract
```

An uncollected road is returned with `null` speeds and
`freshness: "unavailable"` rather than being omitted, so the frontend never has
to invent a placeholder. `require_fresh=true` turns stale data into a `409`
instead of silently serving it.

## Read path (frontend, Stage 7)

```
frontend/dashboard.html            single responsive page, edit-free
  js/config.js                     TRAFFIC_CONFIG: API_BASE_URL, refresh, tiles
  js/api.js                        fetch wrapper -> /api/dashboard/*, /api/traffic/*
  js/ui.js                         shared escape / format / tone helpers
  js/dashboard.js                  controller: state, refresh cycle, retry, selection
  js/map.js                        Leaflet view (window.TrafficMap)
  js/charts.js                     Chart.js view (window.TrafficCharts)
  js/app.js                        shared chrome + health probe (index + dashboard)
  vendor/                          local Leaflet + Chart.js, no CDN
```

The frontend is static and talks only to the documented API endpoints; its base
URL is a config value, so no credential or endpoint is embedded in shipped HTML/JS.
Each panel owns its loading, error and empty state and renders only values the API
returned (the model's own labels, `N/A` for missing data, simulated-data notices).
Auto-refresh polls the endpoints as one batch per cycle (never per-location),
pauses in hidden tabs, and shows the API reachability/health in the header.

## Request flow (current)

```
GET /api/health
  -> api/router.py         matches /api/health
  -> api/health.py         delegates to the service
  -> services/health_service.py   reads SERVICE_NAME from settings
  -> schemas/health.py     validates the response shape
  -> {"status": "healthy", "service": "traffic-prediction-api"}
```

Database access is not on this path, which is deliberate: the health probe must
answer even when PostgreSQL is unavailable, and reporting `"healthy"` while the
database is down would be misleading. Database readiness will be exposed by a
separate endpoint. It now exists: `GET /api/dashboard/status` reports database, model, provider and scheduler readiness with `connected` / `unavailable` / `not_configured`.

## Layering rules

| Layer | May import | Must not import |
| --- | --- | --- |
| `api/` | `services`, `schemas`, `core` | `database` directly, other route modules |
| `services/` | `schemas`, `core`, `database` | `api`, FastAPI route types |
| `schemas/` | `core` | `api`, `services`, `database` |
| `database/` | `core`, `models` | `api`, `services` |
| `ml/` | nothing from `backend/` | `app`, FastAPI, SQLAlchemy |

## Error contract

All non-2xx responses share one envelope, so the frontend needs a single parser:

```json
{
  "error": {
    "code": "model_not_available",
    "message": "No trained model found at ml/models/random_forest.joblib.",
    "details": null
  }
}
```

| Code | HTTP | Meaning |
| --- | --- | --- |
| `validation_error` | 422 | Malformed request body or query parameters |
| `not_found` | 404 | Unknown route |
| `method_not_allowed` | 405 | Known route, wrong HTTP verb |
| `configuration_error` | 500 | A required environment variable is missing |
| `provider_not_configured` | 503 | `TRAFFIC_PROVIDER=real` without a key or a usable locations file |
| `database_unavailable` | 503 | PostgreSQL unreachable or transaction failed |
| `model_not_available` | 503 | Prediction requested before training |
| `insufficient_history` | 422 | Fewer prior readings than the model's lag depth needs |
| `duplicate_observation` | 409 | An observation already exists for that location and timestamp |
| `no_prediction_available` | 404 | No forecast exists yet for that location |
| `invalid_location` | 404 | The dashboard does not know that location |
| `invalid_time_range` | 422 | Trend window outside the allowed bounds |
| `stale_data` | 409 | `require_fresh=true` and the newest reading is older than the freshness threshold |
| `upstream_provider_error` | 502 | A vendor failed, rejected the request, or returned an unusable body |
| `provider_invalid_response` | 502 | A 2xx vendor body missing a field the record requires |
| `upstream_timeout` | 504 | A vendor request exceeded its timeout |
| `provider_rate_limited` | 503 | The vendor reported throttling or exhausted quota |
| `internal_error` | 500 | Unexpected failure; details stay in server logs |

The four upstream codes distinguish *whose* fault a failure was, because that is
what decides whether to retry. A timeout is worth retrying; a rejected key is a
configuration problem that no amount of retrying will fix. Vendor response bodies
are never forwarded to the client — they echo the request URL, which carries the
API key as a query parameter.

## Adding a new endpoint

1. Add or extend a Pydantic schema in `app/schemas/`.
2. Put the logic in a service in `app/services/`; it returns schema objects.
3. Add a router in `app/api/` and include it in `app/api/router.py`.
4. Raise a `TrafficPredictionError` subclass for expected failures — never return
   a fabricated success payload.
5. Add tests under `tests/`.

## Deployment shape

The target is a single process running the API behind a reverse proxy, with
PostgreSQL as a managed service and the model artifact shipped alongside the
code. Migrations run as a separate step before the new revision starts, via
`alembic upgrade head`, with the DSN supplied as an environment variable. The
full walkthrough is [`docs/deployment.md`](deployment.md).

Three Stage 8 facts shape that deployment:

- **Production is a hard guard.** With `ENVIRONMENT=production` the application
  refuses to boot unless `TRAFFIC_PROVIDER=real`, `TRAFFIC_API_KEY` and a
  PostgreSQL `DATABASE_URL` are set — a misconfigured deployment fails at start
  naming the missing setting instead of silently serving simulated traffic.
- **Run one worker.** The scheduler and the engine pool are in-process; several
  uvicorn workers would run several schedulers against one database. Scale behind
  the proxy or drive collection externally.
- **Liveness and readiness are distinct.** `GET /api/health` proves the process
  is up; `GET /api/dashboard/status` reports the readiness of each dependency
  (database, provider, scheduler, model, freshness). `DB_FAIL_FAST=true` makes an
  unreachable database abort boot rather than report "healthy".

The background collection scheduler is **opt-in**. A process that starts with
`COLLECTION_ENABLED=true` spends vendor quota whether or not anyone is watching,
so the flag belongs in the deployment config, not in the code default.

API keys and the database DSN come from the environment. Neither is baked into
an image, and neither is written to `alembic.ini` or any migration, so a CI log
capturing Alembic output cannot leak one.
