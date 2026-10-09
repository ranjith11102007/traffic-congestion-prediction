# API reference

Base URL: `http://127.0.0.1:8000`

Interactive documentation is generated from the running service:

- Swagger UI — <http://127.0.0.1:8000/docs>
- ReDoc — <http://127.0.0.1:8000/redoc>
- OpenAPI schema — <http://127.0.0.1:8000/openapi.json>

Every path below is implemented. Where a document elsewhere in the repository
describes a plan rather than the current contract, this file is authoritative.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Service pointer (not in the OpenAPI schema) |
| GET | `/api/health` | Liveness probe |
| POST | `/api/traffic/observations` | Record an observation |
| GET | `/api/traffic/observations` | List observations |
| GET | `/api/traffic/observations/{id}` | Fetch one observation |
| GET | `/api/traffic/latest` | Most recent observation |
| POST | `/api/traffic/predict` | Store, predict, store the forecast |
| POST | `/api/traffic/observations/{id}/predict` | Predict from a stored observation |
| GET | `/api/traffic/predictions` | Prediction history, newest first |
| GET | `/api/traffic/predictions/latest` | Most recent stored prediction |
| GET | `/api/traffic/predictions/status` | ML integration status |
| GET | `/api/traffic/collector/status` | Active provider status |
| POST | `/api/traffic/collect` | Collect, then predict automatically |
| POST | `/api/traffic/collect/history` | Backfill history for prediction |
| GET | `/api/dashboard/current` | Current state of every location |
| GET | `/api/dashboard/trends` | Speed and forecast series |
| GET | `/api/dashboard/locations` | Per-location readiness |
| GET | `/api/dashboard/status` | Database, model, provider, scheduler |
| GET | `/api/dashboard/model` | Features, provenance, metrics, limits |

---

## `GET /api/health`

Liveness probe. Confirms the API process is serving requests. It does **not**
report database or model state — for that, use `GET /api/dashboard/status`,
which reports each dependency separately.

```json
{ "status": "healthy", "service": "traffic-prediction-api" }
```

---

## Traffic and prediction

### `POST /api/traffic/collect`

Fetches current readings from the configured provider, stores them, then scores
each one.

The order matters: **store first, predict second.** A prediction failure can
therefore never cost an observation, because the reading is already committed.

```bash
curl -X POST http://127.0.0.1:8000/api/traffic/collect \
  -H "Content-Type: application/json" \
  -d '{"location_id": "LOC-001"}'
```

**Request body**

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `location_id` | string \| null | `null` | Location to fetch; `null` uses the provider's first |
| `road_segment_id` | string \| null | `null` | Segment to attribute readings to, applied before scoring |
| `persist` | bool | `true` | `false` for a dry run — nothing is written, nothing is predicted |
| `predict` | bool | `true` | `false` to collect without scoring |

**Response — 201**

```json
{
  "provider": "tomtom",
  "is_simulation": false,
  "observations": [ ... ],
  "count": 1,
  "persisted": true,
  "predictions": [
    {
      "observation_id": 8,
      "location_id": "LOC-001",
      "status": "predicted",
      "predicted_congestion": "moderate",
      "confidence": 0.71,
      "model_version": "4.0.0",
      "prediction_id": 3,
      "required_readings": 6,
      "available_readings": 8,
      "reason": null
    }
  ],
  "predictions_stored": 1
}
```

`predictions[].status` is the field that carries the outcome:

| Status | Meaning |
| --- | --- |
| `predicted` | Scored and stored. `prediction_id` is set. |
| `insufficient_history` | Fewer prior readings than the model needs. `reason` says how many. |
| `model_unavailable` | No model artifact. An operator must train one; collecting more readings will not help. |
| `unscoreable_location` | The reading has no `road_segment_id` the model knows. |
| `failed` | Scored and threw. Logged with a traceback; the response carries no internals. |

**Why these are distinguished:** `insufficient_history` and `model_unavailable`
call for different fixes — waiting versus training. Collapsing them into one
"could not predict" would send an operator to collect readings that will never
be scored.

**Possible responses**

| Status | When |
| --- | --- |
| 201 | Readings stored. Predictions attempted; per-reading outcomes reported. |
| 503 | The configured provider is not usable (for example, `TRAFFIC_PROVIDER=real` with no `TRAFFIC_API_KEY`). Never falls back to simulated data. |
| 500 | The database failed. |

### `GET /api/traffic/predictions`

Stored forecasts, newest first, joined to the observation each describes.

```bash
curl "http://127.0.0.1:8000/api/traffic/predictions?location_id=LOC-001&limit=25"
```

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `location_id` | string | all locations | |
| `start_time` | ISO-8601 | — | Bounds the **observation** time |
| `end_time` | ISO-8601 | — | Bounds the **observation** time |
| `limit` | int 1–500 | 50 | Hard cap; this endpoint is not paginated |

`start_time` filters on the observation rather than the moment the forecast was
generated, so the history list and the trend chart read the same window. Both
timestamps are returned on every row, so lead time is computable.

An empty list is a `200`, not an error: a young database legitimately has no
forecasts yet. Only `predictions/latest` turns emptiness into a `404`.

### `GET /api/traffic/predictions/latest`

The newest stored forecast, optionally for one location.

| Status | When |
| --- | --- |
| 200 | A forecast exists |
| 404 | `no_prediction_available` — nothing has been predicted yet |

### `POST /api/traffic/predict`

Stores a supplied observation, scores it, stores the forecast, returns both. The
complete prediction flow in one call.

Requires `road_segment_id`, `temperature_c` and `rainfall_mm`: the model cannot
score without them. Derived values such as speed ratio are **not** accepted as
input — they are computed by the feature pipeline, and accepting them from a
client would let a caller supply an answer.

**`422 insufficient_history`** with:

```json
{
  "error": {
    "code": "insufficient_history",
    "details": {
      "location_id": "LOC-001",
      "road_segment_id": "SEG-01",
      "required_readings": 6,
      "available_readings": 0
    }
  }
}
```

### `GET /api/traffic/predictions/status`

Whether a model is present, its version, whether it was trained on simulated
data, the columns it requires, the segments and weather labels it knows, and how
much history each location needs. Returns `200` with `model_available: false`
rather than raising — reporting the state is the point.

---

## Dashboard

Five read-only endpoints. Each is polled by a browser, so a single call must
cover a whole panel and a partial outage must still render something useful.

### `GET /api/dashboard/current`

The newest observation, newest forecast and a freshness verdict for every
configured location, plus any location that has data but is not configured.

```json
{
  "generated_at": "2026-10-05T18:00:00Z",
  "freshness_threshold_seconds": 900,
  "locations": [
    {
      "location_id": "LOC-001",
      "road_name": "Anna Salai",
      "avg_speed_kph": 32.4,
      "prediction": {
        "predicted_congestion": "moderate",
        "confidence": 0.71,
        "model_version": "4.0.0",
        "observation_timestamp": "2026-10-05T17:55:00Z",
        "prediction_timestamp": "2026-10-05T17:55:03Z"
      },
      "prediction_status": "available",
      "freshness": "fresh",
      "observation_age_seconds": 300.0,
      "data_source": "tomtom",
      "is_simulation": false
    },
    {
      "location_id": "LOC-004",
      "avg_speed_kph": null,
      "prediction": null,
      "prediction_status": "insufficient_history",
      "freshness": "unavailable",
      "data_source": null,
      "is_simulation": false
    }
  ],
  "location_count": 6,
  "locations_with_data": 5,
  "predictions_available": 4
}
```

**Fields worth reading carefully**

- `freshness` is `fresh`, `stale` or `unavailable`, computed from the observation
  timestamp against `DATA_FRESHNESS_THRESHOLD_SECONDS`. The threshold is echoed in
  the response so a client reaches the same verdict instead of guessing.
- `prediction_status` is one of `available`, `insufficient_history`,
  `unscoreable_location`, `model_unavailable`, `unavailable`.
- An uncollected location reports `null` speed, never `0`. Zero km/h renders as a
  standstill, which is a claim about traffic that no reading supports.
- `confidence` is `null` unless the estimator produced a real probability in 0..1.

Resolves to a fixed number of queries regardless of how many locations are
monitored.

### `GET /api/dashboard/trends`

Speed and forecast per point, oldest first, ready to plot.

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `location_id` | string | all locations | Unknown location → `404 invalid_location` |
| `hours` | int | 24 | Window length when no explicit bounds |
| `start_time` | ISO-8601 | — | Wins over `hours` |
| `end_time` | ISO-8601 | — | Wins over `hours` |
| `limit` | int ≤ 2000 | 2000 | Newest points win; `truncated: true` when trimmed |
| `require_fresh` | bool | `false` | `true` → `409 stale_data` instead of stale data |

An observation with no forecast carries `congestion_prediction: null`. The chart
draws a gap. Interpolating across it would render a congestion trend nobody
predicted.

A reversed window is `422 invalid_time_range`, not an empty series — an empty
result would be indistinguishable from a quiet road.

By default stale data is returned with `freshness: "stale"` and a `note`,
because history stays useful after it stops being current. `require_fresh=true`
is the opt-out for a panel that must never show old numbers as current.

### `GET /api/dashboard/locations`

Per-location operational state. `prediction_status` plus
`available_readings`/`required_readings` is what lets a client show
**"5 of 6 readings"** instead of a bare "no prediction" — the difference between
an unactionable message and an operator who knows to wait.

`available_readings` counts readings *ahead of* the newest one, since that is
what the lag features are built from.

### `GET /api/dashboard/status`

Readiness of every dependency, plus row counts and freshness.

```json
{
  "status": "degraded",
  "database_status": "connected",
  "model_version": "4.0.0",
  "model_status": "available",
  "model_trained_on_simulated_data": true,
  "traffic_provider": "tomtom",
  "traffic_provider_configured": false,
  "weather_provider": "open-meteo",
  "scheduler_enabled": true,
  "scheduler_running": true,
  "observation_count": 120,
  "prediction_count": 114,
  "data_freshness": "fresh"
}
```

`status` is `healthy`, `degraded` (usable but impaired — no model, an
unconfigured provider, stale data, failing collection cycles) or `unhealthy`
(database unreachable).

This endpoint must not be the one that fails, so a database outage is reported
as a field rather than raised as a 500.

**It exposes no configuration.** No DSN, no API key and no vendor response body
appears here, because this payload is read by a browser.

### `GET /api/dashboard/model`

The loaded artifact: version, class labels, every feature with its origin
(measured / lag / rolling / calendar / one-hot), how much history a location
needs, dataset provenance, held-out metrics, and `known_limitations`.

Two fields belong next to any forecast a client renders:

- `dataset_is_simulated` — currently `true`.
- `validated_on_real_traffic` — currently `false`.

`majority_class_baseline_accuracy` sits beside the model's accuracy, because
that accuracy means very little without it.

**`503 model_not_available`** when no artifact exists, rather than describing a
model that is not there.

---

## Errors

Every non-2xx response uses one envelope:

```json
{
  "error": {
    "code": "not_found",
    "message": "Human-readable and safe to display",
    "details": null
  }
}
```

```bash
curl -i http://127.0.0.1:8000/api/not-a-route
# HTTP/1.1 404 Not Found
```

Codes you are likely to meet:

| Status | Code | Raised by |
| --- | --- | --- |
| 404 | `not_found` | Missing observation or route |
| 404 | `no_prediction_available` | `predictions/latest` with nothing stored |
| 404 | `invalid_location` | Trends for an unknown location |
| 409 | `stale_data` | `require_fresh=true` on a stale window |
| 422 | `insufficient_history` | Too few prior readings to score |
| 422 | `invalid_time_range` | `start_time` after `end_time` |
| 503 | `provider_not_configured` | Real provider without an API key |
| 503 | `model_not_available` | No model artifact |

Domain errors are logged with their message and code. Stack traces, DSNs and
vendor response bodies stay in the server logs and never reach the envelope.

## CORS

The API allows the origins listed in `CORS_ORIGINS`. The defaults cover the local
API and a static file server on port 5500, plus `"null"` so the frontend also works
when `index.html` is opened directly from disk. Credentials are never allowed,
because the API is stateless and unauthenticated in this phase.

## Versioning

There is no URL version prefix yet. When a breaking change arrives, routes move
to `/api/v1/...` and the old prefix stays for one release.
