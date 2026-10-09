# Deployment guide

How to run the AI-Based Urban Traffic Congestion Prediction System outside a
developer machine. Every operational decision defaults to **safe and explicit**:
production refuses to start on demo configuration, no credential is committed,
and the dashboard always makes the model's limitations visible.

This guide is **documentation, not automation** — there is no Docker image yet.
Containerising the API is left for the team that owns a registry, so the
instructions here run against plain Python on a server. The same environment
variables drive every deployment.

---

## 1. What you are deploying

| Component | Runs where | Notes |
| --- | --- | --- |
| FastAPI backend | Python server (`uvicorn`) | Reads `.env`, owns the scheduler |
| PostgreSQL 14+ | Your database host | Owns every observation and forecast |
| Static frontend | Any static file server | `frontend/` — no build step, no backend process |
| ML artifacts | `ml/models/` on the API host | Trained **offline**; shipped as files |

The API and the ML layer are deliberately separable. The API server needs the
`ml/models/` artifact bundle at runtime, but does not need scikit-learn to
*import* — `app.services.prediction_service` pulls it in lazily — so a deployment
can serve pre-trained artifacts with only `backend/requirements.txt` installed.

---

## 2. Production is a hard guard, not a soft fallback

Every runtime setting has a **safe default** so a development checkout starts
with nothing configured. Those same defaults are dangerous in production. When
the environment label is `production`, the application refuses to start unless
**all** of these are set:

| Requirement | Why |
| --- | --- |
| `ENVIRONMENT=production` | The guard switches on |
| `TRAFFIC_PROVIDER=real` | The collector must **not** serve simulated data as live traffic |
| `TRAFFIC_API_KEY=<TomTom key>` | The real provider needs a credential; boot fails loudly, not at first collection |
| `DATABASE_URL=postgresql+psycopg2://...` | A production stack without a database is a broken one |
| PostgreSQL DSN — `sqlite://` is rejected | SQLite is the unit-test target, never a production store |

A missing setting fails at start-up with a message naming the variable — never
echoing the secret's value — instead of half-working. `DB_FAIL_FAST=true` is
recommended alongside it so an unreachable database also aborts boot.

---

## 3. Prerequisites

- Python **3.11 or newer** on the API host.
- PostgreSQL **14 or newer**, reachable and with a created database:
  ```sql
  CREATE DATABASE traffic_prediction;
  CREATE USER appuser WITH PASSWORD '<pick-a-password>';
  GRANT ALL PRIVILEGES ON DATABASE traffic_prediction TO appuser;
  ```
- A **TomTom API key** (Flow Segment Data product): <https://developer.tomtom.com/>
- Your own database schema ownership is managed by the repo's Alembic revision
  chain — you do not hand-write DDL.

> Live TomTom verification is impossible without a `TRAFFIC_API_KEY`, and is not
> something this repository can do on your behalf. The provider path is covered by
> mock-transport contract tests; the weather provider (Open-Meteo, no key) has
> additionally been verified live. Wire in your key and run `POST
> /api/traffic/collect` once to confirm end to end.

---

## 4. Prepare the API host

```powershell
# Windows PowerShell
git clone <repository-url> traffic-prediction-system
cd traffic-prediction-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

```bash
# Linux / macOS
git clone <repository-url> traffic-prediction-system
cd traffic-prediction-system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Ship the trained artifacts too if this host does not train them: copy the whole
`ml/models/` directory from the training box, or point `ML_MODELS_DIR` at the
directory you copy them to.

---

## 5. Configure `.env`

Every value is read from the process environment or the untracked `.env` file in
the project root. A production-shaped `.env`:

```dotenv
# --- Service ------------------------------------------------------------
SERVICE_NAME=traffic-prediction-api
ENVIRONMENT=production

# --- Database ------------------------------------------------------------
DATABASE_URL=postgresql+psycopg2://appuser:<password>@<db-host>:5432/traffic_prediction
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_CONNECT_TIMEOUT=5
DB_FAIL_FAST=true
DB_ECHO=false

# --- API server ----------------------------------------------------------
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
API_LOG_LEVEL=INFO

# --- CORS ----------------------------------------------------------------
# List every origin that really calls the API: the dashboard origin, never "*"
# unless you also whitelist a building-wide proxy.
CORS_ORIGINS=["https://dashboard.example.com"]

# --- Traffic provider ----------------------------------------------------
TRAFFIC_PROVIDER=real
TRAFFIC_API_VENDOR=tomtom
TRAFFIC_API_KEY=<your-tomtom-key>
TRAFFIC_API_BASE_URL=https://api.tomtom.com
TRAFFIC_ZOOM=10
TRAFFIC_REQUEST_TIMEOUT=10
TRAFFIC_MIN_REQUEST_INTERVAL=0.15
TRAFFIC_MAX_LOCATIONS=25

# --- Monitored roads -----------------------------------------------------
# Absolute path is used as given; a relative one resolves against the project root.
TRAFFIC_LOCATIONS_FILE=/etc/traffic-prediction/monitored_locations.json

# --- Weather -------------------------------------------------------------
WEATHER_ENABLED=true
WEATHER_API_BASE_URL=https://api.open-meteo.com/v1
WEATHER_REQUEST_TIMEOUT=10

# --- Collection ----------------------------------------------------------
# Real deployments need this: TomTom reports current conditions only, and the
# model needs six prior readings per segment before it will score.
COLLECTION_ENABLED=true
COLLECTION_INTERVAL_SECONDS=300
AUTO_PREDICT_AFTER_COLLECTION=true

# --- Freshness / dashboard ------------------------------------------------
DATA_FRESHNESS_THRESHOLD_SECONDS=900
DASHBOARD_TREND_MAX_HOURS=168

# --- ML -------------------------------------------------------------------
ML_MODELS_DIR=/srv/traffic-prediction/ml/models
ML_PREDICTION_HISTORY=12
```

All other knobs (and their defaults) are documented in [`.env.example`](../.env.example).

---

## 6. Apply the schema

Alembic refuses to run with no database target rather than guessing:

```bash
alembic upgrade head
```

Review the exact PostgreSQL DDL without touching the server first:

```bash
alembic upgrade head --sql > /tmp/schema.sql
```

The full round-trip (upgrade → verify → downgrade → re-upgrade) is exercised by
the test suite against SQLite and by the migrations tests against the PostgreSQL
dialect offline; a live PostgreSQL round-trip was not possible in the
verification environment because no server was available. If you do have one, the
same `alembic upgrade head` is the whole procedure — the chain is three revisions
and idempotent under Alembic's version table.

---

## 7. Start the API

Run from the **project root** (`--app-dir backend`). Running from inside
`backend/` breaks the `ml` import because the repository root is not on
`sys.path`:

```bash
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

**One worker.** The background scheduler and the engine pool live in this
process; running multiple uvicorn workers would run several schedulers against
one database. For capacity, scale behind a proxy and run the scheduler on one
worker, or run collection as a separate externally-scheduled `POST
/api/traffic/collect` while keeping `COLLECTION_ENABLED=false`.

### Liveness vs readiness

| Probe | Meaning |
| --- | --- |
| `GET /api/health` | The process is up and serving. `{"status":"healthy","service":...}` |
| `GET /api/dashboard/status` | Readiness of every dependency: database status, model availability & validation, provider configured/simulated, scheduler status & cycle counts, data freshness (`fresh` / `stale` / `unavailable`) and overall `status` |

`/api/health` is deliberately liveness-only: lying "healthy" while PostgreSQL is
down would be misleading, and `/api/dashboard/status` is where a load balancer or
an operator should decide readiness. With `DB_FAIL_FAST=true` the process refuses
to boot on a bad DSN.

### Behind a reverse proxy

The API is **unauthenticated by design** (noted in the roadmap as a pre-public
requirement). Put it behind a TLS-terminating reverse proxy and restrict who can
call it:

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;
    # ssl_certificate ...; ssl_certificate_key ...;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }
}
```

Before any **public** exposure, add gateway-level authentication and rate
limiting. The upstream request throttles inside the collector
(`TRAFFIC_MIN_REQUEST_INTERVAL`, `TRAFFIC_MAX_LOCATIONS`) protect the vendor
quota, not your API surface area.

---

## 8. Serve the frontend

The frontend is static — point any file server at `frontend/`:

```nginx
server {
    listen 443 ssl;
    server_name dashboard.example.com;
    root /srv/traffic-prediction/frontend;
    index index.html;
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

Tell the browser where the API lives in `frontend/js/config.js`
(`window.TRAFFIC_CONFIG.API_BASE_URL`), and list the dashboard origin in the
backend's `CORS_ORIGINS`. The dashboard is safe-by-default: every API-derived
value is HTML-escaped before it is inserted, Leaflet and Chart.js are vendored
(no CDN), and a failing tile server degrades to a notice rather than breaking
other panels.

---

## 9. Verify a production-style deployment

```bash
# The process boots only if the production guard passes.
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 &

curl http://127.0.0.1:8000/api/health
# {"status":"healthy","service":"traffic-prediction-api"}

# Readiness: database, provider, scheduler, model, freshness.
curl http://127.0.0.1:8000/api/dashboard/status
# {"status":"degraded",...,"database_status":"connected",
#  "traffic_provider":"real",...,"model_available":true,"data_freshness":"stale"}

# Authenticated with YOUR key — a live one-shot collect + auto-predict.
curl -X POST http://127.0.0.1:8000/api/traffic/collect \
  -H "Content-Type: application/json" -d '{"location_id":"LOC-001"}'
```

An expected first-run state: the dashboard reports `unavailable`/`insufficient_history`
until the scheduler has gathered six readings per segment — the seventh produces
the first forecast. That is the honest behaviour, not a fault.

---

## 10. Operations notes

- **Vendor quotas.** TomTom documents ~20,000 requests/month for Flow Segment
  Data; Open-Meteo ~10,000 calls/day. At the default 300 s interval with 6
  locations that is ~288 traffic + weather calls per day, comfortably inside both.
  `TRAFFIC_MAX_LOCATIONS` caps a misconfigured locations file.
- **Upstream failures are classified, not surfaced.** 429 → 503
  `provider_rate_limited`; other 4xx/5xx → 502; timeout → 504; an unusable body →
  502 `provider_invalid_response`. Vendor bodies are never echoed (they contain
  the request URL and the API key).
- **Scheduler status** is in `/api/dashboard/status`
  (`collection_cycles_completed`, `collection_cycles_failed`); a failed cycle is
  counted, logged, and never fatal.
- **Backfill** a fresh location with `POST /api/traffic/collect/history` (simulated
  data, clearly labelled) rather than waiting six cycles.
- **Logs** never contain credentials: the DSN, the API key, and vendor bodies
  are excluded, and Alembic reports a `<redacted>` URL scheme.

---

## 11. Known limitations you must take into your planning

- **The model is trained on simulated data.** Predictions are demonstrative.
  `GET /api/dashboard/model` reports `dataset_is_simulated: true` and
  `validated_on_real_traffic: false`, and the dashboard prints that next to every
  forecast. Require a real dataset and a retrained, re-validated model before
  treating a forecast as an operational number.
- **`vehicle_count` is `NULL` on every live reading.** No self-serve traffic API
  publishes throughput; the column is null with the reason in the observation's
  metadata, and model 4.0.0 dropped that feature.
- **No historical traffic feed exists**; a fresh deployment accumulates its own
  history via the scheduler.
- **Live PostgreSQL round-trip** was not executed in the verification environment
  (no server). The migration chain is verified against SQLite online and the
  PostgreSQL dialect offline.

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Boot aborts with "ENVIRONMENT=production requires TRAFFIC_PROVIDER=real" | Guard tripped | Set `TRAFFIC_PROVIDER=real` and a `TRAFFIC_API_KEY` |
| Boot aborts with a `DATABASE_URL` error | Empty or `sqlite://` DSN in production | Set a `postgresql+psycopg2://` DSN |
| `503 provider_not_configured` | Real provider, no key | Set `TRAFFIC_API_KEY` |
| `503 provider_rate_limited` | Vendor quota | Increase `COLLECTION_INTERVAL_SECONDS`, trim locations |
| Dashboard says `stale` | No recent stored observation | Enable the scheduler or backfill |
| `insufficient_history` on a location forever | Fewer than six prior readings on that segment | Backfill or wait six cycles |
| `ModuleNotFoundError: ml` | uvicorn run from `backend/` | Run from the project root with `--app-dir backend` |
| Dashboard cannot reach the API | Wrong `API_BASE_URL` or CORS | Fix `frontend/js/config.js`; add the origin to `CORS_ORIGINS` |