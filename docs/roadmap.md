# Roadmap

The project is delivered in stages. Each stage ends with something genuinely
runnable, and nothing downstream is built on an unverified assumption.

Terminology note: earlier drafts of this document used "phases" and a four-phase
data plan. The work is now tracked as **stages**, which is the wording the
implementation and the API descriptions use.

## Stage 1 — Foundation *(done)*

| Item | State |
| --- | --- |
| FastAPI application with app factory and lifespan hooks | Done |
| `GET /api/health` returning the agreed two-field payload | Done |
| Environment-driven configuration via `.env`, no hard-coded secrets | Done |
| PostgreSQL + SQLAlchemy connection layer (lazy engine, sessions, pool tuning) | Done |
| Centralised error handling with a uniform envelope | Done |
| CORS configured for the local frontend | Done |
| Frontend landing page, navigation shell, live health indicator | Done |
| ML module layout, data contract, training/evaluation/inference entry points | Done |
| Automated tests for API, config and DB wiring | Done |

Deliberately excluded from this stage: authentication, payments, fake data.

## Stage 2 — Data pipeline and model training *(done)*

- Six-stage pipeline: load, validate, clean, featurise, target, write.
- Leakage controls: contemporaneous measurements excluded and refused at training
  time *and* at the API boundary; backward-only lags and rolling means.
- Random Forest on 26 features, chronological 80/20 split, macro F1 primary.
- Evaluation report with the majority-class baseline for comparison.
- Artifacts: model, preprocessor, metadata sidecar, feature importance.

Exit criteria met: reproducible training from a clean checkout, metrics published
with their split strategy.

## Stage 3 — Prediction API and ORM *(done)*

- `traffic_observations` and `traffic_predictions` with CHECK constraints
  mirroring the API validation.
- Alembic migrations with no DSN committed; downgrade drops tables, never wipes.
- `POST /api/traffic/predict` stores a reading, scores it against stored history,
  persists the forecast, returns both.
- `422 insufficient_history` naming required versus available readings, rather
  than a fabricated number.
- Artifact loader service with startup validation.

## Stage 4 — Ingestion pipeline *(done)*

- `TrafficDataProvider` interface with a deterministic simulated implementation
  always stamped `simulation`, and a real implementation that fails loudly rather
  than substituting synthetic data.
- Validation of provider responses; a 2xx body missing a usable speed is an error.

## Stage 5 — Live traffic, weather and configuration *(done)*

- TomTom Flow Segment Data per monitored road; `vehicle_count` nullable because
  no self-serve API publishes throughput.
- Open-Meteo weather, no API key, WMO codes mapped to the model's labels.
- `config/monitored_locations.json` maps coordinates to `SEG-01..06`.
- Opt-in background collection with failure counted, not fatal.
- Provider error mapping: 429 → `provider_rate_limited`, timeout → 504, and so on.
- Vendor bodies never echoed — they contain the request URL and the API key.

## Stage 6 — Real-time prediction engine and dashboard API *(done)*

| Item | State |
| --- | --- |
| Collection stores readings first, then scores each one | Done |
| Per-reading outcome: `predicted`, `insufficient_history`, `model_unavailable`, `unscoreable_location`, `failed` | Done |
| Prediction history and latest endpoints | Done |
| `GET /api/dashboard/current` — one poll for every location | Done |
| `GET /api/dashboard/trends` — chart series with `require_fresh` opt-out | Done |
| `GET /api/dashboard/locations` — readiness, e.g. "5 of 6 readings" | Done |
| `GET /api/dashboard/status` — database, model, provider, scheduler | Done |
| `GET /api/dashboard/model` — provenance, metrics, known limitations | Done |
| Freshness measured from stored timestamps against a configurable threshold | Done |
| Fixed query count per dashboard request (window functions, never N+1) | Done |
| Composite index on `(observation_id, prediction_timestamp)` | Done |
| Tests for every outcome, failure mode and freshness boundary | Done |

Exit criteria met: a location converges on its own — six readings stored and
reported as unscoreable, the seventh produces a forecast — and no endpoint
reports a value it did not measure.

## Stage 7 — Frontend dashboard *(done)*

- Leaflet map of monitored segments coloured by predicted congestion level.
- Chart.js views: speed and congestion over time, observed versus predicted.
- Auto-refresh against the dashboard endpoints.
- Honest empty states: "no data for this segment", "5 of 6 readings", "stale
  reading" — never a placeholder number.
- Surfaces `dataset_is_simulated` and `validated_on_real_traffic` next to any
  forecast it renders.

Delivered in `frontend/dashboard.html` (`js/{api,ui,map,charts,dashboard,config}.js`,
`css/dashboard.css`, vendored Leaflet/Chart.js). Landing page (`index.html`) and
shared chrome (`app.js`) repointed to it.

Exit criteria met: the dashboard renders only values returned by the API, and makes
the model's limitations visible without the reader having to ask. Verified by
15 frontend contract tests, a live end-to-end run (all dashboard endpoints, all
panels), and the documented manual checklist. Full suite: 437 passing tests.

## Stage 8 — Hardening, deployment and presentation *(done)*

- Production guard in settings: `ENVIRONMENT=production` requires
  `TRAFFIC_PROVIDER=real`, a `TRAFFIC_API_KEY` and a PostgreSQL `DATABASE_URL`
  (no `sqlite://`), else start-up fails naming the missing setting — a missing
  key is a hard boot error, never a silent fallback to simulation, and the error
  never echoes the secret.
- Security review completed: secrets only from `.env` (scan-tested), vendor
  bodies and DSNs never echoed or logged, CORS explicit, outbound HTTPS +
  timeouts, upstream status classification, and frontend HTML-escaping contract
  tests for XSS (`UI.esc` covers `& < > " '`, every `innerHTML` template).
- Health vs readiness distinguished: `GET /api/health` is liveness;
  `GET /api/dashboard/status` is readiness (database, provider, scheduler,
  model and freshness) with `DB_FAIL_FAST=true` recommended for boot-time
  failures.
- Environment template updated (`.env.example`) with the production-guard
  contract; `backend/app/main.py` description updated for Stages 6–8.
- Deployment guide: [`docs/deployment.md`](deployment.md) — prerequisites,
  production `.env`, server startup from the project root (`--app-dir backend`),
  Alembic application, reverse-proxy and static-frontend serving, verification
  commands, quotas and a troubleshooting table.
- Project report (24 sections): [`docs/project-report.md`](project-report.md).
- Presentation deck (12 slides): [`docs/presentation.md`](presentation.md).
- Docker images and a CI pipeline are intentionally **documented as follow-up,
  not shipped untested** — see the roadmap note below.

### Verification performed

- Baseline suite 437 → **450 tests** with the new config-guard and XSS tests.
- Alembic end-to-end round-trip on a fresh database:
  `upgrade head` → verify (2 tables, 12 indexes, unique constraint) →
  `downgrade base` → `upgrade head` again.
- Live Open-Meteo call verified (HTTP 200, contract fields present). Live
  TomTom and live PostgreSQL round-trips require credentials/a server and are
  stated as such in the docs.

> **Note on Docker and CI:** the roadmap's earlier list assumed container images
> and a CI pipeline in this stage. Those are real dependencies (a registry, a
> build runner) that this environment does not own, so the stage ships the
> complete deployment **documentation** and the verified startup/verification
> commands instead. Containerising the API on top of this guide is mechanical
> follow-up work listed in §23 of the project report rather than presented as
> done.

## Explicitly out of scope

Authentication and user accounts, payments and billing, multi-tenant support, and
mobile applications. None of these serve the stated research objective.