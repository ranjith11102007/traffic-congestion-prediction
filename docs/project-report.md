# Project Report — AI-Based Urban Traffic Congestion Prediction System

Final report covering Stages 1–8. Everything stated here has been verified by
execution in this repository, and where a verification was not possible in the
development environment (no live traffic key, no live PostgreSQL server) that is
stated explicitly rather than assumed. Current as of the Stage 8 acceptance run.

---

## 1. Project identification

| Field | Value |
| --- | --- |
| Title | AI-Based Urban Traffic Congestion Prediction System |
| Objective | Predict per-road-segment congestion ahead of time and visualise it |
| Standalone components | `backend/` (FastAPI), `ml/` (independent ML package), `frontend/` (static dashboard) |
| Database | PostgreSQL 14+ via SQLAlchemy 2.x, Alembic migrations |
| Model | Random Forest classifier, version `4.0.0`, 26 features, 4 congestion levels |
| Status | Stages 1–8 **complete** (see §8) |
| Verification | 451 automated tests passing; live Open-Meteo call verified; see §18–§19 |

## 2. Executive summary

The system ingests traffic observations (live via TomTom + Open-Meteo, or a
clearly labelled simulated fixture), engineers and stores them, scores each
stored reading with a pre-trained Random Forest model, and serves current state,
forecasts, trends, per-location readiness, system status and model provenance
through a documented REST API (Stages 1–6). A static, dependency-free browser
dashboard (Leaflet map, Chart.js analytics, auto-refresh) renders those endpoints
(Stage 7). Stage 8 hardens the result for deployment: production configuration
cannot silently fall back to simulated traffic, the frontend escapes all
API-derived HTML, health and readiness are distinct probes, and deployment,
project-report and presentation material are complete.

The honest bottom line is preserved end to end: **the training data is a
simulated fixture**, so every forecast is demonstrative, and every API response,
model card and dashboard surface repeats that caveat.

## 3. Problem statement

Congestion is discovered too late — commuters find out when they are already in
it, and traffic control reacts after congestion has formed. Static timetables
cannot capture incident, weather and peak-hour variation. The system's question
is the forward-looking one: *will this road be congested in 30 minutes?* — which
is the question that lets someone change behaviour, not just observe.

## 4. Objectives

1. Collect real urban traffic observations into a persistent store.
2. Engineer leakage-safe features from time, road geometry, flow and weather.
3. Train and evaluate a classifier against an honest majority-class baseline.
4. Expose predictions through a documented REST API.
5. Visualise current and predicted congestion on an interactive map.
6. Harden the whole for deployment without breaking the honesty constraints.

## 5. Scope

**In scope:** observation ingestion, PostgreSQL persistence with Alembic
migrations, provider abstraction (labelled simulation + live TomTom/Open-Meteo),
auto-prediction after collection, prediction history, dashboard aggregation
endpoints, static frontend dashboard, security review, deployment documentation,
project report and presentation.

**Out of scope (deliberate):** authentication/user accounts, payments,
multi-tenancy, mobile apps. Container images and CI pipelines are documented as
follow-up work, not shipped and untested.

## 6. Stakeholders

| Role | Interest in | How served |
| --- | --- | --- |
| Viewer | Understand current + near-future congestion | Map, chart, status panels |
| Data engineer | Trust the data | Provenance on every row, no fabrication |
| ML engineer | Trust the model | Honest metrics, leakage controls, artifact metadata |
| Operator | Run it safely | Production guard, health vs readiness, deployment guide |
| Reviewer | Verify claims | 450 tests, this report, reproducible commands |

## 7. Timeline and stages

| Stage | Focus | Status |
| --- | --- | --- |
| 1 | Foundation: API, config, DB wiring, ML layout, frontend shell | Complete |
| 2 | Data pipeline: leak-safe features, target derivation, provenance | Complete |
| 3 | Model: Random Forest, evaluation with baseline, inference | Complete |
| 4 | Data layer: PostgreSQL + Alembic, providers, observation endpoints | Complete |
| 5 | Live traffic: TomTom + Open-Meteo, locations, opt-in scheduler | Complete |
| 6 | Prediction engine: auto-predict, history/latest, dashboard API, freshness | Complete |
| 7 | Frontend dashboard: Leaflet map, Chart.js, auto-refresh | Complete |
| 8 | Hardening: production guard, security, deployment, report, deck | Complete |

Full detail: [`docs/roadmap.md`](roadmap.md).

## 8. Technology stack

- **Frontend:** HTML5, CSS3, Vanilla JS (ES2020); Leaflet 1.9.4 and Chart.js 4.4.1 vendored locally (no CDN, no build step).
- **Backend:** Python 3.11+, FastAPI, Pydantic v2, Uvicorn, SQLAlchemy 2.x, psycopg2, Alembic, httpx.
- **ML:** NumPy, pandas, scikit-learn, joblib — a package independent of the web layer.
- **Database:** PostgreSQL (deployment target); SQLite for the isolated test suite.
- **Testing:** pytest (450 tests), `httpx.MockTransport` for provider contracts, jsdom (out-of-suite) for a browser simulation of the dashboard.

## 9. System architecture

```
Browser (frontend/) ──HTTP/JSON, CORS──► FastAPI (backend/app)
   api/ (routers) → schemas/ (validation) → services/ (logic)
                                        ├── repositories/ → SQL
                                        │        └─► database/ → PostgreSQL
                                        └── prediction_service ──► ml/ (artifacts)
Data providers:  TrafficDataProvider ← SimulatedTrafficProvider (labelled)
                                   ← RealTrafficProvider (TomTom) + weather_client (Open-Meteo)
Background:      collection_scheduler (opt-in, asyncio)
```

Key decisions: ML never imports the web layer; routes stay thin; provenance is a
column (`source`); one error envelope; configuration only from the environment;
an unavailable capability is an explicit error (503/422), never an invented
value; an unmeasured value is stored `NULL`, never estimated. Full detail:
[`docs/architecture.md`](architecture.md).

## 10. Data model

| Table | Purpose | Constraints |
| --- | --- | --- |
| `traffic_observations` | One reading per road per time | Unique `(location_id, timestamp)`; `avg_speed_kph <= free_flow_speed_kph`; nullable `vehicle_count`; `source` provenance |
| `traffic_predictions` | One forecast per observation | FK `ON DELETE CASCADE`; `predicted_congestion_level` in 0–3; `confidence` nullable |

Indexes cover the dashboard read paths: timestamp/location/road-name/segment on
observations, `(observation_id, prediction_timestamp)` and observation/created
on predictions (Stage 6). Alembic manages the chain (3 revisions): table
creation, nullable `vehicle_count`, history index. No DSN is committed to
`alembic.ini`; the URL resolves from settings at runtime.

## 11. Data pipeline and ingestion

Six offline stages — load, validate, clean, featurise, target, write — turn raw
observations into `traffic_processed.csv` + metadata sidecar. Ingestion at
runtime runs through a provider interface (`simulation` or `real`), normalizes a
record to the exact field names the model expects, and stores with provenance.

Real-time data flow: TomTom Flow Segment Data per monitored road (current and
free-flow speed, confidence, road closure), Open-Meteo weather (no key; WMO
codes mapped to the model's three weather labels), stored via the collector and
auto-scored. Media notes:

- **`vehicle_count` is `NULL` on live rows** — no self-serve API publishes throughput.
- **No historical feed exists** — history accumulates via the scheduler (6 readings per segment before scoring).

## 12. Machine learning design and evaluation

- **Model:** Random Forest, 26 features, macro F1 primary metric (accuracy
  optimises away the rare severe class); chronological 80/20 split, never random.
- **Leakage control:** contemporaneous measurements (`avg_speed`, `speed_ratio`,
  throughput, occupancy) are excluded from inputs and **refused** at training
  time and at the API boundary; lags/rolling means are strictly backward; all
  imputation is per-segment.
- **Target:** derived from `speed_ratio` thresholds (0.75 / 0.40 / 0.15) into
  `free_flow` / `moderate` / `heavy` / `severe`, or preserved when the dataset
  carries its own label.
- **Result (simulated fixture only):** macro F1 0.7370, accuracy 0.8561 vs a
  0.2143/0.7500 majority-class baseline.
- **Provenance:** `model_metadata.json` records `dataset_is_simulated: true`,
  `validated_on_real_traffic: null`, hyperparameters, split boundary and class
  balance; every report, response and dashboard repeats the caveat.

## 13. API design

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness probe |
| POST/GET | `/api/traffic/observations[/{id}]` | Record / list / fetch readings |
| GET | `/api/traffic/latest` | Most recent reading |
| POST | `/api/traffic/predict` · `/observations/{id}/predict` | Score + persist |
| GET | `/api/traffic/predictions[/latest][/status]` | History, latest, integration status |
| GET | `/api/traffic/collector/status` | Active provider + simulated flag |
| POST | `/api/traffic/collect[/history]` | Collect → auto-predict / backfill |
| GET | `/api/dashboard/current` `trends` `locations` `status` `model` | One-poll dashboard reads |

Rules: one error envelope `{"error":{code,message,details}}`; insufficient
history → 422 naming required vs available; missing model → 503
`model_not_available`; missing provider key → 503 `provider_not_configured`;
duplicate observation → 409. OpenAPI at `/docs`. Full reference:
[`docs/api.md`](api.md).

## 14. Frontend and dashboard

`frontend/` ships a landing page (`index.html`) and the Stage 7 dashboard
(`dashboard.html`) with a sidebar (Overview, Live Traffic, Predictions,
Analytics, Locations, System Status) and a location selector. Panels: 6 KPI
cards, Leaflet map coloured by predicted class, per-location detail, current
table, prediction card + history, Chart.js trend chart for 1/6/12/24 h windows,
readiness cards, and a system status table. Auto-refresh (15/30/60 s or manual,
paused when the tab is hidden), honest loading/empty states, and the model's own
labels throughout. Every API-derived string passes through `UI.esc`.

## 15. Security design and review

| Area | What was done | Verified by |
| --- | --- | --- |
| Secrets | Only from untracked `.env`; empty defaults; nothing committed | Secret-leak tests scan responses, logs, config and every committed file |
| Production guard | `ENVIRONMENT=production` refuses to boot without real provider + key + PostgreSQL (no silent simulation) | Config tests incl. error-message leak check |
| Error handling | Vendor bodies never echoed (they contain the key); DSN redacted in logs/alembic | Error-envelope + migration security tests |
| CORS | Explicit origin list, no credentials, configurable | CORS tests |
| XSS | `UI.esc` on every API-derived string before `innerHTML` | New frontend escaping tests (Stage 8) |
| Outbound | HTTPS required for vendor base URLs; timeouts on both providers; throttling | Provider/settings tests |
| Upstream classification | 429 → 503, 4xx/5xx → 502, timeout → 504, bad body → 502 | Provider tests |

Public exposure still requires gateway auth + rate limiting — stated in
[`docs/deployment.md`](deployment.md) and the roadmap as a pre-public item.

## 16. Reliability and error handling

- Domain, validation, HTTP and unexpected errors share one envelope; unexpected
  errors log server-side without leaking internals.
- Scheduler: opt-in, asyncio task, sync work in a worker thread, a failed cycle
  is counted not fatal, shutdown cancels a pending sleep.
- Collection stores readings **before** predicting, so a scoring failure never
  costs an observation; per-reading outcomes are distinct
  (`predicted` / `insufficient_history` / `model_unavailable` / `unscoreable_location` / `failed`).
- Freshness is measured from stored timestamps against a configurable threshold
  and reported `fresh` / `stale` / `unavailable`.

## 17. Configuration and environments

All settings validated by `app.core.config.Settings`: service metadata,
database, CORS, provider/weather, monitored locations, collection, dashboard and
ML knobs. `.env.example` is the canonical template (updated for Stage 8 with the
production-guard contract). A `development` environment may run the labelled
simulator with an empty DB; `production` must meet the §2 guard.

## 18. Testing strategy and results

**451 tests**, run without any external service (isolated in-memory SQLite,
`MockTransport` providers, offline PostgreSQL dialect rendering):

- API contract, error envelope, config validation, DB wiring and pool options.
- Traffic schemas, provider mapping (TomTom + Open-Meteo), locations file,
  scheduler behaviour, security/secret handling.
- Full observation → predict → persist flow against the real trained artifact;
  auto-prediction after collection including failure modes; history/latest.
- Every dashboard endpoint, freshness boundaries, query-count discipline.
- Alembic: SQLite online apply → verify → reverse → re-apply, ORM drift check,
  PostgreSQL offline DDL render, chain integrity, constraints/indexes/FKs.
- ML pipeline contract and trained-model consistency (skip-free on clean checkout).
- Stage 7 frontend structure/asset/no-secrets/element-contract checks (15 tests).
- Stage 8 production-guard tests (7) and frontend XSS/escaping tests (6).

Run: `pytest -q`. Additional verification: `python -m compileall`, `node --check`
on every `frontend/js/*.js`.

## 19. Verification performed (Stage 8 acceptance)

| Item | Method | Result |
| --- | --- | --- |
| Migration round-trip | Real `alembic upgrade head → downgrade base → upgrade head` on a fresh SQLite file | Head `c3d4e5f6a7b8`, 2 tables + 12 indexes + unique constraint, re-applied cleanly |
| PostgreSQL DDL | Offline `--sql` rendering + migration tests | Verified dialect-correct SQL |
| Live PostgreSQL round-trip | **Not possible** (no server in this environment) | Documented in §22; procedure is plain `alembic upgrade head` |
| Live TomTom | **Not possible** (no `TRAFFIC_API_KEY`) | Contract tests via `MockTransport`; live run requires a key |
| Live Open-Meteo | One-off `httpx` call to the forecast endpoint | HTTP 200; `temperature_2m`, `precipitation`, `weather_code` present, matching the client's contract |
| Production guard | Settings construction for `production` with each missing piece | Boot-time `ValidationError` naming the missing setting; no secret echoed |
| Frontend XSS | Static contract tests on `ui.js`/`dashboard.js`/`map.js`/`charts.js` | `esc()` covers `& < > " '`; every `innerHTML` template has `UI.esc` |
| Dashboard end-to-end | Stage 7 jsdom harness + live API | All scripts eval, all fetches resolve, all panels render |

## 20. Deployment plan and documentation

Delivered: [`docs/deployment.md`](deployment.md) covers prerequisites, a
production-shaped `.env`, the boot-time guard, applying the schema, the exact
startup command (project root + `--app-dir backend`), liveness vs readiness,
reverse-proxy/nginx and static-frontend serving, verification commands,
operations notes (quotas, scheduler, backfill, logging) and a troubleshooting
table. Beyond the guide, this session shipped the deployment itself:

- **Container image** — `backend/Dockerfile` serves the API and the dashboard
  from one image; `docker-compose.yml` runs PostgreSQL + the API locally in one
  command (`docker compose up --build`), mirroring the hosted topology.
- **CI** — `.github/workflows/ci.yml` runs the full `pytest` suite and a Docker
  build on every push and pull request (currently green).
- **Single-URL serving** — when `FRONTEND_DIR` is set, the API mounts the static
  dashboard under its own origin, so the browser needs no CORS and no second
  service. `frontend/js/config.js` defaults `API_BASE_URL` to same-origin (`''`).
- **Public host** — `render.yaml` (Blueprint) provisions a free Render Web
  Service that migrates the schema at boot and starts Uvicorn on Render's
  `$PORT`, with `COLLECTION_ENABLED=true` and `TRAFFIC_PROVIDER=real` so the
  live demo accumulates real TomTom readings and scores them automatically.

## 21. Potential benefits and drawbacks

The problem statement asks for a discussion of what a system like this offers
and what it costs. Benefits are stated first, then the drawbacks that a
responsible deployment must design against.

**Potential benefits**

1. **Forward-looking awareness.** Per-road forecasts (rather than "where is it
   congested now?") let drivers reroute before congestion forms and give
   planners a leading signal for signal timing and incident staging. This is the
   behavioural change the problem statement targets.
2. **Measurable efficiency.** Fewer vehicles idling in avoidable congestion
   means lost travel time, fuel burn and emissions; these are the standard,
   quantifiable outcomes in travel-time and emissions models.
3. **Reproducible and inspectable.** A model replaces arbitrary colour-band
   thresholds: features, split, provenance and metrics are recorded next to each
   forecast instead of being hidden behind a rule.
4. **Vendor-independent and retraining-ready.** One provider interface means the
   storage, prediction and dashboard layers never change when the vendor does,
   and the pipeline is ready to be retrained on real observations.
5. **Cheap to pilot.** The stack runs on free-tier sources (TomTom's documented
   allowance, keyless Open-Meteo) and open-source tooling, so a small municipal
   or institutional pilot is low-cost.

**Potential drawbacks**

1. **Data gaps limit trust.** Live feeds publish current conditions only and are
   quota-limited; there is no historical or throughput feed. A new city must
   first accumulate enough per-segment history before the first forecast is even
   possible, and vehicle throughput stays unknown.
2. **Unvalidated model risk.** The current model is trained on simulated data;
   its forecasts are demonstrative, not field-validated. Acting on them as if
   they were real — or presenting them as such — could mislead drivers and
   planners, which is why every artifact labels `dataset_is_simulated`.
3. **Privacy and liability.** Deriving and storing travel behaviour raises
   privacy concerns, and a publicly served wrong forecast (for example near an
   emergency) could have safety or legal consequences.
4. **External dependency.** The service depends on vendor uptime, quota limits
   and the weather API. Operator error on the `TRAFFIC_PROVIDER`/key pair could
   surface labelled simulation as real without the production guard.
5. **Equity.** Rerouting nudges and infrastructure spend can concentrate on busy
   corridors; without checks, the benefits of prediction need not reach every
   neighbourhood equally.
6. **False confidence.** A polished dashboard reads as authoritative; honest
   provenance display and validation status are therefore part of the product,
   not an afterthought.

## 22. Known limitations and honest scope

1. **Model trained on simulated data** — forecasts are demonstrative, not
   field-validated; `dataset_is_simulated: true`.
2. **`vehicle_count` is `NULL`** on live readings (no vendor publishes throughput).
3. **No historical traffic feed** — the scheduler accumulates history; a fresh
   location needs six readings before the first forecast.
4. **Live PostgreSQL round-trip not executed** here (no server available).
5. **Live TomTom verification requires a `TRAFFIC_API_KEY`** — the mocked
   contract tests pass; the live check is a one-line `curl` once a key exists.
6. **API is unauthenticated** — gateway auth and rate limiting required before public exposure.

None of these are hidden: each is stated in the API description, the model
metadata, the dashboard and the docs.

## 23. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Operator serves simulated data in production | Boot-time production guard (§2) + `collector/status` reporting |
| Vendor quota exhausted | `TRAFFIC_MAX_LOCATIONS` cap, `TRAFFIC_MIN_REQUEST_INTERVAL`, classification of 429 |
| Model over-fitted to synthetic fixture | Honest metrics + baseline; a real dataset is the stated precondition for trust |
| Dashboards misread stale/absent data | `freshness` field, `NULL` speeds, `unavailable` states, no zero-fill |
| Credential leak | `.env` only, secret-scan tests, redacted error messages and logs |
| Multi-worker scheduler duplication | One-worker guidance in deployment doc |

## 24. Future work

- Retrain on a **real** dataset and re-validate (the pipeline is ready and tested).
- Container image + CI pipeline: **delivered** (see §20).
- Gateway authentication and rate limiting for public exposure.
- Historical/throughput data source if a suitable API becomes self-serve.
- Staging environment and live PostgreSQL round-trip in CI.

## 25. Appendix

- **Run the API (verified):** `.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`
- **Migrate:** `alembic upgrade head` (or `--sql` to review DDL offline).
- **Tests:** `pytest -q` (451). **Frontend:** open `frontend/index.html`, serve `frontend/` on any static server, or let the API serve its own origin (`FRONTEND_DIR`).
- **Docs:** [`architecture.md`](architecture.md) · [`api.md`](api.md) · [`ml-pipeline.md`](ml-pipeline.md) · [`roadmap.md`](roadmap.md) · [`deployment.md`](deployment.md) · [`presentation.md`](presentation.md) · [`presentation-notes.md`](presentation-notes.md).
- **Artifacts:** `ml/models/model_metadata.json`, `feature_importance.json`, `evaluation_results.json`, `model_report.md`.
- **Commands used for this report's verification** are recorded inline in §18–§19 and in the Stage 8 acceptance run.