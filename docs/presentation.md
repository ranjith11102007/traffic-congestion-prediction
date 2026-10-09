# Presentation — AI-Based Urban Traffic Congestion Prediction System

Slide deck outline (12 slides). The talking points are in
[`presentation-notes.md`](presentation-notes.md); the architecture, API and
deployment references are linked from each slide so the deck stays short.

---

## Slide 1 — Title

**AI-Based Urban Traffic Congestion Prediction System**

A machine-learning pipeline that predicts per-road-segment congestion ahead of
time and visualises it on a live dashboard. Stages 1–8 complete.

_Footer: repo, date._

---

## Slide 2 — The problem

Congestion is discovered too late. Reactive traffic control answers "where is it
congested *now*?" — not "will this road be congested in 30 minutes?", which is the
question that changes behaviour.

_Image suggestion: photo of a congested arterial with a caption "feedback happens
after you're already inside it"._

---

## Slide 3 — Objective and approach

1. **Collect** real urban traffic observations into PostgreSQL.
2. **Engineer** leakage-safe features from time, road geometry and weather.
3. **Train** a Random Forest and score it against an honest baseline.
4. **Serve** forecasts through a documented REST API.
5. **Show** current and predicted congestion on a map and charts.

Every result traceable to data; nothing fabricated.

---

## Slide 4 — System architecture

Browser → FastAPI (`api`/`schemas`/`services`/`repositories`) → PostgreSQL · ML
pipeline (`ml/`) independent of the web layer. Data providers behind one
interface: labelled simulation, or live TomTom + Open-Meteo. Opt-in background
scheduler.

_Link: [`docs/architecture.md`](architecture.md) for the full diagram._

---

## Slide 5 — Data and the model

- Six-stage offline pipeline: load → validate → clean → featurise → target → write.
- Leakage controls: contemporaneous measurements refused at training **and** at
  the API boundary; backward-only lags; chronological split.
- Random Forest, 26 features, 4 levels (free flow / moderate / heavy / severe),
  macro F1 as the headline metric.
- **Honesty: the training fixture is simulated and labelled as such in every
  artifact, report and API response.** The pipeline is retraining-ready for a
  real dataset.

---

## Slide 6 — The API

One curl-able endpoint per capability: observations, predict, history, latest,
collector status, and five dashboard reads (current, trends, locations, status,
model). One error envelope; insufficient history is an explicit 422, a missing
model a 503 — never an invented forecast.

_Link: [`docs/api.md`](api.md)._

---

## Slide 7 — Live demo

Start `uvicorn`, seed six readings for `LOC-001`, collect a seventh (stored
first, predicted second), then show `/predictions/latest` and
`/dashboard/current`, `/dashboard/status` and `/dashboard/model`.
Live traffic tab → map → analytics → system status.

_Demo script: [`presentation-notes.md`](presentation-notes.md) §5._

---

## Slide 8 — Dashboard highlights

- Leaflet map coloured by predicted congestion class.
- Chart.js trend chart (1/6/12/24 h) of speed vs forecast.
- Auto-refresh, honest empty states, "N/A" never a zero.
- Model provenance shown next to every forecast
  (`dataset_is_simulated: true`).

---

## Slide 9 — Security and deployment readiness

- Production boot guard: no silent simulation; a real provider key and
  PostgreSQL are required, or start-up fails naming the missing setting.
- Secrets only from `.env`; no DSN in `alembic.ini`; vendor bodies and DSNs
  never echoed or logged.
- Every API-derived string HTML-escaped before it reaches the dashboard.
- Health (`/api/health`) vs readiness (`/api/dashboard/status`).
- Deployment guide shipped: [`docs/deployment.md`](deployment.md).

---

## Slide 10 — Testing and verification

**450 automated tests**, no external services required:

- Observation → predict → persist flow against the real artifact.
- Every dashboard endpoint, freshness boundaries, fixed query counts.
- Alembic: SQLite apply → verify → reverse → re-apply **and** offline PostgreSQL
  DDL rendering.
- Provider contracts via mocked transports; security and secret-leak tests;
  Stage 8 production-guard and XSS tests.
- Live outside the suite: Open-Meteo call verified (HTTP 200); TomTom requires
  a real key; live PostgreSQL requires a server.

---

## Slide 11 — Results and honest scope

On the simulated fixture: **macro F1 0.737, accuracy 0.856** vs a 0.214/0.750
majority-class baseline. These numbers demonstrate that the pipeline runs — they
are not a statement about real traffic.

Deliberate gaps: `vehicle_count` is `NULL` on live readings (no vendor publishes
throughput); no historical feed (the scheduler accumulates history); the API is
unauthenticated until a gateway is added.

---

## Slide 12 — Next steps and Q&A

1. Retrain on a **real** dataset and re-validate (pipeline is ready).
2. Docker image + CI running the suite; staging environment.
3. Gateway authentication and rate limiting for public exposure.
4. Live TomTom and PostgreSQL round-trip once credentials and a server exist.

**Questions.**

_See [`docs/roadmap.md`](roadmap.md) and [`docs/project-report.md`](project-report.md)._