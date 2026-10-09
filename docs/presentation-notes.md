# Presentation support

Talking points for the project, current as of **Stage 8** (all stages complete).
The slide outline lives in `presentation.md`; this file carries the detail.

## 1. Problem (30 seconds)

Urban congestion is discovered too late. Commuters find out when they are already
in it, and traffic control reacts after congestion has already formed. Static
timetables cannot capture incident, weather and peak-hour variation.

The need is a **forward-looking** congestion estimate per road segment.

## 2. Objective (30 seconds)

Build a system that ingests real traffic observations, learns congestion patterns
with machine learning, and serves short-horizon congestion forecasts through an
API and an interactive map.

## 3. What the staging guarantees (30 seconds)

Each stage produces something that actually runs, and nothing downstream is built
on an assumption. Concretely, this repository does **not** contain:

- a model validated on real traffic — the training data is a clearly labelled
  simulated fixture, and every artifact and API response says so;
- fabricated traffic data standing in for a real provider — a missing key fails
  loudly instead;
- dashboard numbers that were not measured — an uncollected road reports `null`
  and `unavailable`, never a plausible zero.

That is a deliberate engineering position and the strongest honesty signal in the
demo.

## 4. Architecture (60 seconds)

- **Frontend** — static HTML5/CSS3/JavaScript; Leaflet and Chart.js arrive in
  Stage 7. No build step, no framework lock-in.
- **Backend** — FastAPI with a strict layering: `api` → `services` → `schemas`,
  plus `database` and `core` for infrastructure.
- **ML** — an independent package. It never imports the web layer; the only link
  between them is a serialized model artifact.
- **Database** — PostgreSQL through SQLAlchemy, configured entirely by
  environment variables.
- **Read side** — the dashboard endpoints are aggregation queries: fixed count per
  request (window functions, never N+1), freshness measured from stored
  timestamps.

The ML/API separation is the point to emphasise: models can be retrained and
benchmarked with no API running, and the API can be deployed without scikit-learn.

## 5. Live demo (Stage 6 script)

```bash
# 1. Start the API
cd backend
uvicorn app.main:app --reload

# 2. Show the health contract
curl http://127.0.0.1:8000/api/health
# {"status":"healthy","service":"traffic-prediction-api"}

# 3. Seed six readings, then collect one more: stored first, scored second
curl -X POST http://127.0.0.1:8000/api/traffic/collect/history \
  -H "Content-Type: application/json" -d '{"location_id":"LOC-001"}'
curl -X POST http://127.0.0.1:8000/api/traffic/collect \
  -H "Content-Type: application/json" -d '{"location_id":"LOC-001"}'

# 4. The forecast, and the dashboard view of it
curl http://127.0.0.1:8000/api/traffic/predictions/latest
curl http://127.0.0.1:8000/api/dashboard/current
curl http://127.0.0.1:8000/api/dashboard/model   # dataset_is_simulated: true

# 5. Show the tests
pytest -q
```

Worth demonstrating deliberately: call `predictions/latest` for a location that
has never been collected and it returns `404`; the dashboard reports
`freshness: "unavailable"` with a `null` speed rather than an estimate.

## 6. ML design discussion

- Random Forest chosen first because it is strong on tabular data, needs little
  tuning, and gives feature importances that help explain predictions.
- Macro F1 is the headline metric, not accuracy, because severe congestion is rare
  and an accuracy-optimising model would ignore it.
- Splitting is leakage-safe: chronological, with contemporaneous measurements
  refused at training time *and* at the API boundary.
- Every artifact stores its own feature order and provenance, which removes the
  train/serve feature-skew failure mode.
- `POST /predict` needs six prior readings per segment and answers
  `422 insufficient_history` — naming required versus available — instead of
  guessing.

## 7. Stages (30 seconds)

See `docs/roadmap.md`.

| Stage | Focus | Status |
| --- | --- | --- |
| 1 | Foundation — API, config, DB wiring | **Done** |
| 2 | Data pipeline, leakage control, provenance | **Done** |
| 3 | Training, evaluation, prediction API, ORM | **Done** |
| 4 | Data providers, observation endpoints | **Done** |
| 5 | TomTom + Open-Meteo, monitored locations, scheduler | **Done** |
| 6 | Auto-prediction, history endpoints, dashboard API | **Done** |
| 7 | Leaflet map dashboard and Chart.js analytics | **Done** |
| 8 | Production guard, security review, deployment guide, report, deck | **Done** |

## 8. Anticipated questions

**What are the benefits of such a system?**
Forward-looking per-road forecasts let drivers reroute before congestion forms
(not after they are inside it) and give planners a leading signal for timing and
incident staging. The measurable outcomes are travel time, fuel and emissions
saved. The engineering benefits follow directly: the model is reproducible and
inspectable (features, split and metrics are recorded), the provider interface
makes it vendor-independent and retraining-ready, and the free-tier stack makes
a pilot cheap.

**What are the drawbacks?**
Live feeds publish current conditions only and are quota-limited, so a new city
must accumulate history before any forecast is possible and vehicle throughput
stays unknown. The current model is trained on simulated data, so its forecasts
are demonstrative, not field-validated — acting on them as real would mislead.
There are also privacy and liability concerns around derived travel behaviour, a
dependence on vendor uptime/quota, an equity risk that benefits concentrate on
busy corridors, and a general risk that a polished dashboard reads as
authoritative. The project addresses the honest ones (labelling, provenance,
production guard) and names the rest.

**Why no real accuracy claims?**
The model is trained on simulated data. Presenting synthetic metrics as
field-validated performance would be unverifiable, so every artifact, report and
API response states `dataset_is_simulated: true` and
`validated_on_real_traffic: false`. The pipeline that would let a real dataset
flow through exists and is tested; the dataset does not.

**How do you know the model will work?**
Evaluation is fixed before training: a majority-class baseline must be beaten,
splits are leakage-safe, and metrics ship with their split strategy. The
baseline is exposed on `/api/dashboard/model` alongside the model's own scores.

**Why FastAPI over Flask/Django?**
Automatic validation from Pydantic schemas, typed dependency injection that keeps
sessions request-scoped, and native OpenAPI output for the dashboard team.

**How will real-time data be collected?**
Stage 5 integrates TomTom Flow Segment Data behind a provider interface, with
Open-Meteo for weather, throttled per cycle and opt-in via
`COLLECTION_ENABLED=true`. The same path runs on a manual `POST /collect`, so
scheduled imports and live polling share one code path.

**How is this secured?**
No secrets in the repository; everything arrives through `.env`, which is
git-ignored. Vendor responses are never echoed, because they contain the request
URL and the API key. In production (`ENVIRONMENT=production`) the application
refuses to start on the demo defaults: it requires `TRAFFIC_PROVIDER=real`, a
`TRAFFIC_API_KEY` and a PostgreSQL DSN, so a misconfigured deployment fails at
boot instead of silently serving simulated traffic. The dashboard HTML-escapes
every API-derived value. The API itself is unauthenticated on purpose at this
stage; before any public deployment, gateway auth and rate limiting are required.

## 9. Slide deck checklist

- [ ] Problem and objective
- [ ] System architecture diagram (`docs/architecture.md`)
- [ ] Live demo: health, collect → predict, dashboard current and model
- [ ] Test suite output
- [ ] ML design: leakage control, metric choice, honest baseline
- [ ] Honest scope statement: what is done, what is simulated, what is planned
- [ ] Roadmap and next steps
