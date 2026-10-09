"""FastAPI application factory and entry point.

Run in development::

    uvicorn app.main:app --reload

The application object is created through :func:`create_application` so tests can
build an isolated instance instead of importing a module-level singleton.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import (
    ConfigurationError,
    register_exception_handlers,
)
from app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Backend for the **AI-Based Urban Traffic Congestion Prediction System**.

### Current phase

Stages 6 and 7 are implemented: every collected reading is scored automatically
by the Stage 3 model (version 4.0.0) and stored, stored forecasts are readable as
history and latest, five dashboard endpoints serve the current view, trends, per-
location readiness, system status and model metadata, and a bundled static
dashboard renders those endpoints in the browser. Stage 8 hardens the whole
deployment: production configuration refuses to fall back to the simulation
provider, health vs. readiness are distinct, and vendor data is never exposed
untrusted to the dashboard.

The ingestion path is unchanged from Stage 5: live traffic readings from the TomTom
Flow Segment Data API, weather from Open-Meteo, a configurable set of monitored
roads, and PostgreSQL storage with provenance on every row.

### Data provenance

Every stored row carries a `source` label, and the collector endpoints report
whether the active provider is serving synthetic data.

- `simulation` — synthetic development data. **Not real traffic.**
- `tomtom` — a live reading from TomTom, with weather from Open-Meteo.
- any other value — an observation POSTed through the API.

The model in this repository was trained on **simulated data**. Predictions are
demonstrative, not validated real-world forecasts, and remain so once live data
flows in: real readings feed the same unvalidated model. `GET /api/dashboard/model`
reports this in `dataset_is_simulated` and `validated_on_real_traffic` so a client
cannot present a forecast as field-tested.

### Two limitations worth reading before trusting any number

**`vehicle_count` is null on every live reading.** No self-serve traffic API
publishes vehicle throughput — not TomTom, not HERE, not Mapbox. Rather than
store an estimate under a real vendor's name, the column is nullable and left
empty. Model version 4.0.0 therefore dropped `flow_veh_per_hr` as a feature.

**There is no historical traffic feed.** TomTom reports current conditions only.
The model needs six prior observations per segment before it will score, so a real
deployment accumulates its own history: set `COLLECTION_ENABLED=true`, or POST
observations yourself. `fetch_historical_traffic` raises rather than returning an
empty history. Collection and prediction then converge on their own: readings one
through six are stored and reported as unscoreable, and the seventh produces a
forecast with no further intervention.

### Reading the data honestly

Freshness is measured from the stored observation timestamp against
`DATA_FRESHNESS_THRESHOLD_SECONDS` and reported as `fresh`, `stale` or
`unavailable`. A location that has never been collected reports null measurements
and `prediction_status: "insufficient_history"` — never a zero speed or a
placeholder congestion level, which would read as "a standstill" and "normal"
respectively.

`confidence` is null unless the estimator produced a real probability in 0..1.

### Endpoints

| Method | Path                                        | Purpose |
| ------ | ------------------------------------------- | ------- |
| GET    | `/api/health`                               | Liveness probe |
| POST   | `/api/traffic/observations`                 | Record an observation |
| GET    | `/api/traffic/observations`                 | List observations |
| GET    | `/api/traffic/observations/{id}`            | Fetch one observation |
| GET    | `/api/traffic/latest`                       | Most recent observation |
| POST   | `/api/traffic/predict`                      | Store, predict, store the forecast |
| POST   | `/api/traffic/observations/{id}/predict`    | Predict from a stored observation |
| GET    | `/api/traffic/predictions`                  | Prediction history, newest first |
| GET    | `/api/traffic/predictions/latest`           | Most recent stored prediction |
| GET    | `/api/traffic/predictions/status`           | ML integration status |
| GET    | `/api/traffic/collector/status`             | Active provider status |
| POST   | `/api/traffic/collect`                      | Collect, then predict automatically |
| POST   | `/api/traffic/collect/history`              | Backfill history for prediction |
| GET    | `/api/dashboard/current`                    | Current state of every location |
| GET    | `/api/dashboard/trends`                     | Speed and forecast series |
| GET    | `/api/dashboard/locations`                   | Per-location readiness |
| GET    | `/api/dashboard/status`                     | Database, model, provider, scheduler |
| GET    | `/api/dashboard/model`                      | Features, provenance, metrics, limits |
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shut-down hooks for the application."""

    settings = get_settings()
    logger.info(
        "Starting %s v%s in %s mode",
        settings.SERVICE_NAME,
        app.version,
        settings.ENVIRONMENT,
    )

    if settings.DB_FAIL_FAST:
        from app.database.session import verify_database_connection

        try:
            verify_database_connection()
        except ConfigurationError as error:
            logger.error("Startup aborted: %s", error.message)
            raise
        logger.info("Database connectivity verified")

    # Opt-in only. Built inside the lifespan rather than at import time so a
    # scheduler never creates a database engine for a process that has opted out,
    # and so COLLECTION_ENABLED is read as late as possible. Registered as the
    # active scheduler so the dashboard status endpoint can report it without
    # reaching into this function.
    scheduler = None
    if settings.COLLECTION_ENABLED:
        from app.services.collection_scheduler import (
            build_scheduler,
            set_active_scheduler,
        )

        scheduler = build_scheduler(settings)
        scheduler.start()
        set_active_scheduler(scheduler)
        logger.info(
            "Background collection enabled: every %ds",
            scheduler.interval,
        )

    try:
        yield
    finally:
        from app.services.collection_scheduler import set_active_scheduler

        # Cleared before the worker is awaited: a status request arriving during
        # shutdown should not be told a scheduler is running while it stops.
        set_active_scheduler(None)
        if scheduler is not None:
            # Cancels the pending sleep and waits for the worker to finish, so an
            # in-flight cycle completes before the connection pool closes.
            await scheduler.stop()
            logger.info("Scheduler status: %s", scheduler.status())

        from app.database.session import dispose_engine

        await dispose_engine()
        logger.info("%s stopped", settings.SERVICE_NAME)


def create_application() -> FastAPI:
    """Build and configure the FastAPI application."""

    settings = get_settings()
    configure_logging(settings.API_LOG_LEVEL)

    application = FastAPI(
        title="AI-Based Urban Traffic Congestion Prediction API",
        description=API_DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    register_exception_handlers(application)
    application.include_router(api_router)

    frontend_dir = settings.frontend_dir
    if frontend_dir is not None:
        # Single-URL deployment (Docker, Render): serve the static dashboard
        # from the API's own origin so the demo needs no CORS and no second
        # service. Registered last, so /api/*, /docs, /redoc and /openapi.json
        # are always matched before the catch-all mount, and the openapi-schema
        # and the API description stay reachable from the hosted UI.
        @application.get("/", include_in_schema=False)
        async def dashboard_index() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

        application.mount(
            "/",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )
    else:
        @application.get("/", include_in_schema=False)
        async def root() -> dict[str, str]:
            """Minimal pointer to the docs and the health probe."""

            return {
                "service": settings.SERVICE_NAME,
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/api/health",
            }

    return application


app = create_application()