"""Application settings.

Every runtime value is read from the process environment and, when present,
from a local ``.env`` file located in the project root. Secrets are never
defined in code: ``DATABASE_URL`` and any future provider credentials must be
supplied through the environment.

If ``DATABASE_URL`` is left empty the API still starts, because Phase 1 has no
persisted domain data yet. Database-backed features raise a clear
``ConfigurationError`` instead of silently degrading.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationError

# backend/app/core/config.py -> core -> app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Service metadata ---------------------------------------------------
    SERVICE_NAME: str = Field(
        default="traffic-prediction-api",
        description="Identifier reported by the health endpoint.",
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="Deployment environment label, e.g. development|staging|production.",
    )

    # --- HTTP server --------------------------------------------------------
    API_HOST: str = Field(default="127.0.0.1")
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    API_RELOAD: bool = Field(
        default=True, description="Enable uvicorn auto-reload during development."
    )
    API_LOG_LEVEL: str = Field(default="INFO")

    # --- Database (PostgreSQL via SQLAlchemy) --------------------------------
    DATABASE_URL: str = Field(
        default="",
        description=(
            "SQLAlchemy PostgreSQL DSN, e.g. "
            "postgresql+psycopg2://user:password@host:5432/traffic_prediction. "
            "Intentionally empty by default so no credential is stored in source."
        ),
    )
    DB_ECHO: bool = Field(default=False, description="Log all emitted SQL.")
    DB_POOL_SIZE: int = Field(default=5, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0)
    DB_POOL_TIMEOUT: int = Field(default=30, ge=0)
    DB_CONNECT_TIMEOUT: int = Field(
        default=5, ge=1, description="Seconds to wait for a new database connection."
    )
    DB_FAIL_FAST: bool = Field(
        default=False,
        description=(
            "When true the application refuses to start if the database is "
            "unreachable. Keep false during Phase 1 development."
        ),
    )

    # --- CORS ---------------------------------------------------------------
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            # Browsers report "Origin: null" for pages opened via file://.
            "null",
        ],
        description="Browser origins allowed to call the API.",
    )

    # --- External data providers ---------------------------------------------
    # Real keys belong in .env, which is git-ignored. These are read at runtime
    # and never echoed back to a client.
    #
    # Vendor selection (Stage 5)
    # -------------------------
    # Verified against current vendor documentation:
    #   * TomTom Flow Segment Data - currentSpeed, freeFlowSpeed, confidence,
    #     roadClosure, frc, openlr. JSON. kmph by default. India covered.
    #     Docs: docs.tomtom.com/pricing states 20,000 free requests/month for
    #     "Traffic Flow API Segment Data", no credit card. Endpoint:
    #     docs.tomtom.com/traffic-api/.../flow-segment-data (service version 4).
    #   * HERE Traffic API v7 - richer (bbox query, jamFactor, traversability,
    #     server timestamp) but speeds are metres/second and no self-serve
    #     historical endpoint exists.
    #   * Mapbox - route annotations only, and the bulk Traffic Data product is
    #     Enterprise-only.
    #
    # TomTom is the default because it is the only candidate whose free allowance
    # is documented, and because a point query per monitored location fits a
    # project that monitors a fixed set of roads.
    #
    # None of the three publishes vehicle throughput. That is why
    # ``vehicle_count`` is nullable and why model version 4.0.0 dropped the
    # ``flow_veh_per_hr`` feature.
    TRAFFIC_PROVIDER: str = Field(
        default="simulation",
        description=(
            "Which provider the collector uses: 'simulation' for the synthetic "
            "development fixture, or 'real' for an external traffic API."
        ),
    )
    TRAFFIC_API_KEY: str = Field(
        default="",
        description="API key for the real traffic provider. Empty means unconfigured.",
    )
    TRAFFIC_API_BASE_URL: str = Field(
        default="https://api.tomtom.com",
        description=(
            "Base URL of the external traffic API. Defaults to TomTom's global "
            "endpoint; the provider appends the documented service path."
        ),
    )
    TRAFFIC_API_VENDOR: str = Field(
        default="tomtom",
        description=(
            "Which vendor's request/response mapping to use. Only 'tomtom' is "
            "implemented. Unknown values are rejected at start-up rather than "
            "guessed at, because the response shapes are not interchangeable."
        ),
    )
    TRAFFIC_ZOOM: int = Field(
        default=10,
        ge=0,
        le=22,
        description=(
            "TomTom zoom level for the flow segment query. Higher values see more "
            "minor roads but shift the returned segment geometry."
        ),
    )
    TRAFFIC_REQUEST_TIMEOUT: float = Field(
        default=10.0,
        gt=0,
        le=120,
        description="Seconds to wait for a single upstream traffic request.",
    )
    TRAFFIC_MIN_REQUEST_INTERVAL: float = Field(
        default=0.15,
        ge=0,
        le=60,
        description=(
            "Minimum seconds between upstream traffic requests. TomTom documents "
            "10 queries/second for non-tile traffic endpoints; this keeps a "
            "multi-location collection comfortably inside that."
        ),
    )
    TRAFFIC_MAX_LOCATIONS: int = Field(
        default=25,
        ge=1,
        le=200,
        description=(
            "Upper bound on how many monitored locations one collection may query, "
            "so a misconfigured file cannot spend a month's free quota in a cycle."
        ),
    )

    # --- Weather provider ----------------------------------------------------
    # No traffic vendor publishes weather, and the model requires it. Open-Meteo
    # is used instead: its forecast API needs no API key for non-commercial use
    # and documents a 10,000 calls/day, 300,000/month allowance
    # (open-meteo.com/en/pricing).
    WEATHER_API_KEY: str = Field(
        default="",
        description=(
            "Unused. Open-Meteo's free tier requires no key; the parameter exists "
            "only for a future commercial endpoint, which also needs a different host."
        ),
    )
    WEATHER_API_BASE_URL: str = Field(
        default="https://api.open-meteo.com/v1",
        description="Open-Meteo forecast endpoint base URL.",
    )
    WEATHER_REQUEST_TIMEOUT: float = Field(
        default=10.0,
        gt=0,
        le=120,
        description="Seconds to wait for a single upstream weather request.",
    )
    WEATHER_ENABLED: bool = Field(
        default=True,
        description=(
            "Fetch weather alongside traffic. When false the provider omits "
            "temperature and rainfall rather than substituting values."
        ),
    )

    # --- Monitored locations -------------------------------------------------
    TRAFFIC_LOCATIONS_FILE: str = Field(
        default="",
        description=(
            "Path to the JSON file describing the monitored roads. Empty means "
            "config/monitored_locations.json beside the project root."
        ),
    )

    # --- Automated collection ------------------------------------------------
    COLLECTION_ENABLED: bool = Field(
        default=False,
        description=(
            "Run periodic background collection. Off by default so a deployment "
            "never spends a vendor quota without an explicit opt-in."
        ),
    )
    COLLECTION_INTERVAL_SECONDS: int = Field(
        default=300,
        ge=30,
        le=86400,
        description=(
            "Seconds between scheduled collections. The default is 300s (12 calls "
            "per hour at 3 locations), well inside both vendors' documented limits."
        ),
    )

    # --- Automated prediction -------------------------------------------------
    AUTO_PREDICT_AFTER_COLLECTION: bool = Field(
        default=True,
        description=(
            "Score every freshly stored observation with the trained model and keep "
            "the forecast. A location with too little history is left unpredicted and "
            "the reason is recorded: no forecast is invented to fill the gap."
        ),
    )

    # --- Dashboard reads ------------------------------------------------------
    # A dashboard that labels old data as live is worse than one that admits it is
    # old, so freshness is derived from the newest stored timestamp and reported
    # with the threshold that produced it.
    DATA_FRESHNESS_THRESHOLD_SECONDS: int = Field(
        default=900,
        ge=60,
        le=86400,
        description=(
            "Seconds after which the newest observation is reported as stale rather "
            "than fresh. The default 900s is twice the 300s collection interval, so "
            "one missed cycle is tolerated and two are not."
        ),
    )
    DASHBOARD_TREND_MAX_HOURS: int = Field(
        default=168,
        ge=1,
        le=8760,
        description=(
            "Upper bound on the `hours` window of GET /api/dashboard/trends. One "
            "year of five-minute readings is already a large response, and an "
            "unbounded window would let one request read the whole table."
        ),
    )

    # --- ML inference --------------------------------------------------------
    ML_MODELS_DIR: str = Field(
        default="",
        description=(
            "Directory holding the Stage 3 artifacts. Empty means ml/models, "
            "resolved from the repository layout."
        ),
    )
    ML_PREDICTION_HISTORY: int = Field(
        default=12,
        ge=6,
        le=240,
        description=(
            "How many prior observations per road segment to load from the "
            "database when scoring. Must be at least the trained model's "
            "min_history_per_segment, because lag and rolling features read "
            "backwards."
        ),
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """Reject an obviously wrong DSN early instead of at first query.

        PostgreSQL is the deployment target. SQLite is tolerated so unit tests
        can exercise the persistence layer without a running server; it is not a
        supported production database.
        """

        value = value.strip()
        if value and not value.startswith(("postgresql", "postgres", "sqlite")):
            raise ValueError(
                "DATABASE_URL must be a PostgreSQL DSN, e.g. "
                "postgresql+psycopg2://user:password@localhost:5432/traffic_prediction. "
                "A sqlite:// DSN is accepted for local tests only."
            )
        return value

    @field_validator("API_LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        level = value.strip().upper()
        if level not in allowed:
            raise ValueError(f"API_LOG_LEVEL must be one of {sorted(allowed)}")
        return level

    @property
    def sqlalchemy_dsn(self) -> str:
        """Return the configured DSN or explain how to provide one."""

        if not self.DATABASE_URL:
            raise ConfigurationError(
                "DATABASE_URL is not configured. Copy .env.example to .env and set "
                "DATABASE_URL=postgresql+psycopg2://<user>:<password>@<host>:5432/"
                "<database> before using database-backed features."
            )
        return self.DATABASE_URL

    @field_validator("TRAFFIC_PROVIDER")
    @classmethod
    def validate_traffic_provider(cls, value: str) -> str:
        """Restrict the provider name to the two implemented options.

        Silently accepting an unknown name would fall back to simulated data and
        let it pass as real traffic, so an unrecognised value is rejected at
        start-up instead.
        """

        allowed = {"simulation", "real"}
        name = value.strip().lower()
        if name not in allowed:
            raise ValueError(
                f"TRAFFIC_PROVIDER must be one of {sorted(allowed)}, got {value!r}"
            )
        return name

    @property
    def has_database_configured(self) -> bool:
        return bool(self.DATABASE_URL)

    @property
    def uses_simulation(self) -> bool:
        """True when the collector is serving synthetic development data."""

        return self.TRAFFIC_PROVIDER == "simulation"

    @property
    def has_real_provider_configured(self) -> bool:
        """True when the base URL and key for a real provider are both present.

        The base URL now has a TomTom default, so this effectively tests the key.
        That is the correct signal: a URL alone cannot authenticate a request, and
        an authenticated request without a URL is equally impossible.
        """

        return bool(self.TRAFFIC_API_BASE_URL.strip() and self.TRAFFIC_API_KEY.strip())

    @field_validator("TRAFFIC_API_VENDOR")
    @classmethod
    def validate_traffic_api_vendor(cls, value: str) -> str:
        """Reject vendors whose response shape is not implemented.

        The provider classes map a specific vendor's JSON onto TrafficRecord. A
        HERE or Mapbox payload read as if it were TomTom's would produce plausible
        but wrong speeds, so an unknown name must fail rather than be assumed.
        """

        implemented = {"tomtom"}
        name = value.strip().lower()
        if name not in implemented:
            raise ValueError(
                f"TRAFFIC_API_VENDOR must be one of {sorted(implemented)}, got {value!r}. "
                "Only the TomTom response mapping is implemented."
            )
        return name

    @field_validator("TRAFFIC_API_BASE_URL", "WEATHER_API_BASE_URL")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Require HTTPS for an outbound API base URL, when one is set.

        Both vendors accept an API key as a query parameter, so an http:// base URL
        would put a credential on the wire in the clear. Requiring https is the
        cheapest way to make that mistake impossible.

        An empty value stays legal and means "no vendor configured". The provider
        turns that into an actionable
        :class:`~app.core.exceptions.ProviderNotConfiguredError` at fetch time,
        which is a more useful place for the message than at settings load: a
        deployment running the labelled simulation provider should start cleanly
        with no traffic configuration at all.
        """

        url = value.strip().rstrip("/")
        if not url:
            return url
        if not url.startswith("https://"):
            raise ValueError(
                f"API base URL must use https:// so credentials are not sent in "
                f"the clear, got {value!r}"
            )
        return url

    @model_validator(mode="after")
    def enforce_production_constraints(self) -> "Settings":
        """Refuse to run a production environment on demo defaults.

        Every secret has an empty default so a development checkout starts clean.
        That same default is dangerous in production: an operator who forgets
        ``TRAFFIC_PROVIDER`` would silently serve simulated numbers as if they
        were real traffic, and an empty ``TRAFFIC_API_KEY`` with the real
        provider fails at first collection instead of failing loudly at boot.
        ``ENVIRONMENT=production`` therefore requires the real provider, its
        credential, and a PostgreSQL database, or start-up fails with a message
        naming the missing setting without echoing its value.
        """

        if self.ENVIRONMENT.strip().lower() != "production":
            return self
        if self.TRAFFIC_PROVIDER != "real":
            raise ValueError(
                "ENVIRONMENT=production requires TRAFFIC_PROVIDER=real; the "
                "synthetic simulation provider must never serve production data."
            )
        if not self.TRAFFIC_API_KEY.strip():
            raise ValueError(
                "ENVIRONMENT=production with TRAFFIC_PROVIDER=real requires "
                "TRAFFIC_API_KEY for the configured traffic vendor."
            )
        if not self.DATABASE_URL.strip():
            raise ValueError(
                "ENVIRONMENT=production requires DATABASE_URL; empty is only "
                "legal for Phase-1 development."
            )
        if self.DATABASE_URL.strip().startswith("sqlite"):
            raise ValueError(
                "ENVIRONMENT=production does not accept a sqlite:// DATABASE_URL; "
                "PostgreSQL is the deployment target."
            )
        return self

    @property
    def models_dir(self) -> str:
        """Resolve the ML artifact directory, defaulting to ``ml/models``."""

        if self.ML_MODELS_DIR.strip():
            return self.ML_MODELS_DIR.strip()
        return str(PROJECT_ROOT / "ml" / "models")

    @property
    def traffic_locations_path(self) -> Path:
        """Resolve the monitored-locations JSON path.

        An absolute path is used as given, so a deployment can keep its location
        file outside the repository. A relative one is resolved against the
        project root rather than the working directory, because the working
        directory differs between ``uvicorn backend.app.main:app`` and a test run.
        """

        configured = self.TRAFFIC_LOCATIONS_FILE.strip()
        if configured:
            candidate = Path(configured)
            return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        return PROJECT_ROOT / "config" / "monitored_locations.json"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()