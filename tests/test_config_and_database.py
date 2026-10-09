"""Tests for configuration loading and database wiring.

No live PostgreSQL server is required: these tests exercise validation and
option building only, which is exactly what Phase 1 ships.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.database.session import build_engine_options


def test_defaults_match_agreed_local_settings() -> None:
    settings = Settings(_env_file=None)

    assert settings.API_HOST == "127.0.0.1"
    assert settings.API_PORT == 8000
    assert settings.SERVICE_NAME == "traffic-prediction-api"


def test_database_url_may_be_empty() -> None:
    """The API must start without a database during Phase 1."""

    settings = Settings(_env_file=None, DATABASE_URL="")

    assert settings.has_database_configured is False


def test_sqlalchemy_dsn_raises_when_not_configured() -> None:
    settings = Settings(_env_file=None, DATABASE_URL="")

    with pytest.raises(ConfigurationError) as error:
        _ = settings.sqlalchemy_dsn

    assert "DATABASE_URL" in error.value.message


def test_sqlalchemy_dsn_is_returned_when_configured() -> None:
    dsn = "postgresql+psycopg2://user:secret@localhost:5432/traffic_prediction"
    settings = Settings(_env_file=None, DATABASE_URL=dsn)

    assert settings.sqlalchemy_dsn == dsn


def test_non_postgres_dsn_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DATABASE_URL="mysql://user:secret@localhost/traffic")


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, API_LOG_LEVEL="VERBOSE")


@pytest.mark.parametrize(
    ("dsn", "expects_pool_options"),
    [
        ("postgresql+psycopg2://u:p@localhost:5432/db", True),
        ("sqlite:///./local.db", False),
    ],
)
def test_pool_options_only_for_postgres(
    dsn: str, expects_pool_options: bool
) -> None:
    """SQLite rejects pool_size/max_overflow, so they must not be passed."""

    settings = Settings(_env_file=None, DATABASE_URL=dsn)
    options = build_engine_options(settings)

    assert ("pool_size" in options) is expects_pool_options
    assert ("max_overflow" in options) is expects_pool_options
    assert options["echo"] is False


def test_echo_flag_is_passed_through() -> None:
    settings = Settings(_env_file=None, DATABASE_URL="sqlite:///./local.db", DB_ECHO=True)

    assert build_engine_options(settings)["echo"] is True


# --- Production guard ------------------------------------------------------
#
# Every default is empty so a development checkout starts with no credentials.
# ``ENVIRONMENT=production`` flips the risk: falling back to the simulation
# provider would serve invented numbers as if they were live traffic, so a
# production environment must be explicitly wired to the real provider, its
# credential, and PostgreSQL, or fail at start-up with a message naming the
# missing setting.


def test_production_requires_real_provider() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, ENVIRONMENT="production")

    assert "TRAFFIC_PROVIDER" in str(error.value)


def test_production_real_provider_requires_api_key() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(
            _env_file=None,
            ENVIRONMENT="production",
            TRAFFIC_PROVIDER="real",
        )

    assert "TRAFFIC_API_KEY" in str(error.value)


def test_production_requires_database_url() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(
            _env_file=None,
            ENVIRONMENT="production",
            TRAFFIC_PROVIDER="real",
            TRAFFIC_API_KEY="secret-key",
        )

    assert "DATABASE_URL" in str(error.value)


def test_production_rejects_sqlite_database() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(
            _env_file=None,
            ENVIRONMENT="production",
            TRAFFIC_PROVIDER="real",
            TRAFFIC_API_KEY="secret-key",
            DATABASE_URL="sqlite:///./local.db",
        )

    assert "PostgreSQL" in str(error.value)


def test_production_valid_configuration_passes() -> None:
    settings = Settings(
        _env_file=None,
        ENVIRONMENT="production",
        TRAFFIC_PROVIDER="real",
        TRAFFIC_API_KEY="secret-key",
        DATABASE_URL="postgresql+psycopg2://appuser:secret@db.internal:5432/traffic",
    )

    assert settings.enforce_production_constraints is not None
    assert settings.uses_simulation is False
    assert settings.has_real_provider_configured is True
    assert settings.has_database_configured is True


def test_production_error_does_not_echo_secret_value() -> None:
    """The key value must never appear in the configuration error message."""

    with pytest.raises(ValidationError) as error:
        Settings(
            _env_file=None,
            ENVIRONMENT="production",
            TRAFFIC_PROVIDER="real",
            TRAFFIC_API_KEY="super-secret-value",
        )

    messages = [entry.get("msg", "") for entry in error.value.errors()]
    assert messages
    assert all("super-secret-value" not in message for message in messages)


def test_development_defaults_allow_simulated_provider() -> None:
    """The simulation default stays legal for development checkouts."""

    settings = Settings(_env_file=None)

    assert settings.ENVIRONMENT == "development"
    assert settings.uses_simulation is True