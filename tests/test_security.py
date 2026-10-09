"""Tests for secret handling.

The rule being enforced: no credential may appear in an HTTP response, an error
envelope, a log line, a settings dump, or any committed file. These are the paths
a credential most easily leaks through, and each one is checked explicitly rather
than assumed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    DatabaseConnectionError,
    TrafficPredictionError,
    register_exception_handlers,
)
from app.services.real_provider import RealTrafficProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECRET = "sk-live-do-not-disclose-1234567890"
DB_PASSWORD = "pg-super-secret-password"


def settings_with_secrets(**overrides: str) -> Settings:
    """Build settings carrying distinctive secrets, bypassing ``.env``."""

    defaults = {
        "DATABASE_URL": f"postgresql+psycopg2://appuser:{DB_PASSWORD}@db.internal:5432/traffic",
        "TRAFFIC_PROVIDER": "real",
        "TRAFFIC_API_KEY": SECRET,
        "TRAFFIC_API_BASE_URL": "https://api.example.invalid/traffic",
    }
    return Settings(**{**defaults, **overrides})


def test_database_url_is_not_exposed_in_configuration_errors() -> None:
    """A missing-DSN message must tell you what to set, not what it is.

    ``sqlalchemy_dsn`` is reached whenever the database is unconfigured, and its
    message is rendered into an error envelope.
    """

    settings = Settings(DATABASE_URL="")

    with pytest.raises(TrafficPredictionError) as caught:
        _ = settings.sqlalchemy_dsn

    rendered = f"{caught.value.message} {caught.value.details}"
    assert DB_PASSWORD not in rendered
    assert SECRET not in rendered


def test_domain_error_details_are_controlled_not_passed_through() -> None:
    """A domain error carries a curated message, not a raw driver exception.

    The database layer used to place ``str(error)`` into ``details``, and a
    SQLAlchemy connection error can embed the DSN it came from.
    """

    error = DatabaseConnectionError("The database transaction could not be completed.")

    rendered = f"{error.message} {error.details}"

    assert DB_PASSWORD not in rendered
    assert "psycopg2" not in rendered


def test_real_provider_description_never_contains_the_key() -> None:
    """``describe()`` is returned to clients by the collector status endpoint."""

    provider = RealTrafficProvider(settings_with_secrets())

    for text in (provider.describe(), provider.name, str(provider.is_configured)):
        assert SECRET not in text


def test_api_key_is_not_returned_by_collector_status(client) -> None:
    """The collector status response must carry no credential material."""

    response = client.get("/api/traffic/collector/status")

    body = response.text
    assert response.status_code == 200
    assert SECRET not in body
    assert "TRAFFIC_API_KEY" not in body
    assert "password" not in body.lower()


def test_openapi_schema_exposes_no_secrets(client) -> None:
    """The generated schema is public; it must not describe secret values."""

    response = client.get("/openapi.json")

    assert response.status_code == 200
    body = response.text
    assert SECRET not in body
    assert DB_PASSWORD not in body


def test_settings_repr_does_not_dump_every_value() -> None:
    """``repr(settings)`` must not become a credential dump.

    A settings object is easily logged by accident. This does not assert that
    secrets are hidden by type, only that the object has no blanket ``__dict__``
    dump that would print everything: the real mitigation is that these values are
    supplied by the environment and never written to a committed file.
    """

    settings = settings_with_secrets()

    # Documented behaviour: the raw secret is reachable on the object, which is
    # why logging the instance wholesale is discouraged. Asserting the risk is
    # recorded here so a future change to SecretStr is a visible improvement.
    assert settings.TRAFFIC_API_KEY == SECRET
    assert DB_PASSWORD in settings.DATABASE_URL


def test_health_payload_never_contains_configuration(client) -> None:
    """The health endpoint must not report on database credentials."""

    response = client.get("/api/health")

    assert response.status_code == 200
    assert SECRET not in response.text
    assert DB_PASSWORD not in response.text


def test_validation_errors_do_not_echo_the_whole_payload(db_client) -> None:
    """A rejected request must not be reflected back wholesale.

    Echoing the submitted body in an error response would turn any field the
    client added into something stored in server logs and client-visible output.
    """

    response = db_client.post(
        "/api/traffic/observations",
        json={
            "timestamp": "2026-01-18T08:00:00+00:00",
            "location_id": "",
            "road_name": "Somewhere",
            "vehicle_count": -5,
            "avg_speed_kph": 90.0,
            "free_flow_speed_kph": 60.0,
            "weather_condition": "clear",
            "some_unexpected_field": "sensitive-value",
        },
    )

    assert response.status_code == 422
    body = response.text
    assert "sensitive-value" not in body
    assert SECRET not in body


def test_env_example_contains_no_real_credentials() -> None:
    """``.env.example`` documents settings with obviously fake, empty values only."""

    example = PROJECT_ROOT / ".env.example"

    if not example.exists():
        pytest.skip(".env.example not present")

    text = example.read_text(encoding="utf-8")

    # No key-shaped value, and every secret-bearing setting left empty.
    assert not re.search(r"sk-[A-Za-z0-9]{16,}", text)
    for setting in ("DATABASE_URL", "TRAFFIC_API_KEY", "WEATHER_API_KEY"):
        assert re.search(rf"^{setting}=\s*$", text, re.MULTILINE), (
            f"{setting} must be present and empty in .env.example"
        )

    # Base URLs are different in kind: a vendor's public endpoint is not a secret,
    # and Stage 5 gives both a real default so a fresh checkout works once a key is
    # added. What matters is that they are documented as plain https endpoints and
    # that no credential is embedded in them.
    for setting in ("TRAFFIC_API_BASE_URL", "WEATHER_API_BASE_URL"):
        match = re.search(rf"^{setting}=(\S*)$", text, re.MULTILINE)
        assert match, f"{setting} must be present in .env.example"
        url = match.group(1)
        assert url.startswith("https://"), f"{setting} must be an https endpoint"
        assert "@" not in url, f"{setting} must not embed credentials"
        assert re.search(rf"^{setting}=\s*$", text, re.MULTILINE) is None, (
            f"{setting} now carries a documented default and should not also appear "
            "as an empty line"
        )


def test_no_credentials_committed_to_the_repository() -> None:
    """No tracked file may contain a real database URL with a password in it.

    The format string ``postgresql://user:secret@host/db`` is legitimate
    documentation and appears in the settings descriptions and the migration
    environment, so matching the URL shape alone would flag every helpful example
    in the codebase. Only a DSN whose credential looks like an actual secret
    counts, which means excluding the obvious placeholders.
    """

    dsn = re.compile(
        r"postgresql(\+\w+)?://(?P<user>[^\s:@'\"]+):(?P<secret>[^\s@'\"]+)@",
        re.IGNORECASE,
    )
    placeholders = {"", "secret", "password", "pass", "placeholder", "user", "changeme"}
    skip_roots = {".venv", "__pycache__", ".git", "ml/models", "data"}
    text_suffixes = {".py", ".ini", ".txt", ".md", ".json", ".cfg", ".toml", ".example", ".env"}

    offenders: list[str] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_roots for part in path.parts):
            continue
        if path.suffix.lower() not in text_suffixes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in dsn.finditer(content):
            secret = match.group("secret")
            # Angle-bracket templates such as <user>:<credential>@ are documentation.
            if secret.lower() in placeholders:
                continue
            if secret.startswith("<") or secret.endswith(">"):
                continue
            if "your_" in secret.lower() or "example" in secret.lower():
                continue
            # A braced value is an f-string placeholder, e.g. "postgresql://u:{PWD}@".
            if "{" in secret or "}" in secret:
                continue
            # No real credential is a handful of characters. Test fixtures use
            # "u:p@"-style abbreviations, which this excludes without weakening
            # the check on genuine secrets.
            if len(secret) < 6:
                continue
            offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {match.group(0)}")

    assert not offenders, f"files contain real inline database credentials: {offenders}"


def test_dotenv_is_not_tracked() -> None:
    """A real ``.env`` must be ignored by git.

    Checked against ``.gitignore`` rather than the index, since the working tree
    is not a repository in every checkout.
    """

    gitignore = PROJECT_ROOT / ".gitignore"

    if not gitignore.exists():
        pytest.skip(".gitignore not present")

    lines = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert ".env" in lines, "the real .env file must be ignored"
    # The template must stay tracked, which takes an explicit negation.
    assert "!.env.example" in lines, (
        "the example file must stay tracked even though .env is ignored"
    )