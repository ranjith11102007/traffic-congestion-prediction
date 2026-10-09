"""Pytest configuration.

``backend`` is added to ``sys.path`` so tests can import ``app.*`` exactly the
way the API does, without installing the project as a package.

The database fixtures use an isolated in-memory SQLite database rather than
PostgreSQL. The suite must run on a clean checkout with no server and no
credentials, and the ORM uses portable column types, so the behaviour under test
is the same. The Alembic migration is verified separately against the PostgreSQL
dialect, which catches dialect-specific mistakes SQLite cannot.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_sessionstart(session: pytest.Session) -> None:
    """Generate the simulated fixture if it is absent.

    The fixture is deterministic and untracked, so regenerating it here keeps the
    end-to-end tests running on a clean checkout instead of skipping.
    """

    from ml import preprocessing

    if preprocessing.DEFAULT_SAMPLE_PATH.exists():
        return
    from ml.data.sample.generate_simulated_dataset import write_dataset

    write_dataset(preprocessing.DEFAULT_SAMPLE_PATH)


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Test client bound to a freshly built application instance.

    This client has no database override, so it suits the tests that must run
    without one, such as the health probe. Tests that touch the database use
    :func:`db_client` instead.
    """

    from app.main import create_application

    with TestClient(create_application()) as test_client:
        yield test_client


@pytest.fixture()
def db_engine():
    """Return a fresh in-memory database with the schema created.

    ``StaticPool`` keeps one connection alive for the whole test. Without it
    SQLAlchemy opens a new connection per checkout, and each new connection to an
    in-memory SQLite database gets its own empty copy of the schema.
    """

    from sqlalchemy.pool import StaticPool

    from app.database.base import Base
    import app.models  # noqa: F401  registers tables on Base.metadata

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def db_session(db_engine) -> Iterator[Session]:
    """Yield a session bound to the throwaway database, rolled back afterwards."""

    factory = sessionmaker(
        bind=db_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def db_session_factory(db_engine):
    """Return a session factory bound to the throwaway database.

    The scheduler needs a factory rather than a single session: it opens its own
    session per cycle, which is what a long-lived background worker must do
    because a SQLAlchemy session is not safe to share across threads.
    """

    return sessionmaker(
        bind=db_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )


@pytest.fixture()
def db_client(db_engine, monkeypatch) -> Iterator[TestClient]:
    """A test client whose ``get_db`` dependency yields the throwaway database.

    Overriding the dependency is what keeps the production engine, and any real
    ``DATABASE_URL``, out of the test suite entirely.
    """

    from app.database.session import get_db
    from app.main import create_application

    factory = sessionmaker(
        bind=db_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )

    def override_get_db() -> Iterator[Session]:
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    application = create_application()
    application.dependency_overrides[get_db] = override_get_db
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


@pytest.fixture()
def observation_payload() -> dict:
    """A minimal valid observation body.

    Free-flow speed is above observed speed and every model-required field is
    present, so the payload passes the ``TrafficObservationBase`` invariants.
    """

    return {
        "timestamp": "2026-01-18T08:00:00+00:00",
        "location_id": "LOC-001",
        "road_name": "Anna Salai",
        "road_segment_id": "SEG-01",
        "latitude": 13.0569,
        "longitude": 80.2425,
        "vehicle_count": 900,
        "avg_speed_kph": 32.0,
        "free_flow_speed_kph": 60.0,
        "weather_condition": "clear",
        "temperature_c": 27.5,
        "rainfall_mm": 0.0,
        "is_incident": False,
        "source": "simulation",
    }