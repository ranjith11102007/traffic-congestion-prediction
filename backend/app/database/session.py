"""Database engine and session management.

The engine is created lazily and cached for the lifetime of the process, so
importing the application never opens a socket to PostgreSQL. That keeps the
API startable (and unit-testable) without a database, which is the expected
state of Phase 1.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.core.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def build_engine_options(settings: Settings) -> dict[str, Any]:
    """Return ``create_engine`` keyword arguments for the configured driver.

    Pool tuning only applies to drivers that support a real connection pool;
    SQLite (used by some tests) rejects ``max_overflow``/``pool_size``, so those
    keys are omitted for non-PostgreSQL DSNs.
    """

    options: dict[str, Any] = {"echo": settings.DB_ECHO, "future": True}
    if settings.DATABASE_URL.startswith(("postgresql", "postgres")):
        options.update(
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_pre_ping=True,
            connect_args={"connect_timeout": settings.DB_CONNECT_TIMEOUT},
        )
    return options


def get_engine() -> Engine:
    """Return the cached engine, creating it on first use."""

    global _engine
    if _engine is None:
        settings = get_settings()
        dsn = settings.sqlalchemy_dsn
        logger.info("Creating database engine for %s", dsn.split("@")[-1])
        _engine = create_engine(dsn, **build_engine_options(settings))
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the cached session factory, creating it on first use."""

    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            class_=Session,
        )
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session that always commits or rolls back.

    The client-facing error is deliberately generic. A SQLAlchemy exception string
    can embed the DSN it was raised from, and ``details`` is serialised straight
    into the JSON response, so passing it through would risk leaking the database
    password to any caller. The full error is logged server-side instead.
    """

    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Database transaction failed")
        raise DatabaseConnectionError(
            "The database transaction could not be completed. Check the server logs."
        ) from None
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""

    with session_scope() as session:
        yield session


def verify_database_connection() -> None:
    """Execute a trivial query to prove the database is reachable.

    Raises:
        DatabaseConnectionError: if the server cannot be reached. The message is
            generic on purpose; the underlying exception, which can carry the DSN,
            is logged rather than returned to the caller.
    """

    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.exception("Could not reach the configured database")
        raise DatabaseConnectionError(
            "Could not connect to the configured database. Check the server logs."
        ) from None


async def dispose_engine() -> None:
    """Close pooled connections during application shutdown."""

    if _engine is not None:
        await run_in_threadpool(_engine.dispose)
        logger.info("Database connection pool disposed")