"""Alembic environment for the traffic prediction database.

Two decisions matter here.

**The DSN comes from application settings, not from alembic.ini.** Credentials
stay in the untracked ``.env`` and are never written to a committed config file.
The URL is passed to the engine, not logged, and error messages deliberately
describe the *scheme* rather than echoing a DSN that embeds a password.

**Metadata is imported from ``app.models``, not re-declared.** Importing the models
registers them on ``Base.metadata``, so autogenerate compares against the real
definitions. Hand-writing a second copy of the schema here would let the migration
and the ORM drift apart silently.

Offline and online modes are both supported: offline emits SQL for a DBA to
review, online runs against a live database.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the ``backend`` and repository roots importable so ``app`` and ``ml`` resolve
# regardless of the working directory Alembic is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from alembic import context as alembic_context  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.database.base import Base  # noqa: E402
import app.models  # noqa: E402,F401  # registers tables on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

#: Autogenerate compares against this.
target_metadata = Base.metadata


def _describe_url(url: str) -> str:
    """Describe a DSN without revealing credentials.

    Only the scheme is reported. A DSN such as
    ``postgresql+psycopg2://user:secret@host/db`` would otherwise be printed in a
    traceback or a log line, and ``sqlalchemy.engine`` logs URLs at DEBUG.
    """

    return url.split("://", 1)[0] + "://<redacted>" if "://" in url else "<redacted>"


def get_url() -> str:
    """Resolve the database URL for this migration run.

    Order of precedence:

    1. ``-x db_url=...`` on the command line, for one-off runs against another
       database without editing configuration;
    2. ``DATABASE_URL`` from the environment or the local ``.env``.

    Raises:
        ConfigurationError: when neither is set. Refusing to run is better than
            defaulting to something, because a migration that guesses its target
            database is destructive by definition.
    """

    x_args = alembic_context.get_x_argument(as_dictionary=True)
    override = x_args.get("db_url")
    if override:
        return override

    settings = get_settings()
    if not settings.has_database_configured:
        from app.core.exceptions import ConfigurationError

        raise ConfigurationError(
            "DATABASE_URL is not configured, so Alembic cannot tell which database "
            "to migrate. Copy .env.example to .env and set DATABASE_URL, or pass "
            "-x db_url=... for a one-off run. Refusing to guess a target database."
        )
    return settings.DATABASE_URL


def run_migrations_offline() -> None:
    """Emit migration SQL to stdout without connecting to a database."""

    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        # PostgreSQL-only schema. Hardcoding it keeps migrations deterministic
        # across environments; a non-PostgreSQL target is a test concern handled by
        # batch_alter_table in the migration itself.
        include_schemas=False,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""

    url = get_url()
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = url

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()