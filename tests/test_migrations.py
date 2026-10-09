"""Tests for the Alembic migration and schema integrity.

Two things are verified that a normal application test cannot cover:

* the migration actually applies, reverses, and re-applies, and leaves no drift
  between the ORM metadata and the schema it created;
* the PostgreSQL DDL compiles to the right types, verified by rendering the
  migration offline against the PostgreSQL dialect.

The second matters because no PostgreSQL server is required for the suite to run,
yet a dialect mistake in the migration would otherwise only surface in production.
The offline check catches it here instead.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _read_revisions() -> list[tuple[str, str | None, Path]]:
    """Return ``(revision, down_revision, path)`` for every migration.

    Parsed from the migration files rather than repeated as literals, so adding a
    migration does not silently invalidate the revision-range assertions below. The
    chain is returned in topological order: an entry appears before anything that
    revises it.
    """

    files = sorted(
        path
        for path in (PROJECT_ROOT / "migrations" / "versions").glob("*.py")
        if not path.name.startswith("__")
    )
    assert files, "expected at least one migration"

    parsed: list[tuple[str, str | None, Path]] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        revision = re.search(r'^revision:\s*str\s*=\s*"([0-9a-fA-F]+)"', source, re.MULTILINE)
        assert revision, f"could not parse the revision id from {path.name}"
        down = re.search(
            r'^down_revision:\s*str\s*\|\s*None\s*=\s*(?:"([0-9a-fA-F]+)"|None)',
            source,
            re.MULTILINE,
        )
        assert down, f"could not parse down_revision from {path.name}"
        parsed.append((revision.group(1), down.group(1), path))

    order: list[tuple[str, str | None, Path]] = []
    remaining = list(parsed)
    while remaining:
        for entry in remaining:
            down_revision = entry[1]
            if down_revision is None or any(
                seen[0] == down_revision for seen in order
            ):
                order.append(entry)
                remaining.remove(entry)
                break
        else:
            raise AssertionError(
                f"migration chain has a cycle or a missing parent: {remaining}"
            )
    return order


REVISIONS = _read_revisions()

#: The revision that created the tables. Still used for the offline downgrade
#: range, which must name a single ``<from>:<to>`` pair.
INITIAL_REVISION = REVISIONS[0][0]


#: Every revision id, oldest first. ``head`` must resolve to the last one.
REVISION_IDS = [revision for revision, _, _ in REVISIONS]

#: A DSN with obvious placeholders. Offline mode renders SQL without connecting,
#: so these values are never used to reach a database and nothing is disclosed.
PLACEHOLDER_DSN = (
    "postgresql+psycopg2://placeholder:placeholder@localhost:5432/placeholder"
)




def _alembic_env() -> dict[str, str]:
    """Build the subprocess environment for an Alembic invocation.

    The current environment is inherited rather than replaced: Windows resolves
    Winsock providers through ``PATH``/``System32``, and importing
    ``_overlapped`` fails outright if those are missing. Only ``PYTHONPATH`` and
    ``DATABASE_URL`` are overridden, the latter cleared so a developer's own
    configuration cannot influence a test.
    """

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'backend'};{PROJECT_ROOT}"
    env.pop("DATABASE_URL", None)
    return env


def run_alembic(*args: str, db_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run one Alembic command against a throwaway SQLite database.

    Invoked as a subprocess so the migration is exercised exactly as an operator
    would run it, including env.py's configuration resolution, rather than through
    in-process shortcuts that could mask a broken entry point.

    Alembic accepts one command per invocation, so a multi-step sequence needs one
    call per step against the same ``db_path``.
    """

    if db_path is None:
        db_path = PROJECT_ROOT / ".pytest_migration.db"
        db_path.unlink(missing_ok=True)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-x",
            f"db_url=sqlite:///{db_path}",
            *args,
        ],
        cwd=PROJECT_ROOT,
        env=_alembic_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def render_postgres_sql(*args: str) -> str:
    """Render migration SQL for PostgreSQL without connecting to a server.

    Offline downgrades require an explicit ``<fromrev>:<torev>`` range, hence the
    separate entry point for them.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-x",
            f"db_url={PLACEHOLDER_DSN}",
            *args,
            "--sql",
        ],
        cwd=PROJECT_ROOT,
        env=_alembic_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_upgrade_applies_cleanly() -> None:
    """``upgrade head`` must succeed from an empty database."""

    db_path = PROJECT_ROOT / ".pytest_migration.db"
    db_path.unlink(missing_ok=True)
    try:
        result = run_alembic("upgrade", "head", db_path=db_path)
        assert result.returncode == 0, result.stderr
    finally:
        db_path.unlink(missing_ok=True)


def test_no_drift_between_migrations_and_orm() -> None:
    """A freshly migrated schema must match the ORM metadata exactly.

    ``alembic check`` fails if any column, index or constraint differs. Without
    this, the models and the migration could drift apart silently and the next
    autogenerate would emit an empty or misleading diff.
    """

    db_path = PROJECT_ROOT / ".pytest_migration.db"
    db_path.unlink(missing_ok=True)
    try:
        upgrade = run_alembic("upgrade", "head", db_path=db_path)
        assert upgrade.returncode == 0, upgrade.stderr

        check = run_alembic("check", db_path=db_path)
        assert check.returncode == 0, check.stdout + check.stderr
        assert "No new upgrade operations detected" in check.stdout
    finally:
        db_path.unlink(missing_ok=True)


def test_downgrade_then_upgrade_round_trips() -> None:
    """The migration must be reversible and re-appliable.

    A migration that cannot be rolled back traps an operator in a broken deploy
    with no way back but a hand-written fix. Each Alembic command needs its own
    invocation against the same database file.
    """

    db_path = PROJECT_ROOT / ".pytest_migration.db"
    db_path.unlink(missing_ok=True)
    try:
        up = run_alembic("upgrade", "head", db_path=db_path)
        assert up.returncode == 0, up.stderr

        down = run_alembic("downgrade", "base", db_path=db_path)
        assert down.returncode == 0, down.stderr

        up_again = run_alembic("upgrade", "head", db_path=db_path)
        assert up_again.returncode == 0, up_again.stderr

        check = run_alembic("check", db_path=db_path)
        assert check.returncode == 0, check.stdout + check.stderr
    finally:
        db_path.unlink(missing_ok=True)


def test_migration_creates_both_tables() -> None:
    """Both tables are created by the single initial migration."""

    sql = render_postgres_sql("upgrade", "head")

    assert "CREATE TABLE traffic_observations" in sql
    assert "CREATE TABLE traffic_predictions" in sql


def test_postgres_uses_timestamptz() -> None:
    """Every time column must be timezone-aware.

    Congestion depends on local time of day; a naive timestamp is ambiguous
    across daylight-saving boundaries.
    """

    sql = render_postgres_sql("upgrade", "head")

    assert "timestamp TIMESTAMP WITH TIME ZONE NOT NULL" in sql
    assert "prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL" in sql
    assert "created_at TIMESTAMP WITH TIME ZONE" in sql


def test_postgres_uses_boolean_for_incidents() -> None:
    """``is_incident`` must be a real BOOLEAN, not a hand-rolled 0/1 integer.

    Three-valued semantics matter: NULL means the provider does not report
    incidents, which is different from false.
    """

    sql = render_postgres_sql("upgrade", "head")

    assert "is_incident BOOLEAN" in sql
    assert "is_incident INTEGER" not in sql


def test_postgres_uses_serial_primary_keys() -> None:
    """Integer primary keys must render as SERIAL on PostgreSQL."""

    sql = render_postgres_sql("upgrade", "head")

    assert "id SERIAL NOT NULL" in sql


def test_foreign_key_cascades() -> None:
    """Deleting an observation must remove its predictions, not orphan them."""

    sql = render_postgres_sql("upgrade", "head")

    assert "FOREIGN KEY(observation_id) REFERENCES traffic_observations (id) ON DELETE CASCADE" in sql


def test_required_indexes_exist() -> None:
    """The indexes the read paths depend on must be in the migration.

    Verified against the requirement list, not against the models: the point is
    that the *deployed* schema has them.
    """

    sql = render_postgres_sql("upgrade", "head")

    for index in (
        "ix_traffic_observations_timestamp",
        "ix_traffic_observations_location_id",
        "ix_traffic_observations_road_name",
        "ix_traffic_observations_segment_timestamp",
        "ix_traffic_observations_road_segment_id",
        "ix_traffic_predictions_observation_id",
        "ix_traffic_predictions_prediction_timestamp",
        "ix_traffic_predictions_model_version",
        "ix_traffic_predictions_observation_created",
        # Stage 6: newest forecast per observation, the trend chart's read.
        "ix_traffic_predictions_observation_prediction",
    ):
        assert f"CREATE INDEX {index} " in sql, f"missing index {index}"


def test_unique_index_prevents_duplicate_observations() -> None:
    """One reading per location per timestamp is enforced in the database."""

    sql = render_postgres_sql("upgrade", "head")

    assert (
        "CREATE UNIQUE INDEX uq_traffic_observations_location_timestamp "
        "ON traffic_observations (location_id, timestamp)" in sql
    )


def test_check_constraints_are_emitted() -> None:
    """Domain rules must be enforced by the schema, not only by validation."""

    sql = render_postgres_sql("upgrade", "head")

    for constraint in (
        "ck_traffic_observations_observations_free_flow_exceeds_avg",
        "ck_traffic_observations_observations_latitude_in_range",
        "ck_traffic_observations_observations_longitude_in_range",
        "ck_traffic_observations_observations_vehicle_count_non_negative",
        "ck_traffic_predictions_predictions_level_in_class_range",
        "ck_traffic_predictions_predictions_confidence_in_unit_range",
    ):
        assert f"CONSTRAINT {constraint} " in sql, f"missing constraint {constraint}"


def test_downgrade_is_not_a_blanket_reset() -> None:
    """The downgrade must drop only these tables.

    A ``DROP SCHEMA`` or ``TRUNCATE`` in a migration is how a mistyped command
    empties a staging database.
    """

    upgrade_sql = render_postgres_sql("upgrade", "head")
    # Offline downgrades need an explicit revision range.
    downgrade_sql = render_postgres_sql("downgrade", f"{INITIAL_REVISION}:base")

    assert "DROP TABLE traffic_predictions" in downgrade_sql
    assert "DROP TABLE traffic_observations" in downgrade_sql

    for forbidden in ("DROP SCHEMA", "TRUNCATE", "DROP DATABASE"):
        assert forbidden not in downgrade_sql
        assert forbidden not in upgrade_sql


def test_predictions_dropped_before_observations() -> None:
    """The child table must go first or the foreign key blocks the drop."""

    downgrade_sql = render_postgres_sql("downgrade", f"{INITIAL_REVISION}:base")

    predictions_at = downgrade_sql.index("DROP TABLE traffic_predictions")
    observations_at = downgrade_sql.index("DROP TABLE traffic_observations")

    assert predictions_at < observations_at


def test_migration_never_embeds_credentials() -> None:
    """No DSN or password may be baked into the migration.

    Alembic output can be captured into CI logs, so a connection string with a
    password committed here would leak it permanently.
    """

    versions = list((PROJECT_ROOT / "migrations" / "versions").glob("*.py"))
    assert versions, "expected at least one migration"

    pattern = re.compile(r"postgresql(\+\w+)?://[^\s'\"]+", re.IGNORECASE)
    for path in versions:
        assert not pattern.search(path.read_text(encoding="utf-8")), (
            f"{path.name} contains what looks like a database URL"
        )


def test_migrations_refuse_to_run_without_a_target_database() -> None:
    """With no DSN configured, Alembic must fail rather than guess.

    A migration that picks its own target database is destructive by definition.
    """

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=PROJECT_ROOT,
        env=_alembic_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "DATABASE_URL" in combined


def test_alembic_ini_does_not_commit_a_dsn() -> None:
    """``alembic.ini`` must leave ``sqlalchemy.url`` empty.

    The DSN is resolved from settings at runtime, so a committed value would put
    credentials in version control.
    """

    text = (PROJECT_ROOT / "alembic.ini").read_text(encoding="utf-8")

    assert re.search(r"^\s*sqlalchemy\.url\s*=\s*$", text, re.MULTILINE), (
        "sqlalchemy.url should be present and empty"
    )


# ---------------------------------------------------------------------------
# Chain integrity
# ---------------------------------------------------------------------------


def test_exactly_one_root_revision() -> None:
    """Alembic requires a single base; two roots make ``downgrade base`` ambiguous."""

    roots = [revision for revision, down, _ in REVISIONS if down is None]

    assert len(roots) == 1, f"expected one root revision, found {roots}"


def test_revision_ids_are_unique() -> None:
    seen: set[str] = set()
    for revision, _, path in REVISIONS:
        assert revision not in seen, f"duplicate revision id {revision} in {path.name}"
        seen.add(revision)


def test_every_down_revision_exists() -> None:
    """A dangling parent breaks the chain at exactly the wrong moment."""

    known = set(REVISION_IDS)

    for revision, down, path in REVISIONS:
        if down is not None:
            assert down in known, (
                f"{path.name} revises {down}, which does not exist in the chain"
            )
        assert down != revision, f"{path.name} revises itself"


def test_heads_matches_the_last_revision() -> None:
    """``upgrade head`` must resolve to the newest migration, with no branches."""

    result = run_alembic("heads", db_path=PROJECT_ROOT / ".pytest_migration.db")

    assert result.returncode == 0, result.stderr
    heads = [line.strip().split()[0] for line in result.stdout.splitlines() if "(" in line]
    assert heads == [REVISION_IDS[-1]], f"expected one head, found {heads}"


# ---------------------------------------------------------------------------
# Stage 5: nullable vehicle_count
# ---------------------------------------------------------------------------


CONSTRAINT = "ck_traffic_observations_observations_vehicle_count_non_negative"


def test_postgres_drops_not_null_for_vehicle_count() -> None:
    """A real observation has no vehicle count, so the column must allow NULL."""

    sql = render_postgres_sql("upgrade", "head")

    assert "ALTER TABLE traffic_observations ALTER COLUMN vehicle_count DROP NOT NULL" in sql


def test_postgres_check_allows_a_null_vehicle_count() -> None:
    """The bound is relaxed to permit NULL, not dropped.

    Dropping the check entirely would also admit negative counts, and the
    constraint is the only thing protecting rows written outside the API.
    """

    sql = render_postgres_sql("upgrade", "head")

    assert (
        f"ADD CONSTRAINT {CONSTRAINT} CHECK (vehicle_count IS NULL OR vehicle_count >= 0)"
        in sql
    )


def test_upgraded_schema_accepts_a_null_vehicle_count() -> None:
    """A row with no throughput must actually store, on a real migrated schema."""

    import sqlite3

    db_path = PROJECT_ROOT / ".pytest_migration.db"
    db_path.unlink(missing_ok=True)
    try:
        assert run_alembic("upgrade", "head", db_path=db_path).returncode == 0

        connection = sqlite3.connect(db_path)
        try:
            schema = connection.execute(
                "SELECT sql FROM sqlite_master"
                " WHERE name='traffic_observations' AND type='table'"
            ).fetchone()[0]
            assert "vehicle_count INTEGER NOT NULL" not in schema

            connection.execute(
                "INSERT INTO traffic_observations"
                " (timestamp, location_id, road_name, vehicle_count, avg_speed_kph,"
                "  free_flow_speed_kph, weather_condition, source)"
                " VALUES ('2026-10-05T12:00:00+00:00', 'LOC-001', 'Anna Salai',"
                "         NULL, 30.0, 60.0, 'clear', 'tomtom')"
            )
            connection.commit()
            stored = connection.execute(
                "SELECT vehicle_count FROM traffic_observations"
            ).fetchone()[0]
            assert stored is None
        finally:
            connection.close()
    finally:
        db_path.unlink(missing_ok=True)


def test_a_negative_vehicle_count_is_still_rejected() -> None:
    """Relaxing the constraint must not weaken it for present values."""

    import sqlite3

    db_path = PROJECT_ROOT / ".pytest_migration.db"
    db_path.unlink(missing_ok=True)
    try:
        assert run_alembic("upgrade", "head", db_path=db_path).returncode == 0

        connection = sqlite3.connect(db_path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO traffic_observations"
                    " (timestamp, location_id, road_name, vehicle_count,"
                    "  avg_speed_kph, free_flow_speed_kph, weather_condition, source)"
                    " VALUES ('2026-10-05T12:00:00+00:00', 'LOC-002', 'Anna Salai',"
                    "         -5, 30.0, 60.0, 'clear', 'tomtom')"
                )
        finally:
            connection.close()
    finally:
        db_path.unlink(missing_ok=True)


def test_downgrade_restores_the_strict_column() -> None:
    """Reversing the change must put the original guarantee back.

    Rendered from ``head`` rather than from the initial revision, so the range
    actually includes this migration's ``downgrade``.
    """

    sql = render_postgres_sql("downgrade", "head:base")

    assert "ALTER COLUMN vehicle_count SET NOT NULL" in sql
    assert f"ADD CONSTRAINT {CONSTRAINT} CHECK (vehicle_count >= 0)" in sql