"""Declarative base shared by every ORM model.

``Base.metadata`` carries an explicit naming convention. Without it SQLAlchemy
emits anonymous constraints, which causes two concrete problems: Alembic cannot
autogenerate stable, reversible migrations, and SQLite cannot ``ALTER`` a
constraint at all, so the batch-alter path used for tests fails. Naming them up
front means the same models work on PostgreSQL and in the SQLite test database.
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        columns = ", ".join(f"{name}={value!r}" for name, value in vars(self).items())
        return f"<{type(self).__name__}({columns})>"