"""Allow a null vehicle_count.

Revision ID: b1f2c3d4e5a6
Revises: 467d498f7e77
Created: 2026-10-05

Stage 5 connected a real traffic provider, which surfaced a schema assumption
that only held for synthetic data: ``vehicle_count`` was ``NOT NULL``.

Why
---
The column held "vehicles passing in the interval", which the Stage 3 model read
as the ``flow_veh_per_hr`` feature. Stage 5 verified against current vendor
documentation that **no self-serve traffic API publishes vehicle throughput**.
TomTom Flow Segment Data returns current speed, free-flow speed, confidence and
road closure; HERE Traffic v7 adds jam factor and traversability; Mapbox
Directions annotates a route. None returns a count, volume or density.

Forcing a value would have meant one of two dishonest outcomes: storing a
fabricated number under a real provider's name, or refusing to record genuine
measurements at all. Both are worse than recording nothing, so the column
becomes nullable and a real observation simply leaves it NULL.

The matching change is model version 4.0.0, which dropped ``flow_veh_per_hr``
from the feature set. Those two changes have to ship together: keeping the column
mandatory while the model ignores it would be pointless, and making it nullable
while the model still demanded it would leave every real row unscorable.

Constraint
----------
The ``>= 0`` check is relaxed to ``IS NULL OR >= 0`` rather than dropped.
SQL's three-valued logic would let a plain ``>= 0`` pass on NULL anyway, but
spelling out the null case keeps the intent readable and keeps the predicate
true under a future change to the column's nullability.

Existing rows
-------------
Rows written by the simulated provider keep their counts untouched. No backfill,
no default, and no reinterpretation of a real 0 as "unknown": a NULL is only ever
written by a provider that never reported throughput in the first place.

Downgrade
---------
Restores ``NOT NULL`` and the strict check. Fails loudly if real rows are present
with a NULL count, which is the correct outcome — dropping those rows to force a
downgrade would destroy genuine observations.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision: str = "b1f2c3d4e5a6"
down_revision: str | None = "467d498f7e77"
branch_labels: str | None = None
depends_on: str | None = None

#: Constraint name *without* the ``ck_<table>_`` prefix.
#:
#: The target metadata carries ``ck = "ck_%(table_name)s_%(constraint_name)s"``, so
#: this short name resolves to
#: ``ck_traffic_observations_observations_vehicle_count_non_negative`` — the name
#: the previous revision created. Passing the already-prefixed name instead would
#: have the convention applied a second time, producing a name that matches nothing
#: and failing the drop.
CONSTRAINT = "observations_vehicle_count_non_negative"

#: The fully qualified name, useful in assertions and log messages.
CONSTRAINT_QUALIFIED = f"ck_traffic_observations_{CONSTRAINT}"


def upgrade() -> None:
    """Drop the NOT NULL and accept a null vehicle count.

    Uses batch mode, which is required for SQLite: the dialect cannot ``ALTER`` a
    constraint, so Alembic rewrites the table into a temporary one and moves the
    rows across. PostgreSQL, the deployment target, applies this as plain
    ``ALTER TABLE``.
    """

    with op.batch_alter_table("traffic_observations") as batch:
        batch.alter_column(
            "vehicle_count",
            existing_type=sa.Integer(),
            nullable=True,
        )
        batch.drop_constraint(CONSTRAINT, type_="check")
        batch.create_check_constraint(
            CONSTRAINT,
            "vehicle_count IS NULL OR vehicle_count >= 0",
        )


def downgrade() -> None:
    """Require a vehicle count again.

    Raises if any row has a NULL count rather than inventing a value for it: the
    restored NOT NULL rejects them, and silently substituting a number would turn
    a missing measurement into a fabricated one.
    """

    with op.batch_alter_table("traffic_observations") as batch:
        batch.drop_constraint(CONSTRAINT, type_="check")
        batch.create_check_constraint(
            CONSTRAINT,
            "vehicle_count >= 0",
        )
        batch.alter_column(
            "vehicle_count",
            existing_type=sa.Integer(),
            nullable=False,
        )