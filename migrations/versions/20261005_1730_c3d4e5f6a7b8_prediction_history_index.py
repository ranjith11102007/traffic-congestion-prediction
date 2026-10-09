"""Index the prediction history read path.

Revision ID: c3d4e5f6a7b8
Revises: b1f2c3d4e5a6
Created: 2026-10-05

Stage 6 added two read paths that this schema could not serve efficiently.

What changed
------------
The dashboard's trend endpoint resolves the newest forecast for every observation
in a chart window, and the history endpoint sorts stored forecasts by the
observation each one describes. Both walk ``observation_id`` and then order by a
timestamp, which is what ``ix_traffic_predictions_observation_created`` (on
``observation_id, created_at``) does not provide.

Adding a composite index on ``(observation_id, prediction_timestamp)`` lets both
reads walk the index in order and stop at the first row per partition instead of
sorting every forecast in the window. This matters as history accumulates rather
than at the current scale, which is why it is worth doing before the data is
large, not after.

Why a new index rather than changing the existing one
-----------------------------------------------------
``created_at`` and ``prediction_timestamp`` answer different questions.
``created_at`` is the database-side insert time; ``prediction_timestamp`` is when
the forecast was made, set in Python. They diverge for a forecast written with an
explicit generation time, and the Stage 6 endpoints all order by the latter.
Replacing the old index would silently degrade any future query about insert
order, so this adds rather than renames.

Column choice
-------------
``prediction_timestamp`` rather than ``id`` for the tie-break column. Both are
monotonic, but the index is read by queries that filter and order on the
timestamp itself, so including it lets the database satisfy the filter, the
ordering and the newest-per-group lookup from one structure.

Note on PostgreSQL
------------------
This index is not created concurrently. Alembic runs inside a transaction, and
``CREATE INDEX CONCURRENTLY`` cannot run in one. For a table at this size that is
the right trade: the table holds one row per prediction, and taking a brief write
lock during an index build on a table of that shape is cheaper than the operational
complexity of a concurrent build. Revisit if the predictions table grows large
enough for the lock to matter.

Downgrade
---------
Drops the index. No data is involved, so this is safe in both directions.
"""

from __future__ import annotations

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b1f2c3d4e5a6"
branch_labels: str | None = None
depends_on: str | None = None

INDEX_NAME = "ix_traffic_predictions_observation_prediction"
TABLE_NAME = "traffic_predictions"
COLUMNS = ("observation_id", "prediction_timestamp")


def upgrade() -> None:
    """Create the composite index for the prediction history reads."""

    op.create_index(INDEX_NAME, TABLE_NAME, list(COLUMNS), unique=False)


def downgrade() -> None:
    """Drop the composite index."""

    op.drop_index(INDEX_NAME, table_name=TABLE_NAME)