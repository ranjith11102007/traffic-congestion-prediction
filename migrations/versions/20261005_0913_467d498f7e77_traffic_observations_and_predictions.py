"""Create the traffic observation and prediction tables.

Revision ID: 467d498f7e77
Revises:
Created: 2026-10-05

Two tables, in dependency order.

``traffic_observations``
    One row per reading from a location. Named for what was measured rather than
    what was concluded, so ``congestion_level`` stays absent: that is the derived
    ML target, and storing it next to the features invites a future feature builder
    to read the answer back as an input.

``traffic_predictions``
    One row per forecast, linked to the observation it describes.

PostgreSQL notes
----------------
``TIMESTAMP WITH TIME ZONE`` on every time column, because congestion depends on
local time of day and a naive timestamp is ambiguous across daylight-saving
boundaries.

``BOOLEAN`` for ``is_incident`` rather than an integer flag. Three-valued here:
true, false, or NULL when the provider does not report incidents. NULL is
distinct from false and must not be collapsed to zero.

``TEXT`` for ``class_probabilities`` holding JSON. The class count is small and
fixed by the model artifact, so ``JSONB`` would buy nothing but a migration
every time the class set changed; the service encodes and decodes it.

CHECK constraints mirror the Pydantic validation rules. Validation only protects
requests that pass through the API; these protect rows written by a migration, an
import job, or a future service that forgets to call a validator.

Downgrade
---------
Drops the tables and their indexes, children first. There is no ``TRUNCATE``,
``DROP SCHEMA`` or other blanket reset: a migration that can empty the whole
database on a mistyped command is how a staging environment loses its data.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "467d498f7e77"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create both tables with their constraints and indexes."""

    op.create_table(
        "traffic_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location_id", sa.String(length=64), nullable=False),
        sa.Column("road_name", sa.String(length=255), nullable=False),
        sa.Column("road_segment_id", sa.String(length=64), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("vehicle_count", sa.Integer(), nullable=False),
        sa.Column("avg_speed_kph", sa.Float(), nullable=False),
        sa.Column("free_flow_speed_kph", sa.Float(), nullable=False),
        sa.Column("weather_condition", sa.String(length=32), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("rainfall_mm", sa.Float(), nullable=True),
        sa.Column("is_incident", sa.Boolean(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "avg_speed_kph >= 0 AND free_flow_speed_kph >= 0",
            name=op.f("ck_traffic_observations_observations_speeds_non_negative"),
        ),
        sa.CheckConstraint(
            "free_flow_speed_kph > avg_speed_kph",
            name=op.f("ck_traffic_observations_observations_free_flow_exceeds_avg"),
        ),
        sa.CheckConstraint(
            "latitude IS NULL OR (latitude >= -90 AND latitude <= 90)",
            name=op.f("ck_traffic_observations_observations_latitude_in_range"),
        ),
        sa.CheckConstraint(
            "longitude IS NULL OR (longitude >= -180 AND longitude <= 180)",
            name=op.f("ck_traffic_observations_observations_longitude_in_range"),
        ),
        sa.CheckConstraint(
            "rainfall_mm IS NULL OR rainfall_mm >= 0",
            name=op.f("ck_traffic_observations_observations_rainfall_non_negative"),
        ),
        sa.CheckConstraint(
            "vehicle_count >= 0",
            name=op.f("ck_traffic_observations_observations_vehicle_count_non_negative"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_traffic_observations")),
    )
    # Indexes are created after the table so each column-level index can be dropped
    # independently on downgrade.
    op.create_index(
        op.f("ix_traffic_observations_location_id"),
        "traffic_observations",
        ["location_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_traffic_observations_road_name"),
        "traffic_observations",
        ["road_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_traffic_observations_road_segment_id"),
        "traffic_observations",
        ["road_segment_id"],
        unique=False,
    )
    op.create_index(
        "ix_traffic_observations_segment_timestamp",
        "traffic_observations",
        ["road_segment_id", "timestamp"],
        unique=False,
    )
    op.create_index(
        op.f("ix_traffic_observations_source"),
        "traffic_observations",
        ["source"],
        unique=False,
    )
    op.create_index(
        op.f("ix_traffic_observations_timestamp"),
        "traffic_observations",
        ["timestamp"],
        unique=False,
    )
    # Uniqueness enforced by an index rather than a UNIQUE constraint so it can be
    # dropped without altering the table.
    op.create_index(
        "uq_traffic_observations_location_timestamp",
        "traffic_observations",
        ["location_id", "timestamp"],
        unique=True,
    )

    op.create_table(
        "traffic_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("observation_id", sa.Integer(), nullable=False),
        # Generation time, not observation time. The observation's own time is one
        # join away, and keeping the two separate makes a prediction's lead time
        # answerable.
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_congestion_level", sa.Integer(), nullable=False),
        sa.Column("predicted_congestion", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("class_probabilities", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name=op.f("ck_traffic_predictions_predictions_confidence_in_unit_range"),
        ),
        sa.CheckConstraint(
            "predicted_congestion_level >= 0 AND predicted_congestion_level <= 3",
            name=op.f("ck_traffic_predictions_predictions_level_in_class_range"),
        ),
        # ON DELETE CASCADE: a prediction has no meaning without its observation,
        # so deleting the observation should not orphan the forecast.
        sa.ForeignKeyConstraint(
            ["observation_id"],
            ["traffic_observations.id"],
            name=op.f("fk_traffic_predictions_observation_id_traffic_observations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_traffic_predictions")),
    )
    op.create_index(
        op.f("ix_traffic_predictions_model_version"),
        "traffic_predictions",
        ["model_version"],
        unique=False,
    )
    op.create_index(
        "ix_traffic_predictions_observation_created",
        "traffic_predictions",
        ["observation_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_traffic_predictions_observation_id"),
        "traffic_predictions",
        ["observation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_traffic_predictions_prediction_timestamp"),
        "traffic_predictions",
        ["prediction_timestamp"],
        unique=False,
    )


def downgrade() -> None:
    """Drop both tables and their indexes, children before parents.

    This destroys stored traffic and forecast data and is not reversible in terms
    of content. It exists so a deployment can be rolled back to a pre-Stage-4
    schema; it is never the correct way to clear data between environments.
    """

    op.drop_index(
        op.f("ix_traffic_predictions_prediction_timestamp"),
        table_name="traffic_predictions",
    )
    op.drop_index(
        op.f("ix_traffic_predictions_observation_id"), table_name="traffic_predictions"
    )
    op.drop_index(
        "ix_traffic_predictions_observation_created", table_name="traffic_predictions"
    )
    op.drop_index(
        op.f("ix_traffic_predictions_model_version"), table_name="traffic_predictions"
    )
    op.drop_table("traffic_predictions")

    op.drop_index(
        "uq_traffic_observations_location_timestamp", table_name="traffic_observations"
    )
    op.drop_index(op.f("ix_traffic_observations_timestamp"), table_name="traffic_observations")
    op.drop_index(op.f("ix_traffic_observations_source"), table_name="traffic_observations")
    op.drop_index(
        "ix_traffic_observations_segment_timestamp", table_name="traffic_observations"
    )
    op.drop_index(
        op.f("ix_traffic_observations_road_segment_id"), table_name="traffic_observations"
    )
    op.drop_index(op.f("ix_traffic_observations_road_name"), table_name="traffic_observations")
    op.drop_index(op.f("ix_traffic_observations_location_id"), table_name="traffic_observations")
    op.drop_table("traffic_observations")