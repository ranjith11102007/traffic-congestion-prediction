"""SQLAlchemy ORM models for the traffic data layer.

Two tables, deliberately:

``traffic_observations``
    One row per reading from a sensor: when, where, and what was measured.

``traffic_predictions``
    One row per congestion prediction made from an observation, recording which
    model version produced it so historical forecasts stay attributable.

Column names deliberately mirror the Stage 3 ML contract where a concept exists
there (``avg_speed_kph``, ``free_flow_speed_kph``, ``weather_condition``), so a
stored row maps onto the model's input frame without translation. Fields that
only make sense for storage (``location_id``, ``road_name``, ``latitude``,
``longitude``) are included because the API needs to answer "what happened on
this road" and ``road_segment_id`` alone carries no location.

Leakage note: ``congestion_level`` is **not** stored on the observation. That
column is the derived target in the ML pipeline, computed from ``speed_ratio``.
Persisting it alongside the observation would invite a future feature builder to
read it back as an input, which is exactly the leak Stage 3 removed. Predictions
live in their own table instead.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base

#: Value stored in ``traffic_observations.source`` for synthetic development data.
SIMULATION_SOURCE = "simulation"

#: Origin labels for data that came from an external provider.
EXTERNAL_SOURCE = "external"

#: Origin label when a client posted the observation over the API.
API_SOURCE = "api"

#: Cap on a free-text source label, matching the column width.
SOURCE_MAX_LENGTH = 64


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time.

    Used as the Python-side default for ``prediction_timestamp`` so a prediction
    carries a real generation time even on a backend whose ``func.now()``
    server default has not been refreshed yet.
    """

    return datetime.now(timezone.utc)


class TrafficObservation(Base):
    """A single traffic reading from a location at a point in time.

    Uses timezone-aware ``TIMESTAMP WITH TIME ZONE`` because congestion patterns
    depend on local time-of-day, and storing naive timestamps makes "which
    morning is this" ambiguous across daylight-saving boundaries.
    """

    __tablename__ = "traffic_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- When the reading was taken ----------------------------------------
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="When the traffic state was observed (timezone-aware).",
    )

    # --- Where -------------------------------------------------------------
    location_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Stable sensor or junction identifier for the reporting location.",
    )
    road_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        doc="Human-readable road name.",
    )

    #: Maps onto the model's ``road_segment_id`` feature. NULL for readings that
    #: cannot be scored, e.g. a location the model has never seen.
    road_segment_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc=(
            "ML road segment this reading belongs to, e.g. SEG-01. Nullable "
            "because a reading can be stored for a location the model cannot score."
        ),
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Latitude in decimal degrees (-90..90).",
    )
    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Longitude in decimal degrees (-180..180).",
    )

    # --- What was measured -------------------------------------------------
    # Names match the Stage 3 feature frame exactly.
    vehicle_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc=(
            "Vehicles passing in the interval. Nullable because no self-serve "
            "traffic API publishes throughput; null means 'not reported by the "
            "provider', not zero."
        ),
    )
    avg_speed_kph: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Mean observed speed in km/h.",
    )
    free_flow_speed_kph: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Uncongested reference speed for the road, in km/h.",
    )

    # --- Context -----------------------------------------------------------
    weather_condition: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Weather label; must match the levels the model was trained on.",
    )
    temperature_c: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Air temperature in Celsius.",
    )
    rainfall_mm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Rainfall in millimetres.",
    )
    #: Native ``BOOLEAN``, not ``INTEGER``. A hand-rolled 0/1 flag invites
    #: out-of-range writes and forces every reader to remember the encoding; the
    #: database should enforce the domain. NULL stays meaningful: it means the
    #: provider does not report incidents, which is different from "none".
    is_incident: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        doc="True if an incident or roadworks affected the road; NULL if unreported.",
    )

    # --- Provenance --------------------------------------------------------
    source: Mapped[str] = mapped_column(
        String(SOURCE_MAX_LENGTH),
        nullable=False,
        default=API_SOURCE,
        index=True,
        doc=(
            "Origin of the record: 'simulation' for synthetic development data, "
            "'api' for a client POST, or the provider name for external data."
        ),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="Database-side insert time.",
    )

    predictions: Mapped[list["TrafficPrediction"]] = relationship(
        back_populates="observation",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="TrafficPrediction.id",
    )

    __table_args__ = (
        # Domain rules enforced in the database as well as in Pydantic.
        #
        # The API already rejects a free-flow speed below the observed speed, but
        # validation only covers requests that go through it. Anything writing via
        # SQL, a migration, or a future import job would otherwise be able to store
        # a contradictory row, and the label derived from that ratio would be
        # meaningless. Cheap to enforce here, and the constraint names are explicit
        # because the naming convention requires them for reversible migrations.
        CheckConstraint(
            # NULL means the provider does not report throughput, which SQL's
            # three-valued logic already treats as passing. Requiring >= 0
            # unconditionally would reject every genuine real-traffic row.
            "vehicle_count IS NULL OR vehicle_count >= 0",
            name="observations_vehicle_count_non_negative",
        ),
        CheckConstraint(
            "avg_speed_kph >= 0 AND free_flow_speed_kph >= 0",
            name="observations_speeds_non_negative",
        ),
        CheckConstraint(
            "free_flow_speed_kph > avg_speed_kph",
            name="observations_free_flow_exceeds_avg",
        ),
        CheckConstraint(
            "latitude IS NULL OR (latitude >= -90 AND latitude <= 90)",
            name="observations_latitude_in_range",
        ),
        CheckConstraint(
            "longitude IS NULL OR (longitude >= -180 AND longitude <= 180)",
            name="observations_longitude_in_range",
        ),
        CheckConstraint(
            "rainfall_mm IS NULL OR rainfall_mm >= 0",
            name="observations_rainfall_non_negative",
        ),
        # One reading per location per timestamp. Enforced in the database rather
        # than only in the service, so a concurrent POST cannot duplicate a row.
        Index(
            "uq_traffic_observations_location_timestamp",
            "location_id",
            "timestamp",
            unique=True,
        ),
        # Supports the "latest for this segment" read that prediction needs.
        Index(
            "ix_traffic_observations_segment_timestamp",
            "road_segment_id",
            "timestamp",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<TrafficObservation(id={self.id}, location={self.location_id!r}, "
            f"timestamp={self.timestamp!r}, source={self.source!r})>"
        )


class TrafficPrediction(Base):
    """A congestion forecast derived from one stored observation.

    Stores the full class distribution rather than only the winner, because the
    distribution is what tells you whether the prediction was decisive or a
    coin-flip between two adjacent levels.
    """

    __tablename__ = "traffic_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    observation_id: Mapped[int] = mapped_column(
        ForeignKey("traffic_observations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Observation this forecast was produced from.",
    )

    #: When the forecast was *made*. The observation it describes is linked via
    #: ``observation_id``, so its timestamp stays reachable without being
    #: overloaded into this column. Keeping the two separate is what makes
    #: "forecast run time minus observation time" answerable, which is how far
    #: ahead a prediction was made.
    prediction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=utc_now,
        doc="When the forecast was generated (timezone-aware UTC).",
    )

    predicted_congestion_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Predicted class id, 0=free_flow .. 3=severe.",
    )
    predicted_congestion: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Human-readable label for the predicted class.",
    )

    #: ``None`` when the estimator cannot report probabilities, rather than 0.0,
    #: so "unknown" is distinguishable from "no confidence".
    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Highest predicted class probability, or NULL if unavailable.",
    )

    class_probabilities: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Full per-class probability distribution, JSON-encoded.",
    )

    #: Taken from ml/models/model_metadata.json so every stored forecast is
    #: attributable to the exact artifact that produced it.
    model_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc="Version of the Stage 3 model artifact used, from model_metadata.json.",
    )

    #: Free-text reason when no model was available, e.g. insufficient history.
    note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Optional diagnostic recorded alongside the prediction.",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="Database-side insert time.",
    )

    observation: Mapped[TrafficObservation] = relationship(
        back_populates="predictions",
        lazy="selectin",
    )

    __table_args__ = (
        # A stored level outside the model's class range would mean the artifact
        # and the database disagree about what classes exist.
        CheckConstraint(
            "predicted_congestion_level >= 0 AND predicted_congestion_level <= 3",
            name="predictions_level_in_class_range",
        ),
        # A confidence above 1.0 is impossible; storing one would mean a
        # probability was mistaken for a confidence score.
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="predictions_confidence_in_unit_range",
        ),
        Index("ix_traffic_predictions_observation_created", "observation_id", "created_at"),
        # Supports the "newest forecast for these observations" read that the trend
        # endpoint does once per chart window. Without it that query sorts every
        # forecast for the window rather than walking the index, which is the
        # difference between a chart and a full scan as history accumulates.
        Index(
            "ix_traffic_predictions_observation_prediction",
            "observation_id",
            "prediction_timestamp",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<TrafficPrediction(id={self.id}, observation_id={self.observation_id}, "
            f"level={self.predicted_congestion_level}, model={self.model_version!r})>"
        )


__all__ = [
    "API_SOURCE",
    "EXTERNAL_SOURCE",
    "SIMULATION_SOURCE",
    "SOURCE_MAX_LENGTH",
    "TrafficObservation",
    "TrafficPrediction",
]