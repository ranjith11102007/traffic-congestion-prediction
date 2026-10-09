"""SQLAlchemy ORM models.

Stage 4 implements the two tables the traffic pipeline actually needs. They are
defined in :mod:`app.models.traffic`, which imports :class:`~app.database.base.Base`
so Alembic's ``target_metadata`` and ``Base.metadata.create_all`` both see them.

=====================================  ==========================================
Table                                  Purpose
=====================================  ==========================================
``traffic_observations``               Timestamped speed/flow readings per location.
``traffic_predictions``                Stored forecasts, linked to their observation.
=====================================  ==========================================

Importing this package registers both tables with ``Base.metadata``. Alembic's
``env.py`` relies on that side effect, so it must not be removed.

The Phase 2 wish list (``road_segments``, ``congestion_labels``,
``weather_observations``, ``model_registry``, ``prediction_runs``) is deliberately
still unimplemented. ``congestion_level`` is not stored on observations on
purpose: it is the derived target, and keeping it out of the observation table
prevents a future feature builder from reading the answer back as an input.
"""

from app.models.traffic import (
    API_SOURCE,
    EXTERNAL_SOURCE,
    SIMULATION_SOURCE,
    TrafficObservation,
    TrafficPrediction,
)

__all__ = [
    "API_SOURCE",
    "EXTERNAL_SOURCE",
    "SIMULATION_SOURCE",
    "TrafficObservation",
    "TrafficPrediction",
]