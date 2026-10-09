"""Database access for the traffic tables.

All SQL lives here so the service layer stays free of query construction and the
routes stay free of both. Every function takes a :class:`~sqlalchemy.orm.Session`
rather than opening one, so callers own the transaction boundary and tests can
share a single connection.

Reads are ordered newest-first by default because the overwhelmingly common
question is "what is happening now".
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import DuplicateObservationError
from app.models.traffic import TrafficObservation, TrafficPrediction

#: Columns the dashboard reads for an observation. Deliberately a subset of the
#: table: the map and the status panel never show ``vehicle_count``, ``created_at``
#: or ``is_incident``, and selecting only what is rendered keeps a 200-location
#: dashboard to two narrow queries instead of twenty full rows.
LATEST_OBSERVATION_COLUMNS = (
    TrafficObservation.id,
    TrafficObservation.timestamp,
    TrafficObservation.location_id,
    TrafficObservation.road_name,
    TrafficObservation.road_segment_id,
    TrafficObservation.latitude,
    TrafficObservation.longitude,
    TrafficObservation.avg_speed_kph,
    TrafficObservation.free_flow_speed_kph,
    TrafficObservation.weather_condition,
    TrafficObservation.temperature_c,
    TrafficObservation.rainfall_mm,
    TrafficObservation.source,
)

#: Columns a dashboard needs from a forecast. ``confidence`` is included so a
#: consumer can check whether the estimator genuinely produced one, rather than
#: inferring a value from its absence.
LATEST_PREDICTION_COLUMNS = (
    TrafficPrediction.id,
    TrafficPrediction.observation_id,
    TrafficPrediction.prediction_timestamp,
    TrafficPrediction.predicted_congestion_level,
    TrafficPrediction.predicted_congestion,
    TrafficPrediction.confidence,
    TrafficPrediction.model_version,
    TrafficPrediction.created_at,
)


def _rank_desc(partition: Any, *order_columns: Any) -> Any:
    """Return a ``row_number()`` that orders each partition from newest first.

    Used instead of the ``GROUP BY MAX(...)`` trick so the whole row comes back
    rather than just the key, and instead of a Python-side ``max()`` over a full
    table scan so the database still only touches the rows it returns. Ties are
    broken by the primary key, so the result is deterministic even if two rows
    somehow share a timestamp.
    """

    return func.row_number().over(
        partition_by=partition,
        order_by=[column.desc() for column in order_columns],
    )


def _base_query() -> Select[tuple[TrafficObservation]]:
    return select(TrafficObservation)


def build_filter_query(
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> Select[tuple[TrafficObservation]]:
    """Compose the observation filter query.

    Every filter is optional and they combine with AND. Keeping the composition in
    one place means the list, latest and prediction-history paths cannot drift
    apart in how they treat time bounds.
    """

    query = _base_query()
    if location_id is not None:
        query = query.where(TrafficObservation.location_id == location_id)
    if road_segment_id is not None:
        query = query.where(TrafficObservation.road_segment_id == road_segment_id)
    if start_time is not None:
        query = query.where(TrafficObservation.timestamp >= start_time)
    if end_time is not None:
        query = query.where(TrafficObservation.timestamp <= end_time)
    return query


def list_observations(
    session: Session,
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 50,
) -> list[TrafficObservation]:
    """Return matching observations, newest first.

    Args:
        limit: maximum rows to return; clamped to ``1..MAX_PAGE_SIZE`` by the
            caller, and the query always applies a hard LIMIT.
    """

    query = build_filter_query(
        location_id=location_id,
        road_segment_id=road_segment_id,
        start_time=start_time,
        end_time=end_time,
    )
    query = query.order_by(TrafficObservation.timestamp.desc()).limit(limit)
    return list(session.execute(query).scalars())


def count_observations(
    session: Session,
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> int:
    """Count observations matching the same filters as :func:`list_observations`."""

    query = build_filter_query(
        location_id=location_id,
        road_segment_id=road_segment_id,
        start_time=start_time,
        end_time=end_time,
    )
    return int(
        session.execute(
            select(func.count()).select_from(query.subquery())
        ).scalar_one()
    )


def get_observation(session: Session, observation_id: int) -> TrafficObservation | None:
    """Return one observation by primary key, or ``None``."""

    return session.get(TrafficObservation, observation_id)


def get_latest_observation(
    session: Session,
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
) -> TrafficObservation | None:
    """Return the most recent observation, optionally for one location."""

    query = build_filter_query(
        location_id=location_id, road_segment_id=road_segment_id
    )
    query = query.order_by(TrafficObservation.timestamp.desc()).limit(1)
    return session.execute(query).scalars().first()


def add_observation(
    session: Session, observation: TrafficObservation
) -> TrafficObservation:
    """Persist one observation and return it with its assigned id.

    The unique constraint on ``(location_id, timestamp)`` is enforced by the
    database, so a duplicate arriving through two concurrent requests is caught
    here and converted into a domain error rather than surfacing as a 500.

    Raises:
        DuplicateObservationError: if a reading already exists for that location
            and timestamp.
    """

    session.add(observation)
    try:
        session.flush()
    except IntegrityError as error:
        session.rollback()
        raise DuplicateObservationError(
            "An observation already exists for this location and timestamp.",
            details={
                "location_id": observation.location_id,
                "timestamp": observation.timestamp.isoformat(),
            },
        ) from error
    session.refresh(observation)
    return observation


def get_recent_history(
    session: Session,
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
    limit: int,
    exclude_observation_id: int | None = None,
) -> list[TrafficObservation]:
    """Return the most recent observations, oldest first.

    Oldest-first ordering matches what the model's lag features expect: the last
    row becomes the row being scored and earlier rows become its history.

    Args:
        location_id: restrict to one sensor or junction.
        road_segment_id: restrict to one ML road segment. The model builds its
            features *per segment*, so scoring a segment against another
            segment's history would mix incompatible references. Pass this when
            the caller knows the segment; rows whose segment is NULL are excluded
            because they cannot be scored.
        limit: maximum rows to return.
        exclude_observation_id: a row to leave out, typically the observation
            about to be predicted, so it is not both history and target.

    Raises:
        ValueError: if neither ``location_id`` nor ``road_segment_id`` is given,
            since an unfiltered history would span every segment in the database.
    """

    if location_id is None and road_segment_id is None:
        raise ValueError(
            "get_recent_history requires location_id or road_segment_id; an "
            "unfiltered history would mix unrelated road segments."
        )

    query = select(TrafficObservation)
    if location_id is not None:
        query = query.where(TrafficObservation.location_id == location_id)
    if road_segment_id is not None:
        # An equality filter already excludes NULL rows, so unsegmented readings
        # can never enter a segment's history.
        query = query.where(TrafficObservation.road_segment_id == road_segment_id)
    if exclude_observation_id is not None:
        query = query.where(TrafficObservation.id != exclude_observation_id)

    query = query.order_by(TrafficObservation.timestamp.desc()).limit(limit)
    rows = list(session.execute(query).scalars())
    rows.reverse()
    return rows


def has_enough_history(
    session: Session,
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
    required: int,
) -> bool:
    """Report whether a location or segment has at least ``required`` readings.

    Args:
        required: 1 is a convenient "does anything exist at all" check.
    """

    query = select(func.count()).select_from(TrafficObservation)
    if location_id is not None:
        query = query.where(TrafficObservation.location_id == location_id)
    if road_segment_id is not None:
        query = query.where(TrafficObservation.road_segment_id == road_segment_id)
    return int(session.execute(query).scalar_one()) >= required


def add_prediction(
    session: Session, prediction: TrafficPrediction
) -> TrafficPrediction:
    """Persist one prediction and return it with its assigned id."""

    session.add(prediction)
    session.flush()
    session.refresh(prediction)
    return prediction


def list_predictions(
    session: Session,
    *,
    observation_id: int | None = None,
    limit: int = 50,
) -> list[TrafficPrediction]:
    """Return stored predictions, newest first."""

    query = select(TrafficPrediction)
    if observation_id is not None:
        query = query.where(TrafficPrediction.observation_id == observation_id)
    query = query.order_by(TrafficPrediction.id.desc()).limit(limit)
    return list(session.execute(query).scalars())


def count_predictions_for_observation(session: Session, observation_id: int) -> int:
    """Count predictions recorded for one observation."""

    return int(
        session.execute(
            select(func.count())
            .select_from(TrafficPrediction)
            .where(TrafficPrediction.observation_id == observation_id)
        ).scalar_one()
    )


def reset_database(session: Session) -> None:
    """Truncate both tables. Test-only helper, never called by the API."""

    session.execute(TrafficPrediction.__table__.delete())
    session.execute(TrafficObservation.__table__.delete())
    session.commit()


def observed_segments(session: Session) -> Sequence[str]:
    """Return every distinct road segment that has at least one observation."""

    rows = session.execute(
        select(TrafficObservation.road_segment_id)
        .where(TrafficObservation.road_segment_id.is_not(None))
        .distinct()
        .order_by(TrafficObservation.road_segment_id)
    ).scalars()
    return tuple(rows)


# ---------------------------------------------------------------------------
# Dashboard reads
#
# Every function below serves a page that renders many locations at once. They all
# resolve to a fixed number of queries — never one per location — because an N+1
# here would turn a single dashboard poll into hundreds of round trips, which is
# exactly what a browser refreshes every few seconds.
# ---------------------------------------------------------------------------


def latest_observations_by_location(
    session: Session,
    *,
    location_ids: Sequence[str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Return the newest observation for each location in one query.

    Args:
        location_ids: restrict to these locations. ``None`` means every location
            that has any stored observation.

    Returns:
        ``location_id -> row mapping``. Locations with no observations are absent
        rather than mapped to a fabricated row.
    """

    rank = _rank_desc(
        TrafficObservation.location_id,
        TrafficObservation.timestamp,
        TrafficObservation.id,
    ).label("row_rank")
    inner = select(*LATEST_OBSERVATION_COLUMNS, rank)
    if location_ids is not None:
        inner = inner.where(TrafficObservation.location_id.in_(tuple(location_ids)))
    ranked = inner.subquery()

    rows = session.execute(select(ranked).where(ranked.c.row_rank == 1)).mappings()
    return {row["location_id"]: row for row in rows}


def latest_predictions_by_location(
    session: Session,
    *,
    location_ids: Sequence[str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Return the newest forecast for each location in one query.

    Joins through ``traffic_observations`` because a prediction is stored against
    an observation, and a location only exists on the observation side. Forecasts
    over an observation with no location cannot happen — the column is NOT NULL —
    so nothing is lost by scoping through the join.
    """

    rank = _rank_desc(
        TrafficObservation.location_id,
        TrafficPrediction.prediction_timestamp,
        TrafficPrediction.id,
    ).label("row_rank")
    inner = (
        select(
            TrafficObservation.location_id.label("location_id"),
            TrafficObservation.road_name.label("road_name"),
            TrafficObservation.road_segment_id.label("road_segment_id"),
            TrafficObservation.timestamp.label("observation_timestamp"),
            TrafficObservation.source.label("source"),
            *LATEST_PREDICTION_COLUMNS,
            rank,
        )
        .select_from(TrafficPrediction)
        .join(TrafficObservation, TrafficPrediction.observation_id == TrafficObservation.id)
    )
    if location_ids is not None:
        inner = inner.where(TrafficObservation.location_id.in_(tuple(location_ids)))
    ranked = inner.subquery()

    rows = session.execute(select(ranked).where(ranked.c.row_rank == 1)).mappings()
    return {row["location_id"]: row for row in rows}


def build_prediction_filter_query(
    *,
    location_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> Select[tuple[Any, ...]]:
    """Compose the prediction-history filter query.

    ``start_time`` and ``end_time`` bound the **observation** timestamp, not the
    moment the forecast was generated. A dashboard plots a forecast against the
    traffic state it describes, so filtering on the state keeps the history list,
    the trend series and the map reading the same window. Both timestamps are
    returned on every row, so a caller that cares about lead time can compute it.
    """

    query = select(
        TrafficPrediction.id,
        TrafficPrediction.observation_id,
        TrafficPrediction.prediction_timestamp,
        TrafficPrediction.predicted_congestion_level,
        TrafficPrediction.predicted_congestion,
        TrafficPrediction.confidence,
        TrafficPrediction.class_probabilities,
        TrafficPrediction.model_version,
        TrafficPrediction.note,
        TrafficPrediction.created_at,
        TrafficObservation.location_id,
        TrafficObservation.road_name,
        TrafficObservation.road_segment_id,
        TrafficObservation.timestamp.label("observation_timestamp"),
        TrafficObservation.source,
    ).select_from(TrafficPrediction).join(
        TrafficObservation, TrafficPrediction.observation_id == TrafficObservation.id
    )

    if location_id is not None:
        query = query.where(TrafficObservation.location_id == location_id)
    if start_time is not None:
        query = query.where(TrafficObservation.timestamp >= start_time)
    if end_time is not None:
        query = query.where(TrafficObservation.timestamp <= end_time)
    return query


def list_prediction_history(
    session: Session,
    *,
    location_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 50,
) -> list[Mapping[str, Any]]:
    """Return stored forecasts newest first, with their observation's location.

    The window is always applied in SQL. A history endpoint without a hard LIMIT
    is an invitation to read the whole table into one response.
    """

    query = build_prediction_filter_query(
        location_id=location_id, start_time=start_time, end_time=end_time
    )
    query = query.order_by(
        TrafficObservation.timestamp.desc(), TrafficPrediction.id.desc()
    ).limit(limit)
    return list(session.execute(query).mappings())


def latest_prediction(
    session: Session,
    *,
    location_id: str | None = None,
) -> Mapping[str, Any] | None:
    """Return the newest forecast overall, or for one location."""

    query = build_prediction_filter_query(location_id=location_id).order_by(
        TrafficObservation.timestamp.desc(), TrafficPrediction.id.desc()
    ).limit(1)
    return session.execute(query).mappings().first()


def count_predictions(
    session: Session,
    *,
    location_id: str | None = None,
) -> int:
    """Count stored forecasts, optionally for one location."""

    query = select(func.count()).select_from(TrafficPrediction).join(
        TrafficObservation, TrafficPrediction.observation_id == TrafficObservation.id
    )
    if location_id is not None:
        query = query.where(TrafficObservation.location_id == location_id)
    return int(session.execute(query).scalar_one())


def observation_series(
    session: Session,
    *,
    location_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 500,
) -> list[Mapping[str, Any]]:
    """Return the speed series a trend chart plots, oldest first.

    Oldest first because that is the order a time axis needs; the list and latest
    endpoints deliberately go the other way.
    """

    query = select(
        TrafficObservation.id,
        TrafficObservation.timestamp,
        TrafficObservation.location_id,
        TrafficObservation.road_name,
        TrafficObservation.avg_speed_kph,
        TrafficObservation.free_flow_speed_kph,
        TrafficObservation.weather_condition,
        TrafficObservation.source,
    )
    if location_id is not None:
        query = query.where(TrafficObservation.location_id == location_id)
    if start_time is not None:
        query = query.where(TrafficObservation.timestamp >= start_time)
    if end_time is not None:
        query = query.where(TrafficObservation.timestamp <= end_time)

    # The newest rows win when the window is trimmed, then the result is reversed
    # for charting. Ordering before the LIMIT is what makes the truncation keep the
    # most recent readings rather than an arbitrary slice.
    query = query.order_by(TrafficObservation.timestamp.desc()).limit(limit)
    return list(reversed(list(session.execute(query).mappings())))


def predictions_for_observations(
    session: Session, observation_ids: Sequence[int]
) -> dict[int, Mapping[str, Any]]:
    """Return the newest forecast for each of ``observation_ids``, in one query.

    Trends join forecasts onto a series of observations. Doing that per point
    would be one query per reading; this resolves the whole window at once and
    returns ``observation_id -> row``. Ids with no forecast are simply absent, so
    the caller can leave that point's prediction null rather than inventing one.
    """

    if not observation_ids:
        return {}

    rank = _rank_desc(
        TrafficPrediction.observation_id,
        TrafficPrediction.prediction_timestamp,
        TrafficPrediction.id,
    ).label("row_rank")
    ranked = (
        select(
            TrafficPrediction.observation_id.label("observation_id"),
            TrafficPrediction.predicted_congestion_level,
            TrafficPrediction.predicted_congestion,
            TrafficPrediction.confidence,
            TrafficPrediction.prediction_timestamp,
            TrafficPrediction.model_version,
            rank,
        )
        .select_from(TrafficPrediction)
        .where(TrafficPrediction.observation_id.in_(tuple(observation_ids)))
        .subquery()
    )

    rows = session.execute(select(ranked).where(ranked.c.row_rank == 1)).mappings()
    return {row["observation_id"]: row for row in rows}


def observation_counts_by_location(
    session: Session,
) -> dict[str, int]:
    """Return how many observations each location has, in one query.

    Lets the dashboard say *why* a location has no prediction — "3 of 6 readings"
    — instead of only reporting the absence.
    """

    rows = session.execute(
        select(TrafficObservation.location_id, func.count(TrafficObservation.id))
        .group_by(TrafficObservation.location_id)
    ).all()
    return {location: int(count) for location, count in rows}


def latest_observation_time(session: Session) -> datetime | None:
    """Return the newest observation timestamp, or ``None`` when the table is empty."""

    return session.execute(
        select(func.max(TrafficObservation.timestamp))
    ).scalar_one()


def latest_observation_write_time(session: Session) -> datetime | None:
    """Return when the newest row was written, regardless of when it was observed.

    This is the closest thing the database records to "the collector last ran": a
    vendor may report a reading stamped several minutes before it arrived, so
    ``created_at`` and ``timestamp`` answer different questions.
    """

    return session.execute(select(func.max(TrafficObservation.created_at))).scalar_one()


def latest_prediction_time(session: Session) -> datetime | None:
    """Return the newest forecast generation time, or ``None``."""

    return session.execute(
        select(func.max(TrafficPrediction.prediction_timestamp))
    ).scalar_one()


__all__ = [
    "LATEST_OBSERVATION_COLUMNS",
    "LATEST_PREDICTION_COLUMNS",
    "add_observation",
    "add_prediction",
    "build_filter_query",
    "build_prediction_filter_query",
    "count_observations",
    "count_observations_by_location",
    "count_predictions",
    "count_predictions_for_observation",
    "get_latest_observation",
    "get_observation",
    "get_recent_history",
    "has_enough_history",
    "latest_observation_time",
    "latest_observation_write_time",
    "latest_observations_by_location",
    "latest_prediction",
    "latest_predictions_by_location",
    "latest_prediction_time",
    "list_observations",
    "list_prediction_history",
    "list_predictions",
    "observation_counts_by_location",
    "observation_series",
    "observed_segments",
    "predictions_for_observations",
    "reset_database",
]