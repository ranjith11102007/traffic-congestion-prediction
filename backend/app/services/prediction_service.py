"""Prediction service: the bridge from stored observations to the Stage 3 model.

This is the only module that talks to both the database and ``ml``. It follows
three rules that keep the integration honest:

1. **The ML contract is read, not restated.** Required columns, known road
   segments, known weather labels and the history depth all come from
   ``ml/schema.py`` via the saved :class:`~ml.schema.FeatureSchema`. Nothing here
   invents a feature name, so retraining with a different feature set cannot
   leave this service guessing.
2. **The model is called, never simulated.** If the artifacts are missing, this
   raises :class:`ModelNotAvailableError`; if the provider supplied
   ``source="simulation"``, the response says so.
3. **Derived target quantities are never accepted from the caller.** The
   observation is rebuilt into raw inputs and handed to
   :func:`ml.predict.predict_traffic`, which recomputes ``speed_ratio`` itself.
   The API has no field for it, and this service never adds one.

History comes from the database, not from the request: a caller posting one
observation cannot manufacture the lag features a prediction depends on.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    InsufficientHistoryError,
    InvalidTimeRangeError,
    ModelNotAvailableError,
    ResourceNotFoundError,
)
from app.models.traffic import SIMULATION_SOURCE, TrafficObservation, TrafficPrediction
from app.repositories import traffic_repository as repository
from app.schemas.traffic import (
    PredictionHistoryPage,
    PredictionResponse,
    PredictionRunStatus,
    PredictionSummary,
)

logger = logging.getLogger(__name__)

#: Ceiling on one prediction-history page. A dashboard asks for a chart window,
#: not the whole forecast table, and a request without a bound would let a single
#: call read every prediction ever stored.
MAX_PREDICTION_PAGE_SIZE = 500

#: ML error types translated into API domain errors. The ML package deliberately
#: knows nothing about FastAPI, so the translation happens on this side.
try:  # pragma: no cover - import shape depends on the ML package being present
    from ml.preprocessing import DatasetError as MLDatasetError
    from ml.train import ModelArtifactNotFoundError
except ImportError:  # pragma: no cover
    MLDatasetError = RuntimeError  # type: ignore[assignment,misc]
    ModelArtifactNotFoundError = RuntimeError  # type: ignore[assignment,misc]


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """The Stage 3 artifacts, loaded once and reused.

    Loading per request would re-read a multi-megabyte pickle on every call.
    """

    estimator: Any
    schema: Any
    metadata: dict[str, Any]

    @property
    def model_version(self) -> str:
        """Return the version recorded in ``model_metadata.json``.

        Falls back to the ML package version rather than a placeholder, so a
        prediction is never stored with a made-up version string.
        """

        version = self.metadata.get("model_version")
        if version:
            return str(version)
        return str(self.metadata.get("ml_package_version", "unknown"))

    @property
    def label_mapping(self) -> dict[str, str]:
        mapping = self.metadata.get("label_mapping") or {}
        return {str(key): str(value) for key, value in mapping.items()}

    @property
    def known_segments(self) -> tuple[str, ...]:
        levels = getattr(self.schema, "categorical_levels", {}).get("road_segment_id", ())
        return tuple(str(level) for level in levels)

    @property
    def known_weather(self) -> tuple[str, ...]:
        levels = getattr(self.schema, "categorical_levels", {}).get(
            "weather_condition", ()
        )
        return tuple(str(level) for level in levels)

    @property
    def min_history(self) -> int:
        return int(getattr(self.schema, "min_history_per_segment", 1))

    @property
    def required_inputs(self) -> tuple[str, ...]:
        return tuple(getattr(self.schema, "required_input_columns", ()))


_cache: ModelBundle | None = None


def _models_dir(settings: Settings) -> str:
    return settings.models_dir


def load_model_bundle(settings: Settings | None = None, *, refresh: bool = False) -> ModelBundle:
    """Load and cache the Stage 3 artifacts.

    Args:
        refresh: bypass the cache, used by tests and after a retrain.

    Raises:
        ModelNotAvailableError: if the artifacts are absent or unreadable.
    """

    global _cache
    if _cache is not None and not refresh:
        return _cache

    active = settings or get_settings()
    from ml.train import load_metadata, load_model, load_preprocessor

    directory = _models_dir(active)
    try:
        bundle = ModelBundle(
            estimator=load_model(directory),
            schema=load_preprocessor(directory),
            metadata=load_metadata(directory),
        )
    except (ModelArtifactNotFoundError, MLDatasetError, OSError) as error:
        raise ModelNotAvailableError(
            "No trained model is available. Train one with 'python -m ml.train'.",
            details={"models_dir": directory, "reason": type(error).__name__},
        ) from error

    logger.info("Loaded model %s from %s", bundle.model_version, directory)
    _cache = bundle
    return bundle


def reset_model_cache() -> None:
    """Drop the cached artifacts. Test helper."""

    global _cache
    _cache = None


def model_status(settings: Settings | None = None) -> PredictionRunStatus:
    """Report whether prediction is available, without raising.

    Used by the status endpoint so a client can find out a model is missing
    instead of discovering it through a failed prediction.
    """

    active = settings or get_settings()
    try:
        bundle = load_model_bundle(active)
    except ModelNotAvailableError:
        return PredictionRunStatus(
            model_available=False,
            model_version=None,
            dataset_is_simulated=None,
            required_input_columns=[],
            known_road_segments=[],
            known_weather_conditions=[],
            min_history_per_segment=None,
        )

    return PredictionRunStatus(
        model_available=True,
        model_version=bundle.model_version,
        dataset_is_simulated=bool(bundle.metadata.get("dataset_is_simulated")),
        required_input_columns=list(bundle.required_inputs),
        known_road_segments=list(bundle.known_segments),
        known_weather_conditions=list(bundle.known_weather),
        min_history_per_segment=bundle.min_history,
    )


def _observation_to_row(observation: TrafficObservation) -> dict[str, Any]:
    """Render a stored observation as a raw ML input row.

    Only the columns the schema requires. ``is_incident`` becomes 0 when the
    provider does not report incidents, which is the honest encoding of
    "unknown": the feature means "an incident is known to be happening", and no
    report means none is known.
    """

    return {
        "timestamp": observation.timestamp,
        "road_segment_id": observation.road_segment_id,
        "avg_speed_kph": observation.avg_speed_kph,
        "free_flow_speed_kph": observation.free_flow_speed_kph,
        "weather_condition": observation.weather_condition,
        "temperature_c": observation.temperature_c,
        "precipitation_mm": observation.rainfall_mm,
        "is_incident": 0 if observation.is_incident is None else int(observation.is_incident),
    }


def build_model_frame(
    observations: Sequence[TrafficObservation],
) -> pd.DataFrame:
    """Assemble the raw ML input frame from stored observations, oldest first."""

    return pd.DataFrame([_observation_to_row(item) for item in observations])


def validated_confidence(value: Any) -> float | None:
    """Return a usable confidence value, or ``None``.

    Public so the read-side services apply the same rule to stored values as the
    write side applied to computed ones. A confidence of ``1.4`` is nonsense in the
    database for the same reason it is in a response, and this is the single place
    that says so.
    """

    return _validated_confidence(value)


def _validated_confidence(value: Any) -> float | None:
    """Return a usable confidence value, or ``None``.

    The estimator's highest class probability is only reported when it is a real
    number in 0..1. Anything else — a missing value, ``NaN``, a value outside the
    unit interval — becomes ``None`` rather than being clamped, because clamping a
    broken number produces a plausible-looking confidence that means nothing.
    ``None`` is the honest answer and the API documents it as "not available".
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def predict_from_observations(
    observations: Sequence[TrafficObservation],
    *,
    settings: Settings | None = None,
    history_limit: int | None = None,
) -> dict[str, Any]:
    """Score the last observation in ``observations`` using the rest as history.

    Args:
        observations: oldest first. The final row is the one being scored; earlier
            rows supply its lag and rolling features.
        settings: configuration; the default is read from the environment.
        history_limit: overrides ``ML_PREDICTION_HISTORY``.

    Raises:
        ModelNotAvailableError: if no model artifact exists.
        InsufficientHistoryError: if there are too few prior rows.
        ResourceNotFoundError: if the observations are empty.
    """

    active = settings or get_settings()
    bundle = load_model_bundle(active)

    if not observations:
        raise ResourceNotFoundError("No observations were supplied to score.")

    target = observations[-1]
    history = observations[:-1]
    required = bundle.min_history

    if len(history) < required:
        raise InsufficientHistoryError(
            f"Location {target.location_id!r} has {len(history)} prior observation(s) "
            f"but the model needs {required} to build its lag and rolling features. "
            "Collect more history for this location, or wait for more readings.",
            details={
                "location_id": target.location_id,
                "road_segment_id": target.road_segment_id,
                # Named for the Stage 6 contract: how many readings the model
                # needs versus how many exist. Never zero-filled to make a
                # prediction possible.
                "required_readings": required,
                "available_readings": len(history),
            },
        )

    depth = history_limit or active.ML_PREDICTION_HISTORY
    frame_rows = [item for item in list(history)[-depth:]] + [target]
    frame = build_model_frame(frame_rows)

    from ml.predict import predict_traffic

    try:
        outcome = predict_traffic(
            frame,
            models_dir=_models_dir(active),
            include_probabilities=True,
        )
    except (MLDatasetError, ModelArtifactNotFoundError) as error:
        raise InsufficientHistoryError(
            "The observation could not be scored from the supplied history: "
            f"{error}",
            details={"location_id": target.location_id},
        ) from error

    entry = outcome["predictions"][-1]
    probabilities = entry.get("class_probabilities")
    if isinstance(probabilities, dict):
        # An unusable probability is dropped rather than replaced: a partial
        # distribution is honest, a padded one is not.
        checked = {
            str(label): number
            for label, value in probabilities.items()
            if (number := _validated_confidence(value)) is not None
        }
        probabilities = checked or None
    else:
        probabilities = None

    return {
        "predicted_congestion_level": int(entry["predicted_congestion_level"]),
        "predicted_congestion": str(entry["predicted_congestion"]),
        "confidence": _validated_confidence(entry.get("confidence")),
        "class_probabilities": probabilities,
        "model_version": bundle.model_version,
    }


def store_prediction(
    session: Session,
    observation: TrafficObservation,
    outcome: dict[str, Any],
    *,
    note: str | None = None,
) -> TrafficPrediction:
    """Persist one forecast, linked to its observation.

    ``prediction_timestamp`` is when the forecast was *made*, not the time of the
    observation; the latter is reachable through the observation link, so the two
    are not overloaded into one column. Provenance is copied from the observation,
    which keeps a prediction over simulated data permanently labelled.
    """

    probabilities = outcome.get("class_probabilities")
    row = TrafficPrediction(
        observation_id=observation.id,
        prediction_timestamp=utc_now(),
        predicted_congestion_level=int(outcome["predicted_congestion_level"]),
        predicted_congestion=str(outcome["predicted_congestion"]),
        confidence=outcome.get("confidence"),
        class_probabilities=json.dumps(probabilities) if probabilities else None,
        model_version=str(outcome["model_version"]),
        note=note,
    )
    return repository.add_prediction(session, row)


def predict_for_observation(
    session: Session,
    observation: TrafficObservation,
    *,
    settings: Settings | None = None,
    history_limit: int | None = None,
    store: bool = True,
    note: str | None = None,
) -> PredictionResponse:
    """Score a stored observation, optionally persisting the forecast.

    History is loaded from the database, scoped to the observation's road segment
    so the lag features are built from the same reference road rather than another
    segment's. The observation is excluded from its own history: including it would
    let the lag features read the very values they are meant to precede.
    """

    active = settings or get_settings()
    bundle = load_model_bundle(active)

    if observation.road_segment_id is None:
        raise ResourceNotFoundError(
            f"Observation {observation.id} has no road_segment_id, so the model "
            "cannot score it. Supply the segment this reading belongs to.",
            details={"observation_id": observation.id},
        )

    history = repository.get_recent_history(
        session,
        location_id=observation.location_id,
        road_segment_id=observation.road_segment_id,
        limit=bundle.min_history + 1,
        exclude_observation_id=observation.id,
    )
    outcome = predict_from_observations(
        list(history) + [observation],
        settings=active,
        history_limit=history_limit,
    )

    saved: TrafficPrediction | None = None
    if store:
        saved = store_prediction(session, observation, outcome, note=note)

    return PredictionResponse(
        id=saved.id if saved else None,
        observation_id=observation.id,
        road_segment_id=observation.road_segment_id,
        observation_timestamp=observation.timestamp,
        prediction_timestamp=saved.prediction_timestamp if saved else utc_now(),
        predicted_congestion_level=outcome["predicted_congestion_level"],
        predicted_congestion=outcome["predicted_congestion"],
        confidence=outcome["confidence"],
        class_probabilities=outcome["class_probabilities"],
        model_version=outcome["model_version"],
        source=observation.source,
        note=note
        or (
            "Predicted from simulated development data."
            if observation.source == SIMULATION_SOURCE
            else None
        ),
        created_at=saved.created_at if saved else None,
    )


def predict_and_store(
    session: Session,
    observation: TrafficObservation,
    *,
    settings: Settings | None = None,
    note: str | None = None,
) -> PredictionResponse:
    """Convenience wrapper: score a stored observation and keep the forecast."""

    return predict_for_observation(
        session, observation, settings=settings, store=True, note=note
    )


def prediction_count(session: Session, observation_id: int) -> int:
    """Return how many forecasts exist for an observation."""

    return repository.count_predictions_for_observation(session, observation_id)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Stage 6: reading stored forecasts
# ---------------------------------------------------------------------------


def _summary_from_row(row: Any) -> PredictionSummary:
    """Build a dashboard-facing summary from a prediction/observation join row."""

    return PredictionSummary(
        id=row["id"],
        observation_id=row["observation_id"],
        location_id=row["location_id"],
        road_name=row["road_name"],
        road_segment_id=row["road_segment_id"],
        observation_timestamp=row["observation_timestamp"],
        prediction_timestamp=row["prediction_timestamp"],
        predicted_congestion_level=row["predicted_congestion_level"],
        predicted_congestion=row["predicted_congestion"],
        confidence=_validated_confidence(row["confidence"]),
        model_version=row["model_version"],
        source=row["source"],
        created_at=row["created_at"],
    )


def list_prediction_history(
    session: Session,
    *,
    location_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int | None = None,
) -> PredictionHistoryPage:
    """Return stored forecasts newest first, with the filters echoed back.

    ``start_time`` and ``end_time`` bound the observation each forecast describes.
    A start after the end is rejected rather than answered with an empty list, so
    a mistyped range is not mistaken for a road with no traffic.
    """

    if start_time is not None and end_time is not None and start_time > end_time:
        raise InvalidTimeRangeError(
            f"start_time ({start_time.isoformat()}) must not be after end_time "
            f"({end_time.isoformat()}).",
            details={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
        )

    effective_limit = max(1, min(int(limit or 50), MAX_PREDICTION_PAGE_SIZE))
    rows = repository.list_prediction_history(
        session,
        location_id=location_id,
        start_time=start_time,
        end_time=end_time,
        limit=effective_limit,
    )

    filters: dict[str, Any] = {"limit": effective_limit}
    if location_id is not None:
        filters["location_id"] = location_id
    if start_time is not None:
        filters["start_time"] = start_time.isoformat()
    if end_time is not None:
        filters["end_time"] = end_time.isoformat()

    return PredictionHistoryPage(
        predictions=[_summary_from_row(row) for row in rows],
        count=len(rows),
        limit=effective_limit,
        filters=filters,
    )


def get_latest_prediction(
    session: Session, *, location_id: str | None = None
) -> PredictionSummary | None:
    """Return the newest stored forecast, or ``None`` if there is none.

    Returning ``None`` is not a fallback: the caller decides whether that is a 404
    or a documented empty field. A placeholder congestion level is never invented.
    """

    row = repository.latest_prediction(session, location_id=location_id)
    return _summary_from_row(row) if row is not None else None


def predict_latest_observation(
    session: Session,
    *,
    location_id: str | None = None,
    road_segment_id: str | None = None,
    settings: Settings | None = None,
    store: bool = True,
) -> PredictionResponse:
    """Score the newest stored observation, optionally keeping the forecast.

    The observation is chosen here, from the database. There is no request field
    for a client to point at a reading of its choosing, so the target and its
    history can never be supplied from outside the stored record.

    Raises:
        ResourceNotFoundError: nothing has been stored yet.
        InsufficientHistoryError: the newest reading still has too little history.
    """

    observation = repository.get_latest_observation(
        session, location_id=location_id, road_segment_id=road_segment_id
    )
    if observation is None:
        raise ResourceNotFoundError(
            "No traffic observation has been recorded yet"
            + (f" for location_id={location_id!r}" if location_id else "")
            + ", so there is nothing to predict. Collect readings first.",
            details={"location_id": location_id, "road_segment_id": road_segment_id},
        )
    return predict_for_observation(
        session, observation, settings=settings, store=store
    )


# ---------------------------------------------------------------------------
# Stage 6: prediction as part of collection
# ---------------------------------------------------------------------------

#: Outcome statuses for one observation that was offered to the model. Chosen so
#: a client can tell "not yet scorable" apart from "we tried and it failed", and
#: both apart from "there is no model to score with".
PREDICTED = "predicted"
SKIPPED_INSUFFICIENT_HISTORY = "insufficient_history"
SKIPPED_NO_SEGMENT = "unscoreable_location"
SKIPPED_MODEL_UNAVAILABLE = "model_unavailable"
FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """What happened to one freshly collected observation.

    ``prediction`` is ``None`` whenever a forecast could not be produced. That is
    the common case for the first six readings of a new location, and it is
    reported as a reason rather than smoothed over.
    """

    observation_id: int
    location_id: str
    status: str
    reason: str | None = None
    required_readings: int | None = None
    available_readings: int | None = None
    prediction: PredictionResponse | None = None

    @property
    def stored(self) -> bool:
        return self.status == PREDICTED and self.prediction is not None


def predict_collected_observations(
    session: Session,
    observations: Sequence[TrafficObservation],
    *,
    settings: Settings | None = None,
    store: bool = True,
) -> list[PredictionOutcome]:
    """Score each freshly stored observation, and report what happened to each.

    Collection must survive prediction, so nothing here raises: an insufficient
    history, a missing segment and an outright failure are all recorded as an
    outcome. The observation is always kept — it is the history the next cycle
    needs.

    This is what makes the system converge. Reading one is stored and reported as
    unscoreable; readings two through six likewise; by the seventh the lag features
    can be built and the forecast appears without any further intervention.
    """

    active = settings or get_settings()
    try:
        bundle = load_model_bundle(active)
    except ModelNotAvailableError as error:
        # No artifact at all. Reported as its own status rather than folded into
        # "insufficient_history": the readings are fine, the model is missing, and
        # only an operator can fix that. Blaming the history would send them to
        # collect readings that will never be scored.
        logger.warning("Skipping automatic prediction: %s", error.message)
        return [
            PredictionOutcome(
                observation_id=observation.id or 0,
                location_id=observation.location_id,
                status=SKIPPED_MODEL_UNAVAILABLE,
                reason=error.message,
            )
            for observation in observations
        ]

    required = bundle.min_history
    outcomes: list[PredictionOutcome] = []

    for observation in observations:
        if observation.id is None:
            outcomes.append(
                PredictionOutcome(
                    observation_id=0,
                    location_id=observation.location_id,
                    status=SKIPPED_INSUFFICIENT_HISTORY,
                    reason="The observation was not stored, so it cannot be scored.",
                )
            )
            continue

        if observation.road_segment_id is None:
            reason = (
                f"Observation {observation.id} has no road_segment_id. Model version "
                f"{bundle.model_version} scores per segment, so this reading cannot be "
                "scored. Add the location to config/monitored_locations.json with a "
                "known segment."
            )
            logger.info("%s", reason)
            outcomes.append(
                PredictionOutcome(
                    observation_id=observation.id,
                    location_id=observation.location_id,
                    status=SKIPPED_NO_SEGMENT,
                    reason=reason,
                )
            )
            continue

        try:
            prediction = predict_for_observation(
                session, observation, settings=active, store=store
            )
        except InsufficientHistoryError as error:
            details = error.details if isinstance(error.details, dict) else {}
            logger.info(
                "Location %s has %s of %s prior readings; observation %s is stored "
                "but not predicted yet.",
                observation.location_id,
                details.get("available_readings", 0),
                required,
                observation.id,
            )
            outcomes.append(
                PredictionOutcome(
                    observation_id=observation.id,
                    location_id=observation.location_id,
                    status=SKIPPED_INSUFFICIENT_HISTORY,
                    reason=error.message,
                    required_readings=details.get("required_readings", required),
                    available_readings=details.get("available_readings", 0),
                )
            )
        except Exception as error:  # noqa: BLE001 - collection must not fail here
            logger.error(
                "Automatic prediction failed for observation %s: %s: %s",
                observation.id,
                type(error).__name__,
                error,
                exc_info=True,
            )
            outcomes.append(
                PredictionOutcome(
                    observation_id=observation.id,
                    location_id=observation.location_id,
                    status=FAILED,
                    reason=(
                        "The model could not score this observation "
                        f"({type(error).__name__}). Check the server logs."
                    ),
                )
            )
        else:
            outcomes.append(
                PredictionOutcome(
                    observation_id=observation.id,
                    location_id=observation.location_id,
                    status=PREDICTED,
                    prediction=prediction,
                )
            )

    return outcomes


__all__ = [
    "FAILED",
    "MAX_PREDICTION_PAGE_SIZE",
    "ModelBundle",
    "PREDICTED",
    "PredictionOutcome",
    "SKIPPED_INSUFFICIENT_HISTORY",
    "SKIPPED_MODEL_UNAVAILABLE",
    "SKIPPED_NO_SEGMENT",
    "build_model_frame",
    "get_latest_prediction",
    "list_prediction_history",
    "load_model_bundle",
    "model_status",
    "predict_and_store",
    "predict_collected_observations",
    "predict_for_observation",
    "predict_from_observations",
    "predict_latest_observation",
    "prediction_count",
    "reset_model_cache",
    "store_prediction",
    "utc_now",
    "validated_confidence",
]