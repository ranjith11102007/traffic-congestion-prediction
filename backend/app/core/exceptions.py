"""Domain exceptions and FastAPI exception handlers.

Every error the API returns to a client uses one consistent envelope:

    {
        "error": {
            "code": "machine_readable_code",
            "message": "Human readable summary.",
            "details": {...}          # optional
        }
    }

Handlers registered here never leak stack traces or connection strings to the
client; unexpected exceptions are logged server-side and reported as a generic
500 response.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

#: Starlette renamed the 422 constant in the version this project pins. Resolving
#: it once here keeps the name out of the exception classes and avoids emitting a
#: deprecation warning on every validation failure.
UNPROCESSABLE_ENTITY: int = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", 0
) or status.HTTP_422_UNPROCESSABLE_ENTITY


class ErrorDetail(BaseModel):
    """Machine and human readable description of a single failure."""

    code: str = Field(description="Stable, machine-readable error code.")
    message: str = Field(description="Human-readable explanation of the failure.")
    details: Any | None = Field(
        default=None,
        description="Optional structured context, e.g. the offending field names.",
    )


class ErrorResponse(BaseModel):
    """Envelope returned for every non-2xx response produced by this service."""

    error: ErrorDetail


class TrafficPredictionError(Exception):
    """Base class for all expected, domain-specific failures."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_detail(self) -> ErrorDetail:
        return ErrorDetail(code=self.error_code, message=self.message, details=self.details)


class ConfigurationError(TrafficPredictionError):
    """Raised when required runtime configuration is missing or invalid."""

    error_code = "configuration_error"


class ResourceNotFoundError(TrafficPredictionError):
    """Raised when a requested row does not exist.

    A 404 rather than an empty 200: "no such observation" and "an observation
    with no predictions" are different answers, and returning an empty list for a
    bad id would disguise a client mistake as an empty result.
    """

    status_code = status.HTTP_404_NOT_FOUND
    error_code = "not_found"


class DuplicateObservationError(TrafficPredictionError):
    """Raised when an observation already exists for a location and timestamp.

    A 409 rather than a 400: the request was well formed, it conflicts with
    stored state, and retrying it unchanged will keep failing.
    """

    status_code = status.HTTP_409_CONFLICT
    error_code = "duplicate_observation"


class ProviderNotConfiguredError(ConfigurationError):
    """Raised when a data provider is selected but not usable.

    Distinct from a plain configuration error because the fix is specific: supply
    credentials and a base URL for the real provider. It exists so an unconfigured
    real provider fails loudly instead of falling back to synthetic data, which
    would let simulated records pass as live traffic.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "provider_not_configured"


class UpstreamProviderError(TrafficPredictionError):
    """Raised when an external vendor fails, returns junk, or refuses the request.

    A 502 rather than a 500: the request this service received was fine, and the
    failure belongs to a third party. Distinguishing that from our own bugs is
    what tells an operator whether to retry or to page someone.

    The vendor's own error text is deliberately **not** forwarded to the client.
    TomTom and Open-Meteo include the requested URL in their error bodies, and the
    traffic request URL carries ``key=...``. :attr:`vendor_status` and
    :attr:`vendor_error_code` are recorded for the server log instead.
    """

    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "upstream_provider_error"

    def __init__(
        self,
        message: str,
        *,
        vendor: str,
        vendor_status: int | None = None,
        vendor_error_code: str | None = None,
        details: Any | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.vendor = vendor
        self.vendor_status = vendor_status
        self.vendor_error_code = vendor_error_code


class ProviderRateLimitedError(UpstreamProviderError):
    """Raised when a vendor reports the account is out of quota or too fast.

    Separate from a generic upstream failure because it is expected and usually
    self-limiting: the documented remedy is to collect less often, not to retry
    harder. Surfacing it as 503 also keeps the scheduler's backoff honest.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "provider_rate_limited"


class ProviderResponseError(UpstreamProviderError):
    """Raised when a vendor returns a 2xx body that cannot be trusted.

    A 200 with a missing ``currentSpeed`` is worse than an outright error, because
    a lenient parser would substitute a default and write a fabricated observation.
    Every field the provider needs is checked, and an unusable body is a failure.
    """

    error_code = "provider_invalid_response"


class UpstreamTimeoutError(UpstreamProviderError):
    """Raised when a vendor request exceeds the configured timeout.

    Mapped to 504 because the caller waited and the wait produced nothing. Kept
    distinct from a rejected request so the scheduler can retry a timeout
    without treating it as a permanent configuration fault.
    """

    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    error_code = "upstream_timeout"


class InsufficientHistoryError(TrafficPredictionError):
    """Raised when there is not enough prior data to build the model's features.

    A 422 rather than a 500: the request is understood, but it cannot be scored
    yet. Lag and rolling features read backwards, so a location needs a run of
    prior observations before any prediction is meaningful.
    """

    status_code = UNPROCESSABLE_ENTITY
    error_code = "insufficient_history"


class DatabaseConnectionError(TrafficPredictionError):
    """Raised when PostgreSQL cannot be reached."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "database_unavailable"


class ModelNotAvailableError(TrafficPredictionError):
    """Raised when a prediction is requested before a model has been trained.

    This keeps the API honest: until the Phase 3 training run produces a real
    artifact on disk, prediction endpoints must fail explicitly instead of
    returning invented numbers.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "model_not_available"


class NoPredictionAvailableError(TrafficPredictionError):
    """Raised when no stored prediction exists for a requested scope.

    Separate from :class:`ResourceNotFoundError` because "there is no forecast
    here" and "there is no observation here" need different responses from a
    dashboard: one means the collection pipeline is still accumulating history,
    the other means the caller asked for a road that does not exist. A ``null``
    prediction is never substituted for a missing one.
    """

    status_code = status.HTTP_404_NOT_FOUND
    error_code = "no_prediction_available"


class UnknownLocationError(TrafficPredictionError):
    """Raised when a location identifier is neither configured nor observed.

    A 404: the identifier is well formed but names nothing this deployment knows.
    Echoing back the known identifiers would help a caller recover, so the list is
    included in ``details`` when it is small enough to be useful.
    """

    status_code = status.HTTP_404_NOT_FOUND
    error_code = "invalid_location"


class InvalidTimeRangeError(TrafficPredictionError):
    """Raised when ``start_time`` is after ``end_time``.

    A 422 rather than a 500: the request was understood, but the two bounds it
    sent cannot both be satisfied, so returning an empty series would disguise a
    client mistake as "no data in that window".
    """

    status_code = UNPROCESSABLE_ENTITY
    error_code = "invalid_time_range"


class StaleDataError(TrafficPredictionError):
    """Raised when a caller demands fresh data and the newest row is too old.

    A 409: the request was valid and the answer is genuinely stale, which is a
    conflict between what was asked for and what exists. Only raised when a caller
    opts in with ``require_fresh=true`` — the default dashboard response reports
    staleness as a field instead, because a dashboard must still render something
    when collection has stopped.
    """

    status_code = status.HTTP_409_CONFLICT
    error_code = "stale_data"


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details)
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


async def _handle_domain_error(
    request: Request, exc: TrafficPredictionError
) -> JSONResponse:
    logger.warning("Domain error on %s: %s", request.url.path, exc.message)
    return _error_response(exc.status_code, exc.error_code, exc.message, exc.details)


async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "location": list(error.get("loc", [])),
            "message": error.get("msg", "invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return _error_response(
        UNPROCESSABLE_ENTITY,
        "validation_error",
        "The request payload or query parameters are invalid.",
        details,
    )


async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    code = {
        status.HTTP_404_NOT_FOUND: "not_found",
        status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    }.get(exc.status_code, "http_error")
    logger.info("HTTP %s on %s: %s", exc.status_code, request.url.path, message)
    return _error_response(exc.status_code, code, message)


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url.path)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred. Check the server logs for details.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the shared error handlers to the application."""

    app.add_exception_handler(TrafficPredictionError, _handle_domain_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected_error)