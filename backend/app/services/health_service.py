"""Health service.

Phase 1 scope: report that the process is up and serving requests. The service
name comes from configuration so there is exactly one source of truth.

A database probe is deliberately *not* part of this response. Reporting
"healthy" while PostgreSQL is unreachable would be misleading, so database
readiness will be exposed by its own endpoint once real domain data exists.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.schemas.health import HealthResponse


def get_health_status() -> HealthResponse:
    """Return the liveness payload for the running API process."""

    settings = get_settings()
    return HealthResponse(status="healthy", service=settings.SERVICE_NAME)