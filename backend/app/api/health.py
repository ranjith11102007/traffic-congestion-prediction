"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.health import HealthResponse
from app.services import health_service

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Confirms the API process is running and able to serve requests. "
        "Does not report on database connectivity or model availability."
    ),
)
async def health_check() -> HealthResponse:
    return health_service.get_health_status()