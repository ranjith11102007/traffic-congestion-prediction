"""Pydantic response models for the health endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HealthState = Literal["healthy", "degraded", "unhealthy"]


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /api/health``.

    Kept to exactly two fields on purpose: this is the contract agreed for the
    Phase 1 foundation, and richer readiness data will be added in a later
    phase under a separate endpoint so this contract stays stable.
    """

    status: HealthState = Field(description="Overall service state.")
    service: str = Field(description="Service identifier.")