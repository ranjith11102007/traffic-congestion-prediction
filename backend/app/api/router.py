"""Aggregate API router.

All versioned endpoints are mounted here under a single ``/api`` prefix, so the
URL space stays centralised and a resource can be reorganised without touching
``main.py``.

Router order puts health first so it stays the cheapest route, then the traffic
resource, prediction, the collector and finally the dashboard. The traffic router
declares ``/traffic/latest`` after the collection endpoints; it does not collide
with ``/traffic/observations/{observation_id}`` because the first segment differs.

The dashboard router is mounted last and declares its own ``/dashboard`` prefix.
It shares no path with the others, and it reads the same rows they write, so there
is exactly one source of truth for a traffic reading whichever endpoint reports it.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api import collector, dashboard, health, prediction, traffic

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(traffic.router)
api_router.include_router(prediction.router)
api_router.include_router(collector.router)
api_router.include_router(dashboard.router)

__all__ = ["api_router"]