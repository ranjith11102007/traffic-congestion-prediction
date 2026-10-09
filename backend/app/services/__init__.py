"""Domain services.

Business logic lives here, never inside route handlers. Route modules stay
thin: parse input, call a service, return a schema.

Only the cheap health service is re-exported. The heavier modules are imported by
their full path where they are used, on purpose:
``app.services.prediction_service`` pulls in pandas and the ``ml`` package, and
``app.services.traffic_collector`` pulls in the provider implementations. Eagerly
importing them here would make ``import app.services.providers`` load the whole
scientific stack, and would fail outright on a deployment that has the API but not
the model artifacts.
"""

from app.services.health_service import get_health_status

__all__ = ["get_health_status"]