"""Tests for the health endpoint, which is the Phase 1 acceptance contract."""

from __future__ import annotations

from fastapi.testclient import TestClient

EXPECTED_HEALTH_PAYLOAD = {
    "status": "healthy",
    "service": "traffic-prediction-api",
}


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200


def test_health_returns_exact_agreed_payload(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.json() == EXPECTED_HEALTH_PAYLOAD


def test_health_uses_json_content_type(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.headers["content-type"].startswith("application/json")


def test_health_is_advertised_in_openapi_schema(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/api/health" in schema["paths"]
    assert "get" in schema["paths"]["/api/health"]


def test_root_points_to_health_and_docs(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["health"] == "/api/health"
    assert response.json()["docs"] == "/docs"