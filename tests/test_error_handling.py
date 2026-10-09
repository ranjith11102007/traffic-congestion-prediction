"""Tests for the shared API error envelope."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_unknown_route_returns_structured_404(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert "message" in body["error"]


def test_method_not_allowed_returns_structured_405(client: TestClient) -> None:
    response = client.post("/api/health")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_error_response_never_leaks_internals(client: TestClient) -> None:
    """Client payloads must not contain tracebacks or file paths."""

    response = client.get("/api/does-not-exist")
    body = response.text

    assert "Traceback" not in body
    assert ".py" not in body