from fastapi.testclient import TestClient

from app.main import app


def test_health_check_returns_service_status() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        versioned_response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "myra-api"}
    assert versioned_response.json() == response.json()


def test_health_check_allows_local_web_origin() -> None:
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
