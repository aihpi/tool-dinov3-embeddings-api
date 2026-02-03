from fastapi.testclient import TestClient

from app.main import app


def test_health(monkeypatch):
    monkeypatch.setattr("app.api.routes.load_model_and_processor", lambda: None)
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_id" in payload
