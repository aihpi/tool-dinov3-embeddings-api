from fastapi.testclient import TestClient

from app.main import app


def test_embeddings_base64(monkeypatch):
    monkeypatch.setattr("app.api.routes.compute_embeddings", lambda images: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr("app.api.routes.load_image_from_base64", lambda payload: object())

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={"images": [{"image_base64": "aGVsbG8=", "id": "img-1"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dim"] == 3
    assert payload["ids"] == ["img-1"]
