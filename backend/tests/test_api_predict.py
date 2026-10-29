"""Tests for the FastAPI serving layer (Task 6)."""

import io
import os
from datetime import UTC, datetime

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from cnn_classifier.api import dependencies as api_dependencies
from cnn_classifier.api import main as api_main
from cnn_classifier.api.dependencies import clear_predictor_cache, get_predictor
from cnn_classifier.api.main import app
from cnn_classifier.inference.predictor import Prediction

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class FakePredictor:
    """Predictor stand-in so route tests never load a real checkpoint."""

    def __init__(self) -> None:
        self.model_name = "cnn_scratch"
        self.input_size = (32, 32)
        self.device = "cpu"
        self.class_names = list(CLASS_NAMES)
        self.checkpoint_path = None
        self.received = None
        self.error = None

    def predict(self, tensor):
        self.received = tensor
        if self.error is not None:
            raise self.error
        return [Prediction("cat", 0.9), Prediction("dog", 0.1)]


def _png_bytes(size=(32, 32), color=(128, 128, 128)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    """TestClient with the predictor dependency overridden by a fake."""
    fake = FakePredictor()
    app.dependency_overrides[get_predictor] = lambda: fake
    try:
        with TestClient(app) as test_client:
            yield test_client, fake
    finally:
        app.dependency_overrides.clear()


class TestHealth:
    def test_health_endpoint_returns_ok(self, client):
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_does_not_load_the_model(self):
        """Health stays available even when no checkpoint can be loaded."""

        def explode():
            raise AssertionError("health must not resolve the predictor")

        app.dependency_overrides[get_predictor] = explode
        try:
            with TestClient(app) as test_client:
                response = test_client.get("/health")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 200


class TestPredict:
    def test_predict_returns_ranked_predictions(self, client):
        test_client, fake = client
        response = test_client.post(
            "/predict",
            files={"file": ("cat.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "cnn_scratch"
        assert body["latency_ms"] >= 0
        assert body["predictions"] == [
            {"class_name": "cat", "confidence": 0.9},
            {"class_name": "dog", "confidence": 0.1},
        ]
        assert tuple(fake.received.shape) == (1, 3, 32, 32)
        assert fake.received.dtype == torch.float32

    def test_predict_uses_checkpoint_input_size(self, client):
        """Serving follows the predictor's checkpoint provenance, not Config."""
        test_client, fake = client
        fake.input_size = (64, 64)
        response = test_client.post(
            "/predict",
            files={"file": ("cat.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert tuple(fake.received.shape) == (1, 3, 64, 64)

    @pytest.mark.parametrize("content_type", ["text/plain", "application/pdf"])
    def test_predict_rejects_non_image_content_type(self, client, content_type):
        test_client, _ = client
        response = test_client.post(
            "/predict",
            files={"file": ("notes.txt", b"hello", content_type)},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "file must be an image"

    def test_predict_rejects_missing_content_type(self):
        """A direct UploadFile with no Content-Type header is rejected."""
        from fastapi import HTTPException, UploadFile

        upload = UploadFile(file=io.BytesIO(_png_bytes()), filename="cat.png")
        assert upload.content_type is None
        with pytest.raises(HTTPException) as exc_info:
            api_main.predict(file=upload, predictor=FakePredictor())
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "file must be an image"

    def test_predict_rejects_empty_file(self, client):
        test_client, _ = client
        response = test_client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "empty image file"

    def test_predict_rejects_invalid_image_bytes(self, client):
        test_client, _ = client
        response = test_client.post(
            "/predict",
            files={"file": ("broken.png", b"not an image", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "invalid image file"

    def test_predict_rejects_image_over_pixel_cap(self, client):
        test_client, _ = client
        response = test_client.post(
            "/predict",
            files={"file": ("huge.png", _png_bytes((2001, 1000)), "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "invalid image file"

    def test_predict_rejects_upload_over_byte_cap(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setattr(api_main, "_MAX_UPLOAD_BYTES", 10)
        response = test_client.post(
            "/predict",
            files={"file": ("big.png", b"x" * 11, "image/png")},
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "image file too large"

    def test_predict_sanitizes_model_errors(self, client):
        test_client, fake = client
        fake.error = RuntimeError("checkpoint /secret/path.pt exploded")
        response = test_client.post(
            "/predict",
            files={"file": ("cat.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "model inference failed"
        assert "secret" not in response.text
        assert "/secret/path.pt" not in response.text

    def test_predict_requires_file_field(self, client):
        test_client, _ = client
        response = test_client.post("/predict")
        assert response.status_code == 422


class TestModelInfo:
    def test_model_info_returns_predictor_metadata(self, client):
        test_client, _ = client
        response = test_client.get("/model-info")
        assert response.status_code == 200
        assert response.json() == {
            "model": "cnn_scratch",
            "input_size": [32, 32],
            "classes": CLASS_NAMES,
            "device": "cpu",
            "trained_at": None,
        }

    def test_model_info_reports_checkpoint_mtime(self, client, tmp_path):
        test_client, fake = client
        checkpoint = tmp_path / "best_model.pt"
        checkpoint.write_bytes(b"checkpoint")
        os.utime(checkpoint, (1_700_000_000, 1_700_000_000))
        fake.checkpoint_path = checkpoint

        body = test_client.get("/model-info").json()

        expected = datetime.fromtimestamp(1_700_000_000, tz=UTC)
        assert datetime.fromisoformat(body["trained_at"]) == expected

    def test_model_info_tolerates_deleted_checkpoint(self, client, tmp_path):
        test_client, fake = client
        fake.checkpoint_path = tmp_path / "deleted.pt"

        body = test_client.get("/model-info").json()

        assert body["trained_at"] is None


class TestCors:
    def test_preflight_allows_localhost(self, client):
        test_client, _ = client
        response = test_client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_preflight_rejects_foreign_origin(self, client):
        test_client, _ = client
        response = test_client.options(
            "/predict",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 400


class TestDependencies:
    def test_get_config_delegates_to_singleton(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(api_dependencies, "_get_config", lambda: sentinel)

        assert api_dependencies.get_config() is sentinel

    def test_get_predictor_loads_configured_checkpoint_once(self, monkeypatch, tmp_path):
        calls = []
        fake = FakePredictor()
        checkpoint = tmp_path / "best_model.pt"

        class FakeConfig:
            model_path_resolved = checkpoint

        monkeypatch.setattr(api_dependencies, "get_config", lambda: FakeConfig())

        def fake_from_checkpoint(path, config):
            calls.append((path, config))
            return fake

        monkeypatch.setattr(
            api_dependencies.Predictor, "from_checkpoint", staticmethod(fake_from_checkpoint)
        )
        clear_predictor_cache()
        try:
            first = api_dependencies.get_predictor()
            second = api_dependencies.get_predictor()
        finally:
            clear_predictor_cache()

        assert first is second is fake
        assert len(calls) == 1
        assert calls[0][0] == checkpoint
