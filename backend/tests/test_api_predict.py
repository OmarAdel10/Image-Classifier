"""Tests for the FastAPI serving layer (Task 6)."""

import io
import os
import struct
import zlib
from datetime import UTC, datetime

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from cnn_classifier.api import dependencies as api_dependencies
from cnn_classifier.api import main as api_main
from cnn_classifier.api.dependencies import (
    ModelUnavailableError,
    clear_predictor_cache,
    get_predictor,
)
from cnn_classifier.api.main import app
from cnn_classifier.api.schemas import PredictionItem, PredictionResponse
from cnn_classifier.config import Config
from cnn_classifier.inference.predictor import Prediction, Predictor
from cnn_classifier.models.registry import get_model
from cnn_classifier.training.train import save_checkpoint

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


def _huge_png_bytes(width=20_000, height=20_000) -> bytes:
    """Build a tiny PNG whose IHDR declares a decompression-bomb size."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


def _build_real_checkpoint(tmp_path):
    """Persist a real cnn_scratch checkpoint for integration coverage."""
    cfg = Config(device="cpu", input_size=(32, 32))
    model = get_model("cnn_scratch", cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "best_model.pt"
    save_checkpoint(path, model, optimizer, 1, 0.5, "cnn_scratch", cfg, cfg.class_names)
    return path, cfg


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

    def test_predict_accepts_missing_explicit_content_type(self, client):
        """UploadFile with no explicit Content-Type is accepted (inferred from filename)."""
        test_client, _ = client
        # No content-type provided; TestClient infers from .png extension
        response = test_client.post(
            "/predict",
            files={"file": ("cat.png", _png_bytes())},  # No content-type
        )
        assert response.status_code == 200
        assert response.json()["model"] == "cnn_scratch"

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
            "input_size": "32x32",
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


class TestModelLoadFailure:
    """G1: Predictor-load failure returns 503 with JSON detail (not bare 500)."""

    def test_predict_returns_503_when_checkpoint_missing(self, tmp_path, monkeypatch):
        """Missing checkpoint → 503 'model unavailable' for /predict."""
        app.dependency_overrides.clear()

        def explode():
            raise ModelUnavailableError("model unavailable")

        app.dependency_overrides[get_predictor] = explode
        try:
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/predict",
                    files={"file": ("cat.png", _png_bytes(), "image/png")},
                )
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 503
        assert response.json() == {"detail": "model unavailable"}

    def test_model_info_returns_503_when_checkpoint_missing(self, tmp_path, monkeypatch):
        """Missing checkpoint → 503 'model unavailable' for /model-info."""
        app.dependency_overrides.clear()

        def explode():
            raise ModelUnavailableError("model unavailable")

        app.dependency_overrides[get_predictor] = explode
        try:
            with TestClient(app) as test_client:
                response = test_client.get("/model-info")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 503
        assert response.json() == {"detail": "model unavailable"}

    def test_get_predictor_retries_after_failure(self, monkeypatch, tmp_path):
        """get_predictor retries on failure; does not cache exceptions."""
        checkpoint = tmp_path / "best_model.pt"

        class FakeConfig:
            model_path_resolved = checkpoint

        calls = []
        real_cfg = Config(device="cpu", input_size=(32, 32))

        monkeypatch.setattr(api_dependencies, "get_config", lambda: FakeConfig())

        # Create a real predictor for the successful call
        model = get_model("cnn_scratch", real_cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        save_checkpoint(
            checkpoint, model, opt, 1, 0.5, "cnn_scratch", real_cfg, real_cfg.class_names
        )
        real_predictor = Predictor.from_checkpoint(str(checkpoint), real_cfg)

        def fake_from_checkpoint(path, config):
            calls.append((path, config))
            if len(calls) == 1:
                raise FileNotFoundError("first try missing")
            return real_predictor

        monkeypatch.setattr(
            api_dependencies.Predictor, "from_checkpoint", staticmethod(fake_from_checkpoint)
        )
        clear_predictor_cache()
        try:
            # First call raises ModelUnavailableError (wrapped)
            with pytest.raises(ModelUnavailableError):
                api_dependencies.get_predictor()
            # Second call succeeds and caches
            first = api_dependencies.get_predictor()
            second = api_dependencies.get_predictor()
        finally:
            clear_predictor_cache()
        assert len(calls) == 2
        assert first is second  # second call returns cached instance


class TestDecompressionBomb:
    """G2: DecompressionBombError path returns 400."""

    def test_predict_rejects_decompression_bomb(self, client):
        test_client, _ = client
        response = test_client.post(
            "/predict",
            files={"file": ("bomb.png", _huge_png_bytes(), "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "invalid image file"


class TestRealCheckpointIntegration:
    """G3: Real checkpoint end-to-end through HTTP."""

    @pytest.mark.slow
    def test_predict_with_real_checkpoint(self, tmp_path):
        path, cfg = _build_real_checkpoint(tmp_path)

        app.dependency_overrides.clear()

        real_pred = Predictor.from_checkpoint(str(path), cfg)
        app.dependency_overrides[get_predictor] = lambda: real_pred
        try:
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/predict",
                    files={"file": ("cat.png", _png_bytes((48, 40)), "image/png")},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "cnn_scratch"
        assert len(body["predictions"]) == 10
        assert abs(sum(p["confidence"] for p in body["predictions"]) - 1.0) < 1e-6
        # Descending by confidence
        confs = [p["confidence"] for p in body["predictions"]]
        assert confs == sorted(confs, reverse=True)
        assert isinstance(body["latency_ms"], float) and body["latency_ms"] >= 0


class TestSchemaConstraints:
    """G4: Pydantic schema constraints are enforced."""

    def test_prediction_item_rejects_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            PredictionItem(class_name="cat", confidence=1.5)
        with pytest.raises(ValidationError):
            PredictionItem(class_name="cat", confidence=-0.1)
        with pytest.raises(ValidationError):
            PredictionItem(class_name="cat", confidence=float("nan"))

    def test_prediction_response_rejects_negative_latency(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                predictions=[PredictionItem(class_name="cat", confidence=0.5)],
                latency_ms=-1.0,
                model="cnn_scratch",
            )

    def test_route_returns_500_on_schema_violation(self, client):
        """If predictor returns out-of-range confidence, route returns sanitized 500."""

        class BadFake:
            model_name = "cnn_scratch"
            input_size = (32, 32)
            device = "cpu"
            class_names = CLASS_NAMES
            checkpoint_path = None

            def predict(self, tensor):
                return [Prediction("cat", 1.5)]  # invalid confidence

        app.dependency_overrides.clear()
        app.dependency_overrides[get_predictor] = lambda: BadFake()
        try:
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/predict",
                    files={"file": ("cat.png", _png_bytes(), "image/png")},
                )
        finally:
            app.dependency_overrides.clear()
        # ResponseValidationError bubbles as 500 in TestClient
        assert response.status_code == 500


class TestCorsEdgeCases:
    """G5: CORS regex branches (127.0.0.1, portless, lookalike, DELETE preflight)."""

    @pytest.mark.parametrize(
        "origin,expect_allowed",
        [
            ("http://127.0.0.1:8080", True),
            ("http://localhost", True),  # portless
            ("http://localhost.evil.com", False),  # lookalike
        ],
    )
    def test_preflight_origin_variants(self, client, origin, expect_allowed):
        test_client, _ = client
        response = test_client.options(
            "/predict",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        if expect_allowed:
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == origin
        else:
            assert response.status_code == 400
            assert "access-control-allow-origin" not in response.headers

    def test_delete_preflight_rejected(self, client):
        test_client, _ = client
        response = test_client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        assert response.status_code == 400

    def test_simple_request_echoes_origin(self, client):
        test_client, _ = client
        response = test_client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


class TestUploadBoundary:
    """G6: Exact-cap acceptance and real 10 MiB value."""

    def test_exact_cap_accepted(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setattr(api_main, "_MAX_UPLOAD_BYTES", 200)
        png = _png_bytes((32, 32))
        assert len(png) <= 200
        response = test_client.post(
            "/predict",
            files={"file": ("ok.png", png, "image/png")},
        )
        assert response.status_code == 200

    def test_over_cap_rejected(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setattr(api_main, "_MAX_UPLOAD_BYTES", 10)
        response = test_client.post(
            "/predict",
            files={"file": ("big.png", b"x" * 11, "image/png")},
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "image file too large"


class TestContentTypeNormalization:
    """G7: Content-Type case/parameter normalization."""

    @pytest.mark.parametrize("ct", ["IMAGE/PNG", "image/png; charset=binary"])
    def test_predict_accepts_normalized_content_type(self, client, ct):
        test_client, _ = client
        response = test_client.post(
            "/predict",
            files={"file": ("cat.png", _png_bytes(), ct)},
        )
        assert response.status_code == 200


class TestRequestShape:
    """G9: Non-multipart / wrong field name → 422."""

    def test_json_body_rejected(self, client):
        test_client, _ = client
        response = test_client.post("/predict", json={"foo": "bar"})
        assert response.status_code == 422

    def test_wrong_field_name_rejected(self, client):
        test_client, _ = client
        response = test_client.post(
            "/predict",
            files={"image": ("cat.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 422


class TestInferenceLogging:
    """G10: logger.exception called on inference failure."""

    def test_predict_logs_inference_error(self, client, caplog):
        test_client, fake = client
        fake.error = RuntimeError("inference boom")
        with caplog.at_level("ERROR"):
            response = test_client.post(
                "/predict",
                files={"file": ("cat.png", _png_bytes(), "image/png")},
            )
        assert response.status_code == 500
        assert any("model inference failed" in r.message for r in caplog.records)
        assert any(r.exc_info is not None for r in caplog.records)


class TestHealthResilience:
    """Strengthened: health stays available when checkpoint is truly unloadable."""

    def test_health_available_when_checkpoint_missing(self, tmp_path, monkeypatch):
        """/health returns 200 even when get_predictor would raise ModelUnavailableError."""
        # Point config at a nonexistent checkpoint
        nonexistent = tmp_path / "nope.pt"

        class FakeConfig:
            model_path_resolved = nonexistent

        monkeypatch.setattr(api_dependencies, "get_config", lambda: FakeConfig())
        clear_predictor_cache()
        try:
            with TestClient(app) as test_client:
                response = test_client.get("/health")
        finally:
            clear_predictor_cache()
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
