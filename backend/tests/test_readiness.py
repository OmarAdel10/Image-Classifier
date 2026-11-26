"""Tests for the /ready endpoint."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure backend/src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cnn_classifier.api.dependencies import clear_predictor_cache
from cnn_classifier.api.main import app


class TestReadinessEndpoint:
    """Tests for the /ready endpoint."""

    def setup_method(self) -> None:
        """Clear predictor cache before each test."""
        clear_predictor_cache()

    def teardown_method(self) -> None:
        """Clear predictor cache after each test."""
        clear_predictor_cache()

    def test_ready_when_model_loaded(self) -> None:
        """Test /ready returns 200 when predictor is loaded."""
        # Create a mock predictor
        mock_predictor = MagicMock()
        mock_predictor.model_name = "cnn_scratch"
        mock_predictor.device = "cpu"
        mock_predictor.class_names = ["a", "b", "c"]
        mock_predictor.checkpoint_path = Path("/fake/checkpoints/best_model.pt")
        mock_predictor.input_size = (32, 32)

        with patch("cnn_classifier.api.main.get_predictor", return_value=mock_predictor):
            client = TestClient(app)
            response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["service"] == "cnn-classifier"
        assert data["model_loaded"] is True
        assert data["model_path"] == "/fake/checkpoints/best_model.pt"
        assert data["device"] == "cpu"
        assert data["classes"] == 3
        assert data["error"] is None

    def test_ready_when_model_not_loaded(self) -> None:
        """Test /ready returns 503 when predictor fails to load."""
        with patch(
            "cnn_classifier.api.main.get_predictor",
            side_effect=RuntimeError("Checkpoint not found"),
        ):
            client = TestClient(app)
            response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["service"] == "cnn-classifier"
        assert data["model_loaded"] is False
        assert data["error"] == "Checkpoint not found"

    def test_ready_includes_request_id_header(self) -> None:
        """Test /ready response includes X-Request-ID header."""
        mock_predictor = MagicMock()
        mock_predictor.model_name = "cnn_scratch"
        mock_predictor.device = "cpu"
        mock_predictor.class_names = ["a", "b"]
        mock_predictor.checkpoint_path = Path("/fake/checkpoints/best_model.pt")

        with patch("cnn_classifier.api.main.get_predictor", return_value=mock_predictor):
            client = TestClient(app)
            response = client.get("/ready", headers={"X-Request-ID": "test-req-123"})

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == "test-req-123"

    def test_ready_generates_request_id_when_missing(self) -> None:
        """Test /ready generates request ID when not provided."""
        mock_predictor = MagicMock()
        mock_predictor.model_name = "cnn_scratch"
        mock_predictor.device = "cpu"
        mock_predictor.class_names = ["a", "b"]
        mock_predictor.checkpoint_path = Path("/fake/checkpoints/best_model.pt")

        with patch("cnn_classifier.api.main.get_predictor", return_value=mock_predictor):
            client = TestClient(app)
            response = client.get("/ready")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8  # UUID prefix


class TestHealthEndpoint:
    """Tests for the /health endpoint (liveness probe)."""

    def test_health_returns_ok(self) -> None:
        """Test /health always returns 200 with service marker."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "cnn-classifier"

    def test_health_includes_request_id_header(self) -> None:
        """Test /health response includes X-Request-ID header."""
        client = TestClient(app)
        response = client.get("/health", headers={"X-Request-ID": "health-req-456"})

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == "health-req-456"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
