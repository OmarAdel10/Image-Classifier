"""Tests for cnn_classifier.inference.predictor (Task 5)."""

import pytest
import torch

from cnn_classifier.config import Config
from cnn_classifier.models.registry import get_model
from cnn_classifier.training.train import save_checkpoint


def _build_checkpoint(tmp_path, cfg, model_name="cnn_scratch"):
    """Persist a real 6-key checkpoint and return its path."""
    model = get_model(model_name, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "best_model.pt"
    save_checkpoint(path, model, opt, 1, 0.5, model_name, cfg, cfg.class_names)
    return path


class TestPredictor:
    """Predictor.from_checkpoint + predict contract."""

    def test_predictor_loads_and_predicts(self, tmp_path):
        """Spec test: from_checkpoint + predict -> 10 softmax predictions."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        x = torch.randn(1, 3, 32, 32)
        results = pred.predict(x)
        assert len(results) == 10
        assert all(0 <= r.confidence <= 1 for r in results)
        assert abs(sum(r.confidence for r in results) - 1.0) < 1e-4

    def test_predictions_sorted_descending(self, tmp_path):
        """predict returns predictions sorted by confidence, highest first."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        results = pred.predict(torch.randn(1, 3, 32, 32))
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_prediction_names_cover_all_classes(self, tmp_path):
        """Every class name appears exactly once in the prediction list."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        results = pred.predict(torch.randn(1, 3, 32, 32))
        names = [r.class_name for r in results]
        assert set(names) == set(cfg.class_names)
        assert len(names) == 10

    def test_predict_is_deterministic(self, tmp_path):
        """Same input twice -> identical confidences (eval mode, no dropout)."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        torch.manual_seed(0)
        x = torch.randn(1, 3, 32, 32)
        first = pred.predict(x)
        second = pred.predict(x)
        assert [r.confidence for r in first] == [r.confidence for r in second]

    def test_model_in_eval_mode_after_load(self, tmp_path):
        """from_checkpoint leaves the rebuilt model in eval mode."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        assert pred.model.training is False

    def test_rejects_batch_size_above_one(self, tmp_path):
        """predict only accepts a single-image tensor (1, C, H, W)."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        with pytest.raises(ValueError, match="single-image"):
            pred.predict(torch.randn(2, 3, 32, 32))

    def test_missing_checkpoint_raises(self, tmp_path):
        """A nonexistent checkpoint path raises FileNotFoundError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        with pytest.raises(FileNotFoundError):
            Predictor.from_checkpoint(str(tmp_path / "nope.pt"), cfg)

    def test_corrupt_checkpoint_raises_clean_error(self, tmp_path):
        """Garbage files raise a clean ValueError, not a raw pickle traceback."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        bad = tmp_path / "corrupt.pt"
        bad.write_bytes(b"this is not a torch checkpoint")
        with pytest.raises(ValueError, match="not a valid torch checkpoint"):
            Predictor.from_checkpoint(str(bad), cfg)

    def test_checkpoint_missing_required_keys_raises(self, tmp_path):
        """Valid torch files missing required keys are rejected."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        bad = tmp_path / "incomplete.pt"
        torch.save({"model_state": {}, "epoch": 1}, bad)
        with pytest.raises(ValueError, match="missing required keys"):
            Predictor.from_checkpoint(str(bad), cfg)
