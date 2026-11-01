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

    def test_exposes_checkpoint_provenance(self, tmp_path):
        """Predictor exposes model name and checkpoint path for /model-info."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        assert pred.model_name == "cnn_scratch"
        assert pred.checkpoint_path == path

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


class _Evil:
    """Module-level unpicklable-by-weights-only class for the security test."""


class TestFixRoundOne:
    """Fix-round-1: provenance, unsafe-pickle, pipeline, validation gaps."""

    def test_resnet18_provenance_rebuild(self, tmp_path):
        """Checkpoint provenance wins over drifted runtime config."""
        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.models.transfer import TransferModel

        ckpt_cfg = Config(device="cpu", input_size=(224, 224))
        model = TransferModel("resnet18", ckpt_cfg, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        names = [f"c{i}" for i in range(10)]
        path = tmp_path / "resnet_ckpt.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "resnet18", ckpt_cfg, names)

        drifted = Config(device="cpu", input_size=(32, 32))
        pred = Predictor.from_checkpoint(str(path), drifted)
        assert isinstance(pred.model, TransferModel)
        assert pred.input_size == (224, 224)
        results = pred.predict(torch.randn(1, 3, 224, 224))
        assert len(results) == 10
        assert set(r.class_name for r in results) == set(names)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_unsafe_pickle_rejected(self, tmp_path):
        """A valid pickle containing a disallowed object is rejected."""
        import pickle

        from cnn_classifier.inference.predictor import Predictor

        bad = tmp_path / "bad.pt"
        bad.write_bytes(
            pickle.dumps({"model_state": {}, "config": {}, "class_names": [], "x": _Evil()})
        )
        cfg = Config(device="cpu", input_size=(32, 32))
        with pytest.raises(ValueError, match="not a valid torch checkpoint"):
            Predictor.from_checkpoint(str(bad), cfg)

    def test_pipeline_end_to_end(self, tmp_path):
        """preprocess_image feeds Predictor.predict with matching contract."""
        from PIL import Image

        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.inference.preprocess import preprocess_image

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        img = Image.new("RGB", (64, 48), color="red")
        tensor = preprocess_image(img, (32, 32))
        results = pred.predict(tensor)
        assert len(results) == 10
        confidences = [r.confidence for r in results]
        assert all(confidences[i] >= confidences[i + 1] for i in range(9))
        assert abs(sum(confidences) - 1.0) < 1e-4

    def test_rejects_wrong_spatial_shape(self, tmp_path):
        """A 4-D single image with wrong H/W vs checkpoint input size is rejected."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        with pytest.raises(ValueError, match="matching checkpoint input size"):
            pred.predict(torch.randn(1, 3, 64, 64))

    def test_rejects_non_4d_tensor(self, tmp_path):
        """predict rejects 3-D and 5-D tensors."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        with pytest.raises(ValueError, match="single-image"):
            pred.predict(torch.randn(3, 32, 32))
        with pytest.raises(ValueError, match="single-image"):
            pred.predict(torch.randn(1, 1, 3, 32, 32))

    def test_unknown_checkpoint_model_name_rejected(self, tmp_path):
        """Unknown model_name in checkpoint propagates a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        model = get_model("cnn_scratch", cfg)
        bad = tmp_path / "bogus.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {"model_name": "bogus", "input_size": (32, 32)},
                "class_names": list(cfg.class_names),
            },
            bad,
        )
        with pytest.raises(ValueError, match="unknown model"):
            Predictor.from_checkpoint(str(bad), cfg)

    def test_class_count_mismatch_rejected(self, tmp_path):
        """class_names shorter than the model head raises ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        model = get_model("cnn_scratch", cfg)
        short = tmp_path / "short.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {"model_name": "cnn_scratch", "input_size": (32, 32)},
                "class_names": ["a", "b"],
            },
            short,
        )
        with pytest.raises(ValueError, match="class count"):
            Predictor.from_checkpoint(str(short), cfg)


class TestFixRoundTwo:
    """Fix-round-2: checkpoint config-shape + payload validation."""

    @pytest.mark.parametrize(
        "config_value",
        [{}, {"model_name": "cnn_scratch"}, {"input_size": (32, 32)}],
    )
    def test_checkpoint_config_missing_inner_keys(self, tmp_path, config_value):
        """Config dicts missing model_name/input_size raise a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        model = get_model("cnn_scratch", cfg)
        path = tmp_path / "bad_inner.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config_value,
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError, match="malformed checkpoint config"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_checkpoint_config_not_a_dict(self, tmp_path):
        """A non-dict config payload raises a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        model = get_model("cnn_scratch", cfg)
        path = tmp_path / "config_list.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": ["a", "b"],
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError, match="malformed checkpoint config"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_non_dict_checkpoint_payload_rejected(self, tmp_path):
        """A bare-primitive checkpoint payload raises a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", input_size=(32, 32))
        path = tmp_path / "int.pt"
        torch.save(5, path)
        with pytest.raises(ValueError, match="not a valid torch checkpoint"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_architecture_mismatch_rejected(self, tmp_path):
        """model_state tensors of the wrong architecture raise a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.models.transfer import TransferModel

        cfg = Config(device="cpu", input_size=(32, 32))
        wrong_state = TransferModel(
            "resnet18", Config(device="cpu", input_size=(224, 224)), pretrained=False
        ).state_dict()
        path = tmp_path / "mismatch.pt"
        torch.save(
            {
                "model_state": wrong_state,
                "config": {"model_name": "cnn_scratch", "input_size": (32, 32)},
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError, match="does not match recorded architecture"):
            Predictor.from_checkpoint(str(path), cfg)


class TestFixRoundThree:
    """Fix-round-3: no pretrained download at inference + payload validation."""

    def test_from_checkpoint_skips_pretrained_download(self, tmp_path, monkeypatch):
        """from_checkpoint builds the model without downloading pretrained weights."""
        import cnn_classifier.models.transfer as transfer_module
        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.models.transfer import TransferModel

        calls = []
        real_resnet = transfer_module.resnet18

        def spy_resnet(*args, **kwargs):
            calls.append(kwargs.get("weights", "MISSING"))
            return real_resnet(*args, **kwargs)

        monkeypatch.setattr(transfer_module, "resnet18", spy_resnet)
        ckpt_cfg = Config(device="cpu", input_size=(224, 224))
        model = TransferModel("resnet18", ckpt_cfg, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "resnet.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "resnet18", ckpt_cfg, ckpt_cfg.class_names)
        pred = Predictor.from_checkpoint(str(path), Config(device="cpu", input_size=(32, 32)))
        assert calls and all(weight is None for weight in calls)
        results = pred.predict(torch.randn(1, 3, 224, 224))
        assert len(results) == 10

    def test_non_dict_model_state_rejected(self, tmp_path):
        """A non-dict model_state payload is rejected with a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = tmp_path / "none_state.pt"
        torch.save(
            {
                "model_state": None,
                "config": {"model_name": "cnn_scratch", "input_size": [32, 32]},
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError, match="does not match recorded architecture"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_non_iterable_input_size_rejected(self, tmp_path):
        """A non-iterable input_size in the checkpoint config is rejected."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        path = tmp_path / "int_size.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {"model_name": "cnn_scratch", "input_size": 32},
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError, match="malformed checkpoint config"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_string_class_names_rejected(self, tmp_path):
        """A string or int class_names payload is rejected, not silently split."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        base = {
            "model_state": model.state_dict(),
            "config": {"model_name": "cnn_scratch", "input_size": [32, 32]},
        }
        for bad_names in ("0123456789", 7):
            path = tmp_path / f"bad_names_{bad_names}.pt"
            torch.save({**base, "class_names": bad_names}, path)
            with pytest.raises(ValueError, match="class_names"):
                Predictor.from_checkpoint(str(path), cfg)

    def test_predict_rejects_non_float_dtype(self, tmp_path):
        """predict rejects float64 and integer tensors with a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        pred = Predictor.from_checkpoint(str(_build_checkpoint(tmp_path, cfg)), cfg)
        with pytest.raises(ValueError, match="float32 tensor"):
            pred.predict(torch.randn(1, 3, 32, 32, dtype=torch.float64))
        with pytest.raises(ValueError, match="float32 tensor"):
            pred.predict(torch.ones(1, 3, 32, 32, dtype=torch.int32))

    def test_predict_rejects_wrong_channels(self, tmp_path):
        """predict rejects a (1, 1, H, W) tensor with a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        pred = Predictor.from_checkpoint(str(_build_checkpoint(tmp_path, cfg)), cfg)
        with pytest.raises(ValueError, match="matching checkpoint input size"):
            pred.predict(torch.randn(1, 1, 32, 32))


class TestFixRoundFour:
    """Fix-round-4: element types, NaN weights, env decoupling, contract pins."""

    @pytest.mark.parametrize("input_size", [["x", "y"], [True, True]])
    def test_checkpoint_input_size_non_int_elements_rejected(self, tmp_path, input_size):
        """Non-int input_size elements are rejected before Config construction."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        path = tmp_path / "bad_elem.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {"model_name": "cnn_scratch", "input_size": input_size},
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError, match="malformed checkpoint config"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_nan_weights_rejected(self, tmp_path):
        """Checkpoints whose weights produce non-finite outputs are rejected."""
        import hashlib

        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = _build_checkpoint(tmp_path, cfg)
        ckpt = torch.load(path, weights_only=True)
        ckpt["model_state"]["head.4.weight"].fill_(float("nan"))
        torch.save(ckpt, path)
        # Update SHA256 sidecar to match tampered checkpoint (IC-32)
        sha256_path = path.with_suffix(path.suffix + ".sha256")
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        sha256_path.write_text(hasher.hexdigest())
        with pytest.raises(ValueError, match="non-finite"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_from_checkpoint_ignores_hostile_model_path_env(self, tmp_path, monkeypatch):
        """A hostile CNN_MODEL_PATH env cannot break checkpoint loading."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        monkeypatch.setenv("CNN_MODEL_PATH", "/etc/passwd")
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        results = pred.predict(torch.randn(1, 3, 32, 32))
        assert len(results) == 10

    @pytest.mark.parametrize("names", [[1] * 10, ["a", 2] + ["c"] * 8])
    def test_class_names_list_with_non_string_rejected(self, tmp_path, names):
        """class_names lists containing non-string items are rejected."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        path = tmp_path / "bad_names.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {"model_name": "cnn_scratch", "input_size": [32, 32]},
                "class_names": names,
            },
            path,
        )
        with pytest.raises(ValueError, match="class_names must be a list of strings"):
            Predictor.from_checkpoint(str(path), cfg)

    @pytest.mark.parametrize("input_size", [(32,), (32, 32, 32)])
    def test_checkpoint_input_size_wrong_length_rejected(self, tmp_path, input_size):
        """input_size of wrong arity is rejected by the config guard."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        path = tmp_path / "bad_len.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {"model_name": "cnn_scratch", "input_size": input_size},
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError, match="malformed checkpoint config"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_inference_all_matches_public_names(self):
        """Package __all__ exports exactly the public inference surface."""
        import cnn_classifier.inference as pkg

        assert set(pkg.__all__) == {"preprocess_image", "Prediction", "Predictor"}


class TestFixRoundFive:
    """Fix-round-5: model_name type guard, non-finite input, contract pins."""

    @pytest.mark.parametrize("model_name", [["cnn_scratch"], 42, None])
    def test_checkpoint_model_name_not_string_rejected(self, tmp_path, model_name):
        """Non-string model_name payloads are rejected with a clean ValueError."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = tmp_path / "bad_name.pt"
        torch.save(
            {
                "model_state": get_model("cnn_scratch", cfg).state_dict(),
                "config": {"model_name": model_name, "input_size": [32, 32]},
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError, match="malformed checkpoint config"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_predict_rejects_non_finite_input(self, tmp_path):
        """NaN or Inf input tensors are rejected instead of predicting NaN."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        with pytest.raises(ValueError, match="non-finite"):
            pred.predict(torch.full((1, 3, 32, 32), float("nan")))
        with pytest.raises(ValueError, match="non-finite"):
            pred.predict(torch.full((1, 3, 32, 32), float("inf")))

    @pytest.mark.parametrize("input_size", [[-32, 32], [0, 32], [5000, 32]])
    def test_checkpoint_input_size_out_of_range_rejected(self, tmp_path, input_size):
        """Out-of-range but well-typed input_size is caught by the Config layer."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = tmp_path / "range.pt"
        torch.save(
            {
                "model_state": get_model("cnn_scratch", cfg).state_dict(),
                "config": {"model_name": "cnn_scratch", "input_size": input_size},
                "class_names": list(cfg.class_names),
            },
            path,
        )
        with pytest.raises(ValueError):
            Predictor.from_checkpoint(str(path), cfg)

    def test_model_state_non_tensor_values_rejected(self, tmp_path):
        """model_state dicts with non-tensor values hit the TypeError arm."""
        import hashlib

        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = _build_checkpoint(tmp_path, cfg)
        ckpt = torch.load(path, weights_only=True)
        ckpt["model_state"] = {key: "corrupt" for key in ckpt["model_state"]}
        torch.save(ckpt, path)
        # Update SHA256 sidecar to match tampered checkpoint (IC-32)
        sha256_path = path.with_suffix(path.suffix + ".sha256")
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        sha256_path.write_text(hasher.hexdigest())
        with pytest.raises(ValueError, match="does not match recorded architecture"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_load_checkpoint_directory_and_path_arg(self, tmp_path):
        """load_checkpoint rejects directories and accepts Path arguments."""
        from pathlib import Path

        from cnn_classifier.inference.checkpoints import load_checkpoint

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        with pytest.raises(ValueError, match="not a regular file"):
            load_checkpoint(tmp_path)
        path = _build_checkpoint(tmp_path, cfg)
        loaded = load_checkpoint(Path(path))
        assert {
            "model_state",
            "optimizer_state",
            "epoch",
            "best_val_acc",
            "config",
            "class_names",
        } <= set(loaded)


class TestFixRoundSix:
    """Fix-round-6: probe clamp, env decoupling, finite-weight probes."""

    def test_probe_clamped_for_huge_input_size(self, tmp_path):
        """A hostile huge input_size loads fast via the clamped probe."""
        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.models.transfer import TransferModel

        ckpt_cfg = Config(device="cpu", input_size=(2000, 1000), batch_size=1)
        model = TransferModel("resnet18", ckpt_cfg, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        names = [f"k{i}" for i in range(10)]
        path = tmp_path / "huge.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "resnet18", ckpt_cfg, names)

        pred = Predictor.from_checkpoint(str(path), Config(device="cpu", input_size=(32, 32)))
        assert isinstance(pred.model, TransferModel)
        assert pred.input_size == (2000, 1000)

    def test_from_checkpoint_ignores_hostile_batch_size_env(self, tmp_path, monkeypatch):
        """A hostile CNN_BATCH_SIZE env cannot veto legitimate checkpoints."""
        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.models.transfer import TransferModel

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        monkeypatch.setenv("CNN_BATCH_SIZE", "512")
        ckpt_cfg = Config(device="cpu", input_size=(500, 500), batch_size=1)
        model = TransferModel("resnet18", ckpt_cfg, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "big500.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "resnet18", ckpt_cfg, cfg.class_names)

        pred = Predictor.from_checkpoint(str(path), cfg)
        assert pred.input_size == (500, 500)

    def test_huge_finite_weights_rejected(self, tmp_path):
        """Finite-but-huge weights overflow the ones-probe at load time."""
        import hashlib

        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = _build_checkpoint(tmp_path, cfg)
        ckpt = torch.load(path, weights_only=True)
        ckpt["model_state"] = {
            key: torch.full_like(value, 3e38) for key, value in ckpt["model_state"].items()
        }
        torch.save(ckpt, path)
        # Update SHA256 sidecar to match tampered checkpoint (IC-32)
        sha256_path = path.with_suffix(path.suffix + ".sha256")
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        sha256_path.write_text(hasher.hexdigest())
        with pytest.raises(ValueError, match="non-finite"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_predict_rejects_non_finite_probabilities(self, tmp_path):
        """NaN/Inf probabilities raise instead of reaching the serving layer."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        state = model.state_dict()
        state["head.4.weight"].fill_(float("inf"))
        model.load_state_dict(state)
        model.eval()
        pred = Predictor(model, list(cfg.class_names), "cpu", (32, 32), model_name="cnn_scratch")
        with pytest.raises(ValueError, match="non-finite probabilities"):
            pred.predict(torch.randn(1, 3, 32, 32))


class TestFixRoundSeven:
    """Fix-round-7: probe-clamp pin + edge-contract pins from coverage review."""

    def test_probe_input_clamped_to_512_bound(self, tmp_path, monkeypatch):
        """The load probe runs at min(dim, 512) per dimension, not the raw size."""
        import cnn_classifier.inference.predictor as predictor_module
        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.models.transfer import TransferModel

        ckpt_cfg = Config(device="cpu", input_size=(2000, 1000), batch_size=1)
        model = TransferModel("resnet18", ckpt_cfg, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "huge_probe.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "resnet18", ckpt_cfg, ckpt_cfg.class_names)

        recorded = []
        real_get_model = predictor_module.get_model

        def spy_get_model(name, config, pretrained=True):
            built = real_get_model(name, config, pretrained)
            built.register_forward_pre_hook(
                lambda module, args: recorded.append(tuple(args[0].shape))
            )
            return built

        monkeypatch.setattr(predictor_module, "get_model", spy_get_model)
        pred = Predictor.from_checkpoint(str(path), Config(device="cpu", input_size=(32, 32)))
        assert recorded == [(1, 3, 512, 512)]
        assert pred.input_size == (2000, 1000)

    def test_probe_unclamped_below_512(self, tmp_path, monkeypatch):
        """Probe runs at the recorded size when it is already below the clamp."""
        import cnn_classifier.inference.predictor as predictor_module
        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.models.transfer import TransferModel

        ckpt_cfg = Config(device="cpu", input_size=(64, 64), batch_size=1)
        model = TransferModel("resnet18", ckpt_cfg, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "small_probe.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "resnet18", ckpt_cfg, ckpt_cfg.class_names)

        recorded = []
        real_get_model = predictor_module.get_model

        def spy_get_model(name, config, pretrained=True):
            built = real_get_model(name, config, pretrained)
            built.register_forward_pre_hook(
                lambda module, args: recorded.append(tuple(args[0].shape))
            )
            return built

        monkeypatch.setattr(predictor_module, "get_model", spy_get_model)
        pred = Predictor.from_checkpoint(str(path), Config(device="cpu", input_size=(32, 32)))
        assert recorded == [(1, 3, 64, 64)]
        assert pred.input_size == (64, 64)

    def test_from_checkpoint_accepts_path_object(self, tmp_path):
        """from_checkpoint accepts a pathlib.Path argument end to end."""
        from pathlib import Path

        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        pred = Predictor.from_checkpoint(Path(_build_checkpoint(tmp_path, cfg)), cfg)
        results = pred.predict(torch.randn(1, 3, 32, 32))
        assert len(results) == 10

    def test_class_count_mismatch_long_rejected(self, tmp_path):
        """More class names than model outputs is also rejected cleanly."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        path = tmp_path / "eleven.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": {"model_name": "cnn_scratch", "input_size": [32, 32]},
                "class_names": [f"n{i}" for i in range(11)],
            },
            path,
        )
        with pytest.raises(ValueError, match="class count"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_direct_construction_coerces_input_size_tuple(self):
        """Direct Predictor construction coerces list input_size to a tuple."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        pred = Predictor(model, list(cfg.class_names), "cpu", [32, 32], model_name="cnn_scratch")
        assert pred.input_size == (32, 32)

    def test_predict_rejects_empty_batch(self, tmp_path):
        """A zero-sized batch falls into the single-image rejection branch."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        pred = Predictor.from_checkpoint(str(_build_checkpoint(tmp_path, cfg)), cfg)
        with pytest.raises(ValueError, match="single-image"):
            pred.predict(torch.randn(0, 3, 32, 32))

    def test_zero_byte_checkpoint_rejected(self, tmp_path):
        """An empty file is rejected as an invalid checkpoint."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        empty = tmp_path / "empty.pt"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="not a valid torch checkpoint"):
            Predictor.from_checkpoint(str(empty), cfg)

    def test_vgg16_provenance_rebuild(self, tmp_path):
        """vgg16 checkpoints rebuild through the real serving path."""
        from cnn_classifier.inference.predictor import Predictor
        from cnn_classifier.models.transfer import TransferModel

        ckpt_cfg = Config(device="cpu", input_size=(224, 224), batch_size=1)
        model = TransferModel("vgg16", ckpt_cfg, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        names = [f"v{i}" for i in range(10)]
        path = tmp_path / "vgg.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "vgg16", ckpt_cfg, names)

        drifted = Config(device="cpu", input_size=(32, 32))
        pred = Predictor.from_checkpoint(str(path), drifted)
        assert isinstance(pred.model, TransferModel)
        assert pred.input_size == (224, 224)
        results = pred.predict(torch.randn(1, 3, 224, 224))
        assert len(results) == 10
        assert set(r.class_name for r in results) == set(names)

    def test_from_checkpoint_ignores_hostile_input_size_env(self, tmp_path, monkeypatch):
        """A hostile CNN_INPUT_SIZE env cannot override checkpoint provenance."""
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        monkeypatch.setenv("CNN_INPUT_SIZE", "[16, 16]")
        path = _build_checkpoint(tmp_path, cfg)
        pred = Predictor.from_checkpoint(str(path), cfg)
        assert pred.input_size == (32, 32)
        results = pred.predict(torch.randn(1, 3, 32, 32))
        assert len(results) == 10


class TestNormalizationProvenance:
    """IC-65: from_checkpoint verifies recorded normalization provenance."""

    def _resave_with_normalization(self, path, normalization):
        """Load a checkpoint, replace its normalization payload, re-save with updated SHA256."""
        import hashlib
        import torch

        ckpt = torch.load(path, weights_only=True)
        ckpt["config"]["normalization"] = normalization
        torch.save(ckpt, path)
        # Update SHA256 sidecar (IC-32)
        sha256_path = path.with_suffix(path.suffix + ".sha256")
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        sha256_path.write_text(hasher.hexdigest())
        return path

    def test_mismatched_normalization_raises(self, tmp_path):
        """A checkpoint trained with different stats is refused (wrong inputs)."""
        import pytest

        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = _build_checkpoint(tmp_path, cfg)
        self._resave_with_normalization(
            path,
            {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        )
        with pytest.raises(ValueError, match="does not match inference normalization"):
            Predictor.from_checkpoint(str(path), cfg)

    def test_matching_normalization_loads(self, tmp_path):
        """The recorded CIFAR-10 stats match -> loads without error."""
        from cnn_classifier.data.dataset import CIFAR10_MEAN, CIFAR10_STD
        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = _build_checkpoint(tmp_path, cfg)
        self._resave_with_normalization(
            path,
            {"mean": list(CIFAR10_MEAN), "std": list(CIFAR10_STD)},
        )
        pred = Predictor.from_checkpoint(str(path), cfg)
        assert len(pred.class_names) == 10

    def test_legacy_checkpoint_without_normalization_loads(self, tmp_path):
        """Pre-IC-30 checkpoints (no normalization key, no SHA256) stay loadable."""
        import torch
        import hashlib

        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = _build_checkpoint(tmp_path, cfg)
        ckpt = torch.load(path, weights_only=True)
        del ckpt["config"]["normalization"]
        torch.save(ckpt, path)
        # Remove SHA256 sidecar to simulate pre-IC-32 checkpoint
        sha256_path = path.with_suffix(path.suffix + ".sha256")
        if sha256_path.exists():
            sha256_path.unlink()
        pred = Predictor.from_checkpoint(str(path), cfg)
        assert len(pred.class_names) == 10

    def test_malformed_normalization_raises(self, tmp_path):
        """A non-3-element normalization payload is rejected, not silently used."""
        import pytest

        from cnn_classifier.inference.predictor import Predictor

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        path = _build_checkpoint(tmp_path, cfg)
        self._resave_with_normalization(path, {"mean": [0.5], "std": [0.2]})
        with pytest.raises(ValueError, match="malformed checkpoint normalization"):
            Predictor.from_checkpoint(str(path), cfg)
