"""Tests for cnn_classifier.training module (Task 4 — Training Loop & Evaluation)."""

import json
import logging
import math
import sys

import pytest
import torch
import torch.nn as nn

from cnn_classifier.config import Config
from cnn_classifier.models.registry import get_model
from cnn_classifier.models.transfer import TransferModel

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


def _fake_loader(batch_size: int = 2, batches: int = 1, dim: int = 32):
    """Tiny in-memory loader — no CIFAR dependency, deterministic shape."""
    return [
        (torch.randn(batch_size, 3, dim, dim), torch.randint(0, 10, (batch_size,)))
        for _ in range(batches)
    ]


class TestTrainOneEpoch:
    """Tests for train_one_epoch (spec smoke test + behaviour)."""

    def test_train_one_epoch_runs(self):
        """Spec test: one epoch runs and returns loss/acc metrics."""
        from cnn_classifier.training.train import train_one_epoch

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        loader = [(torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        metrics = train_one_epoch(model, loader, opt, crit, "cpu")
        assert "loss" in metrics and "acc" in metrics
        assert metrics["loss"] >= 0

    def test_train_one_epoch_metrics_in_range(self):
        """Loss is non-negative finite; accuracy is a fraction in [0, 1]."""
        from cnn_classifier.training.train import train_one_epoch

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        loader = _fake_loader(batches=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        metrics = train_one_epoch(model, loader, opt, crit, "cpu")
        assert math.isfinite(metrics["loss"])
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_train_one_epoch_clips_and_steps(self):
        """Parameters actually change after one optimizer step."""
        from cnn_classifier.training.train import train_one_epoch

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        before = model.head[-1].weight.detach().clone()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
        crit = torch.nn.CrossEntropyLoss()
        train_one_epoch(model, _fake_loader(), opt, crit, "cpu")
        after = model.head[-1].weight.detach()
        assert not torch.equal(before, after)

    def test_train_one_epoch_keeps_frozen_backbone_in_eval(self):
        """Frozen-backbone BN layers stay in eval mode (running stats frozen)."""
        from cnn_classifier.training.train import train_one_epoch

        cfg = Config(input_size=(32, 32), device="cpu", batch_size=2, num_workers=0)
        model = TransferModel("resnet18", cfg, pretrained=False)
        running_mean_before = model.backbone.bn1.running_mean.clone()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        loader = _fake_loader(dim=32)
        train_one_epoch(model, loader, opt, crit, "cpu")
        assert not model.backbone.bn1.training
        assert torch.equal(model.backbone.bn1.running_mean, running_mean_before)

    def test_frozen_backbone_all_trainable_stays_train(self):
        """All-trainable backbone stays in train mode (any()-False arc)."""
        from cnn_classifier.training.train import _enforce_frozen_backbone_eval

        cfg = Config(input_size=(224, 224), device="cpu", batch_size=2, num_workers=0)
        model = TransferModel("resnet18", cfg, pretrained=False)
        model.unfreeze_backbone()
        model.train()
        _enforce_frozen_backbone_eval(model)
        assert model.backbone.training is True


class TestValidate:
    """Tests for validate (eval-mode metrics)."""

    def test_validate_returns_loss_and_acc(self):
        """validate runs under no_grad and returns loss/acc."""
        from cnn_classifier.training.train import validate

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        crit = torch.nn.CrossEntropyLoss()
        metrics = validate(model, _fake_loader(batches=2), crit, "cpu")
        assert "loss" in metrics and "acc" in metrics
        assert metrics["loss"] >= 0
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_validate_leaves_model_in_eval_mode(self):
        """validate puts the model in eval mode (no dropout noise)."""
        from cnn_classifier.training.train import validate

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        crit = torch.nn.CrossEntropyLoss()
        validate(model, _fake_loader(), crit, "cpu")
        assert not model.training


class TestEvaluate:
    """Tests for evaluate (full metrics + confusion matrix)."""

    def test_evaluate_metrics_structure(self):
        """evaluate returns acc, per-class metrics, and a confusion matrix."""
        from cnn_classifier.training.evaluate import evaluate

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        metrics = evaluate(model, _fake_loader(batches=2), "cpu", CLASS_NAMES)
        assert 0.0 <= metrics["acc"] <= 1.0
        assert set(metrics["per_class"]) == set(CLASS_NAMES)
        for name in CLASS_NAMES:
            entry = metrics["per_class"][name]
            assert {"precision", "recall", "f1", "support"} <= set(entry)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 10 and all(len(row) == 10 for row in cm)
        assert sum(sum(row) for row in cm) == 4  # 2 batches x batch_size 2

    def test_evaluate_perfect_constant_model(self):
        """A model always predicting class 0 scores perfectly on all-0 labels."""
        from cnn_classifier.training.evaluate import evaluate

        class ConstantModel(nn.Module):
            def forward(self, x):
                out = torch.zeros(x.shape[0], 10)
                out[:, 0] = 10.0
                return out

        loader = [(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))] * 2
        metrics = evaluate(ConstantModel(), loader, "cpu", CLASS_NAMES)
        assert metrics["acc"] == 1.0
        assert metrics["per_class"]["airplane"]["precision"] == 1.0
        assert metrics["per_class"]["airplane"]["recall"] == 1.0
        assert metrics["per_class"]["cat"]["support"] == 0
        assert metrics["confusion_matrix"][0][0] == 4
        assert sum(metrics["confusion_matrix"][1]) == 0

    def test_save_classification_report(self, tmp_path):
        """classification report saves a parseable JSON file."""
        from cnn_classifier.training.evaluate import evaluate, save_classification_report

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        metrics = evaluate(model, _fake_loader(), "cpu", CLASS_NAMES)
        report_path = tmp_path / "classification_report.json"
        save_classification_report(metrics, report_path)
        assert report_path.exists()
        loaded = json.loads(report_path.read_text())
        assert loaded["acc"] == metrics["acc"]
        assert "confusion_matrix" in loaded
        assert not list(report_path.parent.glob("*.tmp*"))


class TestCheckpoint:
    """Tests for checkpoint saving."""

    def test_save_checkpoint_roundtrip(self, tmp_path):
        """Checkpoint dict roundtrips with all spec'd keys."""
        from cnn_classifier.training.train import save_checkpoint

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "best_model.pt"
        save_checkpoint(
            ckpt_path,
            model,
            opt,
            epoch=3,
            best_val_acc=0.42,
            model_name="cnn_scratch",
            config=cfg,
            class_names=CLASS_NAMES,
        )
        checkpoint = torch.load(ckpt_path, weights_only=True)
        assert {
            "model_state",
            "optimizer_state",
            "epoch",
            "best_val_acc",
            "config",
            "class_names",
        } <= set(checkpoint)
        assert checkpoint["epoch"] == 3
        assert checkpoint["best_val_acc"] == 0.42
        assert checkpoint["class_names"] == CLASS_NAMES
        assert checkpoint["config"] == {
            "model_name": "cnn_scratch",
            "input_size": (32, 32),
            "seed": cfg.seed,
            "device": "cpu",
            "batch_size": 2,
        }
        head_weight = model.head[-1].weight.detach()
        restored = checkpoint["model_state"]["head.4.weight"]
        assert torch.equal(head_weight, restored)
        assert not list(ckpt_path.parent.glob("*.tmp*"))

    def test_atomic_write_failure_preserves_old_checkpoint(self, tmp_path, monkeypatch):
        """A failed save leaves the previous checkpoint intact with no .tmp residue."""
        from cnn_classifier.training.train import save_checkpoint

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "best_model.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "cnn_scratch", cfg, CLASS_NAMES)

        def partial_write_then_raise(obj, f):
            with open(f, "wb") as handle:
                handle.write(b"PARTIAL")
            raise RuntimeError("disk full")

        monkeypatch.setattr(torch, "save", partial_write_then_raise)
        with pytest.raises(RuntimeError, match="disk full"):
            save_checkpoint(path, model, opt, 2, 0.7, "cnn_scratch", cfg, CLASS_NAMES)
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["epoch"] == 1
        assert not list(path.parent.glob("*.tmp*"))


class TestScriptEntryPoints:
    """Import + argparse tests for the CLI scripts (full runs are too slow)."""

    def test_train_scratch_script_parses_args(self):
        """train_scratch CLI accepts --epochs, --lr, --model-name."""
        from scripts.train_scratch import parse_args

        args = parse_args(["--epochs", "1", "--lr", "0.001", "--model-name", "cnn_scratch"])
        assert args.epochs == 1
        assert args.lr == 0.001
        assert args.model_name == "cnn_scratch"

    def test_train_transfer_script_defaults(self):
        """train_transfer defaults: resnet18, lr 1e-4, input size 32x32."""
        from scripts.train_transfer import parse_args

        args = parse_args([])
        assert args.model_name == "resnet18"
        assert args.lr == 1e-4
        assert args.input_size == (32, 32)


class _AlwaysZero(nn.Module):
    """Stub model whose forward always predicts class 0 with logit 10."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(10.0))
        self.moved_to: list[str] = []

    def to(self, device, **kwargs):
        """Record device placement (asserts the run_training .to() fix)."""
        self.moved_to.append(str(device))
        return super().to(device, **kwargs)

    def forward(self, x):
        out = torch.zeros(x.shape[0], 10)
        out[:, 0] = self.bias
        return out


class TestRunTraining:
    """Tests for the run_training orchestrator (normal + early stop)."""

    @staticmethod
    def _install_stubs(monkeypatch, tmp_path):
        """Point config paths at tmp_path and stub model/data loaders."""
        from cnn_classifier.training import cli as cli_module

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        stub = _AlwaysZero()
        monkeypatch.setattr(cli_module, "get_model", lambda name, config: stub)
        val_loader = [(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))]
        monkeypatch.setattr(
            cli_module, "get_dataloaders", lambda config: (_fake_loader(), val_loader, val_loader)
        )
        return cfg, stub

    def test_run_training_saves_checkpoint(self, tmp_path, monkeypatch, caplog):
        """run_training saves a best checkpoint and completes normally."""
        from cnn_classifier.training.cli import run_training

        cfg, stub = self._install_stubs(monkeypatch, tmp_path)
        with caplog.at_level(logging.INFO, logger="cnn_classifier.training.cli"):
            run_training(cfg, "stub", epochs=2, lr=1e-3)
        ckpt_path = tmp_path / "checkpoints" / "best_model.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=True)
        assert ckpt["epoch"] == 1
        assert ckpt["best_val_acc"] == 1.0
        assert stub.moved_to == ["cpu"]
        assert (tmp_path / "checkpoints" / "labels.json").exists()
        assert (tmp_path / "checkpoints" / "classification_report.json").exists()
        assert "training done" in caplog.text
        assert "early stopping" not in caplog.text

    def test_run_training_early_stops(self, tmp_path, monkeypatch, caplog):
        """run_training breaks early when val acc stalls past patience."""
        from cnn_classifier.training.cli import run_training

        cfg, stub = self._install_stubs(monkeypatch, tmp_path)
        with caplog.at_level(logging.INFO, logger="cnn_classifier.training.cli"):
            run_training(cfg, "stub", epochs=15, lr=1e-3)
        assert "early stopping at epoch 11" in caplog.text
        ckpt_path = tmp_path / "checkpoints" / "best_model.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=True)
        assert ckpt["epoch"] == 1

    def test_cosine_lr_and_checkpoint_overwrite(self, tmp_path, monkeypatch, caplog):
        """Cosine LR anneals strictly; a better later epoch overwrites the checkpoint."""
        from cnn_classifier.training import cli as cli_module
        from cnn_classifier.training.cli import run_training

        cfg, stub = self._install_stubs(monkeypatch, tmp_path)
        captured_lrs = []
        real_save = cli_module.save_checkpoint

        def spy_save(path, model, optimizer, *rest):
            captured_lrs.append(optimizer.param_groups[0]["lr"])
            return real_save(path, model, optimizer, *rest)

        monkeypatch.setattr(cli_module, "save_checkpoint", spy_save)
        accs = iter([0.5, 0.7, 0.9])
        monkeypatch.setattr(
            cli_module, "validate", lambda *a, **k: {"loss": 0.1, "acc": next(accs)}
        )
        with caplog.at_level(logging.INFO, logger="cnn_classifier.training.cli"):
            run_training(cfg, "stub", epochs=3, lr=1e-3)
        assert captured_lrs == pytest.approx([7.5e-4, 2.5e-4, 0.0])
        assert captured_lrs[0] > captured_lrs[1] > captured_lrs[2]
        ckpt = torch.load(tmp_path / "checkpoints" / "best_model.pt", weights_only=True)
        assert ckpt["epoch"] == 3
        assert ckpt["best_val_acc"] == pytest.approx(0.9)
        assert "training done: best val acc 0.9000 at epoch 3" in caplog.text

    def test_trainable_only_optimizer(self, tmp_path, monkeypatch):
        """The optimizer only receives params with requires_grad=True."""

        class _FrozenTrainable(nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = nn.Parameter(torch.zeros(1))
                self.frozen.requires_grad = False
                self.trainable = nn.Parameter(torch.tensor(10.0))

            def forward(self, x):
                out = torch.zeros(x.shape[0], 10)
                out[:, 0] = self.trainable
                return out

        from cnn_classifier.training import cli as cli_module
        from cnn_classifier.training.cli import run_training

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        stub = _FrozenTrainable()
        monkeypatch.setattr(cli_module, "get_model", lambda name, config: stub)
        val_loader = [(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))]
        monkeypatch.setattr(
            cli_module, "get_dataloaders", lambda config: (_fake_loader(), val_loader, val_loader)
        )
        captured = []
        real_save = cli_module.save_checkpoint

        def spy_save(path, model, optimizer, *rest):
            captured.append(optimizer)
            return real_save(path, model, optimizer, *rest)

        monkeypatch.setattr(cli_module, "save_checkpoint", spy_save)
        run_training(cfg, "stub", epochs=1, lr=1e-3)
        params = captured[0].param_groups[0]["params"]
        assert stub.frozen not in params
        assert stub.trainable in params


class TestEmptyLoaderGuards:
    """Empty loaders return zeroed metrics instead of dividing by zero."""

    def test_train_one_epoch_empty_loader(self):
        from cnn_classifier.training.train import train_one_epoch

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        assert train_one_epoch(model, [], opt, crit, "cpu") == {"loss": 0.0, "acc": 0.0}

    def test_validate_empty_loader(self):
        from cnn_classifier.training.train import validate

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        crit = nn.CrossEntropyLoss()
        assert validate(model, [], crit, "cpu") == {"loss": 0.0, "acc": 0.0}

    def test_evaluate_empty_loader(self):
        from cnn_classifier.training.evaluate import evaluate

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        metrics = evaluate(model, [], "cpu", CLASS_NAMES)
        assert metrics["acc"] == 0.0
        assert metrics["confusion_matrix"] == [[0] * 10 for _ in range(10)]
        for entry in metrics["per_class"].values():
            assert entry["precision"] == 0.0
            assert entry["recall"] == 0.0
            assert entry["f1"] == 0.0
            assert entry["support"] == 0


class TestGradClipping:
    """Gradient clipping caps the global norm at 1.0."""

    def test_grad_clipping_caps_norm(self, monkeypatch):
        from cnn_classifier.training.train import train_one_epoch

        class _HugeGrad(nn.Module):
            def __init__(self):
                super().__init__()
                self.big_param = nn.Parameter(torch.tensor(1.0))

            def forward(self, x):
                out = torch.zeros(x.shape[0], 10)
                out[:, 3] = 50.0 * self.big_param
                return out

        model = _HugeGrad()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        loader = [(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))]
        captured = []
        real_clip = torch.nn.utils.clip_grad_norm_

        def spy_clip(parameters, max_norm, **kwargs):
            params = list(parameters)
            grads = [p.grad for p in params if p.grad is not None]
            if grads:
                total = sum(g.norm().item() ** 2 for g in grads) ** 0.5
                captured.append(total)
            return real_clip(params, max_norm, **kwargs)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
        train_one_epoch(model, loader, opt, crit, "cpu")
        assert captured and captured[0] > 1.0
        assert model.big_param.grad.norm().item() <= 1.0 + 1e-6


class TestEvaluateMetrics:
    """Per-class metric arithmetic on a mixed confusion pattern."""

    def test_fractional_per_class_metrics(self):
        from cnn_classifier.training.evaluate import evaluate

        class _PredictsFixed(nn.Module):
            def __init__(self, preds):
                super().__init__()
                self.preds = list(preds)
                self.i = 0

            def forward(self, x):
                out = torch.zeros(x.shape[0], 10)
                for j in range(x.shape[0]):
                    out[j, self.preds[self.i % len(self.preds)]] = 10.0
                    self.i += 1
                return out

        model = _PredictsFixed([0, 0, 1, 2])
        loader = [(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 1, 1]))]
        metrics = evaluate(model, loader, "cpu", CLASS_NAMES)
        assert metrics["acc"] == pytest.approx(0.5)
        airplane = metrics["per_class"]["airplane"]
        assert airplane["precision"] == pytest.approx(0.5)
        assert airplane["recall"] == pytest.approx(1.0)
        assert airplane["f1"] == pytest.approx(2 / 3)
        assert airplane["support"] == 1
        automobile = metrics["per_class"]["automobile"]
        assert automobile["precision"] == pytest.approx(1.0)
        assert automobile["recall"] == pytest.approx(1 / 3)
        assert automobile["f1"] == pytest.approx(0.5)
        assert automobile["support"] == 3
        bird = metrics["per_class"]["bird"]
        assert bird["precision"] == 0.0
        assert bird["recall"] == 0.0
        assert bird["f1"] == 0.0
        assert bird["support"] == 0
        for name in CLASS_NAMES[3:]:
            entry = metrics["per_class"][name]
            assert entry["precision"] == 0.0
            assert entry["recall"] == 0.0
            assert entry["f1"] == 0.0
            assert entry["support"] == 0


class TestValidateNoGrad:
    """validate() runs under no_grad: parameter grads stay None."""

    def test_validate_leaves_grads_none(self):
        from cnn_classifier.training.train import validate

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        crit = nn.CrossEntropyLoss()
        validate(model, _fake_loader(), crit, "cpu")
        assert all(p.grad is None for p in model.parameters())


class TestReportAndCheckpointExtras:
    """Nested-dir report writes and optimizer state restorability."""

    def test_save_classification_report_nested_dirs(self, tmp_path):
        from cnn_classifier.training.evaluate import save_classification_report

        metrics = {"acc": 0.5, "per_class": {}, "confusion_matrix": [[0] * 10] * 10}
        path = tmp_path / "nested" / "dir" / "report.json"
        save_classification_report(metrics, path)
        assert path.exists()
        assert json.loads(path.read_text())["acc"] == 0.5

    def test_optimizer_state_restorable(self, tmp_path):
        from cnn_classifier.training.train import save_checkpoint

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "best_model.pt"
        save_checkpoint(
            path,
            model,
            opt,
            epoch=1,
            best_val_acc=0.5,
            model_name="cnn_scratch",
            config=cfg,
            class_names=CLASS_NAMES,
        )
        ckpt = torch.load(path, weights_only=True)
        fresh = torch.optim.AdamW(get_model("cnn_scratch", cfg).parameters(), lr=1e-3)
        fresh.load_state_dict(ckpt["optimizer_state"])
        fresh.step()


class TestScriptDefaults:
    """parse_args defaults and custom input-size parsing."""

    def test_train_scratch_defaults(self):
        from scripts.train_scratch import parse_args

        args = parse_args([])
        assert args.epochs == 30
        assert args.lr == pytest.approx(1e-3)
        assert args.model_name == "cnn_scratch"


class TestCliMains:
    """main() wiring: config → run_training with parsed args."""

    def test_train_scratch_main_wires_run_training(self, monkeypatch):
        import cnn_classifier.config as config_module
        import scripts.train_scratch as scratch_module
        from scripts.train_scratch import main

        sentinel = object()
        calls = []
        monkeypatch.setattr(sys, "argv", ["train_scratch.py", "--epochs", "2"])
        monkeypatch.setenv("CNN_MODEL_PATH", "wiring-test.pt")
        monkeypatch.setattr(config_module, "get_config", lambda: sentinel)
        monkeypatch.setattr(
            scratch_module,
            "run_training",
            lambda config, name, epochs, lr: calls.append((config, name, epochs, lr)),
        )
        main()
        assert len(calls) == 1
        assert calls[0][0] is sentinel
        assert calls[0][1] == "cnn_scratch"
        assert calls[0][2] == 2
        assert calls[0][3] == pytest.approx(1e-3)

    def test_train_transfer_main_wires_run_training(self, monkeypatch):
        import cnn_classifier.config as config_module
        import scripts.train_transfer as transfer_module
        from scripts.train_transfer import main

        sentinel = object()
        kwargs_seen = []
        calls = []

        def fake_config(**kwargs):
            kwargs_seen.append(kwargs)
            return sentinel

        monkeypatch.setattr(sys, "argv", ["train_transfer.py"])
        monkeypatch.setenv("CNN_MODEL_PATH", "wiring-test.pt")
        monkeypatch.setattr(config_module, "Config", fake_config)
        monkeypatch.setattr(
            transfer_module,
            "run_training",
            lambda config, name, epochs, lr: calls.append((name, epochs, lr)),
        )
        main()
        assert kwargs_seen == [{"input_size": (32, 32)}]
        assert calls[0][0] == "resnet18"
        assert calls[0][1] == 30
        assert calls[0][2] == pytest.approx(1e-4)


class TestFixRoundTwoSecurity:
    """Fix-round-2 security + robustness tests (H1/H2/H4/row-10)."""

    def test_checkpoint_config_is_primitives_dict(self, tmp_path):
        """Checkpoint config payload is a JSON-safe dict (weights_only=True safe)."""
        import torch

        from cnn_classifier.config import Config
        from cnn_classifier.training.train import save_checkpoint

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "cnn_scratch", cfg, CLASS_NAMES)
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["config"] == {
            "model_name": "cnn_scratch",
            "input_size": (32, 32),
            "seed": cfg.seed,
            "device": "cpu",
            "batch_size": 2,
        }

    def test_report_uses_best_checkpoint(self, tmp_path, monkeypatch, caplog):
        """Post-loop evaluate runs on the reloaded BEST checkpoint, not final weights."""
        import logging

        import torch

        from cnn_classifier.training import cli as cli_module
        from cnn_classifier.training.cli import run_training

        with caplog.at_level(logging.INFO, logger="cnn_classifier.training.cli"):
            cfg, stub = TestRunTraining._install_stubs(monkeypatch, tmp_path)
            accs = iter([0.5, 0.9, 0.4])
            monkeypatch.setattr(
                cli_module,
                "validate",
                lambda model, loader, criterion, device: {"loss": 0.1, "acc": next(accs)},
            )
            real_evaluate = cli_module.evaluate
            captured = {}

            def spy_evaluate(model, loader, device, names):
                captured["model"] = model
                return real_evaluate(model, loader, device, names)

            monkeypatch.setattr(cli_module, "evaluate", spy_evaluate)
            run_training(cfg, "stub", epochs=3, lr=1e-3)

        ckpt = torch.load(tmp_path / "checkpoints" / "best_model.pt", weights_only=True)
        assert ckpt["epoch"] == 2
        assert ckpt["best_val_acc"] == 0.9
        best_bias = ckpt["model_state"]["bias"]
        assert captured["model"].state_dict()["bias"] == best_bias
        assert (tmp_path / "checkpoints" / "classification_report.json").exists()

    def test_train_transfer_input_size_must_be_32(self, capsys):
        """--input-size values other than 32 32 fail fast with a clear error."""
        from scripts import train_transfer

        with pytest.raises(SystemExit, match="2"):
            train_transfer.parse_args(["--input-size", "64", "64"])
        captured = capsys.readouterr()
        assert "supports only 32 32" in captured.err

    def test_epochs_must_be_positive(self):
        """--epochs 0 (or negative) is rejected by argparse."""
        from scripts import train_scratch

        with pytest.raises(SystemExit, match="2"):
            train_scratch.parse_args(["--epochs", "0"])
        with pytest.raises(SystemExit, match="2"):
            train_scratch.parse_args(["--epochs", "-5"])

    def test_lr_must_be_positive(self):
        """--lr 0 (or negative) is rejected by argparse."""
        from scripts import train_transfer

        with pytest.raises(SystemExit, match="2"):
            train_transfer.parse_args(["--lr", "0"])
        with pytest.raises(SystemExit, match="2"):
            train_transfer.parse_args(["--lr", "-1e-3"])


class TestFixRoundTwoBehavioral:
    """Coverage-reviewer Priority-1 behavioral tests."""

    def test_train_one_epoch_exact_metrics_and_types(self):
        """Metrics are exact floats: acc==1.0, loss==CE(all-zero logits)."""
        import math

        import torch

        from cnn_classifier.training.train import train_one_epoch

        model = _AlwaysZero()
        loader = [(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        metrics = train_one_epoch(model, loader, opt, crit, "cpu")
        assert metrics["acc"] == 1.0
        expected_loss = math.log(1 + 9 * math.exp(-10.0))
        assert metrics["loss"] == pytest.approx(expected_loss, abs=1e-6)
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["acc"], float)

    def test_validate_exact_metrics_and_types(self):
        """validate returns exact floats with the same closed-form loss."""
        import math

        import torch

        from cnn_classifier.training.train import validate

        model = _AlwaysZero()
        loader = [(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))]
        crit = torch.nn.CrossEntropyLoss()
        metrics = validate(model, loader, crit, "cpu")
        assert metrics["acc"] == 1.0
        expected_loss = math.log(1 + 9 * math.exp(-10.0))
        assert metrics["loss"] == pytest.approx(expected_loss, abs=1e-6)
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["acc"], float)

    def test_run_training_epochs_zero(self, tmp_path, monkeypatch, caplog):
        """epochs=0: no checkpoint, but labels.json + report still written."""
        import logging

        from cnn_classifier.training.cli import run_training

        with caplog.at_level(logging.INFO, logger="cnn_classifier.training.cli"):
            cfg, _stub = TestRunTraining._install_stubs(monkeypatch, tmp_path)
            run_training(cfg, "stub", epochs=0, lr=1e-3)

        assert not (tmp_path / "checkpoints" / "best_model.pt").exists()
        assert (tmp_path / "checkpoints" / "labels.json").exists()
        assert (tmp_path / "checkpoints" / "classification_report.json").exists()
        assert "best val acc 0.0000 at epoch 0" in caplog.text

    def test_training_reexports(self):
        """cnn_classifier.training re-exports all 5 public names, each callable."""

        import cnn_classifier.training as training_pkg

        for name in (
            "train_one_epoch",
            "validate",
            "evaluate",
            "save_checkpoint",
            "save_classification_report",
        ):
            obj = getattr(training_pkg, name)
            assert callable(obj), name

    def test_run_training_seeds_rng(self, tmp_path, monkeypatch):
        """run_training seeds torch.manual_seed with config.seed."""
        import torch

        from cnn_classifier.training.cli import run_training

        cfg, _stub = TestRunTraining._install_stubs(monkeypatch, tmp_path)
        calls = []
        real_seed = torch.manual_seed
        monkeypatch.setattr(torch, "manual_seed", lambda s: calls.append(s) or real_seed(s))
        import random

        random_calls = []
        real_random_seed = random.seed
        monkeypatch.setattr(random, "seed", lambda s: random_calls.append(s) or real_random_seed(s))
        run_training(cfg, "stub", epochs=1, lr=1e-3)
        assert cfg.seed in calls
        assert cfg.seed in random_calls

    def test_enforce_frozen_backbone_no_backbone_noop(self):
        """Models without a backbone attribute are left unchanged."""
        from cnn_classifier.training.train import _enforce_frozen_backbone_eval

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        model.train()
        _enforce_frozen_backbone_eval(model)
        assert model.training is True


class TestFixRoundThreeSecurity:
    """Fix-round-3: per-module frozen eval + stale-reload guard + validator gaps."""

    def test_vgg16_head_dropout_stays_train(self):
        """M1: per-module eval keeps head-feeding Dropout in train mode for vgg16."""
        from cnn_classifier.models.transfer import TransferModel
        from cnn_classifier.training.train import _enforce_frozen_backbone_eval

        cfg = Config(input_size=(32, 32), device="cpu")
        model = TransferModel("vgg16", cfg, pretrained=False)
        model.train()
        _enforce_frozen_backbone_eval(model)
        assert model.backbone.classifier[2].training is True
        assert model.backbone.classifier[5].training is True
        assert model.backbone.classifier[6].training is True
        for module in model.backbone.modules():
            params = list(module.parameters())
            if params and all(not prm.requires_grad for prm in params):
                assert module.training is False
        for module in model.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                assert module.training is False

    def test_resnet_bn_still_frozen_after_per_module_eval(self):
        """M1 regression: resnet18 BN modules remain eval under per-module loop."""
        from cnn_classifier.models.transfer import TransferModel
        from cnn_classifier.training.train import _enforce_frozen_backbone_eval

        cfg = Config(input_size=(32, 32), device="cpu")
        model = TransferModel("resnet18", cfg, pretrained=False)
        model.train()
        _enforce_frozen_backbone_eval(model)
        assert model.backbone.bn1.training is False
        assert model.backbone.fc.training is True

    def test_stale_checkpoint_not_reloaded_on_zero_improvement(self, tmp_path, monkeypatch, caplog):
        """Row 5: zero-improvement run must not reload a stale best_model.pt."""
        from cnn_classifier.training import cli as cli_module
        from cnn_classifier.training.cli import run_training
        from cnn_classifier.training.train import save_checkpoint

        cfg, stub = TestRunTraining._install_stubs(monkeypatch, tmp_path)
        # Pre-create a stale best_model.pt from a DIFFERENT model (bias=123)
        # so a reload would visibly overwrite the live weights.
        stale = _AlwaysZero()
        stale.bias = torch.nn.Parameter(torch.tensor(123.0))
        stale_opt = torch.optim.AdamW(stale.parameters(), lr=1e-3)
        save_checkpoint(cfg.model_path_resolved, stale, stale_opt, 1, 0.5, "stub", cfg, CLASS_NAMES)

        monkeypatch.setattr(
            cli_module,
            "validate",
            lambda model, loader, criterion, device: {"loss": 0.5, "acc": 0.0},
        )
        captured_models = []
        real_evaluate = cli_module.evaluate

        def spy_evaluate(model, loader, device, class_names):
            captured_models.append(model)
            return real_evaluate(model, loader, device, class_names)

        monkeypatch.setattr(cli_module, "evaluate", spy_evaluate)
        run_training(cfg, "stub", epochs=1, lr=1e-3)
        assert (tmp_path / "checkpoints" / "best_model.pt").exists()
        current_bias = captured_models[0].state_dict()["bias"]
        assert float(current_bias) != 123.0
        assert "no val-acc improvement" in caplog.text
        assert "was not updated" in caplog.text

    def test_checkpoint_model_name_roundtrip(self, tmp_path):
        """Row 9: checkpoint config dict carries model_name for Task-5 rebuild."""
        import torch

        from cnn_classifier.training.train import save_checkpoint

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = tmp_path / "best.pt"
        save_checkpoint(path, model, opt, 1, 0.5, "cnn_scratch", cfg, CLASS_NAMES)
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["config"]["model_name"] == "cnn_scratch"


class TestFixRoundThreeValidators:
    """Fix-round-3: isfinite guards + validator coverage-gap closures."""

    def test_scratch_lr_zero_rejected(self):
        """Scratch script rejects --lr 0 (its own validator error path)."""
        import scripts.train_scratch as scratch

        with pytest.raises(SystemExit, match="2"):
            scratch.parse_args(["--lr", "0"])

    def test_transfer_epochs_zero_rejected(self):
        """Transfer script rejects --epochs 0 (its own _positive_int path)."""
        import scripts.train_transfer as transfer

        with pytest.raises(SystemExit, match="2"):
            transfer.parse_args(["--epochs", "0"])

    def test_transfer_valid_args_accepted(self):
        """Transfer accepts valid --epochs/--lr (happy path both validators)."""
        import scripts.train_transfer as transfer

        args = transfer.parse_args(["--epochs", "5", "--lr", "0.0005"])
        assert args.epochs == 5
        assert args.lr == pytest.approx(5e-4)

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
    def test_nan_inf_lr_rejected(self, bad):
        """Both scripts reject non-finite --lr values (isfinite guard)."""
        import scripts.train_scratch as scratch
        import scripts.train_transfer as transfer

        with pytest.raises(SystemExit, match="2"):
            scratch.parse_args(["--lr", bad])
        with pytest.raises(SystemExit, match="2"):
            transfer.parse_args(["--lr", bad])


class TestFixRoundThreeBehavioral:
    """Fix-round-3: real DataLoader integration + artifact content asserts."""

    def test_real_dataloader_integration(self):
        """M3: train/validate/evaluate with a real torch DataLoader (partial batch)."""
        from torch.utils.data import DataLoader, TensorDataset

        from cnn_classifier.training.evaluate import evaluate
        from cnn_classifier.training.train import train_one_epoch, validate

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()

        class _Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(10.0))

            def forward(self, x):
                out = torch.zeros(x.shape[0], 10)
                out[:, 0] = self.bias
                return out

        const = _Constant()
        data_x = torch.randn(5, 3, 32, 32)
        data_y = torch.zeros(5, dtype=torch.long)
        loader = DataLoader(TensorDataset(data_x, data_y), batch_size=2, shuffle=False)
        train_m = train_one_epoch(const, loader, opt, crit, "cpu")
        assert train_m["acc"] == 1.0
        val_m = validate(const, loader, crit, "cpu")
        assert val_m["acc"] == 1.0
        metrics = evaluate(const, loader, "cpu", CLASS_NAMES)
        assert metrics["acc"] == 1.0

    def test_artifact_contents_asserted(self, tmp_path, monkeypatch, caplog):
        """M4: labels.json and classification_report.json carry correct CONTENTS."""
        import logging

        from cnn_classifier.training import cli as cli_module
        from cnn_classifier.training.cli import run_training

        caplog.at_level(logging.INFO, logger="cnn_classifier.training.cli")
        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        stub = _AlwaysZero()
        monkeypatch.setattr(cli_module, "get_model", lambda name, config: stub)
        val_loader = [(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))]
        monkeypatch.setattr(
            cli_module,
            "get_dataloaders",
            lambda config: (_fake_loader(), val_loader, val_loader),
        )
        run_training(cfg, "stub", epochs=1, lr=1e-3)
        labels_path = tmp_path / "checkpoints" / "labels.json"
        report_path = tmp_path / "checkpoints" / "classification_report.json"
        assert json.loads(labels_path.read_text()) == CLASS_NAMES
        report = json.loads(report_path.read_text())
        assert report["acc"] == pytest.approx(1.0)
        assert report["evaluated_epoch"] == 1
        checkpoint_dir = tmp_path / "checkpoints"
        assert {p.name for p in checkpoint_dir.iterdir()} == {
            "best_model.pt",
            "labels.json",
            "classification_report.json",
        }

    def test_zero_grad_called_per_batch(self):
        """Row 12: optimizer.zero_grad() fires once per batch."""
        from cnn_classifier.training.train import train_one_epoch

        model = _AlwaysZero()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        calls = []
        real_zero = opt.zero_grad

        def counting_zero_grad():
            calls.append(1)
            real_zero()

        opt.zero_grad = counting_zero_grad
        crit = torch.nn.CrossEntropyLoss()
        loader = _fake_loader(batch_size=2, batches=3)
        train_one_epoch(model, loader, opt, crit, "cpu")
        assert len(calls) == 3

    def test_validate_then_train_restores_train_mode(self):
        """Row 12: validate leaves eval; train_one_epoch restores train mode."""
        from cnn_classifier.training.train import train_one_epoch, validate

        model = _AlwaysZero()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        loader = _fake_loader()
        validate(model, loader, crit, "cpu")
        assert model.training is False
        train_one_epoch(model, loader, opt, crit, "cpu")
        assert model.training is True


class TestAtomicWriteHelper:
    """Direct tests for the shared atomic_write helper (M3)."""

    def test_success_replaces_content_no_residue(self, tmp_path):
        """A successful write lands the content and leaves no tmp files."""
        from cnn_classifier.training.atomic import atomic_write

        target = tmp_path / "nested" / "out.txt"
        atomic_write(target, lambda tmp: tmp.write_text("hello"))
        assert target.read_text() == "hello"
        assert not list(tmp_path.rglob("*.tmp*"))

    def test_failure_preserves_old_file_and_cleans_residue(self, tmp_path):
        """A failing write keeps the old file and removes the partial tmp."""
        from cnn_classifier.training.atomic import atomic_write

        target = tmp_path / "out.txt"
        target.write_text("old")

        def writer(tmp):
            tmp.write_text("partial")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            atomic_write(target, writer)
        assert target.read_text() == "old"
        assert not list(tmp_path.glob("*.tmp*"))

    def test_report_write_failure_preserves_previous_report(self, tmp_path, monkeypatch):
        """A failed report write leaves the previous report intact."""
        from pathlib import Path

        from cnn_classifier.training.evaluate import save_classification_report

        report_path = tmp_path / "report.json"
        save_classification_report(
            {"acc": 0.5, "per_class": {}, "confusion_matrix": []}, report_path
        )

        def boom(self, data, *args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        with pytest.raises(RuntimeError, match="disk full"):
            save_classification_report(
                {"acc": 0.9, "per_class": {}, "confusion_matrix": []}, report_path
            )
        assert json.loads(report_path.read_text())["acc"] == 0.5
        assert not list(report_path.parent.glob("*.tmp*"))


class TestFixRoundFiveM4Paths:
    """Per-model default checkpoint paths (user-ruled M4)."""

    def test_scratch_main_sets_per_model_checkpoint_default(self, tmp_path, monkeypatch):
        """train_scratch defaults to best_model_cnn_scratch.pt."""
        import scripts.train_scratch as scratch

        monkeypatch.delenv("CNN_MODEL_PATH", raising=False)
        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["train_scratch.py"])
        from cnn_classifier.config import clear_config_cache

        clear_config_cache()
        recorded = []
        monkeypatch.setattr(
            scratch, "run_training", lambda config, name, epochs, lr: recorded.append(config)
        )
        scratch.main()
        assert recorded[0].model_path == "checkpoints/best_model_cnn_scratch.pt"
        clear_config_cache()
        import os

        os.environ.pop("CNN_MODEL_PATH", None)

    def test_transfer_main_sets_per_model_checkpoint_default(self, tmp_path, monkeypatch):
        """train_transfer defaults to best_model_resnet18.pt."""
        import scripts.train_transfer as transfer

        monkeypatch.delenv("CNN_MODEL_PATH", raising=False)
        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["train_transfer.py"])
        recorded = []
        monkeypatch.setattr(
            transfer, "run_training", lambda config, name, epochs, lr: recorded.append(config)
        )
        transfer.main()
        assert recorded[0].model_path == "checkpoints/best_model_resnet18.pt"
        import os

        os.environ.pop("CNN_MODEL_PATH", None)

    def test_model_path_env_override_respected(self, tmp_path, monkeypatch):
        """An explicit CNN_MODEL_PATH beats the per-model default."""
        import scripts.train_scratch as scratch

        monkeypatch.setenv("CNN_MODEL_PATH", "custom.pt")
        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["train_scratch.py"])
        from cnn_classifier.config import clear_config_cache

        clear_config_cache()
        recorded = []
        monkeypatch.setattr(
            scratch, "run_training", lambda config, name, epochs, lr: recorded.append(config)
        )
        scratch.main()
        assert recorded[0].model_path == "custom.pt"
        clear_config_cache()


class TestFixRoundFiveFrozenEval:
    """Coverage-Low: mixed-freeze module semantics (per-module eval)."""

    def test_enforce_frozen_backbone_mixed_module_stays_train(self):
        """A module mixing frozen and trainable params stays in train mode."""
        from cnn_classifier.training.train import _enforce_frozen_backbone_eval

        class _Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = nn.Parameter(torch.zeros(1))
                self.frozen.requires_grad = False
                self.trainable = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return x

        class _Carrier(nn.Module):
            def __init__(self):
                super().__init__()
                frozen_layer = nn.Linear(2, 2)
                for param in frozen_layer.parameters():
                    param.requires_grad = False
                self.backbone = nn.ModuleList([_Mixed(), frozen_layer])
                self.frozen_layer = frozen_layer

            def forward(self, x):
                return x

        model = _Carrier()
        model.train()
        _enforce_frozen_backbone_eval(model)
        assert model.backbone.training is True
        assert model.frozen_layer.training is False
