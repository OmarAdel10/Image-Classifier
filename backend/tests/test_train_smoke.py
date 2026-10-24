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
            ckpt_path, model, opt, epoch=3, best_val_acc=0.42, config=cfg, class_names=CLASS_NAMES
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
            "input_size": (32, 32),
            "seed": cfg.seed,
            "device": "cpu",
            "batch_size": 2,
        }
        head_weight = model.head[-1].weight.detach()
        restored = checkpoint["model_state"]["head.4.weight"]
        assert torch.equal(head_weight, restored)


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
            path, model, opt, epoch=1, best_val_acc=0.5, config=cfg, class_names=CLASS_NAMES
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
        save_checkpoint(path, model, opt, 1, 0.5, cfg, CLASS_NAMES)
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["config"] == {
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

    def test_train_transfer_input_size_must_be_32(self):
        """--input-size values other than 32 32 fail fast with a clear error."""
        from scripts import train_transfer

        with pytest.raises(SystemExit):
            train_transfer.parse_args(["--input-size", "64", "64"])

    def test_epochs_must_be_positive(self):
        """--epochs 0 (or negative) is rejected by argparse."""
        from scripts import train_scratch

        with pytest.raises(SystemExit):
            train_scratch.parse_args(["--epochs", "0"])
        with pytest.raises(SystemExit):
            train_scratch.parse_args(["--epochs", "-5"])

    def test_lr_must_be_positive(self):
        """--lr 0 (or negative) is rejected by argparse."""
        from scripts import train_transfer

        with pytest.raises(SystemExit):
            train_transfer.parse_args(["--lr", "0"])
        with pytest.raises(SystemExit):
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
        run_training(cfg, "stub", epochs=1, lr=1e-3)
        assert cfg.seed in calls

    def test_enforce_frozen_backbone_no_backbone_noop(self):
        """Models without a backbone attribute are left unchanged."""
        from cnn_classifier.training.train import _enforce_frozen_backbone_eval

        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        model = get_model("cnn_scratch", cfg)
        model.train()
        _enforce_frozen_backbone_eval(model)
        assert model.training is True
