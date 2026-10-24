"""Tests for cnn_classifier.training module (Task 4 — Training Loop & Evaluation)."""

import json
import logging

import pytest
import torch
import torch.nn as nn

from cnn_classifier.config import Config
from cnn_classifier.models.registry import get_model

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
        assert torch.isfinite(torch.tensor(metrics["loss"]))
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
        model = get_model("resnet18", cfg)
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
        model = get_model("resnet18", cfg)
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
        checkpoint = torch.load(ckpt_path, weights_only=False)
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
        """train_transfer defaults: resnet18, lr 1e-4, input size 224."""
        from scripts.train_transfer import parse_args

        args = parse_args([])
        assert args.model_name == "resnet18"
        assert args.lr == 1e-4
        assert args.input_size == (224, 224)


class _AlwaysZero(nn.Module):
    """Stub model whose forward always predicts class 0 with logit 10."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        out = torch.zeros(x.shape[0], 10)
        out[:, 0] = self.bias
        return out


class TestRunTraining:
    """Tests for the run_training orchestrator (normal + early stop)."""

    def _install_stubs(self, monkeypatch, tmp_path):
        """Point config paths at tmp_path and stub model/data loaders."""
        from cnn_classifier.training import cli as cli_module

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        cfg = Config(device="cpu", batch_size=2, num_workers=0)
        monkeypatch.setattr(cli_module, "get_model", lambda name, config: _AlwaysZero())
        val_loader = [(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))]
        monkeypatch.setattr(
            cli_module, "get_dataloaders", lambda config: (_fake_loader(), val_loader, val_loader)
        )
        return cfg

    def test_run_training_saves_checkpoint(self, tmp_path, monkeypatch, caplog):
        """run_training saves a best checkpoint and completes normally."""
        from cnn_classifier.training.cli import run_training

        cfg = self._install_stubs(monkeypatch, tmp_path)
        with caplog.at_level(logging.INFO, logger="cnn_classifier.training.cli"):
            run_training(cfg, "stub", epochs=2, lr=1e-3)
        ckpt_path = tmp_path / "checkpoints" / "best_model.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["epoch"] == 1
        assert ckpt["best_val_acc"] == 1.0
        assert "training done" in caplog.text
        assert "early stopping" not in caplog.text

    def test_run_training_early_stops(self, tmp_path, monkeypatch, caplog):
        """run_training breaks early when val acc stalls past patience."""
        from cnn_classifier.training.cli import run_training

        cfg = self._install_stubs(monkeypatch, tmp_path)
        with caplog.at_level(logging.INFO, logger="cnn_classifier.training.cli"):
            run_training(cfg, "stub", epochs=15, lr=1e-3)
        assert "early stopping at epoch 11" in caplog.text
        ckpt_path = tmp_path / "checkpoints" / "best_model.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["epoch"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
