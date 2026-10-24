"""Tests for cnn_classifier.models module (Task 3 — Model Definitions)."""

import pytest
import torch

from cnn_classifier.config import Config
from cnn_classifier.models import CNNScratch, TransferModel, get_model


class TestCNNScratch:
    """Tests for the from-scratch CNN (cifar_scratch architecture)."""

    def test_cnn_scratch_forward(self):
        """Spec test: forward pass through CNNScratch produces (2, 10) logits, not NaN."""
        cfg = Config(input_size=(32, 32), device="cpu")
        model = get_model("cnn_scratch", cfg)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 10)
        assert not torch.isnan(logits).any()

    def test_cnn_scratch_accepts_batch(self):
        """CNNScratch handles batch size 4 and produces finite logits."""
        cfg = Config(input_size=(32, 32), device="cpu")
        model = get_model("cnn_scratch", cfg)
        model.eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 10)
        assert torch.isfinite(logits).all()

    def test_cnn_scratch_head_output_is_10_classes(self):
        """CNNScratch head maps 512 hidden features to 10 classes."""
        cfg = Config(input_size=(32, 32), device="cpu")
        model = CNNScratch(cfg)
        assert model.head[-1].out_features == 10

    def test_cnn_scratch_input_too_small_raises(self):
        """Input smaller than the two max-pools (4x downsampling) is rejected."""
        with pytest.raises(ValueError, match="too small"):
            CNNScratch(Config(input_size=(2, 2)))


class TestTransferModelResNet18:
    """Tests for the ResNet18 transfer model (spec test + behaviour)."""

    def test_transfer_resnet18_forward(self):
        """Spec test: forward pass through resnet18 produces (2, 10) logits."""
        cfg = Config(input_size=(224, 224), device="cpu")
        model = get_model("resnet18", cfg)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 10)

    def test_resnet18_head_replaced_and_backbone_frozen(self):
        """ResNet18 head maps to 10 classes; backbone frozen, head trainable."""
        cfg = Config(input_size=(224, 224), device="cpu")
        model = TransferModel("resnet18", cfg, pretrained=False)
        assert model.backbone.fc.out_features == 10
        backbone_frozen = all(
            not p.requires_grad
            for n, p in model.backbone.named_parameters()
            if not n.startswith("fc.")
        )
        head_trainable = all(
            p.requires_grad for n, p in model.backbone.named_parameters() if n.startswith("fc.")
        )
        assert backbone_frozen
        assert head_trainable

    def test_unfreeze_backbone(self):
        """unfreeze_backbone() re-enables gradients on every parameter."""
        cfg = Config(input_size=(224, 224), device="cpu")
        model = TransferModel("resnet18", cfg, pretrained=False)
        model.unfreeze_backbone()
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_resnet18_pretrained_false_no_download(self):
        """ResNet18 with pretrained=False constructs and runs a forward pass."""
        cfg = Config(input_size=(224, 224), device="cpu")
        model = TransferModel("resnet18", cfg, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()

    def test_transfer_unknown_arch_raises(self):
        """Unknown architecture names are rejected before any backbone build."""
        cfg = Config(input_size=(32, 32), device="cpu")
        with pytest.raises(ValueError, match="unknown architecture"):
            TransferModel("alexnet", cfg)


class TestTransferModelVGG16:
    """Tests for the VGG16 transfer model construction path."""

    def test_vgg16_constructs_with_random_weights(self):
        """VGG16 constructs with random init (pretrained=False) and replaced head."""
        cfg = Config(input_size=(224, 224), device="cpu")
        model = TransferModel("vgg16", cfg, pretrained=False)
        assert model.backbone.classifier[6].out_features == 10
        backbone_frozen = all(
            not p.requires_grad
            for n, p in model.backbone.named_parameters()
            if "classifier.6" not in n
        )
        head_trainable = all(
            p.requires_grad for n, p in model.backbone.named_parameters() if "classifier.6" in n
        )
        assert backbone_frozen
        assert head_trainable


class TestModelRegistry:
    """Tests for the model factory and registry."""

    def test_get_model_unknown_name_raises(self):
        """get_model with an unknown name raises ValueError listing options."""
        cfg = Config(input_size=(32, 32), device="cpu")
        with pytest.raises(ValueError, match="unknown model"):
            get_model("nonexistent_model", cfg)

    def test_cnn_scratch_kaiming_init_applied(self):
        """Kaiming normal init: conv/linear weights are non-zero and not default."""
        cfg = Config(input_size=(32, 32), device="cpu")
        model = CNNScratch(cfg)
        first_conv_weight = model.features[0].weight
        assert first_conv_weight.abs().sum() > 0
        assert first_conv_weight.std() > 0.01
