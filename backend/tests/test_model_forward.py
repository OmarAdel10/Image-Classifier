"""Tests for cnn_classifier.models module (Task 3 — Model Definitions)."""

import pytest
import torch
import torch.nn as nn

from cnn_classifier.config import Config
from cnn_classifier.models import MODEL_REGISTRY, CNNScratch, TransferModel, get_model


class TestCNNScratch:
    """Tests for the from-scratch CNN (cnn_scratch architecture)."""

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

    def test_cnn_scratch_width_too_small_raises(self):
        """Width below one pool step is rejected (height operand tested above)."""
        with pytest.raises(ValueError, match="too small"):
            CNNScratch(Config(input_size=(32, 2)))

    def test_cnn_scratch_flatten_too_large_raises(self):
        """Oversized input (flattened size > 65,536) is rejected before Linear."""
        with pytest.raises(ValueError, match="too large"):
            CNNScratch(Config(input_size=(256, 256)))

    @pytest.mark.parametrize("size", [(37, 29), (4, 4), (128, 128)])
    def test_cnn_scratch_non_default_input_sizes(self, size):
        """Dynamic flatten handles odd/mixed and boundary sizes (128,128 == cap)."""
        cfg = Config(input_size=size, device="cpu")
        model = get_model("cnn_scratch", cfg)
        model.eval()
        x = torch.randn(2, 3, *size)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()

    def test_cnn_scratch_head_has_dropout(self):
        """Head pins the spec architecture: Dropout(0.5) between FC layers."""
        cfg = Config(input_size=(32, 32), device="cpu")
        model = CNNScratch(cfg)
        dropout = model.head[3]
        assert isinstance(dropout, nn.Dropout)
        assert dropout.p == 0.5

    def test_cnn_scratch_kaiming_init_applied(self):
        """Kaiming init zeroes biases; PyTorch default init leaves them nonzero."""
        cfg = Config(input_size=(32, 32), device="cpu")
        model = CNNScratch(cfg)
        checked_conv = 0
        checked_linear = 0
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert module.weight.abs().sum() > 0
                assert module.bias.eq(0).all()
                if isinstance(module, nn.Conv2d):
                    checked_conv += 1
                else:
                    checked_linear += 1
        assert checked_conv >= 2
        assert checked_linear >= 2


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
            if not n.startswith("classifier.6.")
        )
        head_trainable = all(
            p.requires_grad
            for n, p in model.backbone.named_parameters()
            if n.startswith("classifier.6.")
        )
        assert backbone_frozen
        assert head_trainable

    def test_vgg16_forward_with_random_weights(self):
        """VGG16 random-init forward produces (2, 10) finite logits via 4096-dim head."""
        cfg = Config(input_size=(224, 224), device="cpu")
        model = TransferModel("vgg16", cfg, pretrained=False)
        model.eval()
        assert model.backbone.classifier[6].in_features == 4096
        x = torch.randn(2, 3, 64, 64)  # AdaptiveAvgPool accepts >= 32; faster than 224
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()

    def test_vgg16_unfreeze_backbone(self):
        """unfreeze_backbone() re-enables gradients on every VGG16 parameter."""
        cfg = Config(input_size=(224, 224), device="cpu")
        model = TransferModel("vgg16", cfg, pretrained=False)
        model.unfreeze_backbone()
        assert all(p.requires_grad for p in model.backbone.parameters())


class TestModelRegistry:
    """Tests for the model factory and registry."""

    def test_get_model_unknown_name_raises(self):
        """get_model with an unknown name raises ValueError listing options."""
        cfg = Config(input_size=(32, 32), device="cpu")
        with pytest.raises(ValueError, match="unknown model"):
            get_model("nonexistent_model", cfg)

    def test_registry_vgg16_lambda_constructs_transfer_model(self, monkeypatch):
        """MODEL_REGISTRY['vgg16'] wiring builds TransferModel('vgg16', config)."""
        import cnn_classifier.models.registry as registry_module

        recorded = []

        def record(arch_name, config):
            recorded.append((arch_name, config))
            return object()

        monkeypatch.setattr(registry_module, "TransferModel", record)
        cfg = Config(input_size=(224, 224), device="cpu")
        MODEL_REGISTRY["vgg16"](cfg)
        assert recorded == [("vgg16", cfg)]

    def test_models_all_matches_public_names(self):
        """Package __all__ exports exactly the public model surface."""
        import cnn_classifier.models as models_package

        assert set(models_package.__all__) == {
            "CNNScratch",
            "TransferModel",
            "get_model",
            "MODEL_REGISTRY",
        }
