"""Transfer-learning models (ResNet18, VGG16) with replaced heads.

Backbone weights come from torchvision (pretrained ImageNet by default);
the classification head is replaced to match the configured class count
and the backbone is frozen by default (feature-extractor mode). Call
``unfreeze_backbone()`` for full fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, VGG16_Weights, resnet18, vgg16

from cnn_classifier.config import Config

# Head module name for each supported architecture, with the default
# ImageNet feature dim feeding that head.
_HEAD_FEATURES = {"resnet18": 512, "vgg16": 4096}


class TransferModel(nn.Module):
    """Pretrained torchvision backbone with a task-specific head."""

    def __init__(
        self,
        arch_name: str,
        config: Config,
        pretrained: bool = True,
        num_classes: int = 10,
    ) -> None:
        """Build the transfer model.

        Args:
            arch_name: One of ``'resnet18'`` or ``'vgg16'``.
            config: Application config (kept for interface parity).
            pretrained: Load ImageNet weights (downloads on first use).
            num_classes: Number of output classes (default 10, CIFAR-10).

        Raises:
            ValueError: If ``arch_name`` is not supported.
        """
        super().__init__()
        if arch_name not in _HEAD_FEATURES:
            raise ValueError(
                f"unknown architecture: {arch_name} " f"(supported: {sorted(_HEAD_FEATURES)})"
            )
        if arch_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            self.backbone.fc = nn.Linear(_HEAD_FEATURES[arch_name], num_classes)
            head_name = "fc"
        else:
            weights = VGG16_Weights.DEFAULT if pretrained else None
            self.backbone = vgg16(weights=weights)
            self.backbone.classifier[6] = nn.Linear(_HEAD_FEATURES[arch_name], num_classes)
            head_name = "classifier.6"

        self._freeze_backbone(head_name)

    def _freeze_backbone(self, head_name: str) -> None:
        """Freeze all backbone params except the replaced head."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith(f"{head_name}.")

    def unfreeze_backbone(self) -> None:
        """Re-enable gradients on every backbone parameter (fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return class logits."""
        return self.backbone(x)
