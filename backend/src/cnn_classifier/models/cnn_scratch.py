"""From-scratch CNN for CIFAR-10 classification.

Architecture (spec): Conv(3->32, 3x3) -> ReLU -> MaxPool(2) ->
Conv(32->64, 3x3) -> ReLU -> MaxPool(2) -> Flatten ->
FC(64*8*8 -> 512) -> ReLU -> Dropout(0.5) -> FC(512 -> 10).

Convs use padding=1 to preserve spatial dims (32x32 -> 16x16 -> 8x8),
so the flatten size for the default CIFAR-10 input is 64*8*8 = 4096.
"""

import torch
import torch.nn as nn

from cnn_classifier.config import Config

# Spatial downsampling applied by the two MaxPool(2) layers.
_DOWNSAMPLE = 4


class CNNScratch(nn.Module):
    """Small from-scratch CNN for CIFAR-10 (32x32 RGB, 10 classes)."""

    def __init__(self, config: Config, num_classes: int = 10) -> None:
        """Build the CNN from the config's input size.

        Args:
            config: Application config (consumes ``input_size``).
            num_classes: Number of output classes (default 10, CIFAR-10).

        Raises:
            ValueError: If ``input_size`` is smaller than one pool step.
        """
        super().__init__()
        height, width = config.input_size
        if height < _DOWNSAMPLE or width < _DOWNSAMPLE:
            raise ValueError(
                f"input_size {config.input_size} too small for CNNScratch "
                f"(each dim must be >= {_DOWNSAMPLE})"
            )
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flatten_size = 64 * (height // _DOWNSAMPLE) * (width // _DOWNSAMPLE)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming-normal init to conv/linear weights, zero biases."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return class logits."""
        return self.head(self.features(x))
