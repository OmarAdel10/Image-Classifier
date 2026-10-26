"""Image preprocessing for inference (Task 5).

Converts any PIL image to RGB, resizes + center-crops to the model's
input size, and normalizes with the CIFAR-10 training statistics
(adjudicated M3 ruling: CIFAR10_MEAN/STD apply to ALL models, including
pretrained transfer backbones, so inference matches training exactly).
"""

import torch
from PIL import Image
from torchvision import transforms

from cnn_classifier.data.dataset import CIFAR10_MEAN, CIFAR10_STD


def preprocess_image(image: Image.Image, input_size: tuple) -> torch.Tensor:
    """Convert a PIL image to a normalized (1, C, H, W) tensor.

    Args:
        image: Input PIL image (any mode; non-RGB is converted).
        input_size: Target (H, W) for the model input.

    Returns:
        Float tensor of shape (1, 3, H, W) normalized with the
        CIFAR-10 mean/std statistics.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return pipeline(image).unsqueeze(0)
