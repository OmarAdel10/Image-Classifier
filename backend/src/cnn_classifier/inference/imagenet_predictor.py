"""ImageNet pre-trained predictor: load a torchvision model and run inference."""

from pathlib import Path
from typing import Literal

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from cnn_classifier.config import Config
from cnn_classifier.inference.predictor import Prediction

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_MODELS = Literal[
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
]


class ImageNetPredictor:
    """Runs a pre-trained ImageNet model on preprocessed image tensors."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list,
        device: str,
        input_size: tuple,
        *,
        model_name: str,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.device = device
        self.input_size = tuple(input_size)
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: Config,
        class_names: list | None = None,
    ) -> "ImageNetPredictor":
        """Load a pre-trained ImageNet model from torchvision."""
        weights_enum = cls._get_weights_enum(model_name)
        model_fn = getattr(models, model_name)
        model = model_fn(weights=weights_enum)

        # Use provided class names or default to ImageNet 1000 classes
        if class_names is None:
            class_names = cls._load_imagenet_labels(config)

        model = model.to(config.device)
        model.eval()

        # Verify output dimension matches class count
        probe_size = (min(config.input_size[0], 512), min(config.input_size[1], 512))
        with torch.no_grad():
            probe = model(torch.ones(1, 3, *probe_size, device=config.device))
        if len(class_names) != probe.shape[1]:
            raise ValueError(f"class count {len(class_names)} != model outputs {probe.shape[1]}")

        return cls(
            model,
            class_names,
            config.device,
            tuple(config.input_size),
            model_name=model_name,
            checkpoint_path=config.model_path_resolved,
        )

    @staticmethod
    def _get_weights_enum(model_name: str):
        """Get the default weights enum for a model."""
        # Map model names to their correct weights enum names (torchvision uses PascalCase)
        weights_map = {
            "resnet18": "ResNet18_Weights",
            "resnet34": "ResNet34_Weights",
            "resnet50": "ResNet50_Weights",
            "resnet101": "ResNet101_Weights",
            "resnet152": "ResNet152_Weights",
            "mobilenet_v2": "MobileNet_V2_Weights",
            "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
            "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
            "efficientnet_b0": "EfficientNet_B0_Weights",
            "efficientnet_b1": "EfficientNet_B1_Weights",
            "efficientnet_b2": "EfficientNet_B2_Weights",
            "vit_b_16": "ViT_B_16_Weights",
            "vit_b_32": "ViT_B_32_Weights",
            "vit_l_16": "ViT_L_16_Weights",
            "vit_l_32": "ViT_L_32_Weights",
        }
        weights_attr = weights_map.get(model_name.lower())
        if weights_attr is None:
            raise ValueError(f"Unsupported model for ImageNet weights: {model_name}")
        try:
            weights_cls = getattr(models, weights_attr)
            return weights_cls.DEFAULT
        except AttributeError:
            raise ValueError(f"No default weights found for model: {model_name}")

    @staticmethod
    def _load_imagenet_labels(config: Config) -> list[str]:
        """Load ImageNet 1000 class labels."""
        labels_path = config.labels_path_resolved
        if labels_path.exists():
            import json
            with open(labels_path) as f:
                labels = json.load(f)
            if isinstance(labels, list) and len(labels) == 1000:
                return labels
        raise RuntimeError(
            f"ImageNet labels not found at {labels_path}. "
            "Run the label download script or ensure checkpoints/imagenet_labels.json exists."
        )

    def predict(self, tensor: torch.Tensor) -> list[Prediction]:
        """Classify one preprocessed image tensor."""
        if tensor.dim() != 4 or tensor.shape[0] != 1:
            raise ValueError(
                f"expected a single-image tensor (1, C, H, W), got shape {tuple(tensor.shape)}"
            )
        if tuple(tensor.shape[1:]) != (3, *self.input_size):
            raise ValueError(
                f"expected tensor (1, 3, {self.input_size[0]}, {self.input_size[1]}) "
                f"matching checkpoint input size {self.input_size}, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(f"expected a float32 tensor, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite input tensor values")
        tensor = tensor.to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        if not torch.isfinite(probabilities).all():
            raise ValueError("model produced non-finite probabilities")
        ranked = torch.argsort(probabilities, descending=True)
        return [
            Prediction(self.class_names[int(index)], float(probabilities[int(index)]))
            for index in ranked
        ]

    def get_top_k(self, tensor: torch.Tensor, k: int = 5) -> list:
        """Get top-k predictions."""
        all_preds = self.predict(tensor)
        return all_preds[:k]


def get_imagenet_transform(input_size: tuple) -> transforms.Compose:
    """Get the standard ImageNet preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
