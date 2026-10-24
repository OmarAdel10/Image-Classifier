"""Model registry and factory.

Maps model names to constructors so the rest of the codebase can build
models from configuration alone.
"""

import torch.nn as nn

from cnn_classifier.config import Config
from cnn_classifier.models.cnn_scratch import CNNScratch
from cnn_classifier.models.transfer import TransferModel

MODEL_REGISTRY = {
    "cnn_scratch": CNNScratch,
    "resnet18": lambda config: TransferModel("resnet18", config),
    "vgg16": lambda config: TransferModel("vgg16", config),
}


def get_model(name: str, config: Config) -> nn.Module:
    """Instantiate a model by registry name.

    Args:
        name: Model name (must exist in ``MODEL_REGISTRY``).
        config: Application config consumed by the model constructor.

    Returns:
        The instantiated ``nn.Module``.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {name} (available: {sorted(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[name](config)
