"""FastAPI dependency providers (Task 6).

Testing utilities:
    clear_predictor_cache() -- Clears the lru_cache on get_predictor() to allow
    reloading a different checkpoint between tests. Call in test fixtures or
    test setup/teardown when you need a fresh predictor instance.
"""

from functools import lru_cache
from pathlib import Path

from cnn_classifier.config import Config
from cnn_classifier.config import get_config as _get_config
from cnn_classifier.inference.predictor import Predictor
from cnn_classifier.inference.imagenet_predictor import ImageNetPredictor


class ModelUnavailableError(RuntimeError):
    """The configured checkpoint could not be loaded."""


def get_config() -> Config:
    """Return the process-wide Config singleton."""
    return _get_config()


def _is_imagenet_model(model_path: Path) -> bool:
    """Check if the model path points to an ImageNet pre-trained model."""
    return "imagenet" in model_path.name.lower()


@lru_cache(maxsize=1)
def get_predictor():
    """Load the configured model (CIFAR-10 checkpoint or ImageNet pre-trained) once."""
    config = get_config()
    model_path = config.model_path_resolved
    
    try:
        if _is_imagenet_model(model_path):
            # Load ImageNet pre-trained model
            from cnn_classifier.inference.imagenet_predictor import ImageNetPredictor
            return ImageNetPredictor.from_pretrained(
                model_name="resnet18",
                config=config,
            )
        else:
            # Load CIFAR-10 checkpoint
            return Predictor.from_checkpoint(model_path, config)
    except Exception as exc:  # Boundary: any load failure means no model.
        raise ModelUnavailableError("model unavailable") from exc


class ModelUnavailableError(RuntimeError):
    """The configured checkpoint could not be loaded."""


def get_config() -> Config:
    """Return the process-wide Config singleton."""
    return _get_config()


def clear_predictor_cache() -> None:
    """Clear the predictor singleton (tests/hot-reload)."""
    get_predictor.cache_clear()
