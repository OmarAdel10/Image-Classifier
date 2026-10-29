"""FastAPI dependency providers (Task 6)."""

from functools import lru_cache

from cnn_classifier.config import Config
from cnn_classifier.config import get_config as _get_config
from cnn_classifier.inference.predictor import Predictor


def get_config() -> Config:
    """Return the process-wide Config singleton."""
    return _get_config()


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load the configured checkpoint once and reuse it for all requests.

    Raises:
        FileNotFoundError: If the configured checkpoint does not exist.
        ValueError: If the checkpoint is corrupt or malformed.
    """
    config = get_config()
    return Predictor.from_checkpoint(config.model_path_resolved, config)


def clear_predictor_cache() -> None:
    """Clear the predictor singleton (tests/hot-reload)."""
    get_predictor.cache_clear()
