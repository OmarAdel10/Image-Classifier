"""FastAPI dependency providers (Task 6).

Testing utilities:
    clear_predictor_cache() -- Clears the lru_cache on get_predictor() to allow
    reloading a different checkpoint between tests. Call in test fixtures or
    test setup/teardown when you need a fresh predictor instance.
"""

from functools import lru_cache

from cnn_classifier.config import Config
from cnn_classifier.config import get_config as _get_config
from cnn_classifier.inference.predictor import Predictor


class ModelUnavailableError(RuntimeError):
    """The configured checkpoint could not be loaded."""


def get_config() -> Config:
    """Return the process-wide Config singleton."""
    return _get_config()


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load the configured checkpoint once and reuse it for all requests.

    Raises:
        ModelUnavailableError: If the checkpoint is missing, corrupt, or
            otherwise cannot be loaded. The original error is chained as
            the cause for server-side logs.
    """
    config = get_config()
    try:
        return Predictor.from_checkpoint(config.model_path_resolved, config)
    except Exception as exc:  # Boundary: any load failure means no model.
        raise ModelUnavailableError("model unavailable") from exc


def clear_predictor_cache() -> None:
    """Clear the predictor singleton (tests/hot-reload)."""
    get_predictor.cache_clear()
