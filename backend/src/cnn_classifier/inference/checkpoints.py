"""Safe checkpoint loading for inference (CT2 ruling).

All inference-side checkpoint reads go through this helper so every
``torch.load`` in the serving path uses ``weights_only=True`` with an
explicit ``map_location`` and fails with informative errors on
missing/corrupt files.
"""

from pathlib import Path

import torch

# Keys inference needs from any checkpoint written by save_checkpoint.
_REQUIRED_KEYS = {"model_state", "config", "class_names"}


def load_checkpoint(path: str | Path) -> dict:
    """Load a training checkpoint safely.

    Args:
        path: Checkpoint file path (explicit; no default locations).

    Returns:
        The checkpoint dict (model_state, config, class_names, ...).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a loadable torch checkpoint or
            is missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"not a valid torch checkpoint: {path}") from exc
    missing = _REQUIRED_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing required keys: {sorted(missing)}")
    return checkpoint
