"""Safe checkpoint loading for inference (CT2 ruling).

All inference-side checkpoint reads go through this helper so every
``torch.load`` in the serving path uses ``weights_only=True`` with an
explicit ``map_location`` and fails with informative errors on
missing/corrupt files.
"""

from pathlib import Path
import json

import torch

# Keys inference needs from any checkpoint written by save_checkpoint.
_REQUIRED_KEYS = {"model_state", "config", "class_names"}


def _load_labels_sidecar(model_path: Path) -> list | None:
    """Try to load class names from sidecar labels.json next to the model checkpoint."""
    labels_path = model_path.with_name(model_path.stem + "_labels.json")
    if not labels_path.exists():
        return None
    try:
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and all(isinstance(name, str) for name in labels):
            return labels
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        pass
    return None


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
    if not isinstance(checkpoint, dict):
        raise ValueError(f"not a valid torch checkpoint: {path}")
    missing = _REQUIRED_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing required keys: {sorted(missing)}")
    return checkpoint


def load_checkpoint_with_sidecar(path: str | Path) -> dict:
    """Load a checkpoint and merge sidecar labels.json if available.

    This is the preferred entry point for inference. If a sidecar
    ``<model_stem>_labels.json`` exists next to the checkpoint, its
    class_names replace the checkpoint's embedded class_names.

    Args:
        path: Checkpoint file path (explicit; no default locations).

    Returns:
        The checkpoint dict with class_names possibly overridden by sidecar.
    """
    checkpoint = load_checkpoint(path)
    model_path = Path(path)
    sidecar_labels = _load_labels_sidecar(model_path)
    if sidecar_labels is not None:
        checkpoint["class_names"] = sidecar_labels
    return checkpoint
