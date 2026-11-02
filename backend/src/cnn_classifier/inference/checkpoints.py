"""Safe checkpoint loading for inference (CT2 ruling).

All inference-side checkpoint reads go through this helper so every
``torch.load`` in the serving path uses ``weights_only=True`` with an
explicit ``map_location`` and fails with informative errors on
missing/corrupt files.
"""

import hashlib
import json
import stat
from pathlib import Path

import torch

# Keys inference needs from any checkpoint written by save_checkpoint.
_REQUIRED_KEYS = {"model_state", "config", "class_names"}


def _verify_sha256_sidecar(model_path: Path) -> None:
    """Verify the checkpoint against its SHA256 sidecar (IC-32).

    The sidecar file is ``<model_stem>.sha256`` next to the checkpoint.
    If the sidecar exists, its content must match the computed SHA256
    of the checkpoint file. If the sidecar is missing, the check is
    skipped (backward compatibility for checkpoints created before IC-32).
    This provides integrity verification to mitigate supply-chain and
    tampering risks (CWE-345).
    """
    sha256_path = model_path.with_suffix(model_path.suffix + ".sha256")
    if not sha256_path.exists():
        return  # No sidecar -> skip (backward compat)
    try:
        expected = sha256_path.read_text().strip()
    except OSError:
        return  # Unreadable sidecar -> skip (backward compat)
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"malformed SHA256 sidecar: {sha256_path}")
    hasher = hashlib.sha256()
    with model_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if not _constant_time_compare(actual, expected):
        raise ValueError(f"checkpoint integrity check failed: {model_path}")


def _constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    if len(a) != len(b):
        return False
    result = 0
    for ca, cb in zip(a, b):
        result |= ord(ca) ^ ord(cb)
    return result == 0


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
    # Regular-file gate (IC-3): path.stat() follows symlinks, so this rejects
    # directories/FIFOs/dev nodes while accepting symlinks that target regular
    # files. weights_only=True is the operative deserialization control.
    stat_result = path.stat()
    if not stat.S_ISREG(stat_result.st_mode):
        raise ValueError(f"not a regular file: {path}")
    # SHA256 sidecar integrity verification (IC-32): verifies checkpoint
    # hasn't been tampered with since training. Sidecar missing = skip
    # (backward compat for pre-IC-32 checkpoints).
    _verify_sha256_sidecar(path)
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
