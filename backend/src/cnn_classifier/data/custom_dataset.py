"""Custom image-folder dataset for user-provided class folders."""

import os
import stat
from collections.abc import Callable
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torchvision import datasets

from cnn_classifier.config import _MAX_IMAGE_PIXELS

# Hardened allowlist: user folders are untrusted input at inference time.
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _is_regular_file(path: str) -> bool:
    """Return True only for regular files (never FIFOs, devices, sockets)."""
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def _safe_loader(path: str) -> Image.Image:
    """Hardened image loader for untrusted user folders.

    Rejects symlinked files, non-allowlisted extensions, special files
    (FIFOs and devices block ``Image.open`` indefinitely), unreadable
    or truncated payloads, decompression bombs, and images whose pixel
    count exceeds the config cap, then converts to RGB. Load-time
    defense-in-depth: scanning already filters by suffix, file type,
    and containment.
    """
    p = Path(path)
    if p.is_symlink():
        raise ValueError(f"symlinked images are not allowed: {p}")
    if p.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported image extension: {p.suffix}")
    if not _is_regular_file(path):
        raise ValueError(f"not a regular image file: {p}")
    try:
        with Image.open(path) as img:
            if img.width * img.height > _MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"image exceeds pixel cap ({img.width}x{img.height} > "
                    f"{_MAX_IMAGE_PIXELS} pixels): {p}"
                )
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"unreadable image file: {p}") from exc


class CustomImageFolder(datasets.ImageFolder):
    """Loads images arranged as ``root/class_name/*.jpg``.

    Thin wrapper over ``torchvision.datasets.ImageFolder`` giving later
    tasks (inference over user folders) a stable, project-owned entry
    point. Hardening for untrusted folders: scan-time rejection of
    symlinked directories (torchvision walks class dirs with
    ``followlinks=True``, so a symlinked dir can hang init on a cycle
    or pull in outside-root images), regular-file + extension scan
    filter (jpg/jpeg/png only, so ``len()`` matches loadable files and
    stray formats or special files never abort iteration), post-init
    containment check (no sample may resolve outside ``root``), plus a
    hardened load-time loader (symlink rejection, pixel cap, RGB
    convert, unreadable-file ``ValueError``). Accepts an optional
    transform callable.
    """

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """Reject symlinked directories before the walk can cycle on them.

        ``os.walk(followlinks=False)`` never descends into symlinked
        dirs, so this scan is itself cycle-safe while still surfacing
        every symlinked directory anywhere under ``directory``.
        """
        for dirpath, dirnames, _ in os.walk(directory, followlinks=False):
            for name in dirnames:
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    raise ValueError(f"symlinked class directory not allowed: {full}")
        return super().find_classes(directory)

    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        super().__init__(
            root=str(root),
            transform=transform,
            loader=_safe_loader,
            is_valid_file=lambda p: Path(p).suffix.lower() in _ALLOWED_SUFFIXES
            and _is_regular_file(p),
        )
        real_root = Path(root).resolve()
        for sample_path, _ in self.samples:
            resolved = Path(sample_path).resolve()
            if not resolved.is_relative_to(real_root):
                raise ValueError(f"sample escapes dataset root: {sample_path}")
