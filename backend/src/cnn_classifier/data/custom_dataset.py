"""Custom image-folder dataset for user-provided class folders."""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torchvision import datasets

from cnn_classifier.config import _MAX_IMAGE_PIXELS

# Hardened allowlist: user folders are untrusted input at inference time.
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _safe_loader(path: str) -> Image.Image:
    """Hardened image loader for untrusted user folders.

    Rejects symlinked files, non-allowlisted extensions, and images whose
    pixel count exceeds the config cap (decompression-bomb guard), then
    converts to RGB.
    """
    p = Path(path)
    if p.is_symlink():
        raise ValueError(f"symlinked images are not allowed: {p}")
    if p.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported image extension: {p.suffix}")
    with Image.open(path) as img:
        if img.width * img.height > _MAX_IMAGE_PIXELS:
            raise ValueError(
                f"image exceeds pixel cap ({img.width}x{img.height} > "
                f"{_MAX_IMAGE_PIXELS} pixels): {p}"
            )
        return img.convert("RGB")


class CustomImageFolder(datasets.ImageFolder):
    """Loads images arranged as ``root/class_name/*.jpg``.

    Thin wrapper over ``torchvision.datasets.ImageFolder`` giving later
    tasks (inference over user folders) a stable, project-owned entry
    point. Uses a hardened loader: symlinks are rejected, extensions are
    restricted to jpg/jpeg/png, and oversized images (decompression
    bombs) raise ``ValueError``. Accepts an optional transform callable.
    """

    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        super().__init__(root=str(root), transform=transform, loader=_safe_loader)
