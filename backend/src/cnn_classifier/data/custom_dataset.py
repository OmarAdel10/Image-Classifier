"""Custom image-folder dataset for user-provided class folders."""

from collections.abc import Callable
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torchvision import datasets

from cnn_classifier.config import _MAX_IMAGE_PIXELS

# Hardened allowlist: user folders are untrusted input at inference time.
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _safe_loader(path: str) -> Image.Image:
    """Hardened image loader for untrusted user folders.

    Rejects symlinked files, non-allowlisted extensions, unreadable
    payloads, and images whose pixel count exceeds the config cap
    (decompression-bomb guard), then converts to RGB. Load-time
    defense-in-depth: scanning already filters by suffix and containment.
    """
    p = Path(path)
    if p.is_symlink():
        raise ValueError(f"symlinked images are not allowed: {p}")
    if p.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported image extension: {p.suffix}")
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"unreadable image file: {p}") from exc
    with img:
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
    point. Hardening for untrusted folders: scan-time extension filter
    (jpg/jpeg/png only, so ``len()`` matches loadable files and stray
    formats never abort iteration), post-init containment check (no
    sample may resolve outside ``root``, blocking symlinked class
    directories), plus a hardened load-time loader (symlink rejection,
    pixel cap, RGB convert, unreadable-file ``ValueError``). Accepts an
    optional transform callable.
    """

    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        super().__init__(
            root=str(root),
            transform=transform,
            loader=_safe_loader,
            is_valid_file=lambda p: Path(p).suffix.lower() in _ALLOWED_SUFFIXES,
        )
        real_root = Path(root).resolve()
        for sample_path, _ in self.samples:
            resolved = Path(sample_path).resolve()
            if not resolved.is_relative_to(real_root):
                raise ValueError(f"sample escapes dataset root: {sample_path}")
