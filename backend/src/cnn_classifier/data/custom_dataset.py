"""Custom image-folder dataset for user-provided class folders."""

from pathlib import Path

from torchvision import datasets, transforms


class CustomImageFolder(datasets.ImageFolder):
    """Loads images arranged as ``root/class_name/*.jpg``.

    Thin wrapper over ``torchvision.datasets.ImageFolder`` giving later
    tasks (inference over user folders) a stable, project-owned entry
    point. Accepts an optional torchvision transform.
    """

    def __init__(self, root: str | Path, transform: transforms.Compose | None = None) -> None:
        super().__init__(root=str(root), transform=transform)
