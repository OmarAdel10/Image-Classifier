"""CIFAR-10 dataloaders with augmentation and stratified train/val split.

Consumes ``Config`` (input_size, batch_size, num_workers, seed) and builds
three ``DataLoader`` objects (train, val, test) over torchvision CIFAR-10.

The train split applies augmentation (RandomCrop with padding, horizontal
flip) while val/test only apply ToTensor + Normalize, so val metrics are
computed on unaugmented data. The train/val split is stratified per class
and seeded for reproducibility.

Seeding policy: the train DataLoader's shuffle order is seeded from
``config.seed`` via a dedicated ``torch.Generator``. Reproducibility of
augmentation depends on the worker regime: with ``num_workers=0`` the
global torch RNG governs augmentation, so ``torch.manual_seed(seed)``
before iterating gives full reproducibility (the training entry point
is responsible for this; tests cover this regime only). With
``num_workers>0`` each worker derives its augmentation seed from the
loader's base seed, and ``torch.manual_seed`` does NOT reach workers.
Note: each call builds a fresh ``Generator`` from ``config.seed``, so
recreating loaders replays the identical first-epoch shuffle — create
loaders once and reuse them across epochs.
"""

import random
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cnn_classifier.config import Config, _get_project_root

# Standard CIFAR-10 channel statistics (RGB).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _build_train_transform(input_size: tuple[int, int]) -> transforms.Compose:
    """Augmented transform for training: crop + flip + normalize."""
    return transforms.Compose(
        [
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _build_eval_transform() -> transforms.Compose:
    """Plain transform for val/test: ToTensor + Normalize only."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _stratified_split(
    targets: Sequence[int],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split dataset indices into stratified (train, val) lists.

    Each class contributes ``val_ratio`` of its samples to val, so the
    class distribution is preserved in both splits. Deterministic for a
    given seed.
    """
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    by_class: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        by_class.setdefault(target, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for target in sorted(by_class):
        indices = by_class[target]
        rng.shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return sorted(train_idx), sorted(val_idx)


def get_dataloaders(
    config: Config,
    data_dir: str | Path | None = None,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build CIFAR-10 train/val/test dataloaders from config.

    Downloads CIFAR-10 into ``data_dir`` (defaults to ``<project_root>/data``)
    if not already present. Train/val come from the 50k train set via a
    stratified split; test is the 10k official test set.
    """
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    if tuple(config.input_size) != (32, 32):
        raise ValueError(
            f"CIFAR-10 loaders require input_size=(32, 32), got {tuple(config.input_size)}"
        )

    if data_dir is None:
        data_dir = _get_project_root() / "data"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_full = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=_build_train_transform(config.input_size),
    )
    val_full = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=_build_eval_transform(),
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=_build_eval_transform(),
    )

    train_idx, val_idx = _stratified_split(
        train_full.targets, val_ratio=val_ratio, seed=config.seed
    )
    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader
